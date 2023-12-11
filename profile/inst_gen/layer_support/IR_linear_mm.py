import os
import numpy as np
from utils import tools
from utils.config_generator import CFG
from inst_gen import isa

# compile linear mm layer
def generate_layer_inst(
        layer_ir: dict(), 
        fuse_layer_ir: dict(),
        cfg: CFG, 
        layer_first_LD_wait: list(), 
        layer_last_ST_release: list(), 
    ):
    fuse_misc_flag = fuse_layer_ir is not None
    if cfg.DEBUG:
        if fuse_misc_flag:
            print(f"Generate inst for fusing {layer_ir['type']} layer {layer_ir['name']} and {fuse_layer_ir['fuse_layer_num']} {fuse_layer_ir['type']} layer {fuse_layer_ir['name']}")
        else:
            print(f"Generate inst for {layer_ir['type']} layer {layer_ir['name']}")
    N, M, K, N_block_num, M_block_num, K_block_num, bias_flag, relu_flag, sparse_flag, param_weight_idx, param_bias_idx, param_meta_idx, param_fsb_idx = tools.get_linear_layer_info(layer_ir, cfg, "MM")
    if fuse_misc_flag: # 融合linear和silu或者eltwise层
        MISC_BATCH, MISC_M, MISC_K, MISC_K_block_num, MISC_eltwise_flag, MISC_operation_name = tools.get_misc_layer_info(fuse_layer_ir, cfg)
        if MISC_eltwise_flag:
            assert MISC_BATCH in (1, 32)
            assert MISC_BATCH * MISC_M * MISC_K == M * N
            assert layer_ir["output"][0]["name"] in (fuse_layer_ir["input"][0]["name"], fuse_layer_ir["input"][1]["name"])
            assert MISC_operation_name in ("eltwise_add", "eltwise_mul")
        else:
            assert MISC_K_block_num == N_block_num
            assert MISC_BATCH == 1
            assert MISC_M == M
            assert MISC_K == N
            assert layer_ir["output"][0]["name"] == fuse_layer_ir["input"][0]["name"]
            assert MISC_operation_name == "silu"
    last_save_length = M - (M_block_num - 1) * cfg.FSB_BLOCK_SIZE # 考虑M方向不是16整数倍的情况，最后store的时候length变小
    # input: M * K, weight: K * N, Out: M * N, bias: 1 * N
    

    # FSB info: 可能取值2，4，8，16，代表N:16稀疏，只支持这四种，其中16代表稠密
    fsb_info = np.ones((K_block_num, N_block_num), dtype=np.int8) * cfg.FSB_BLOCK_SIZE # 全dense
    assert fsb_info.dtype == np.int8
    # 验证只有这四种数字
    assert np.sum(fsb_info == 2) + np.sum(fsb_info == 4) + np.sum(fsb_info == 8) + np.sum(fsb_info == 16) == N_block_num * K_block_num
    assert fsb_info.shape == (K_block_num, N_block_num)

    each_dense_block_size_B = cfg.FSB_BLOCK_SIZE * cfg.FSB_BLOCK_SIZE # 一个dense FSB块的大小

    # 验证A Buffer满足大小，由于是固定A换B，所以A不需要流水
    A_buffer_addr_of_a_K_block = 1 # 切块后每列K block正好对应A buffer中bank的一个地址
    A_buffer_addr_of_a_K_block_line = K_block_num * A_buffer_addr_of_a_K_block # 为了计算第一条MM，A Buffer中需要的地址长度
    assert cfg.A_BUFFER_DEPTH >= A_buffer_addr_of_a_K_block_line

    # 根据buffer大小对矩阵进行划分，需要考虑B buffer、meta、FSB、Bias Buffer
    # 计算B Buffer能容纳的块数，从而用于处理依赖
    B_block_2_16_sparse_size_B = cfg.FSB_BLOCK_SIZE * 2 # 代表一个块如果是2:16稀疏的数据大小
    B_buffer_addr_of_a_block_sparse_2 = B_block_2_16_sparse_size_B // cfg.B_BUFFER_MM_BANK_WIDTH_B # 一个块如果是2:16占B Buffer的地址数
    assert B_buffer_addr_of_a_block_sparse_2 * cfg.B_BUFFER_MM_BANK_WIDTH_B == B_block_2_16_sparse_size_B # 验证可以整除
    B_buffer_addr_of_a_K_block_line = (np.sum(fsb_info, axis = 0) // 2) * B_buffer_addr_of_a_block_sparse_2 # 所有K列占的B Buffer地址数
    max_B_buffer_addr_of_a_K_block_line = np.max(B_buffer_addr_of_a_K_block_line) # 最坏情况占的B buffer地址数
    # 这里有两种情况，第一种是B Buffer足够大，可以实现放下所有K的时候的流水，这种情况下直接流水即可
    # 第二种情况是B Buffer不足以放下两次计算实现流水，甚至一次都放不下（在FFN中可能出现），这种情况下需要多条MM连续计算再输出，因此需要强制流水。
    # 这种情况下需要计算需要几条MM才能实现input_weights_dependency_len >= 2
    tgt_weights_dependency_len = 2
    target_max_B_buffer_addr_of_a_K_block_line = cfg.B_BUFFER_DEPTH // tgt_weights_dependency_len # 最坏情况下，B buffer中可以放下多少次计算所需的数据
    split_K_min_num = tools.ceil(max_B_buffer_addr_of_a_K_block_line, target_max_B_buffer_addr_of_a_K_block_line) # 至少需要切分MM的数量，这是最好情况，即FSB的分布是均匀的
    split_K_num = split_K_min_num # 目前切分的数量

    # 计算meta buffer能容纳的块数，从而用于处理依赖
    meta_block_2_16_sparse_size_B = cfg.FSB_BLOCK_SIZE # 代表一个块的meta index如果是2:16稀疏的数据大小
    assert meta_block_2_16_sparse_size_B * 2 == B_block_2_16_sparse_size_B # 由于index是4bit，数据是8bit，所以正好是数据的一半
    meta_buffer_addr_of_a_block_sparse_2 = meta_block_2_16_sparse_size_B // cfg.META_BUFFER_BANK_WIDTH_B # 一个块如果是2:16占meta Buffer的地址数
    assert meta_buffer_addr_of_a_block_sparse_2 * cfg.META_BUFFER_BANK_WIDTH_B == meta_block_2_16_sparse_size_B # 验证可以整除

    def get_max_buffer_addr_of_a_K_block_line(split_num, buffer_addr_of_a_block_sparse_2): # 计算K方向切成split_num块以后，最大一块占的B buffer地址
        split_K_start_end_K_block_id_list = tools.tiling_to_list(0, K_block_num, K_block_num // split_num) # 切成split_num块
        max_buffer_addr_of_a_K_block_line = -1
        for (split_K_start_K_block_id, split_K_end_K_block_id) in split_K_start_end_K_block_id_list:
            buffer_addr_of_a_K_block_line = (np.sum(fsb_info[split_K_start_K_block_id: split_K_end_K_block_id, :], axis = 0) // 2) * buffer_addr_of_a_block_sparse_2 # 每块切分的K列部分占的B Buffer地址数
            max_buffer_addr_of_a_K_block_line = max(np.max(buffer_addr_of_a_K_block_line), max_buffer_addr_of_a_K_block_line) # 最坏情况占的B buffer地址数
        return max_buffer_addr_of_a_K_block_line, split_K_start_end_K_block_id_list

    while True: # 最差情况是每个K block都用一条MM
        assert split_K_num <= K_block_num
        max_B_buffer_addr_of_a_K_block_line, split_K_start_end_K_block_id_list = get_max_buffer_addr_of_a_K_block_line(split_K_num, B_buffer_addr_of_a_block_sparse_2)
        max_meta_buffer_addr_of_a_K_block_line, split_K_start_end_K_block_id_list = get_max_buffer_addr_of_a_K_block_line(split_K_num, meta_buffer_addr_of_a_block_sparse_2)
        input_weights_dependency_len = cfg.B_BUFFER_DEPTH // max_B_buffer_addr_of_a_K_block_line
        input_meta_dependency_len = cfg.META_BUFFER_DEPTH // max_meta_buffer_addr_of_a_K_block_line # 最坏情况下，meta buffer中可以放下多少次计算所需的数据
        if input_weights_dependency_len >= 2 and input_meta_dependency_len >= 2: # 满足流水要求
            break
        else:
            split_K_num += 1 # 有块不满足要求
    split_K_num = len(split_K_start_end_K_block_id_list) # 有可能不同，按实际切块数赋值

    input_weights_dependency_len *= cfg.B_BUFFER_HBM_CHANNEL_NUM # 每K列一条LD
    input_meta_dependency_len *= cfg.META_BUFFER_HBM_CHANNEL_NUM # 每K列meta一条LD

    # 计算FSB能容纳的块数，从而用于处理依赖
    fsb_block_size_B = 1 # 一个块的FSB就是1个Byte
    meta_buffer_addr_of_a_K_block_line = tools.ceil(K_block_num, split_K_num) * fsb_block_size_B // cfg.FSB_BUFFER_BANK_WIDTH_B # 所有K列占的meta buffer地址数
    assert meta_buffer_addr_of_a_K_block_line * cfg.FSB_BUFFER_BANK_WIDTH_B == tools.ceil(K_block_num, split_K_num) * fsb_block_size_B # 验证可以整除
    input_fsb_dependency_len = cfg.META_BUFFER_DEPTH // meta_buffer_addr_of_a_K_block_line # 最坏情况下，fsb buffer中可以放下多少次计算所需的数据
    assert input_fsb_dependency_len >= 2 # 能放下多于一行才能流水
    input_fsb_dependency_len *= cfg.FSB_BUFFER_HBM_CHANNEL_NUM # 每K列fsb一条LD

    input_B_dependency_len = min(input_weights_dependency_len, input_meta_dependency_len, input_fsb_dependency_len) # 三者取最小值

    # bias buffer应该足够容纳所有数据
    bias_size_B = N * 4 # bias is int32, 4Bytes
    assert bias_size_B <= cfg.BIAS_BUFFER_DEPTH * cfg.BIAS_BUFFER_BANK_WIDTH_B

    # 计算Global buffer能容纳的块数，从而用于处理依赖
    # 由于输出的顺序是N方向（输出的K方向），所以计算方法和A buffer不同
    out_buffer_addr_of_a_N_block = 1 # 输出的K实际上就是输入的N方向
    out_dependency_len = cfg.GLOBAL_BUFFER_DEPTH // out_buffer_addr_of_a_N_block # Global Buffer中可以放下多少次计算所需的数据
    assert out_dependency_len >= 2 # 输出满足流水要求
    # out_dependency_len代表global buffer放不下的时候算了几次MM（此时ST的次数是out_dependency_len * cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM）
    
    """
    每次M方向计算128
    生成指令: 固定A Buffer (Activation), 动B (Weights)
    生成的时候先不管这一层前后的依赖，等到这一层指令全部生成完再添加
    需要注意每次M_block_tiling_num组与组之前, 最后一条MM要发送一个release LD给下一组的第一条LD Activation

    Load 本层所有 bias * 1
    for M方向每次M_block_tiling_num个block, 走完这一组M
        Load Activation: LD * 8 for each channel
        for N方向每次一K列FSB块, 走完所有N
            for split_K, 走完所有K
                Load FSB * 1 (不跨HBM Channel)
                Load Weights * 1 (不跨HBM Channel)
                Load Meta * 1 (不跨HBM Channel)
                MM * 1 (最后一次才输出)
            if fuse_misc_flag:
                MISC * 1
            Store Out * 8 (每个Channel一个)
    """
    inst_list = list()
    M_block_tiling_num = cfg.MM_PARALLEL_M // cfg.FSB_BLOCK_SIZE # 每次计算的M方向块数
    # M方向根据M_block_tiling_num切分，注意M方向最后行的块可能不满
    M_block_tiling_id_list = tools.tiling_to_list(0, M_block_num, M_block_tiling_num)
    if bias_flag:
        # Load bias * 1
        LD_bias_inst = isa.generate_LD_inst(
            LD_wait                 = [],
            LD_release              = [],
            LD_1d_length            = bias_size_B,
            LD_hbm_addr             = layer_ir["param"][param_bias_idx]["addr"],
            LD_bank_addr            = 0, # 因为是固定A换B，所以A buffer中的地址固定为0
            LD_target_bank_name     = "bias buffer",
            LD_hbm_channel_id       = 0 % cfg.BIAS_BUFFER_HBM_CHANNEL_NUM,
            LD_cross_hbm_channel    = False,
            LD_parallel_channel_num = 1,
        )
        inst_list.append(LD_bias_inst)

    for (start_M_block_id, end_M_block_id) in M_block_tiling_id_list:
        # Load A * 8 for each channel
        LD_A_inst_list = list()
        for M_block_id in range(start_M_block_id, end_M_block_id):
            LD_A_block_hbm_src_addr = tools.get_dense_matrix_hbm_addr(
                addr_base               = layer_ir["input"][0]["addr"], # Activation
                K_block_num             = K_block_num,
                K_block_id              = 0,
                M_or_N_block_id         = M_block_id,
                cfg                     = cfg,
            )
            LD_A_inst = isa.generate_LD_inst(
                LD_wait                 = [],
                LD_release              = [],
                LD_1d_length            = each_dense_block_size_B * K_block_num,
                LD_hbm_addr             = LD_A_block_hbm_src_addr,
                LD_bank_addr            = 0, # 因为是固定A换B，所以A buffer中的地址固定为0
                LD_target_bank_name     = "A buffer",
                LD_hbm_channel_id       = M_block_id % cfg.A_BUFFER_HBM_CHANNEL_NUM,
                LD_cross_hbm_channel    = False,
                LD_parallel_channel_num = cfg.A_BUFFER_HBM_CHANNEL_NUM,
            )
            LD_A_inst_list.append(LD_A_inst)
        LD_A_wait_previous_MM_flag = (start_M_block_id > 0) # 第一组不需要等待，其他组需要等待前一组的MM用完
        LD_A_inst_list = isa.set_first_wait_last_release(
            inst_list       = LD_A_inst_list,
            first_wait      = ["MM"] if LD_A_wait_previous_MM_flag else [],
            last_release    = [],
            first_inst_type = "LD",
            last_inst_type  = "LD",
        )
        inst_list.extend(LD_A_inst_list)
        
        # 一共需要的LD_B, MM, ST的组数，用于处理依赖关系
        # 本质上就是每个N_block都LD，MM，ST一次
        # LD_B_group: LD FSB + LD meta + LD weights
        total_LD_B_group_cal = 0
        total_MM_cal = 0 # 和LD可能发生依赖的MM数量（即所有MM的数量）
        total_MM_output_cal = 0 # 和ST可能发生依赖的MM数量（即输出MM的数量）
        total_ST_group_cal = 0
        for N_block_id in range(N_block_num):
            for (split_K_start_K_block_id, split_K_end_K_block_id) in split_K_start_end_K_block_id_list:
                total_LD_B_group_cal += 1
                total_MM_cal += 1 # 和LD可能发生依赖的MM数量（即所有MM的数量）
            total_MM_output_cal += 1 # 注意！output MM才算MM的有效次数
            total_ST_group_cal += 1

        # 目前的LD_B, MM, ST的组数，用于处理依赖关系
        now_LD_B_group_cal = 0
        now_MM_cal = 0 # 和LD可能发生依赖的MM数量（即所有MM的数量）
        now_MM_output_cal = 0 # 和ST可能发生依赖的MM数量（即输出MM的数量）
        now_ST_group_cal = 0

        now_block_weights_bank_addr = 0
        now_block_weights_hbm_addr = layer_ir["param"][param_weight_idx]["addr"]
        now_block_meta_bank_addr = 0
        now_block_fsb_bank_addr = 0
        if sparse_flag:
            now_block_meta_hbm_addr = layer_ir["param"][param_meta_idx]["addr"]
            now_block_fsb_hbm_addr = layer_ir["param"][param_fsb_idx]["addr"]

        for N_block_id in range(N_block_num):
            now_MM_output_cal += 1 # 从1开始
            now_ST_group_cal += 1 # 从1开始

            MM_out_block_bank_addr = N_block_id * out_buffer_addr_of_a_N_block # MM输出这个块的bank地址

            for (split_K_start_K_block_id, split_K_end_K_block_id) in split_K_start_end_K_block_id_list:
                now_MM_cal += 1 # 从1开始
                now_LD_B_group_cal += 1 # 从1开始

                first_split_K_group_flag = split_K_start_K_block_id == 0 # 是否第一组K切分，用于处理依赖
                last_split_K_group_flag = split_K_end_K_block_id == K_block_num # 是否最后一组K切分，用于处理依赖
                split_K_block_num = split_K_end_K_block_id - split_K_start_K_block_id
                
                this_N_block_col_fsb_info = fsb_info[split_K_start_K_block_id: split_K_end_K_block_id, N_block_id] # (split_K_block_num,)

                weights_N_block_col_size_B = (np.sum(this_N_block_col_fsb_info) // 2) * B_block_2_16_sparse_size_B # 这一K行的weights大小
                assert np.sum(this_N_block_col_fsb_info) % 2 == 0 # 验证可以整除
                weights_N_block_col_addr_increment = weights_N_block_col_size_B // cfg.B_BUFFER_MM_BANK_WIDTH_B # weights buffer 地址增量
                assert weights_N_block_col_size_B % cfg.B_BUFFER_MM_BANK_WIDTH_B == 0 # 验证可以整除

                meta_N_block_col_size_B = weights_N_block_col_size_B // 2 # 这一K行的meta大小，恰好是数据的一半大小
                assert weights_N_block_col_size_B % 2 == 0 # 验证可以整除
                meta_N_block_col_addr_increment = meta_N_block_col_size_B // cfg.META_BUFFER_BANK_WIDTH_B # meta buffer 地址增量
                assert meta_N_block_col_size_B % cfg.META_BUFFER_BANK_WIDTH_B == 0 # 验证可以整除

                fsb_N_block_col_size_B = split_K_block_num * fsb_block_size_B
                fsb_N_block_col_addr_increment = fsb_N_block_col_size_B // cfg.FSB_BUFFER_BANK_WIDTH_B # FSB buffer 地址增量
                assert fsb_N_block_col_size_B % cfg.FSB_BUFFER_BANK_WIDTH_B == 0 # 验证可以整除

                assert now_LD_B_group_cal == now_MM_cal # LD_B_group_cal和MM_cal一一对应，用于处理MM和LD之间的依赖
                LD_B_wait_MM_flag = (now_LD_B_group_cal > input_B_dependency_len) # LD wait MM / MM release LD 的条件是B/FSB/meta某个 buffer满了，需要等MM用完buffer里的数据才能启动下一次LD
                tmp_MM_release_LD_B_flag = (now_MM_cal + input_B_dependency_len <= total_LD_B_group_cal)
                tmp_MM_release_next_group_LD_A_flag = (now_MM_cal == total_MM_cal) and (end_M_block_id < M_block_num) and last_split_K_group_flag # 这一组和下一组的依赖，条件是本组的最后一条MM且存在下一组
                assert not (tmp_MM_release_LD_B_flag and tmp_MM_release_next_group_LD_A_flag) # 两个条件不可能同时满足
                MM_release_LD_flag = tmp_MM_release_LD_B_flag or tmp_MM_release_next_group_LD_A_flag
                MM_wait_ST_flag = (now_MM_output_cal > out_dependency_len) and last_split_K_group_flag and (not fuse_misc_flag) # MM wait ST / ST release MM 的条件是out buffer满了，需要等ST存完global buffer里的数据才能启动下一次MM，【由于可能有K切分，所以需要判断K切分组数是否最后一组】
                MM_wait_MISC_flag = (now_MM_output_cal > out_dependency_len) and last_split_K_group_flag and fuse_misc_flag

                LD_B_group_inst_list = list()
                if sparse_flag:
                    # Load FSB
                    LD_fsb_inst = isa.generate_LD_inst(
                        LD_wait                 = [],
                        LD_release              = [],
                        LD_1d_length            = fsb_N_block_col_size_B,
                        LD_hbm_addr             = now_block_fsb_hbm_addr,
                        LD_bank_addr            = now_block_fsb_bank_addr % cfg.FSB_BUFFER_DEPTH,
                        LD_target_bank_name     = "FSB buffer",
                        LD_hbm_channel_id       = N_block_id % cfg.FSB_BUFFER_HBM_CHANNEL_NUM, # 0
                        LD_cross_hbm_channel    = False,
                        LD_parallel_channel_num = 1,
                    )
                    LD_B_group_inst_list.append(LD_fsb_inst)

                    # Load meta
                    LD_meta_inst = isa.generate_LD_inst(
                        LD_wait                 = [],
                        LD_release              = [],
                        LD_1d_length            = meta_N_block_col_size_B,
                        LD_hbm_addr             = now_block_meta_hbm_addr,
                        LD_bank_addr            = now_block_meta_bank_addr % cfg.META_BUFFER_DEPTH,
                        LD_target_bank_name     = "meta buffer",
                        LD_hbm_channel_id       = N_block_id % cfg.META_BUFFER_HBM_CHANNEL_NUM, # 0
                        LD_cross_hbm_channel    = True,
                        LD_parallel_channel_num = 1,
                    )
                    LD_B_group_inst_list.append(LD_meta_inst)

                # Load B (Weight)
                this_load_bandwidth_ratio = 2 / layer_ir["param"][param_weight_idx]["sparse_ratio"] # 2 stands for INT4
                LD_B_inst = isa.generate_LD_inst(
                    LD_wait                 = [],
                    LD_release              = [],
                    LD_1d_length            = weights_N_block_col_size_B,
                    LD_hbm_addr             = now_block_weights_hbm_addr,
                    LD_bank_addr            = now_block_weights_bank_addr % cfg.B_BUFFER_DEPTH,
                    LD_target_bank_name     = "B buffer",
                    LD_hbm_channel_id       = N_block_id % cfg.B_BUFFER_HBM_CHANNEL_NUM, # 0
                    LD_cross_hbm_channel    = True,
                    LD_parallel_channel_num = 1,
                    LD_bandwidth_ratio      = this_load_bandwidth_ratio,
                )
                LD_B_group_inst_list.append(LD_B_inst)

                LD_B_group_inst_list = isa.set_first_wait_last_release(
                    inst_list       = LD_B_group_inst_list,
                    first_wait      = ["MM"] if LD_B_wait_MM_flag else [],
                    last_release    = ["MM"],
                    first_inst_type = "LD",
                    last_inst_type  = "LD",
                )
                inst_list.extend(LD_B_group_inst_list)

                # MM
                this_MM_wait = ["LD"]
                if MM_wait_ST_flag:
                    this_MM_wait.append("ST")
                if MM_wait_MISC_flag:
                    this_MM_wait.append("MISC")
                this_MM_release = []
                if last_split_K_group_flag:
                    if fuse_misc_flag:
                        this_MM_release.append("MISC")
                    else:
                        this_MM_release.append("ST")
                if MM_release_LD_flag:
                    this_MM_release.append("LD")
                MM_inst = isa.generate_MM_inst(
                    MM_wait                 = this_MM_wait, 
                    MM_release              = this_MM_release,
                    MM_A_start_addr         = 0,
                    MM_B_start_addr         = now_block_weights_bank_addr % cfg.B_BUFFER_DEPTH,
                    MM_out_start_addr       = MM_out_block_bank_addr % cfg.GLOBAL_BUFFER_DEPTH,
                    MM_K_block_num          = split_K_block_num,
                    MM_bias_start_addr      = 0,
                    MM_meta_start_addr      = (now_block_meta_bank_addr % cfg.META_BUFFER_DEPTH) if sparse_flag else 0,
                    MM_fsb_start_addr       = (now_block_fsb_bank_addr % cfg.FSB_BUFFER_DEPTH) if sparse_flag else 0,
                    MM_bias_flag            = bias_flag,
                    MM_relu_flag            = relu_flag,
                    MM_sparse_flag          = layer_ir["param"][param_weight_idx]["sparse_flag"], # sparse_flag
                    MM_output_flag          = last_split_K_group_flag, # 最后一组K再输出
                    MM_sparse_ratio         = layer_ir["param"][param_weight_idx]["sparse_ratio"],
                )
                inst_list.append(MM_inst)

                # increse fsb, meta, weights bank & hbm addr
                if sparse_flag:
                    now_block_fsb_hbm_addr += fsb_N_block_col_size_B
                    now_block_fsb_bank_addr += fsb_N_block_col_addr_increment
                    now_block_meta_hbm_addr += meta_N_block_col_size_B
                    now_block_meta_bank_addr += meta_N_block_col_addr_increment
                now_block_weights_hbm_addr += weights_N_block_col_size_B
                now_block_weights_bank_addr += weights_N_block_col_addr_increment
            
            # 由于切分K以后需要每个K为一组处理依赖，因此ST的依赖判断等到这一列K的指令全部生成再计算
            assert now_ST_group_cal == now_MM_output_cal # ST_group_cal和MM_output_cal一一对应，用于处理MM和ST之间的依赖
            ST_release_MM_flag = (now_ST_group_cal + out_dependency_len <= total_MM_output_cal) and (not fuse_misc_flag)
            MISC_release_MM_flag = (now_ST_group_cal + out_dependency_len <= total_MM_output_cal) and fuse_misc_flag

            # Store out * 8 for each channel [after all K_blocks done]
            # K维度切分完之后MM输出，然后可以开始融合MISC层指令
            if fuse_misc_flag:
                fuse_layer_num = fuse_layer_ir["fuse_layer_num"]
                if MISC_eltwise_flag:
                    # Load another input
                    LD_eltwise_input_inst_list = list()
                    for M_block_id in range(start_M_block_id, end_M_block_id):
                        LD_eltwise_inst = isa.generate_LD_inst(
                            LD_wait                 = [],
                            LD_release              = [],
                            LD_1d_length            = each_dense_block_size_B,
                            LD_hbm_addr             = fuse_layer_ir["input"][1]["addr"],
                            LD_bank_addr            = 0,
                            LD_target_bank_name     = "global buffer",
                            LD_hbm_channel_id       = M_block_id % cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
                            LD_cross_hbm_channel    = False,
                            LD_parallel_channel_num = cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
                        )
                        LD_eltwise_input_inst_list.append(LD_eltwise_inst)
                    LD_eltwise_input_inst_list = isa.set_first_wait_last_release(
                        inst_list       = LD_eltwise_input_inst_list,
                        first_wait      = [],
                        last_release    = ["MISC"],
                        first_inst_type = "LD",
                        last_inst_type  = "LD",
                    )
                    if fuse_layer_num == 1:
                        inst_list.extend(LD_eltwise_input_inst_list)
                    elif fuse_layer_num == 3:
                        inst_list.extend(LD_eltwise_input_inst_list)
                        inst_list.extend(LD_eltwise_input_inst_list)
                    else:
                        raise ValueError
                if fuse_layer_num == 1:
                    MISC_inst = isa.generate_MISC_inst(
                        MISC_wait               = ["LD", "MM"] if MISC_eltwise_flag else ["MM"],
                        MISC_release            = ["ST", "MM"] if MISC_release_MM_flag else ["ST"],
                        MISC_in_a_start_addr    = 0 % cfg.GLOBAL_BUFFER_DEPTH,
                        MISC_in_b_start_addr    = 0,
                        MISC_out_start_addr     = 0 % cfg.GLOBAL_BUFFER_DEPTH,
                        MISC_K_block_num        = end_M_block_id - start_M_block_id,
                        MISC_operation_name     = MISC_operation_name,
                    )
                    inst_list.append(MISC_inst)
                elif fuse_layer_num == 3:
                    assert MISC_eltwise_flag
                    MISC_0_inst = isa.generate_MISC_inst(
                        MISC_wait               = ["LD", "MM"],
                        MISC_release            = [],
                        MISC_in_a_start_addr    = 0,
                        MISC_in_b_start_addr    = 0,
                        MISC_out_start_addr     = 0,
                        MISC_K_block_num        = end_M_block_id - start_M_block_id,
                        MISC_operation_name     = MISC_operation_name,
                    )
                    inst_list.append(MISC_0_inst)
                    MISC_1_inst = isa.generate_MISC_inst(
                        MISC_wait               = ["LD"],
                        MISC_release            = [],
                        MISC_in_a_start_addr    = 0,
                        MISC_in_b_start_addr    = 0,
                        MISC_out_start_addr     = 0, # 原地操作
                        MISC_K_block_num        = end_M_block_id - start_M_block_id,
                        MISC_operation_name     = MISC_operation_name,
                    )
                    inst_list.append(MISC_1_inst)
                    MISC_2_inst = isa.generate_MISC_inst(
                        MISC_wait               = [],
                        MISC_release            = ["ST", "MM"] if MISC_release_MM_flag else ["ST"],
                        MISC_in_a_start_addr    = 0,
                        MISC_in_b_start_addr    = 0,
                        MISC_out_start_addr     = 0, # 原地操作
                        MISC_K_block_num        = end_M_block_id - start_M_block_id,
                        MISC_operation_name     = MISC_operation_name,
                    )
                    inst_list.append(MISC_2_inst)
                else:
                    raise ValueError

            ST_inst_list = list()
            for M_block_id in range(start_M_block_id, end_M_block_id):
                ST_out_block_hbm_dst_addr = tools.get_dense_matrix_hbm_addr(
                    addr_base               = layer_ir["output"][0]["addr"], # Out
                    K_block_num             = N_block_num, # 输出矩阵的K方向是输入的N方向
                    K_block_id              = N_block_id, # 输出的K方向是输入的N方向
                    M_or_N_block_id         = M_block_id, # 输出的M/N方向是输入的M方向
                    cfg                     = cfg,
                )
                ST_inst = isa.generate_ST_inst(
                    ST_wait                 = [],
                    ST_release              = [],
                    ST_1d_length            = each_dense_block_size_B if M_block_id < M_block_num - 1 else last_save_length, # 每条Save对应一个FSB块的大小，最后一行M对应的可能不到一个FSB块
                    ST_hbm_addr             = ST_out_block_hbm_dst_addr,
                    ST_bank_addr            = MM_out_block_bank_addr % cfg.GLOBAL_BUFFER_DEPTH, # 一条MM计算出来的bank addr相同
                    ST_hbm_channel_id       = M_block_id % cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
                    ST_parallel_channel_num = cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
                )
                ST_inst_list.append(ST_inst)
            ST_first_wait = ["MISC"] if fuse_misc_flag else ["MM"]
            ST_inst_list = isa.set_first_wait_last_release( # 一条MM对应8条ST，因此生成完以后再处理依赖
                inst_list       = ST_inst_list,
                first_wait      = ST_first_wait,
                last_release    = ["MM"] if ST_release_MM_flag else [],
                first_inst_type = "ST",
                last_inst_type  = "ST",
            ) 
            inst_list.extend(ST_inst_list)

        assert now_LD_B_group_cal == total_LD_B_group_cal
        assert now_MM_cal == total_MM_cal
        assert now_MM_output_cal == total_MM_output_cal
        assert now_ST_group_cal == total_ST_group_cal
            
    inst_list = isa.set_first_wait_last_release(
        inst_list       = inst_list,
        first_wait      = layer_first_LD_wait,
        last_release    = layer_last_ST_release,
        first_inst_type = "LD",
        last_inst_type  = "ST",
    )
    return inst_list

