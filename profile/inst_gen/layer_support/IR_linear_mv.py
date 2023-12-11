import os
import numpy as np
from utils import tools
from utils.config_generator import CFG
from inst_gen import isa

# compile linear mv layer
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
    N, M, K, N_block_num, M_block_num, K_block_num, bias_flag, relu_flag, sparse_flag, param_weight_idx, param_bias_idx, param_meta_idx, param_fsb_idx = tools.get_linear_layer_info(layer_ir, cfg, "MV")
    # input: M(1) * K, weight: K * N, Out: M(1) * N, bias: 1 * N
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

    assert M == 1
    assert N % 16 == 0 # N是16的整数倍
    assert K % 16 == 0 # K是16的整数倍

    each_dense_block_size_B = cfg.FSB_BLOCK_SIZE * cfg.FSB_BLOCK_SIZE # 一个dense FSB块的大小

    each_MV_N_block_num = cfg.MV_START_N_NUM // cfg.FSB_BLOCK_SIZE # 每次计算的N方向块数
    assert each_MV_N_block_num * cfg.FSB_BLOCK_SIZE == cfg.MV_START_N_NUM # 验证可以整除
    
    need_N_block_num = tools.ceil(N_block_num, each_MV_N_block_num) * each_MV_N_block_num # 按MV需要的N列数整数倍对齐
    out_sequence_actual_size_B = M * N * 1 # 1 for int8

    each_MV_channel_loop = tools.ceil(each_MV_N_block_num, cfg.A_BUFFER_HBM_CHANNEL_NUM) # MV需要的每次计算跨越所有Channel的次数
    assert each_MV_channel_loop * cfg.A_BUFFER_HBM_CHANNEL_NUM >= each_MV_N_block_num # 验证可以整除

    # FSB info: 可能取值2，4，8，16，代表N:16稀疏，只支持这四种，其中16代表稠密
    fsb_info = np.ones((K_block_num, need_N_block_num), dtype=np.int8) * cfg.FSB_BLOCK_SIZE # 全dense
    assert fsb_info.dtype == np.int8
    # 验证只有这四种数字
    assert np.sum(fsb_info == 2) + np.sum(fsb_info == 4) + np.sum(fsb_info == 8) + np.sum(fsb_info == 16) == need_N_block_num * K_block_num
    assert fsb_info.shape == (K_block_num, need_N_block_num)

    # 根据buffer大小对矩阵进行划分，需要考虑A buffer、meta、FSB、Bias Buffer
    # 计算A Buffer能容纳的块数，从而用于处理依赖
    A_block_loop_2_16_sparse_size_B = cfg.FSB_BLOCK_SIZE * 2 * each_MV_channel_loop # 代表一个块如果是2:16稀疏的数据大小
    A_buffer_loop_addr_of_a_block_sparse_2 = A_block_loop_2_16_sparse_size_B // cfg.A_BUFFER_MV_PER_CHANNEL_WIDTH_B # 一个块如果是2:16占A Buffer的地址数
    assert A_buffer_loop_addr_of_a_block_sparse_2 * cfg.A_BUFFER_MV_PER_CHANNEL_WIDTH_B == A_block_loop_2_16_sparse_size_B # 验证可以整除
    A_buffer_addr_of_a_K_block_line = (np.sum(fsb_info, axis = 0) // 2) * A_buffer_loop_addr_of_a_block_sparse_2 # 所有K列占的A Buffer地址数
    max_A_buffer_addr_of_a_K_block_line = np.max(A_buffer_addr_of_a_K_block_line) # 最坏情况占的A buffer地址数，注意MV需要不止一组weights
    # 这里有两种情况，第一种是A Buffer足够大，可以实现放下所有K的时候的流水，这种情况下直接流水即可
    # 第二种情况是A Buffer不足以放下两次计算实现流水，甚至一次都放不下（在FFN中可能出现），这种情况下需要多条MV连续计算再输出，因此需要强制流水。
    # 这种情况下需要计算需要几条MV才能实现input_weights_dependency_len >= 2
    tgt_weights_dependency_len = 2
    target_max_A_buffer_addr_of_a_K_block_line = cfg.A_BUFFER_DEPTH // tgt_weights_dependency_len # 最坏情况下，A buffer中可以放下多少次计算所需的数据
    split_K_min_num = tools.ceil(max_A_buffer_addr_of_a_K_block_line, target_max_A_buffer_addr_of_a_K_block_line) # 至少需要切分MV的数量，这是最好情况，即FSB的分布是均匀的
    split_K_num = split_K_min_num # 目前切分的数量

    # 计算meta buffer能容纳的块数，从而用于处理依赖
    meta_block_loop_2_16_sparse_size_B = cfg.FSB_BLOCK_SIZE * each_MV_channel_loop # 代表一个块的meta index如果是2:16稀疏的数据大小
    assert meta_block_loop_2_16_sparse_size_B * 2 == A_block_loop_2_16_sparse_size_B # 由于index是4bit，数据是8bit，所以正好是数据的一半
    meta_buffer_loop_addr_of_a_block_sparse_2 = meta_block_loop_2_16_sparse_size_B // cfg.META_BUFFER_BANK_WIDTH_B # 一个块如果是2:16占meta Buffer的地址数
    assert meta_buffer_loop_addr_of_a_block_sparse_2 * cfg.META_BUFFER_BANK_WIDTH_B == meta_block_loop_2_16_sparse_size_B # 验证可以整除

    def get_max_buffer_addr_of_a_K_block_line(split_num, buffer_loop_addr_of_a_block_sparse_2): # 计算K方向切成split_num块以后，最大一块占的某 buffer地址
        split_K_start_end_K_block_id_list = tools.tiling_to_list(0, K_block_num, K_block_num // split_num) # 切成split_num块
        max_buffer_addr_of_a_K_block_line = -1
        for (split_K_start_K_block_id, split_K_end_K_block_id) in split_K_start_end_K_block_id_list:
            buffer_addr_of_a_K_block_line = (np.sum(fsb_info[split_K_start_K_block_id: split_K_end_K_block_id, :], axis = 0) // 2) * buffer_loop_addr_of_a_block_sparse_2 # 每块切分的K列部分占的某 Buffer地址数
            max_buffer_addr_of_a_K_block_line = max(np.max(buffer_addr_of_a_K_block_line) // each_MV_channel_loop, max_buffer_addr_of_a_K_block_line) # 最坏情况占的某 buffer地址数，注意MV需要不止一组weights
        return max_buffer_addr_of_a_K_block_line, split_K_start_end_K_block_id_list

    while True: # 最差情况是每个K block都用一条MV
        assert split_K_num <= K_block_num
        max_A_buffer_addr_of_a_K_block_line, split_K_start_end_K_block_id_list = get_max_buffer_addr_of_a_K_block_line(split_K_num, A_buffer_loop_addr_of_a_block_sparse_2)
        max_meta_buffer_addr_of_a_K_block_line, split_K_start_end_K_block_id_list = get_max_buffer_addr_of_a_K_block_line(split_K_num, meta_buffer_loop_addr_of_a_block_sparse_2)
        input_weights_dependency_len = cfg.A_BUFFER_DEPTH // max_A_buffer_addr_of_a_K_block_line
        input_meta_dependency_len = cfg.META_BUFFER_DEPTH // max_meta_buffer_addr_of_a_K_block_line # 最坏情况下，meta buffer中可以放下多少次计算所需的数据
        if input_weights_dependency_len >= 2 and input_meta_dependency_len >= 2: # 满足流水要求
            break
        else:
            split_K_num += 1 # 有块不满足要求
    split_K_num = len(split_K_start_end_K_block_id_list) # 有可能不同，按实际切块数赋值

    # 计算FSB能容纳的块数，从而用于处理依赖
    fsb_block_size_B = 1 # 一个块的FSB就是1个Byte
    meta_buffer_addr_of_a_K_block_line = tools.ceil(K_block_num, split_K_num) * fsb_block_size_B // cfg.FSB_BUFFER_BANK_WIDTH_B # 所有K列占的meta buffer地址数
    assert meta_buffer_addr_of_a_K_block_line * cfg.FSB_BUFFER_BANK_WIDTH_B == tools.ceil(K_block_num, split_K_num) * fsb_block_size_B # 验证可以整除
    input_fsb_dependency_len = cfg.META_BUFFER_DEPTH // meta_buffer_addr_of_a_K_block_line # 最坏情况下，fsb buffer中可以放下多少次计算所需的数据
    assert input_fsb_dependency_len >= 2 # 能放下多于一行才能流水

    input_dependency_len = min(input_weights_dependency_len, input_meta_dependency_len, input_fsb_dependency_len) # 三者取最小值
    assert input_dependency_len >= 2

    # bias buffer应该足够容纳所有数据
    bias_size_B = N * 4 # bias is int32, 4Bytes
    assert bias_size_B <= cfg.BIAS_BUFFER_DEPTH * cfg.BIAS_BUFFER_BANK_WIDTH_B

    # 验证B Buffer大小满足要求
    sequence_batch_1_size_B = M * K * 1 # 1 for int8
    B_buffer_addr_of_sequence_batch_1 = tools.ceil(tools.ceil(sequence_batch_1_size_B, cfg.B_BUFFER_HBM_CHANNEL_NUM), cfg.B_BUFFER_MV_BANK_WIDTH_B)
    assert B_buffer_addr_of_sequence_batch_1 * cfg.B_BUFFER_MV_BANK_WIDTH_B * cfg.B_BUFFER_HBM_CHANNEL_NUM == sequence_batch_1_size_B # 验证可以整除
    B_buffer_addr_of_multi_batch_sequence = B_buffer_addr_of_sequence_batch_1 * 1 # batch = 1
    assert B_buffer_addr_of_multi_batch_sequence <= cfg.B_BUFFER_DEPTH # B Buffer能放下sequence

    # 由于MV时Sequence一直在片上，所以可以考虑无需ST
    # 验证输出buffer（global buffer）大小满足要求
    each_MV_output_size_B = M * cfg.MV_START_N_NUM * 1 # 1 for int8
    out_buffer_addr_of_each_MV_output = tools.ceil(tools.ceil(each_MV_output_size_B, cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM), cfg.GLOBAL_BUFFER_MV_PER_CHANNEL_WIDTH_B)
    assert out_buffer_addr_of_each_MV_output * cfg.GLOBAL_BUFFER_MV_PER_CHANNEL_WIDTH_B * cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM == each_MV_output_size_B # 验证可以整除
    out_buffer_addr_of_multi_batch_sequence = need_N_block_num * out_buffer_addr_of_each_MV_output
    # assert out_buffer_addr_of_multi_batch_sequence <= cfg.GLOBAL_BUFFER_DEPTH # Global Buffer能放下输出sequence
    
    """
    MV的权重矩阵Weights放在A Buffer, 序列放在B Buffer
    生成的时候先不管这一层前后的依赖，等到这一层指令全部生成完再添加

    Load input sequence (Q/Softmax) to B Buffer * 1 [MAYBE DO NOT NEED]
    Load 本层所有 bias * 1
    for N方向每次走each_MV_N_block_num个块, 走完所有N
        for split_K, 走完所有K
            Load FSB * 1 (不跨HBM Channel)
            Load Weights * 1 (不跨HBM Channel)
            Load Meta * 1 (不跨HBM Channel)
        MV * 1 (最后一次才输出)
        if fuse_misc_flag
            MISC * 1
    Store output sequence (QKT) from Global Buffer * 1 [MAYBE DO NOT NEED]
    """
    inst_list = list()
    total_MV_group_num = need_N_block_num // each_MV_N_block_num # 一共需要多少组MV，一组MV算完一溜K
    assert total_MV_group_num * each_MV_N_block_num == need_N_block_num # 验证可以整除
    total_MV_num = total_MV_group_num * split_K_num # 一共需要多少次MV
    
    LD_seq_in_inst = isa.generate_LD_inst( # LD sequence
        LD_wait                 = [],
        LD_release              = [],
        LD_1d_length            = sequence_batch_1_size_B,
        LD_hbm_addr             = layer_ir["input"][0]["addr"],
        LD_bank_addr            = 0, # 因为是固定A换B，所以A buffer中的地址固定为0
        LD_target_bank_name     = "B buffer",
        LD_hbm_channel_id       = 0 % cfg.B_BUFFER_HBM_CHANNEL_NUM,
        LD_cross_hbm_channel    = True,
        LD_parallel_channel_num = 1,
    )
    inst_list.append(LD_seq_in_inst)


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

    now_calc_id = 0

    now_MV_A_bank_addr = 0 # 存储每轮LD_A的起始地址，用于MV
    now_LD_A_bank_addr = 0 # 存储每轮LD_A的当前地址，用于计算LD_A的bank地址
    now_MV_out_bank_addr = 0 # 存储每次MV输出的bank地址
    MV_out_bank_addr_increment = tools.ceil(each_MV_N_block_num * cfg.FSB_BLOCK_SIZE * 1, cfg.GLOBAL_BUFFER_MV_PER_CHANNEL_WIDTH_B)
    now_A_hbm_addr = layer_ir["param"][param_weight_idx]["addr"]

    if sparse_flag:
        now_MV_meta_bank_addr = 0
        now_LD_meta_bank_addr = 0
        now_meta_hbm_addr = layer_ir["param"][param_meta_idx]["addr"]

        now_MV_fsb_bank_addr = 0
        now_LD_fsb_bank_addr = 0
        now_fsb_hbm_addr = layer_ir["param"][param_fsb_idx]["addr"]

    for now_MV_group_id in range(total_MV_group_num):
        for (split_K_start_K_block_id, split_K_end_K_block_id) in split_K_start_end_K_block_id_list:
            now_calc_id += 1 # 从1开始计数
            # depend:
            first_split_K_group_flag = split_K_start_K_block_id == 0 # 是否第一组K切分，用于处理依赖
            last_split_K_group_flag = split_K_end_K_block_id == K_block_num # 是否最后一组K切分，用于处理依赖
            split_K_block_num = split_K_end_K_block_id - split_K_start_K_block_id

            LD_A_wait_MV_flag = (now_calc_id > input_dependency_len) # LD wait MV / MV release LD 的条件是A/meta/fsb buffer满了，需要等MV用完A buffer里的数据才能启动下一次LD
            MV_release_LD_A_flag = (now_calc_id + input_dependency_len <= total_MV_num)
            # MV情况不存在组与组之间LD，因为不需要在组与组之间重复更新weight（sequence）
            MV_release_ST_flag = (now_calc_id == total_MV_num) and last_split_K_group_flag and (not fuse_misc_flag) # 没有融合MISC时，本层最后一条MV需要release ST
            MV_release_MISC_flag = last_split_K_group_flag and fuse_misc_flag # 融合MISC时，每次MV输出时需要计算MISC

            # save start addr
            now_MV_A_bank_addr = now_LD_A_bank_addr
            if sparse_flag:
                now_MV_meta_bank_addr = now_LD_meta_bank_addr
                now_MV_fsb_bank_addr = now_LD_fsb_bank_addr

            LD_A_group_inst_list = list()
            for channel_id in range(cfg.A_BUFFER_HBM_CHANNEL_NUM): # 对应8条LD

                LD_A_data_loop = 0
                LD_A_zero_loop = 0
                for channel_loop_id in range(each_MV_channel_loop):
                    if channel_loop_id * cfg.A_BUFFER_HBM_CHANNEL_NUM + channel_id < N_block_num: # 存在数据
                        LD_A_data_loop += 1
                    else: # LD zerofill
                        LD_A_zero_loop += 1
                assert LD_A_zero_loop + LD_A_data_loop == each_MV_channel_loop

                # 计算load数据的大小和地址偏移
                weights_N_block_col_info = 0
                meta_N_block_col_size_B = 0
                fsb_N_block_col_size_B = 0
                for LD_A_data_loop_id in range(LD_A_data_loop):
                    N_block_id = now_MV_group_id * each_MV_N_block_num + LD_A_data_loop_id * cfg.A_BUFFER_HBM_CHANNEL_NUM + channel_id # 对应的N block id
                    this_N_block_col_fsb_info = fsb_info[split_K_start_K_block_id: split_K_end_K_block_id, N_block_id] # (split_K_block_num,)

                    weights_N_block_col_info += (np.sum(this_N_block_col_fsb_info) // 2) # 这一K行的weights大小
                    assert np.sum(this_N_block_col_fsb_info) % 2 == 0 # 验证可以整除



                weights_N_block_col_size_B = (weights_N_block_col_info * A_block_loop_2_16_sparse_size_B) // each_MV_channel_loop
                assert weights_N_block_col_size_B * each_MV_channel_loop == weights_N_block_col_info * A_block_loop_2_16_sparse_size_B # 验证可以整除
                weights_N_block_col_addr_increment = weights_N_block_col_size_B // cfg.A_BUFFER_MV_PER_CHANNEL_WIDTH_B # weights buffer 地址增量
                assert weights_N_block_col_size_B % cfg.A_BUFFER_MV_PER_CHANNEL_WIDTH_B == 0 # 验证可以整除

                meta_N_block_col_size_B = weights_N_block_col_size_B // 2 # 这一K行的meta大小，恰好是数据的一半大小
                assert weights_N_block_col_size_B % 2 == 0 # 验证可以整除

                meta_N_block_col_addr_increment = meta_N_block_col_size_B // cfg.META_BUFFER_BANK_WIDTH_B # meta buffer 地址增量
                assert meta_N_block_col_size_B % cfg.META_BUFFER_BANK_WIDTH_B == 0 # 验证可以整除
                
                fsb_N_block_col_size_B = split_K_block_num * fsb_block_size_B * each_MV_channel_loop # 这一K行的fsb大小

                fsb_N_block_col_addr_increment = fsb_N_block_col_size_B // cfg.FSB_BUFFER_BANK_WIDTH_B # FSB buffer 地址增量
                assert fsb_N_block_col_size_B % cfg.FSB_BUFFER_BANK_WIDTH_B == 0 # 验证可以整除

                LD_A_zero_size_B = LD_A_zero_loop * each_dense_block_size_B # 补0的大小，按全dense计算
                LD_A_data_size_B = weights_N_block_col_size_B - LD_A_zero_size_B # 实际数据大小
                LD_A_data_bank_addr_increment = LD_A_data_size_B // cfg.A_BUFFER_MV_PER_CHANNEL_WIDTH_B # weights buffer 地址增量
                assert LD_A_data_size_B % cfg.A_BUFFER_MV_PER_CHANNEL_WIDTH_B == 0 # 验证可以整除

                # Load A * 8 for each channel
                # 考虑每次MV需要的数据对应channel中的几轮，例如如果需要16*8个数，就相当于每个channel一次，需要16*16个数就是两次
                # 而MV的矩阵有可能不能对齐到所有channel，因此需要LD zerofill，需要计算LD几次数据，补几次0
                if sparse_flag: # 需要load实际数据且稀疏
                    # Load FSB
                    LD_fsb_inst = isa.generate_LD_inst(
                        LD_wait                 = [],
                        LD_release              = [],
                        LD_1d_length            = fsb_N_block_col_size_B,
                        LD_hbm_addr             = now_fsb_hbm_addr,
                        LD_bank_addr            = now_LD_fsb_bank_addr % cfg.FSB_BUFFER_DEPTH,
                        LD_target_bank_name     = "FSB buffer",
                        LD_hbm_channel_id       = channel_id % cfg.FSB_BUFFER_HBM_CHANNEL_NUM, # 0
                        LD_cross_hbm_channel    = False,
                        LD_parallel_channel_num = 1,
                    )
                    LD_A_group_inst_list.append(LD_fsb_inst)

                    # Load meta
                    LD_meta_inst = isa.generate_LD_inst(
                        LD_wait                 = [],
                        LD_release              = [],
                        LD_1d_length            = meta_N_block_col_size_B,
                        LD_hbm_addr             = now_meta_hbm_addr,
                        LD_bank_addr            = now_LD_meta_bank_addr % cfg.META_BUFFER_DEPTH,
                        LD_target_bank_name     = "meta buffer",
                        LD_hbm_channel_id       = channel_id % cfg.META_BUFFER_HBM_CHANNEL_NUM, # 0
                        LD_cross_hbm_channel    = False,
                        LD_parallel_channel_num = 1,
                    )
                    LD_A_group_inst_list.append(LD_meta_inst)

                # Load A (Weight)
                this_load_bandwidth_ratio = 2 / layer_ir["param"][param_weight_idx]["sparse_ratio"] # 2 stands for INT4
                if LD_A_data_loop > 0:
                    LD_A_inst = isa.generate_LD_inst(
                        LD_wait                 = [],
                        LD_release              = [],
                        LD_1d_length            = LD_A_data_size_B, # 减去补0大小
                        LD_hbm_addr             = now_A_hbm_addr,
                        LD_bank_addr            = now_LD_A_bank_addr % cfg.A_BUFFER_DEPTH,
                        LD_target_bank_name     = "A buffer",
                        LD_hbm_channel_id       = N_block_id % cfg.A_BUFFER_HBM_CHANNEL_NUM, # 0
                        LD_cross_hbm_channel    = False,
                        LD_parallel_channel_num = cfg.A_BUFFER_HBM_CHANNEL_NUM,
                        LD_bandwidth_ratio      = this_load_bandwidth_ratio, # INT4 Weight
                    )
                    LD_A_group_inst_list.append(LD_A_inst)

                # 通过不输出来避免补0

                # increse fsb, meta, weights bank & hbm addr
                if sparse_flag:
                    now_fsb_hbm_addr += fsb_N_block_col_size_B
                    now_LD_fsb_bank_addr += fsb_N_block_col_addr_increment

                    now_meta_hbm_addr += meta_N_block_col_size_B
                    now_LD_meta_bank_addr += meta_N_block_col_addr_increment
                    
                now_A_hbm_addr += weights_N_block_col_size_B
                now_LD_A_bank_addr += weights_N_block_col_addr_increment

            LD_A_group_inst_list = isa.set_first_wait_last_release(
                inst_list       = LD_A_group_inst_list,
                first_wait      = ["MV"] if LD_A_wait_MV_flag else [],
                last_release    = ["MV"],
                first_inst_type = "LD",
                last_inst_type  = "LD",
            )
            inst_list.extend(LD_A_group_inst_list)

            # MV
            this_MV_release = []
            if MV_release_LD_A_flag:
                this_MV_release.append("LD")
            if MV_release_ST_flag:
                this_MV_release.append("ST")
            if MV_release_MISC_flag:
                this_MV_release.append("MISC")
                
            MV_inst = isa.generate_MV_inst(
                MV_wait                 = ["LD"], 
                MV_release              = this_MV_release,
                MV_A_start_addr         = now_MV_A_bank_addr % cfg.A_BUFFER_DEPTH,
                MV_B_start_addr         = 0,
                MV_out_start_addr       = now_MV_out_bank_addr % cfg.GLOBAL_BUFFER_DEPTH,
                MV_K_block_num          = split_K_block_num,
                MV_bias_start_addr      = 0,
                MV_meta_start_addr      = (now_MV_meta_bank_addr % cfg.META_BUFFER_DEPTH) if sparse_flag else 0,
                MV_fsb_start_addr       = (now_MV_fsb_bank_addr % cfg.FSB_BUFFER_DEPTH) if sparse_flag else 0,
                MV_bias_flag            = bias_flag,
                MV_relu_flag            = relu_flag,
                MV_sparse_flag          = layer_ir["param"][param_weight_idx]["sparse_flag"], # sparse_flag
                MV_output_flag          = last_split_K_group_flag, # 最后一组K再输出
                MV_sparse_ratio         = layer_ir["param"][param_weight_idx]["sparse_ratio"],
            )
            inst_list.append(MV_inst)

        # K维度切分完之后MV输出，然后可以开始融合MISC层指令
        if fuse_misc_flag:
            fuse_layer_num = fuse_layer_ir["fuse_layer_num"]
            MISC_release_ST_flag = (now_MV_group_id == total_MV_group_num - 1) # 本层最后一条MISC需要release ST
            if MISC_eltwise_flag:
                # Load another input
                LD_eltwise_input_inst_list = list()
                for channel_id in range(cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM):
                    LD_eltwise_inst = isa.generate_LD_inst(
                        LD_wait                 = [],
                        LD_release              = [],
                        LD_1d_length            = each_MV_N_block_num * cfg.FSB_BLOCK_SIZE * 1, # 1 for int8
                        LD_hbm_addr             = fuse_layer_ir["input"][1]["addr"],
                        LD_bank_addr            = 0,
                        LD_target_bank_name     = "global buffer",
                        LD_hbm_channel_id       = channel_id,
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
                    MISC_wait               = ["LD", "MV"] if MISC_eltwise_flag else ["MV"],
                    MISC_release            = ["ST"] if MISC_release_ST_flag else [],
                    MISC_in_a_start_addr    = now_MV_out_bank_addr % cfg.GLOBAL_BUFFER_DEPTH,
                    MISC_in_b_start_addr    = 0,
                    MISC_out_start_addr     = now_MV_out_bank_addr % cfg.GLOBAL_BUFFER_DEPTH, # 原地操作
                    MISC_K_block_num        = each_MV_N_block_num,
                    MISC_operation_name     = MISC_operation_name,
                )
                inst_list.append(MISC_inst)
            elif fuse_layer_num == 3:
                assert MISC_eltwise_flag
                MISC_0_inst = isa.generate_MISC_inst(
                    MISC_wait               = ["LD", "MV"],
                    MISC_release            = [],
                    MISC_in_a_start_addr    = now_MV_out_bank_addr % cfg.GLOBAL_BUFFER_DEPTH,
                    MISC_in_b_start_addr    = 0,
                    MISC_out_start_addr     = now_MV_out_bank_addr % cfg.GLOBAL_BUFFER_DEPTH, # 原地操作
                    MISC_K_block_num        = each_MV_N_block_num,
                    MISC_operation_name     = MISC_operation_name,
                )
                inst_list.append(MISC_0_inst)
                MISC_1_inst = isa.generate_MISC_inst(
                    MISC_wait               = ["LD"],
                    MISC_release            = [],
                    MISC_in_a_start_addr    = now_MV_out_bank_addr % cfg.GLOBAL_BUFFER_DEPTH,
                    MISC_in_b_start_addr    = 0,
                    MISC_out_start_addr     = now_MV_out_bank_addr % cfg.GLOBAL_BUFFER_DEPTH, # 原地操作
                    MISC_K_block_num        = each_MV_N_block_num,
                    MISC_operation_name     = MISC_operation_name,
                )
                inst_list.append(MISC_1_inst)
                MISC_2_inst = isa.generate_MISC_inst(
                    MISC_wait               = [],
                    MISC_release            = ["ST"] if MISC_release_ST_flag else [],
                    MISC_in_a_start_addr    = now_MV_out_bank_addr % cfg.GLOBAL_BUFFER_DEPTH,
                    MISC_in_b_start_addr    = 0,
                    MISC_out_start_addr     = now_MV_out_bank_addr % cfg.GLOBAL_BUFFER_DEPTH, # 原地操作
                    MISC_K_block_num        = each_MV_N_block_num,
                    MISC_operation_name     = MISC_operation_name,
                )
                inst_list.append(MISC_2_inst)
            else:
                raise ValueError

        now_MV_out_bank_addr += MV_out_bank_addr_increment # MV out bank addr increase
 
        
    assert now_calc_id == total_MV_num

    # Store out for each channel
    ST_seq_out_inst = isa.generate_ST_inst(
        ST_wait                 = ["MISC"] if fuse_misc_flag else ["MV"],
        ST_release              = [],
        ST_1d_length            = out_sequence_actual_size_B,
        ST_hbm_addr             = layer_ir["output"][0]["addr"],
        ST_bank_addr            = 0 % cfg.GLOBAL_BUFFER_DEPTH,
        ST_hbm_channel_id       = 0 % cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
        ST_parallel_channel_num = 1,
    )
    inst_list.append(ST_seq_out_inst)
            
    inst_list = isa.set_first_wait_last_release(
        inst_list       = inst_list,
        first_wait      = layer_first_LD_wait,
        last_release    = layer_last_ST_release,
        first_inst_type = "LD",
        last_inst_type  = "ST",
    )
    return inst_list
