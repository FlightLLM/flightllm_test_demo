import os
import numpy as np
from utils import tools
from utils.config_generator import CFG
from inst_gen import isa

# compile attention QKT or QKTV
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
            print(f"Generate inst for fusing {layer_ir['type']} layer {layer_ir['name']} and {fuse_layer_ir['type']} layer {fuse_layer_ir['name']}")
        else:
            print(f"Generate inst for {layer_ir['type']} layer {layer_ir['name']}")
    BATCH, N, M, K, N_block_num, M_block_num, K_block_num, mask_mode = tools.get_attention_layer_info(layer_ir, cfg, "MM")
    if fuse_misc_flag: # 融合attention和softmax层
        MISC_BATCH, MISC_M, MISC_K, MISC_K_block_num, MISC_eltwise_flag, MISC_operation_name = tools.get_misc_layer_info(fuse_layer_ir, cfg)
        assert MISC_BATCH == BATCH
        assert MISC_M == M
        assert MISC_K == N
        assert MISC_K_block_num == N_block_num
        assert layer_ir["output"][0]["name"] == fuse_layer_ir["input"][0]["name"]
        assert not MISC_eltwise_flag
        assert MISC_operation_name == "softmax"

    assert BATCH == 32
    # Q/Softmax: M * K, KT/V: K * N, Out: M * N
    
    last_save_length = M - (M_block_num - 1) * cfg.FSB_BLOCK_SIZE # 考虑M方向不是16整数倍的情况，最后store的时候length变小

    mask_layout_dir = os.path.join(cfg.OUTPUT_DIR, f"ir_output/attention_mask/{layer_ir['param'][0]['name']}.npy")
    assert os.path.isfile(mask_layout_dir)
    mask_layout = np.load(mask_layout_dir)
    # mask_layout = np.ones((BATCH, cfg.MAX_LEN // cfg.MASK_LAYOUT_BLOCK_SIZE, cfg.MAX_LEN // cfg.MASK_LAYOUT_BLOCK_SIZE), dtype=bool) # 用全1代替mask
    # assert mask_layout.shape == (BATCH, cfg.MAX_LEN // cfg.MASK_LAYOUT_BLOCK_SIZE, cfg.MAX_LEN // cfg.MASK_LAYOUT_BLOCK_SIZE)
    # 将mask转换为block粒度的
    mask_blocks = np.repeat(np.repeat(mask_layout, cfg.FSB_BLOCK_NUM_PER_MASK_LAYOUT_BLOCK, axis=1), cfg.FSB_BLOCK_NUM_PER_MASK_LAYOUT_BLOCK, axis=2)
    if mask_mode == "qkt":
        mask_blocks = mask_blocks[:, :M_block_num, :N_block_num] # (BATCH, M_block_num, N_block_num)
    elif mask_mode == "qktv":
        assert M_block_num == K_block_num
        mask_blocks = mask_blocks[:, :M_block_num, :K_block_num]
    else:
        raise ValueError

    M_mask_layout_block_merge_num = cfg.MM_PARALLEL_M // cfg.MASK_LAYOUT_BLOCK_SIZE # M方向需要merge几个mask layout块
    assert M_mask_layout_block_merge_num * cfg.MASK_LAYOUT_BLOCK_SIZE == cfg.MM_PARALLEL_M
    # 验证合并mask块后M方向恰好是128，对应所有channel里一块，也就是在A buffer中每一列K block是一行（一个地址）
    M_block_tiling_num = M_mask_layout_block_merge_num * cfg.FSB_BLOCK_NUM_PER_MASK_LAYOUT_BLOCK # 每次计算所需的M方向块数
    assert M_block_tiling_num == cfg.A_BUFFER_HBM_CHANNEL_NUM # 一次计算正好用到所有channel
    
    # 验证A Buffer满足大小，由于是固定A换B，所以A不需要流水
    A_buffer_addr_of_a_K_block = 1 # 切块后每列K block正好对应A buffer中bank的一个地址
    A_buffer_addr_of_a_K_block_line = K_block_num * A_buffer_addr_of_a_K_block # 为了计算第一条MM，A Buffer中需要的地址长度
    assert cfg.A_BUFFER_DEPTH >= A_buffer_addr_of_a_K_block_line 
    
    # 根据buffer大小对矩阵进行划分，attention操作不会用到meta、FSB、Bias Buffer
    # B Buffer只有一个Bank，而且形状也不一样，在这里需要跨channel load
    # 计算B Buffer能容纳的块数，从而用于处理依赖
    each_block_size_B = cfg.FSB_BLOCK_SIZE * cfg.FSB_BLOCK_SIZE * 1 # 1代表INT8，每个数1Byte
    assert each_block_size_B % cfg.B_BUFFER_MM_BANK_WIDTH_B == 0
    B_buffer_addr_of_a_K_block = each_block_size_B // cfg.B_BUFFER_MM_BANK_WIDTH_B
    B_buffer_addr_of_a_K_block_line = K_block_num * B_buffer_addr_of_a_K_block # 根据每块数据量算出来每个block占B中的多少地址
    input_B_dependency_len = cfg.B_BUFFER_DEPTH // B_buffer_addr_of_a_K_block_line # B bank中可以放下多少次计算所需的数据
    assert input_B_dependency_len >= 2 # 输入满足流水要求
    input_B_dependency_len *= cfg.B_BUFFER_HBM_CHANNEL_NUM # B的每个K_block_line只需要1条LD
    # input_B_dependency_len代表LD几次会放不下

    # 计算Global buffer能容纳的块数，从而用于处理依赖
    # 由于输出的顺序是N方向（输出的K方向），所以计算方法和A buffer不同
    out_buffer_addr_of_a_N_block = N_block_num # 输出的K实际上就是输入的N方向
    out_dependency_len = cfg.GLOBAL_BUFFER_DEPTH // out_buffer_addr_of_a_N_block # Global Buffer中可以放下多少次计算所需的数据
    assert out_dependency_len >= 2 # 输出满足流水要求
    # out_dependency_len代表global buffer放不下的时候算了几次MM（此时ST的次数是out_dependency_len * cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM）
    
    """
    由于带宽限制，最好是尽量固定A换B，A的复用需要达到4个MM才能匹配带宽
    因此，采用每次计算128*64的大小，即每次计算两个mask块，每个mask块64*64，也就是M方向每次8个FSB块
    生成指令: 固定A Buffer (Q/Softmax), 动B (KT/V)
    生成的时候先不管这一层前后的依赖，等到这一层指令全部生成完再添加
    需要注意每次M_block_tiling_num组与组之前，最后一条MM要发送一个release LD给下一组的第一条LD Q/Softmax
    Attention暂时不支持切K维度, 因为bank够大

    for 所有attention head (BATCH)
        for M方向每次M_block_tiling_num个block, 走完这一组M
            Load Q/Softmax: LD * 8 for each channel
            for N方向每次一K列FSB块, 走完所有N
                Load KT/V * 1 (跨HBM Channel)
                MM * 1
            Store Out * 8 (每个Channel一个)
            if fuse_misc_flag
                MISC * 128
    """
    inst_list = list()
    # M方向根据M_block_tiling_num切分，注意M方向最后行的块可能不满
    M_block_tiling_id_list = tools.tiling_to_list(0, M_block_num, M_block_tiling_num)
    for head_id in range(BATCH):
        M_block_group_cal = 0
        for (start_M_block_id, end_M_block_id) in M_block_tiling_id_list:
            M_block_group_cal += 1
            # Load A * 8 for each channel
            LD_A_inst_list = list()
            for M_block_id in range(start_M_block_id, end_M_block_id):
                if mask_mode == "qkt":
                    this_M_block_row_mask = mask_blocks[head_id, M_block_id, :] # (N_block_num,)
                    if np.sum(this_M_block_row_mask) == 0: # 这一行没有一个mask，无需加载这块Q，直接跳过
                        continue
                    LD_Q_block_hbm_src_addr = tools.get_dense_matrix_hbm_addr(
                        addr_base               = layer_ir["input"][0]["addr"] + head_id * M * K, # Q
                        K_block_num             = K_block_num,
                        K_block_id              = 0,
                        M_or_N_block_id         = M_block_id,
                        cfg                     = cfg,
                    )
                    LD_Q_inst = isa.generate_LD_inst(
                        LD_wait                 = [],
                        LD_release              = [],
                        LD_1d_length            = each_block_size_B * K_block_num,
                        LD_hbm_addr             = LD_Q_block_hbm_src_addr,
                        LD_bank_addr            = 0, # 因为是固定A换B，所以A buffer中的地址固定为0
                        LD_target_bank_name     = "A buffer",
                        LD_hbm_channel_id       = M_block_id % cfg.A_BUFFER_HBM_CHANNEL_NUM,
                        LD_cross_hbm_channel    = False,
                        LD_parallel_channel_num = cfg.A_BUFFER_HBM_CHANNEL_NUM,
                    )
                    LD_A_inst_list.append(LD_Q_inst)
                elif mask_mode == "qktv":
                    this_M_block_row_mask = mask_blocks[head_id, M_block_id, :] # (K_block_num,)
                    if np.sum(this_M_block_row_mask) < K_block_num: # 如果这一行不是全都有数
                        LD_softmax_zero_inst = isa.generate_LD_inst(
                            LD_wait                 = [],
                            LD_release              = [],
                            LD_1d_length            = each_block_size_B * (K_block_num - np.sum(this_M_block_row_mask)),
                            LD_hbm_addr             = 0,
                            LD_bank_addr            = 0, # 因为是固定A换B，所以A buffer中的地址固定为0
                            LD_target_bank_name     = "A buffer",
                            LD_hbm_channel_id       = M_block_id % cfg.A_BUFFER_HBM_CHANNEL_NUM,
                            LD_cross_hbm_channel    = False,
                            LD_parallel_channel_num = cfg.A_BUFFER_HBM_CHANNEL_NUM,
                            LD_zero_fill            = True, # write zero
                        )
                        LD_A_inst_list.append(LD_softmax_zero_inst)
                    LD_softmax_inst_list = list()
                    LD_softmax_start_end_id_list = get_qktv_load_softmax_block_id(this_M_block_row_mask)
                    for (LD_softmax_start_K_block_id, LD_softmax_end_K_block_id) in LD_softmax_start_end_id_list:
                        LD_softmax_block_hbm_src_addr = tools.get_dense_matrix_hbm_addr(
                            addr_base               = layer_ir["input"][0]["addr"] + head_id * M * K, # softmax
                            K_block_num             = K_block_num,
                            K_block_id              = LD_softmax_start_K_block_id,
                            M_or_N_block_id         = M_block_id,
                            cfg                     = cfg,
                        )
                        LD_softmax_inst = isa.generate_LD_inst(
                            LD_wait                 = [],
                            LD_release              = [],
                            LD_1d_length            = each_block_size_B * (LD_softmax_end_K_block_id - LD_softmax_start_K_block_id),
                            LD_hbm_addr             = LD_softmax_block_hbm_src_addr,
                            LD_bank_addr            = LD_softmax_start_K_block_id, # 因为是固定A换B，所以A buffer中的地址固定为0
                            LD_target_bank_name     = "A buffer",
                            LD_hbm_channel_id       = M_block_id % cfg.A_BUFFER_HBM_CHANNEL_NUM,
                            LD_parallel_channel_num = cfg.A_BUFFER_HBM_CHANNEL_NUM,
                            LD_cross_hbm_channel    = False,
                        )
                        LD_softmax_inst_list.append(LD_softmax_inst)
                    assert len(LD_softmax_inst_list) > 0
                    LD_A_inst_list.extend(LD_softmax_inst_list)
                else:
                    raise ValueError
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
            # mask_mode == "qkt": LD_B is load KT
            # mask_mode == "qktv": LD_B is load softmax
            total_LD_B_cal = 0
            total_MM_cal = 0
            for N_block_id in range(N_block_num):
                if mask_mode == "qkt":
                    this_N_block_col_mask = mask_blocks[head_id, start_M_block_id: end_M_block_id, N_block_id]
                    if np.sum(this_N_block_col_mask) == 0: # 这一列没有一个mask，无需LD B, MM, ST，可以整个直接跳过
                        continue
                elif mask_mode == "qktv":
                    pass # qktv的V矩阵是稠密的
                else:
                    raise ValueError
                total_LD_B_cal += 1
                total_MM_cal += 1

            # 目前的LD_B, MM, ST的组数，用于处理依赖关系
            now_LD_B_cal = 0
            now_MM_cal = 0
            for N_block_id in range(N_block_num):
                if mask_mode == "qkt":
                    this_N_block_col_mask = mask_blocks[head_id, start_M_block_id: end_M_block_id, N_block_id]
                    if np.sum(this_N_block_col_mask) == 0: # 这一列没有一个mask，无需LD B, MM, ST，可以整个直接跳过
                        continue
                elif mask_mode == "qktv":
                    pass # qktv的V矩阵是稠密的
                else:
                    raise ValueError
                
                now_LD_B_cal += 1 # 从1开始
                now_MM_cal += 1 # 从1开始

                # depend:
                assert now_LD_B_cal == now_MM_cal # LD_B_cal和MM_cal一一对应，用于处理MM和LD之间的依赖
                LD_B_wait_MM_flag = (now_LD_B_cal > input_B_dependency_len) # LD wait MM / MM release LD 的条件是B buffer满了，需要等MM用完B buffer里的数据才能启动下一次LD
                tmp_MM_release_LD_B_flag = (now_MM_cal + input_B_dependency_len <= total_LD_B_cal)
                tmp_MM_release_next_group_LD_A_flag = (now_MM_cal == total_MM_cal) and (end_M_block_id < M_block_num) # 这一组和下一组的依赖，条件是本组的最后一条MM且存在下一组
                assert not (tmp_MM_release_LD_B_flag and tmp_MM_release_next_group_LD_A_flag) # 两个条件不可能同时满足
                MM_release_LD_flag = tmp_MM_release_LD_B_flag or tmp_MM_release_next_group_LD_A_flag

                MM_release_ST_flag = (now_MM_cal == total_MM_cal) and (not fuse_misc_flag) # 最后一组N_block
                MM_release_MISC_flag = (now_MM_cal == total_MM_cal) and fuse_misc_flag
                MM_wait_ST_flag = (M_block_group_cal > out_dependency_len) and (now_MM_cal == 1) and (not fuse_misc_flag) # MM wait ST / ST release MM 的条件是out buffer满了，需要等ST存完global buffer里的数据才能启动下一次MM
                MM_wait_MISC_flag = (M_block_group_cal > out_dependency_len) and (now_MM_cal == 1) and fuse_misc_flag
                ST_release_MM_flag = (M_block_group_cal + out_dependency_len <= len(M_block_tiling_id_list)) and (not fuse_misc_flag)
                MISC_release_MM_flag = (M_block_group_cal + out_dependency_len <= len(M_block_tiling_id_list)) and fuse_misc_flag

                # Load B (KT for qkt, V for qktv)
                LD_B_this_block_B_bank_addr = (N_block_id * B_buffer_addr_of_a_K_block_line) % cfg.B_BUFFER_DEPTH
                LD_B_block_hbm_src_addr = tools.get_dense_matrix_hbm_addr(
                    addr_base               = layer_ir["input"][1]["addr"] + head_id * K * N,
                    K_block_num             = K_block_num,
                    K_block_id              = 0,
                    M_or_N_block_id         = N_block_id,
                    cfg                     = cfg,
                )
                LD_B_inst = isa.generate_LD_inst(
                    LD_wait                 = ["MM"] if LD_B_wait_MM_flag else [],
                    LD_release              = ["MM"],
                    LD_1d_length            = each_block_size_B * K_block_num,
                    LD_hbm_addr             = LD_B_block_hbm_src_addr,
                    LD_bank_addr            = LD_B_this_block_B_bank_addr,
                    LD_target_bank_name     = "B buffer",
                    LD_hbm_channel_id       = N_block_id % cfg.A_BUFFER_HBM_CHANNEL_NUM, # KT或V矩阵本来存储在A Buffer对应的HBM通道中
                    LD_cross_hbm_channel    = True,
                    LD_parallel_channel_num = 1,
                )
                inst_list.append(LD_B_inst)

                # MM
                MM_out_block_bank_addr = N_block_id * out_buffer_addr_of_a_N_block
                this_MM_wait = ["LD"]
                if MM_wait_ST_flag:
                    this_MM_wait.append("ST")
                if MM_wait_MISC_flag:
                    this_MM_wait.append("MISC")
                this_MM_release = []
                if MM_release_LD_flag:
                    this_MM_release.append("LD")
                if MM_release_ST_flag:
                    this_MM_release.append("ST")
                if MM_release_MISC_flag:
                    this_MM_release.append("MISC")
                MM_inst = isa.generate_MM_inst(
                    MM_wait                 = this_MM_wait, 
                    MM_release              = this_MM_release,
                    MM_A_start_addr         = 0,
                    MM_B_start_addr         = LD_B_this_block_B_bank_addr,
                    MM_out_start_addr       = MM_out_block_bank_addr % cfg.GLOBAL_BUFFER_DEPTH,
                    MM_K_block_num          = K_block_num,
                )
                inst_list.append(MM_inst)
                
            # Store out * 8 for each channel
            for M_block_id in range(start_M_block_id, end_M_block_id):
                if mask_mode == "qkt":
                    nonzero_N_block_num = np.sum(mask_blocks[head_id, M_block_id, :])
                    if nonzero_N_block_num == 0: # 这一行没有要存的块
                        continue
                elif mask_mode == "qktv":
                    nonzero_N_block_num = N_block_num # qktv的结果矩阵是稠密的
                else:
                    raise ValueError
                
                if fuse_misc_flag: # softmax
                    MISC_inst_list = list()
                    this_MISC_group_wait = ["MM"] if M_block_id == start_M_block_id else []
                    this_MISC_group_release = ["ST"]
                    if MISC_release_MM_flag and M_block_id == (end_M_block_id - 1):
                        this_MISC_group_release.append("MM")
                    for _ in range(cfg.FSB_BLOCK_SIZE):
                        MISC_inst = isa.generate_MISC_inst(
                            MISC_wait               = [],
                            MISC_release            = [],
                            MISC_in_a_start_addr    = MM_out_block_bank_addr % cfg.GLOBAL_BUFFER_DEPTH,
                            MISC_in_b_start_addr    = 0, # no use for softmax
                            MISC_out_start_addr     = MM_out_block_bank_addr % cfg.GLOBAL_BUFFER_DEPTH,
                            MISC_K_block_num        = nonzero_N_block_num,
                            MISC_operation_name     = MISC_operation_name,
                        )
                        MISC_inst_list.append(MISC_inst)

                    MISC_inst_list = isa.set_first_wait_last_release(
                        inst_list       = MISC_inst_list,
                        first_wait      = this_MISC_group_wait,
                        last_release    = this_MISC_group_release,
                        first_inst_type = "MISC",
                        last_inst_type  = "MISC",
                    )
                    inst_list.extend(MISC_inst_list)

                ST_out_block_hbm_dst_addr = tools.get_dense_matrix_hbm_addr(
                    addr_base               = layer_ir["output"][0]["addr"] + head_id * N * M, # Out
                    K_block_num             = N_block_num, # 输出矩阵的K方向是输入的N方向
                    K_block_id              = N_block_id, # 输出的K方向是输入的N方向
                    M_or_N_block_id         = M_block_id, # 输出的M/N方向是输入的M方向
                    cfg                     = cfg,
                )
                this_ST_wait = []
                if fuse_misc_flag:
                    this_ST_wait.append("MISC")
                elif M_block_id == start_M_block_id:
                        this_ST_wait.append("MM")
                this_ST_release = []
                if ST_release_MM_flag and M_block_id == (end_M_block_id - 1):
                    this_ST_release.append("MM")
                ST_inst = isa.generate_ST_inst(
                    ST_wait                 = this_ST_wait,
                    ST_release              = this_ST_release,
                    ST_1d_length            = nonzero_N_block_num * each_block_size_B, # 每条Save对应一个FSB块的大小，最后一行M对应的可能不到一个FSB块
                    ST_hbm_addr             = ST_out_block_hbm_dst_addr,
                    ST_bank_addr            = MM_out_block_bank_addr % cfg.GLOBAL_BUFFER_DEPTH, # 一条MM计算出来的bank addr相同
                    ST_hbm_channel_id       = M_block_id % cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
                    ST_parallel_channel_num = cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
                )
                inst_list.append(ST_inst)

            assert now_LD_B_cal == total_LD_B_cal
            assert now_MM_cal == total_MM_cal
            
    inst_list = isa.set_first_wait_last_release(
        inst_list       = inst_list,
        first_wait      = layer_first_LD_wait,
        last_release    = layer_last_ST_release,
        first_inst_type = "LD",
        last_inst_type  = "ST",
    )
    return inst_list

def get_qktv_load_softmax_block_id(mask_list):
    # example input: [0, 1, 1, 1, 0, 1, 0]
    # example output: [(1, 4), (5, 6)] 

    # 不直接把mask_list扩充是因为这样会改变mask_list
    def access_puls_1_mask_list(mask_list, idx):
        assert idx <= len(mask_list)
        if idx < len(mask_list):
            return mask_list[idx]
        else:
            return False
        
    K_block_id = 0
    plus_1_mask_list_len = len(mask_list) + 1
    LD_softmax_start_end_id_list = list()
    while K_block_id < plus_1_mask_list_len:
        if access_puls_1_mask_list(mask_list, K_block_id): # 这个FSB块有数：
            for end_K_block_id in range(K_block_id, plus_1_mask_list_len):
                if not access_puls_1_mask_list(mask_list, end_K_block_id):
                    break
            LD_softmax_start_end_id_list.append((K_block_id, end_K_block_id,))
            K_block_id = end_K_block_id + 1
        else:
            K_block_id += 1
    return LD_softmax_start_end_id_list
