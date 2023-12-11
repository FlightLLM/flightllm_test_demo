import os
import numpy as np
from utils import tools
from utils.config_generator import CFG
from inst_gen import isa

# compile attention QKT or QKTV, support fusing attention with softmax
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
    BATCH, N, M, K, N_block_num, M_block_num, K_block_num, mask_mode = tools.get_attention_layer_info(layer_ir, cfg, "MV")
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

    assert M == 1
    assert N % 16 == 0 # N是16的整数倍
    assert K % 16 == 0 # K是16的整数倍

    each_MV_N_block_num = cfg.MV_START_N_NUM // cfg.FSB_BLOCK_SIZE # 每次计算的N方向块数
    assert each_MV_N_block_num * cfg.FSB_BLOCK_SIZE == cfg.MV_START_N_NUM # 验证可以整除
    
    need_N_block_num = tools.ceil(N_block_num, each_MV_N_block_num) * each_MV_N_block_num # 按MV需要的N列数整数倍对齐
    total_MV_num = need_N_block_num // each_MV_N_block_num # 一共需要多少次MV
    out_sequence_actual_size_B = M * N * 1 # 1 for int8

    each_MV_channel_loop = tools.ceil(each_MV_N_block_num, cfg.A_BUFFER_HBM_CHANNEL_NUM) # MV需要的每次计算跨越所有Channel的次数
    assert each_MV_channel_loop * cfg.A_BUFFER_HBM_CHANNEL_NUM >= each_MV_N_block_num # 验证可以整除

    # 验证A Buffer满足大小，由于是固定A换B，所以A不需要流水
    a_K_block_line_size_B = 1 * cfg.FSB_BLOCK_SIZE * K # 1 for int8
    A_buffer_addr_of_a_K_block_line = (a_K_block_line_size_B * each_MV_channel_loop) // cfg.A_BUFFER_MV_PER_CHANNEL_WIDTH_B # 为了计算第一条MV，A Buffer中需要的地址长度
    assert A_buffer_addr_of_a_K_block_line * cfg.A_BUFFER_MV_PER_CHANNEL_WIDTH_B == a_K_block_line_size_B * each_MV_channel_loop # 验证可以整除
    input_A_dependency_len = cfg.A_BUFFER_DEPTH // A_buffer_addr_of_a_K_block_line # A bank中可以放下多少次计算所需的数据
    # 由于MV时A Buffer存Weight，B buffer存sequence，所以需要A buffer流水
    assert input_A_dependency_len >= 2 # 输入满足流水要求
    
    # 验证B Buffer大小满足要求
    sequence_batch_1_size_B = M * K * 1 # 1 for int8
    B_buffer_addr_of_sequence_batch_1 = tools.ceil(tools.ceil(sequence_batch_1_size_B, cfg.B_BUFFER_HBM_CHANNEL_NUM), cfg.B_BUFFER_MV_BANK_WIDTH_B)
    assert B_buffer_addr_of_sequence_batch_1 * cfg.B_BUFFER_MV_BANK_WIDTH_B * cfg.B_BUFFER_HBM_CHANNEL_NUM == sequence_batch_1_size_B # 验证可以整除
    B_buffer_addr_of_multi_batch_sequence = B_buffer_addr_of_sequence_batch_1 * BATCH
    assert B_buffer_addr_of_multi_batch_sequence <= cfg.B_BUFFER_DEPTH # B Buffer能放下sequence

    # 由于MV时Sequence一直在片上，所以可以考虑无需ST
    # 验证输出buffer（global buffer）大小满足要求
    each_MV_output_size_B = M * cfg.MV_START_N_NUM * 1 # 1 for int8
    out_buffer_addr_of_each_MV_output = tools.ceil(tools.ceil(each_MV_output_size_B, cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM), cfg.GLOBAL_BUFFER_MV_PER_CHANNEL_WIDTH_B)
    assert out_buffer_addr_of_each_MV_output * cfg.GLOBAL_BUFFER_MV_PER_CHANNEL_WIDTH_B * cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM == each_MV_output_size_B # 验证可以整除
    out_buffer_addr_of_multi_batch_sequence = need_N_block_num * out_buffer_addr_of_each_MV_output
    assert out_buffer_addr_of_multi_batch_sequence <= cfg.GLOBAL_BUFFER_DEPTH # Global Buffer能放下输出sequence
    
    """
    MV的矩阵(KT/V)放在A Buffer, 序列(Q/Softmax)放在B Buffer
    生成的时候先不管这一层前后的依赖，等到这一层指令全部生成完再添加
    Attention暂时不支持切K维度, 因为bank够大

    Load input sequence (Q/Softmax) to B Buffer * 1 [MAYBE DO NOT NEED]
    for 所有attention head (BATCH)
        for N方向每次走each_MV_N_block_num个块, 走完所有N
            Load KT/V: LD * 8 for each channel
            MV * 1
        if fuse_misc_flag
            MISC * 1
    Store output sequence (QKT) from Global Buffer * 1 [MAYBE DO NOT NEED]

    """
    inst_list = list()
    assert total_MV_num * each_MV_N_block_num == need_N_block_num # 验证可以整除
    
    LD_seq_in_inst = isa.generate_LD_inst( # LD Q/Softmax
        LD_wait                 = [],
        LD_release              = [],
        LD_1d_length            = sequence_batch_1_size_B * BATCH,
        LD_hbm_addr             = layer_ir["input"][0]["addr"],
        LD_bank_addr            = 0, # 因为是固定A换B，所以A buffer中的地址固定为0
        LD_target_bank_name     = "B buffer",
        LD_hbm_channel_id       = 0 % cfg.B_BUFFER_HBM_CHANNEL_NUM,
        LD_cross_hbm_channel    = True,
        LD_parallel_channel_num = 1,
    )
    inst_list.append(LD_seq_in_inst)
    now_MV_A_bank_start_addr = 0 # 存储每轮LD_A的起始地址，用于MV
    now_LD_A_bank_addr = 0 # 存储每轮LD_A的当前地址，用于计算LD_A的bank地址
    now_MV_out_bank_start_addr = 0 # 存储每轮MV的输出起始地址，用于MV和ST
    now_MISC_out_bank_start_addr = 0 # 存储每个head的MV输出起始地址，用于融合的MISC层
    for head_id in range(BATCH): # 可以把这个任务看成BATCH个很长的MV任务在N方向拼起来流水
        now_MISC_out_bank_start_addr = now_MV_out_bank_start_addr
        for now_MV_id in range(total_MV_num):
            # depend:
            now_calc_id = now_MV_id + 1 # 从1开始计数
            LD_A_wait_MV_flag = (now_calc_id > input_A_dependency_len) # LD wait MV / MV release LD 的条件是A buffer满了，需要等MV用完A buffer里的数据才能启动下一次LD
            MV_release_LD_A_flag = (now_calc_id + input_A_dependency_len <= total_MV_num)
            # MV情况不存在组与组之间LD，因为不需要在组与组之间重复更新weight（sequence）
            MV_release_ST_flag = (now_calc_id == total_MV_num) and (head_id == BATCH - 1) and (not fuse_misc_flag) # MV需要release ST的条件是本层的最后一条MV且没有融合MISC层
            MV_release_MISC_flag = (now_calc_id == total_MV_num) and fuse_misc_flag # MV release MISC的条件是本head最后一条MV且融合了MISC层

            # Load A * 8 for each channel (KT/V)
            # 考虑每次MV需要的数据对应channel中的几轮，例如如果需要16*8个数，就相当于每个channel一次，需要16*16个数就是两次
            # 而MV的矩阵有可能不能对齐到所有channel，因此需要LD zerofill，需要计算LD几次数据，补几次0
            LD_A_inst_list = list()
            for channel_id in range(cfg.A_BUFFER_HBM_CHANNEL_NUM): # 对应8条LD
                LD_A_data_loop = 0
                LD_A_zero_loop = 0
                for channel_loop_id in range(each_MV_channel_loop):
                    if channel_loop_id * cfg.A_BUFFER_HBM_CHANNEL_NUM + channel_id < N_block_num: # 存在数据
                        LD_A_data_loop += 1
                    else: # LD zerofill
                        LD_A_zero_loop += 1
                assert LD_A_zero_loop + LD_A_data_loop == each_MV_channel_loop

                if LD_A_data_loop > 0: # 一定是先数据，再补0，不可能是先补0而后面有数据
                    LD_A_block_hbm_src_addr = tools.get_dense_matrix_hbm_addr(
                        addr_base               = layer_ir["input"][1]["addr"] + head_id * K * N,
                        K_block_num             = K_block_num,
                        K_block_id              = 0,
                        M_or_N_block_id         = channel_id, # 代表N方向的块数
                        cfg                     = cfg,
                    )
                    LD_A_inst = isa.generate_LD_inst(
                        LD_wait                 = [],
                        LD_release              = [],
                        LD_1d_length            = a_K_block_line_size_B * LD_A_data_loop, # 有几块数据，LD多少数据，合并多次LD，因为在HBM的不同Channel是分着存的
                        LD_hbm_addr             = LD_A_block_hbm_src_addr,
                        LD_bank_addr            = now_LD_A_bank_addr % cfg.A_BUFFER_DEPTH,
                        LD_target_bank_name     = "A buffer",
                        LD_hbm_channel_id       = channel_id % cfg.A_BUFFER_HBM_CHANNEL_NUM, # KT或V矩阵本来存储在A Buffer对应的HBM通道中
                        LD_cross_hbm_channel    = False,
                        LD_parallel_channel_num = cfg.A_BUFFER_HBM_CHANNEL_NUM,
                    )
                    LD_A_inst_list.append(LD_A_inst)
                # 通过不输出来避免补0
                now_LD_A_bank_addr += each_MV_channel_loop * A_buffer_addr_of_a_K_block_line # 每次计算的A buffer地址增量
            
            LD_A_inst_list = isa.set_first_wait_last_release(
                inst_list       = LD_A_inst_list,
                first_wait      = ["MV"] if LD_A_wait_MV_flag else [],
                last_release    = ["MV"],
                first_inst_type = "LD",
                last_inst_type  = "LD",
            )
            inst_list.extend(LD_A_inst_list)

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
                MV_A_start_addr         = now_MV_A_bank_start_addr % cfg.A_BUFFER_DEPTH,
                MV_B_start_addr         = (B_buffer_addr_of_sequence_batch_1 * head_id) % cfg.B_BUFFER_DEPTH,
                MV_out_start_addr       = now_MV_out_bank_start_addr % cfg.GLOBAL_BUFFER_DEPTH,
                MV_K_block_num          = K_block_num,
            )
            inst_list.append(MV_inst)

            now_MV_A_bank_start_addr += each_MV_N_block_num * A_buffer_addr_of_a_K_block_line # 每次计算的A buffer地址增量
            assert now_MV_A_bank_start_addr == now_LD_A_bank_addr # 验证两个地址每次增加后仍然相等
            now_MV_out_bank_start_addr += each_MV_N_block_num * out_buffer_addr_of_each_MV_output
        if fuse_misc_flag:
            MISC_release_ST_flag = (head_id == BATCH - 1) # 本层最后一条MISC需要release ST
            MISC_inst = isa.generate_MISC_inst(
                MISC_wait               = ["MV"],
                MISC_release            = ["ST"] if MISC_release_ST_flag else [],
                MISC_in_a_start_addr    = now_MISC_out_bank_start_addr % cfg.GLOBAL_BUFFER_DEPTH,
                MISC_in_b_start_addr    = 0, # no use for softmax
                MISC_out_start_addr     = now_MISC_out_bank_start_addr % cfg.GLOBAL_BUFFER_DEPTH, # 原地操作
                MISC_K_block_num        = MISC_K_block_num,
                MISC_operation_name     = MISC_operation_name,
            )
            inst_list.append(MISC_inst)


    # Store out
    ST_seq_out_inst = isa.generate_ST_inst(
        ST_wait                 = ["MISC"] if fuse_misc_flag else ["MV"],
        ST_release              = [],
        ST_1d_length            = out_sequence_actual_size_B * BATCH,
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
