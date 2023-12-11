import os
import numpy as np
from utils import tools
from utils.config_generator import CFG
from inst_gen import isa

# compile softmax, layernorm, eltwise (add/mul) layer
def generate_layer_inst(
        layer_ir: dict(), 
        cfg: CFG, 
        layer_first_LD_wait: list(), 
        layer_last_ST_release: list(), 
    ):
    if cfg.DEBUG:
        print(f"Generate inst for {layer_ir['type']} layer {layer_ir['name']}")
    BATCH, M, K, K_block_num, eltwise_flag, operation_name = tools.get_misc_layer_info(layer_ir, cfg)
    assert operation_name in isa.ALL_MISC_OP_TYPE
    # input: M * K, Out: M * K
    
    each_K_line_size_B = K * 1 # 一行的大小
    # 由于MISC每次只算一行，因此LD也每次一行

    if eltwise_flag: # eltwise add / mul
        bank_addr_range = cfg.GLOBAL_BUFFER_DEPTH // 2 # eltwise输入a和输出在地址前半段，输入b在地址后半段，相当于输入1，输入2，输出，各可以占一半的global buffer（其中输入1和输出共享）
    else: # softmax / layernorm
        bank_addr_range = cfg.GLOBAL_BUFFER_DEPTH # softmax和layernorm在的输入和输出共用所有bank地址

    # 验证Global Buffer满足大小
    global_buffer_addr_of_a_K_line = tools.ceil(each_K_line_size_B, cfg.GLOBAL_BUFFER_MISC_PER_CHANNEL_WIDTH_B * cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM) # 为了计算第一条MISC，global buffer中需要的地址长度
    assert cfg.GLOBAL_BUFFER_MISC_PER_CHANNEL_WIDTH_B * cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM * global_buffer_addr_of_a_K_line == each_K_line_size_B # 验证可以整除

    input_dependency_len = bank_addr_range // global_buffer_addr_of_a_K_line
    assert input_dependency_len >= 2
    output_dependency_len = input_dependency_len # MISC原地操作

    """
    生成的时候先不管这一层前后的依赖，等到这一层指令全部生成完再添加

    for batch in range(BATCH):
        for M_id in range(M):
            if softmax / layernorm:
                Load Input * 1 (不跨HBM Channel)
            else (eltwise):
                Load Input a * 1 (不跨HBM Channel)
                Load Input b * 1 (不跨HBM Channel)
            MISC * 1
            Store Out * 1 (不跨HBM Channel)
    """
    inst_list = list()
    total_calc_group_num = BATCH * M # 总共需要计算的组数
    now_bank_addr = 0 # 由于MISC原地操作，所以bank地址可以共用，记录每次LD/MISC/ST的地址
    bank_addr_increment = global_buffer_addr_of_a_K_line # 每次MISC需要的bank地址增量

    for head_id in range(BATCH): # 看成是BATCH * M_block_num的大计算
        for M_id in range(M):
            # depend
            # 当前的LD, MISC, ST的组数，用于处理依赖关系
            now_calc_group_id = head_id * M + M_id + 1 # 从1开始计数
            M_block_id = tools.ceil(M_id, cfg.FSB_BLOCK_SIZE)

            LD_wait_MISC_flag = (now_calc_group_id > input_dependency_len) # LD wait MISC / MISC release LD 的条件是Global buffer满了，需要等MISC用完Global buffer里的数据才能启动下一次LD
            MISC_release_LD_flag = (now_calc_group_id + input_dependency_len <= total_calc_group_num)
            MISC_wait_ST_flag = (now_calc_group_id > output_dependency_len) # MISC wait ST / ST release MISC 的条件是Global buffer满了，需要等ST存完Global buffer里的数据才能启动下一次MISC
            ST_release_MISC_flag = (now_calc_group_id + output_dependency_len <= total_calc_group_num)

            LD_ST_hbm_addr_shift = (head_id * M * K + M_id * K) * 1 # 1 for int8

            if eltwise_flag: # eltwise add / mul
                # Load input a
                LD_a_inst = isa.generate_LD_inst(
                    LD_wait                 = ["MISC"] if LD_wait_MISC_flag else [], # 前面的LD需要wait
                    LD_release              = [],
                    LD_1d_length            = each_K_line_size_B,
                    LD_hbm_addr             = layer_ir["input"][0]["addr"] + LD_ST_hbm_addr_shift,
                    LD_bank_addr            = now_bank_addr % bank_addr_range, # 注意这里地址不能超过深度的一半
                    LD_target_bank_name     = "global buffer",
                    LD_hbm_channel_id       = M_block_id % cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
                    LD_cross_hbm_channel    = False,
                    LD_parallel_channel_num = 1,
                )
                inst_list.append(LD_a_inst)

                # Load input b
                LD_b_inst = isa.generate_LD_inst(
                    LD_wait                 = [],
                    LD_release              = ["MISC"], # 后面的LD需要release
                    LD_1d_length            = each_K_line_size_B,
                    LD_hbm_addr             = layer_ir["input"][1]["addr"] + LD_ST_hbm_addr_shift,
                    LD_bank_addr            = bank_addr_range + (now_bank_addr % bank_addr_range), # 注意这里地址需要是后一半
                    LD_target_bank_name     = "global buffer",
                    LD_hbm_channel_id       = M_block_id % cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
                    LD_cross_hbm_channel    = False,
                    LD_parallel_channel_num = 1,
                )
                inst_list.append(LD_b_inst)
            else: # softmax / layernorm
                # Load input
                LD_inst = isa.generate_LD_inst(
                    LD_wait                 = ["MISC"] if LD_wait_MISC_flag else [],
                    LD_release              = ["MISC"],
                    LD_1d_length            = each_K_line_size_B,
                    LD_hbm_addr             = layer_ir["input"][0]["addr"] + LD_ST_hbm_addr_shift,
                    LD_bank_addr            = now_bank_addr % bank_addr_range,
                    LD_target_bank_name     = "global buffer",
                    LD_hbm_channel_id       = M_block_id % cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
                    LD_cross_hbm_channel    = False,
                    LD_parallel_channel_num = 1,
                )
                inst_list.append(LD_inst)
            
            MISC_in_b_start_addr = bank_addr_range + (now_bank_addr % bank_addr_range) if eltwise_flag else 0
            MISC_inst = isa.generate_MISC_inst(
                MISC_wait               = ["LD", "ST"] if MISC_wait_ST_flag else ["LD"],
                MISC_release            = ["LD", "ST"] if MISC_release_LD_flag else ["ST"],
                MISC_in_a_start_addr    = now_bank_addr % bank_addr_range,
                MISC_in_b_start_addr    = MISC_in_b_start_addr, # 仅eltwise非0
                MISC_out_start_addr     = now_bank_addr % bank_addr_range, # 原地操作
                MISC_K_block_num        = K_block_num,
                MISC_operation_name     = operation_name,
            )
            inst_list.append(MISC_inst)

            # Store output
            ST_inst = isa.generate_ST_inst(
                ST_wait                 = ["MISC"],
                ST_release              = ["MISC"] if ST_release_MISC_flag else [],
                ST_1d_length            = each_K_line_size_B,
                ST_hbm_addr             = layer_ir["output"][0]["addr"] + LD_ST_hbm_addr_shift,
                ST_bank_addr            = now_bank_addr % bank_addr_range, # 一条MISC计算出来的bank addr相同
                ST_hbm_channel_id       = M_block_id % cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM,
                ST_parallel_channel_num = 1,
            )
            inst_list.append(ST_inst)

            now_bank_addr += bank_addr_increment # bank addr increse
    
    inst_list = isa.set_first_wait_last_release(
        inst_list       = inst_list,
        first_wait      = layer_first_LD_wait,
        last_release    = layer_last_ST_release,
        first_inst_type = "LD",
        last_inst_type  = "ST",
    )
    return inst_list
