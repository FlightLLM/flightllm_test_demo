from typing import List
import struct
import yaml

ALL_INST_TYPE = ("LD", "ST", "MM", "MV", "MISC", "SYS") # 顺序代表了依赖顺序，不能随便换顺序
INST_OPCODE_DICT = {
    "LD":   0b0001,
    "ST":   0b0010,
    "MM":   0b0011,
    "MV":   0b0100,
    "MISC": 0b0101,
    "SYS":  0b1111,
}
# 指令有多少行（多少个32bit）
INST_LENGTH_DICT = {
    "LD":   7,
    "ST":   3,
    "MM":   4,
    "MV":   4,
    "MISC": 3,
    "SYS":  1,
}
# 所有MISC的名称
ALL_MISC_OP_TYPE = (
    "eltwise_add",  # operation_flag = 0
    "eltwise_mul",  # operation_flag = 1
    "softmax",      # operation_flag = 2
    "layernorm",    # operation_flag = 3
    "RMSlayernorm", # operation_flag = 4
    "silu",         # operation_flag = 5
)

inst_set = dict()
# format: ((line_num, start_bit, end_bit, value), ...)
## LD
inst_set["LD"] = dict()
inst_set["LD"]["opcode"]            = ((0, 31, 28, [INST_OPCODE_DICT["LD"]]),)
inst_set["LD"]["wait"]              = ((0, 27, 23, None),)
inst_set["LD"]["release"]           = ((0, 22, 18, None),)
inst_set["LD"]["mode"]              = ((0, 17, 15, [0, 1, 2, 3, 4, 5]),)
inst_set["LD"]["hbm_channel_id"]    = ((0, 14, 12, None),)
inst_set["LD"]["bank_addr"]         = ((0, 11,  0, None),)
inst_set["LD"]["src_1d_stride"]     = ((1, 31,  0, None), (2,  1,  0, None),)
inst_set["LD"]["src_2d_stride"]     = ((2, 31,  2, None), (3,  3,  0, None),)
inst_set["LD"]["src_3d_stride"]     = ((3, 31,  4, None), (4,  5,  0, None),)
inst_set["LD"]["hbm_addr"]          = ((4, 31,  6, None), (5,  7,  0, None),)
inst_set["LD"]["1d_loop"]           = ((5, 31, 24, None),)
inst_set["LD"]["2d_loop"]           = ((5, 23, 16, None),)
inst_set["LD"]["3d_loop"]           = ((5, 15,  8, None),)
inst_set["LD"]["zero_fill"]         = ((6, 25, 25, [0, 1]),)
inst_set["LD"]["hbm_type"]          = ((6, 24, 23, [0, 1]),)
inst_set["LD"]["1d_length"]         = ((6, 22,  0, None),)
inst_set["LD"]["reserved"]          = ((6, 31, 26, None),) # reserved

## ST
inst_set["ST"] = dict()
inst_set["ST"]["opcode"]            = ((0, 31, 28, [INST_OPCODE_DICT["ST"]]),)
inst_set["ST"]["wait"]              = ((0, 27, 23, None),)
inst_set["ST"]["release"]           = ((0, 22, 18, None),)
inst_set["ST"]["bank_addr"]         = ((0, 17,  6, None),)
inst_set["ST"]["1d_loop"]           = ((0,  5,  1, None),)
inst_set["ST"]["rs_col"]            = ((0,  0,  0, [0, 1]),)
inst_set["ST"]["hbm_addr"]          = ((1, 31,  0, None), (2,  1,  0, None),)
inst_set["ST"]["hbm_type"]          = ((2, 29, 28, [0, 1]),)
inst_set["ST"]["hbm_channel_id"]    = ((2, 27, 25, None),)
inst_set["ST"]["1d_length"]         = ((2, 24,  2, None),)
inst_set["ST"]["reserved"]          = ((2, 31, 30, None),) # reserved

## MM
inst_set["MM"] = dict()
inst_set["MM"]["opcode"]            = ((0, 31, 28, [INST_OPCODE_DICT["MM"]]),)
inst_set["MM"]["wait"]              = ((0, 27, 23, None),)
inst_set["MM"]["release"]           = ((0, 22, 18, None),)
inst_set["MM"]["K"]                 = ((0, 17,  4, None),)
inst_set["MM"]["output_flag"]       = ((0,  3,  3, None),)
inst_set["MM"]["bias_flag"]         = ((0,  2,  2, [0, 1]),)
inst_set["MM"]["relu_flag"]         = ((0,  1,  1, [0, 1]),)
inst_set["MM"]["sparse_flag"]       = ((0,  0,  0, [0, 1]),)
inst_set["MM"]["bias_start_addr"]   = ((1, 31, 24, None), (2,  3,  0, None),)
inst_set["MM"]["meta_start_addr"]   = ((1, 23, 12, None),)
inst_set["MM"]["fsb_start_addr"]    = ((1, 11,  0, None),)
inst_set["MM"]["A_start_addr"]      = ((2, 27, 16, None),)
inst_set["MM"]["B_start_addr"]      = ((2, 15,  4, None),)
inst_set["MM"]["out_start_addr"]    = ((3, 11,  0, None),)
inst_set["MM"]["reserved"]          = ((2, 31, 28, None), (3, 31, 12, None),) # reserved

## MV
inst_set["MV"] = dict()
inst_set["MV"]["opcode"]            = ((0, 31, 28, [INST_OPCODE_DICT["MV"]]),)
inst_set["MV"]["wait"]              = ((0, 27, 23, None),)
inst_set["MV"]["release"]           = ((0, 22, 18, None),)
inst_set["MV"]["K"]                 = ((0, 17,  4, None),)
inst_set["MV"]["output_flag"]       = ((0,  3,  3, None),)
inst_set["MV"]["bias_flag"]         = ((0,  2,  2, [0, 1]),)
inst_set["MV"]["relu_flag"]         = ((0,  1,  1, [0, 1]),)
inst_set["MV"]["sparse_flag"]       = ((0,  0,  0, [0, 1]),)
inst_set["MV"]["bias_start_addr"]   = ((1, 31, 24, None), (2,  3,  0, None),)
inst_set["MV"]["meta_start_addr"]   = ((1, 23, 12, None),)
inst_set["MV"]["fsb_start_addr"]    = ((1, 11,  0, None),)
inst_set["MV"]["A_start_addr"]      = ((2, 27, 16, None),)
inst_set["MV"]["B_start_addr"]      = ((2, 15,  4, None),)
inst_set["MV"]["out_start_addr"]    = ((3, 11,  0, None),)
inst_set["MV"]["reserved"]          = ((2, 31, 28, None), (3, 31, 12, None),) # reserved

## MISC
inst_set["MISC"] = dict()
inst_set["MISC"]["opcode"]          = ((0, 31, 28, [INST_OPCODE_DICT["MISC"]]),)
inst_set["MISC"]["wait"]            = ((0, 27, 23, None),)
inst_set["MISC"]["release"]         = ((0, 22, 18, None),)
inst_set["MISC"]["in_a_start_addr"] = ((0, 17,  6, None),)
inst_set["MISC"]["mask_flag"]       = ((0,  5,  4, [0, 1]),)
inst_set["MISC"]["operation_flag"]  = ((0,  3,  0, [0, 1, 2, 3, 4, 5]),)
inst_set["MISC"]["in_b_start_addr"] = ((1, 31, 20, None),)
inst_set["MISC"]["out_start_addr"]  = ((1, 19,  8, None),)
inst_set["MISC"]["K"]               = ((2, 31, 18, None),)
inst_set["MISC"]["reserved"]        = ((1,  7,  0, None), (2, 17,  0, None),) # reserved

## SYS
inst_set["SYS"] = dict()
inst_set["SYS"]["opcode"]           = ((0, 31, 28, [INST_OPCODE_DICT["SYS"]]),)
inst_set["SYS"]["wait"]             = ((0, 27, 23, None),)
inst_set["SYS"]["release"]          = ((0, 22, 18, None),)
inst_set["SYS"]["reserved"]         = ((0, 17,  0, None),) # reserved


for inst_type in inst_set.keys():
    assert inst_type in ALL_INST_TYPE
    for field_name, field_param in inst_set[inst_type].items():
        # 检查所有字段，确保默认值只可能出现在第一个part
        if not all(map(lambda x: x[3] is None, field_param[1:])):
            raise Exception(f"field_param {field_param} is not in correct format")
        # 检查指令长度正确
        for filed_param_option in field_param:
            assert filed_param_option[0] < INST_LENGTH_DICT[inst_type]

# 遍历检查所有的字段（包括reserved）不重不漏地覆盖了所有的位置
for inst_type in inst_set.keys():
    inst_configuration = inst_set[inst_type]
    # 检查所有的可能位置
    for i in range(INST_LENGTH_DICT[inst_type]):
        for j in range(32):
            j = 31 - j # 反向
            appear_count = 0
            for field_parameter in inst_configuration.values():
                for cover_range in field_parameter:
                    # cover_range是覆盖范围的四元组，测试是否出现
                    if i == cover_range[0] and j <= cover_range[1] and j >= cover_range[2]:
                        appear_count = appear_count + 1
            # appear_count应该永远为1
            if not appear_count == 1:
                raise Exception(f"{inst_type}, {inst_configuration} appear_count is {appear_count}")
            assert appear_count == 1
    # 在不重不漏的前提下，还应保证数量的一致性
    total_count = 0
    for field_parameter in inst_configuration.values():
        for cover_range in field_parameter:
            total_count = total_count + cover_range[1] - cover_range[2] + 1
    assert total_count == INST_LENGTH_DICT[inst_type] * 32

# 利用上述configuration生成所需的encode和decode文件，此时，需要额外的一个参数作为输入配置
# 编码的时候，输入是inst_type 字符串，inst_dict需要的字典，输出是指令列表
def encode_inst(inst_type, inst_param_dict):
    # 先选出对应的配置文件
    inst_configuration = inst_set[inst_type]
    # 根据配置文件的信息确定相关的操作
    inst_list = [int(0)] * INST_LENGTH_DICT[inst_type]
    for key, field_parameter in inst_configuration.items():
        if key == "reserved":
            continue
        field_length = 0
        for cover_range in field_parameter:
            field_length = field_length + cover_range[1] - cover_range[2] + 1
        # 取值，判定，赋值，先出现的是低位
        if key not in inst_param_dict.keys():
            tmp = 0
        else:
            tmp = inst_param_dict[key]
        if not(tmp >= 0 and tmp < 2 ** field_length):
            print(inst_param_dict)
            print(key)
            raise ValueError(f"inst_param_dict[{key}] = {tmp} is not in range(0, {2 ** field_length})")
        for cover_range in field_parameter:
            field_length = cover_range[1] - cover_range[2] + 1
            inst_list[cover_range[0]] += int((tmp % (2 ** field_length)) << cover_range[2])
            tmp = tmp // (2 ** field_length)
    return inst_list

def wait_release_list_to_value(inst_type, wr_list):
    # wr_list是一个列表，里面是字符串
    # 返回一个整数
    assert inst_type in ALL_INST_TYPE
    for wr in wr_list:
        assert wr in ALL_INST_TYPE
    other_inst_type = list() # 创建一个除了自己的inst_type的列表
    for tmp_inst_type in ALL_INST_TYPE:
        if tmp_inst_type != inst_type:
            other_inst_type.append(tmp_inst_type)
    other_inst_num = len(other_inst_type)
    assert other_inst_num == len(ALL_INST_TYPE) - 1
    value = 0
    for wr in wr_list:
        value += (1 << (other_inst_num - 1 - other_inst_type.index(wr)))
    return int(value)

# 编码指令到字典上
def inst_add_type(inst_type, inst_param_dict):
    out_inst_dict = dict()
    out_inst_dict["TYPE"] = inst_type
    out_inst_dict["WAIT_LIST"] = inst_param_dict["wait"]
    out_inst_dict["RELEASE_LIST"] = inst_param_dict["release"]
    out_inst_dict["HEX"] = ""
    out_inst_dict["VALUE"] = []
    out_inst_dict["PARAM"] = dict()

    wait_value = wait_release_list_to_value(inst_type, inst_param_dict["wait"])
    release_value = wait_release_list_to_value(inst_type, inst_param_dict["release"])
        
    inst_param_dict["wait"] = wait_value
    inst_param_dict["release"] = release_value
    
    inst_configuration = inst_set[inst_type]
    for key, field_parameter in inst_configuration.items():
        if key == "reserved":
            continue
        # 如果键在输入内，检查是否满足范围要求，如果不在输入内，那么取一个默认值，默认范围是第一部分的第四块
        option_values = field_parameter[0][3]
        assert key in inst_param_dict.keys()
            # None是没有要求，否则是个列表
        if option_values is not None and inst_param_dict[key] not in option_values:
            print(inst_type)
            print(inst_param_dict)
            print(key)
            raise Exception(f"{inst_param_dict[key]} not in {option_values}")
        value = inst_param_dict[key]
        out_inst_dict["PARAM"][key] = value
    # 对于在inst_dict中，但是不在inst_configuration中的，必须满足为0，否则视为错误
    for key in inst_param_dict.keys():
        assert key in inst_configuration.keys()
    bin_inst = encode_inst(inst_type, inst_param_dict)
    hex_inst = ""
    for i in range(INST_LENGTH_DICT[inst_type] - 1, -1, -1): # 小端序HEX输出
        hex_inst = hex_inst + ("%08x" % bin_inst[i]) + " "
    out_inst_dict["VALUE"] = bin_inst
    out_inst_dict["HEX"] = hex_inst.strip()
    return out_inst_dict

def check_wait_release(inst_type, wait, release):
    assert inst_type in ALL_INST_TYPE
    for wait_inst_type in wait:
        assert wait_inst_type != inst_type
        assert wait_inst_type in ALL_INST_TYPE
    for release_inst_type in release:
        assert release_inst_type != inst_type
        assert release_inst_type in ALL_INST_TYPE

def generate_LD_inst(
    LD_wait: List[str],
    LD_release: List[str],
    LD_1d_length: int,
    LD_hbm_addr: int,
    LD_bank_addr: int,
    LD_target_bank_name: str,
    LD_hbm_channel_id: int,
    LD_cross_hbm_channel: bool, # 用于仿真，指定是否跨channel
    LD_parallel_channel_num: int, # 用于仿真，指定并行的channel数量，仿真时带宽增加为这个倍数
    LD_bandwidth_ratio: int = 1, # 用于仿真，指定数据位宽和稀疏导致的带宽增长系数，INT8为1，INT4为2，以此类推
    LD_zero_fill: bool = False,
    LD_1d_loop: int = 1,
    LD_2d_loop: int = 1,
    LD_3d_loop: int = 1,
    LD_src_1d_stride: int = 0,
    LD_src_2d_stride: int = 0,
    LD_src_3d_stride: int = 0,
    LD_hbm_type: int = 1,
):
    LD_inst_type = "LD"
    all_bank_name = (
        "B buffer",         # LD_mode = 0
        "meta buffer",      # LD_mode = 1
        "bias buffer",      # LD_mode = 2
        "FSB buffer",       # LD_mode = 3
        "A buffer",         # LD_mode = 4
        "global buffer",    # LD_mode = 5
    )
    assert LD_target_bank_name in all_bank_name
    LD_mode = all_bank_name.index(LD_target_bank_name)
    check_wait_release(LD_inst_type, LD_wait, LD_release)
    LD_inst_param_dict = {
        "opcode":           INST_OPCODE_DICT[LD_inst_type],
        "wait":             LD_wait,
        "release":          LD_release,
        "1d_length":        int(LD_1d_length),
        "1d_loop":          int(LD_1d_loop),
        "2d_loop":          int(LD_2d_loop),
        "3d_loop":          int(LD_3d_loop),
        "src_1d_stride":    int(LD_src_1d_stride),
        "src_2d_stride":    int(LD_src_2d_stride),
        "src_3d_stride":    int(LD_src_3d_stride),
        "hbm_addr":         int(LD_hbm_addr),
        "bank_addr":        int(LD_bank_addr),
        "mode":             int(LD_mode),
        "hbm_type":         int(LD_hbm_type),
        "hbm_channel_id":   int(LD_hbm_channel_id),
        "zero_fill":        int(LD_zero_fill),
    }
    LD_inst_dict = inst_add_type(LD_inst_type, LD_inst_param_dict)
    LD_inst_dict["NOTE"] = {
        "cross_hbm_channel": int(LD_cross_hbm_channel),
        "parallel_channel_num": LD_parallel_channel_num,
        "bandwidth_ratio": LD_bandwidth_ratio,
    } # add note for LD inst
    return LD_inst_dict

def generate_ST_inst(
    ST_wait: List[str],
    ST_release: List[str],
    ST_1d_length: int,
    ST_hbm_addr: int,
    ST_bank_addr: int,
    ST_hbm_channel_id: int,
    ST_parallel_channel_num: int, # 用于仿真，指定并行的channel数量，仿真时带宽增加为这个倍数
    ST_cross_hbm_channel: bool = False, # 用于仿真，指定并行的channel数量，仿真时带宽增加为这个倍数
    ST_1d_loop: int = 1,
    ST_rs_col: bool = False,
    ST_hbm_type: int = 1,
):
    ST_inst_type = "ST"
    check_wait_release(ST_inst_type, ST_wait, ST_release)
    ST_inst_param_dict = {
        "opcode":           INST_OPCODE_DICT[ST_inst_type],
        "wait":             ST_wait,
        "release":          ST_release,
        "1d_length":        int(ST_1d_length),
        "1d_loop":          int(ST_1d_loop),
        "hbm_addr":         int(ST_hbm_addr),
        "bank_addr":        int(ST_bank_addr),
        "rs_col":           int(ST_rs_col),
        "hbm_type":         int(ST_hbm_type),
        "hbm_channel_id":   int(ST_hbm_channel_id),
    }
    ST_inst_dict = inst_add_type(ST_inst_type, ST_inst_param_dict)
    ST_inst_dict["NOTE"] = {
        "cross_hbm_channel": int(ST_cross_hbm_channel),
        "parallel_channel_num": int(ST_parallel_channel_num),
    } # add note for ST inst
    return ST_inst_dict

def generate_MM_inst(
    MM_wait: List[str],
    MM_release: List[str],
    MM_A_start_addr: int,
    MM_B_start_addr: int,
    MM_out_start_addr: int,
    MM_K_block_num: int,
    MM_bias_start_addr: int = 0,
    MM_meta_start_addr: int = 0,
    MM_fsb_start_addr: int = 0,
    MM_bias_flag: bool = False,
    MM_relu_flag: bool = False,
    MM_sparse_flag: bool = False,
    MM_output_flag: bool = True,
    MM_sparse_ratio: float = 1, # 用于仿真，指定稀疏导致的计算量降低
):
    MM_inst_type = "MM"
    check_wait_release(MM_inst_type, MM_wait, MM_release)
    MM_inst_param_dict = {
        "opcode":           INST_OPCODE_DICT[MM_inst_type],
        "wait":             MM_wait,
        "release":          MM_release,
        "fsb_start_addr":   int(MM_fsb_start_addr),
        "bias_start_addr":  int(MM_bias_start_addr),
        "meta_start_addr":  int(MM_meta_start_addr),
        "A_start_addr":     int(MM_A_start_addr),
        "B_start_addr":     int(MM_B_start_addr),
        "out_start_addr":   int(MM_out_start_addr),
        "K":                int(MM_K_block_num),
        "bias_flag":        int(MM_bias_flag),
        "relu_flag":        int(MM_relu_flag),
        "sparse_flag":      int(MM_sparse_flag),
        "output_flag":      int(MM_output_flag),
    }
    MM_inst_dict = inst_add_type(MM_inst_type, MM_inst_param_dict)
    MM_inst_dict["NOTE"] = {
        "sparse_ratio": float(MM_sparse_ratio),
    } # add note for MM inst
    return MM_inst_dict

def generate_MV_inst(
    MV_wait: List[str],
    MV_release: List[str],
    MV_A_start_addr: int,
    MV_B_start_addr: int,
    MV_out_start_addr: int,
    MV_K_block_num: int,
    MV_bias_start_addr: int = 0,
    MV_meta_start_addr: int = 0,
    MV_fsb_start_addr: int = 0,
    MV_bias_flag: bool = False,
    MV_relu_flag: bool = False,
    MV_sparse_flag: bool = False,
    MV_output_flag: bool = True,
    MV_sparse_ratio: float = 1, # 用于仿真，指定稀疏导致的计算量降低
):
    MV_inst_type = "MV"
    check_wait_release(MV_inst_type, MV_wait, MV_release)
    MV_inst_param_dict = {
        "opcode":           INST_OPCODE_DICT[MV_inst_type],
        "wait":             MV_wait,
        "release":          MV_release,
        "fsb_start_addr":   int(MV_fsb_start_addr),
        "bias_start_addr":  int(MV_bias_start_addr),
        "meta_start_addr":  int(MV_meta_start_addr),
        "A_start_addr":     int(MV_A_start_addr),
        "B_start_addr":     int(MV_B_start_addr),
        "out_start_addr":   int(MV_out_start_addr),
        "K":                int(MV_K_block_num),
        "bias_flag":        int(MV_bias_flag),
        "relu_flag":        int(MV_relu_flag),
        "sparse_flag":      int(MV_sparse_flag),
        "output_flag":      int(MV_output_flag),
    }
    MV_inst_dict = inst_add_type(MV_inst_type, MV_inst_param_dict)
    MV_inst_dict["NOTE"] = {
        "sparse_ratio": float(MV_sparse_ratio),
    } # add note for MM inst
    return MV_inst_dict

def generate_MISC_inst(
    MISC_wait: List[str],
    MISC_release: List[str],
    MISC_in_a_start_addr: int,
    MISC_in_b_start_addr: int,
    MISC_out_start_addr: int,
    MISC_K_block_num: int,
    MISC_operation_name: str,
    MISC_mask_flag: bool = False,
):
    MISC_inst_type = "MISC"
    assert MISC_operation_name in ALL_MISC_OP_TYPE
    MISC_operation_flag = ALL_MISC_OP_TYPE.index(MISC_operation_name)
    check_wait_release(MISC_inst_type, MISC_wait, MISC_release)
    MISC_inst_param_dict = {
        "opcode":           INST_OPCODE_DICT[MISC_inst_type],
        "wait":             MISC_wait,
        "release":          MISC_release,
        "in_a_start_addr":  int(MISC_in_a_start_addr),
        "in_b_start_addr":  int(MISC_in_b_start_addr),
        "out_start_addr":   int(MISC_out_start_addr),
        "K":                int(MISC_K_block_num),
        "mask_flag":        int(MISC_mask_flag),
        "operation_flag":   int(MISC_operation_flag),
    }
    MISC_inst_dict = inst_add_type(MISC_inst_type, MISC_inst_param_dict)
    return MISC_inst_dict

def generate_SYS_inst(
    SYS_wait: List[str],
    SYS_release: List[str],
):
    SYS_inst_type = "SYS"
    check_wait_release(SYS_inst_type, SYS_wait, SYS_release)
    SYS_inst_param_dict = {
        "opcode":           INST_OPCODE_DICT[SYS_inst_type],
        "wait":             SYS_wait,
        "release":          SYS_release,
    }
    SYS_inst_dict = inst_add_type(SYS_inst_type, SYS_inst_param_dict)
    return SYS_inst_dict

def set_first_wait_last_release(
    inst_list: list,
    first_wait: list,
    last_release: list,
    first_inst_type: str,
    last_inst_type: str,
):
    assert len(inst_list) > 0
    assert first_inst_type in ALL_INST_TYPE
    assert last_inst_type in ALL_INST_TYPE

    for first_wait_type in first_wait:
        assert first_wait_type in ALL_INST_TYPE

    for last_release_type in last_release:
        assert last_release_type in ALL_INST_TYPE

    assert inst_list[0]["TYPE"] == first_inst_type
    assert inst_list[-1]["TYPE"] == last_inst_type

    # 原来没有wait和release
    assert inst_list[0]["PARAM"]["wait"] == 0
    assert inst_list[-1]["PARAM"]["release"] == 0

    new_first_wait_value = wait_release_list_to_value(first_inst_type, first_wait)
    new_last_release_value = wait_release_list_to_value(last_inst_type, last_release)

    inst_list[0]["WAIT_LIST"] = first_wait
    inst_list[-1]["RELEASE_LIST"] = last_release
    inst_list[0]["PARAM"]["wait"] = new_first_wait_value
    inst_list[-1]["PARAM"]["release"] = new_last_release_value

    return inst_list

def dump_inst_yaml_file(inst_yaml_dir, model_inst_list):
    print("Dump readable inst to {}".format(inst_yaml_dir))
    with open(inst_yaml_dir, "w") as f:
        yaml.dump(model_inst_list, f, sort_keys=False)

def dump_inst_bin_file(bin_inst_dir, model_inst_list):
    print("Dump binary inst to {}".format(bin_inst_dir))
    NG = 4 # num of group, must == 4 for this func
    with open(bin_inst_dir, "wb") as f:
        for layer_inst_list in model_inst_list:
            for inst in layer_inst_list:
                for inst_value in inst["VALUE"]:
                    f.write(struct.pack("I", (inst_value % (1 << (NG * 8)))))

def dump_inst_txt_file(txt_inst_dir, model_inst_list):
    print("Dump txt inst to {}".format(txt_inst_dir))
    NG = 4 # num of group, must == 4 for this func
    with open(txt_inst_dir, "w") as f:
        for layer_inst_list in model_inst_list:
            for inst in layer_inst_list:
                for inst_value in inst["VALUE"]:
                    f.write(("%%0%dx\n" % (NG * 2)) % (inst_value % (1 << (NG * 8))))
