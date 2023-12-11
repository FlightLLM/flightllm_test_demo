from utils.config_generator import CFG

def is_hbm_addr_legal(addr, cfg: CFG):
    assert addr is not None
    return addr >= 0 and addr < cfg.HBM_TOTAL_SIZE_B

def get_block_num(N, M, K, cfg: CFG):
    # 代表三个维度分别有多少个block
    N_block_num = ceil(N, cfg.FSB_BLOCK_SIZE)
    M_block_num = ceil(M, cfg.FSB_BLOCK_SIZE)
    K_block_num = ceil(K, cfg.FSB_BLOCK_SIZE)
    
    # N和K必须是FSB大小的整数倍，M方向（Seq Len）不要求
    assert N_block_num * cfg.FSB_BLOCK_SIZE == N
    assert K_block_num * cfg.FSB_BLOCK_SIZE == K

    assert N_block_num > 0
    assert M_block_num > 0
    assert K_block_num > 0

    return N_block_num, M_block_num, K_block_num

def get_attention_layer_info(layer_ir, cfg: CFG, MM_or_MV: str):
    assert layer_ir["type"] in ("attention_mm", "attention_mv")
    assert len(layer_ir["input"]) == 2
    assert len(layer_ir["output"]) == 1
    batch_A, M_A, K_A = layer_ir["input"][0]["shape"]
    batch_B, K_B, N_B = layer_ir["input"][1]["shape"]
    batch_O, M_O, N_O = layer_ir["output"][0]["shape"]
    assert batch_A == batch_B == batch_O
    assert K_A == K_B
    assert M_A == M_O
    assert N_B == N_O
    BATCH = batch_A
    K = K_A
    N = N_O
    M = M_O
    assert MM_or_MV in ("MM", "MV")
    mask_mode = layer_ir["structure"]["mask_mode"]

    if MM_or_MV == "MM":
        assert M > 1
        assert len(layer_ir["param"]) == 1 # 不需要加载到bank，只在编译时候用
        assert layer_ir["structure"]["mask_mode"] in ("qkt", "qktv")
        if mask_mode == "qkt":
            assert N == M
        elif mask_mode == "qktv":
            assert M == K
        else:
            raise ValueError
    elif MM_or_MV == "MV":
        assert M == 1
        assert mask_mode == "none"
        assert len(layer_ir["param"]) == 1 # Decode阶段没有sparse attention，但param依然存了Prefill阶段的mask
        assert layer_ir["structure"]["mask_mode"] == "none"
    else:
        raise ValueError

    N_block_num, M_block_num, K_block_num = get_block_num(N, M, K, cfg)

    assert not layer_ir["structure"]["relu_flag"]
    assert not layer_ir["structure"]["bias_flag"]
    
    assert is_hbm_addr_legal(layer_ir["input"][0]["addr"], cfg)
    assert is_hbm_addr_legal(layer_ir["input"][1]["addr"], cfg)
    assert is_hbm_addr_legal(layer_ir["output"][0]["addr"], cfg)
    # 不要求mask的addr
    return BATCH, N, M, K, N_block_num, M_block_num, K_block_num, mask_mode

def get_linear_layer_info(layer_ir, cfg: CFG, MM_or_MV: str):
    assert layer_ir["type"] in ("linear_mm", "linear_mv")
    assert len(layer_ir["input"]) == 1
    assert len(layer_ir["output"]) == 1
    batch_A, M, K_A = layer_ir["input"][0]["shape"]
    batch_B, K_B, N = layer_ir["param"][0]["shape"] # weight
    assert batch_A == batch_B == 1
    assert K_A == K_B
    assert layer_ir["output"][0]["shape"] == [1, M, N]
    K = K_A
    assert MM_or_MV in ("MM", "MV")
    if MM_or_MV == "MM":
        assert M > 1
    elif MM_or_MV == "MV":
        assert M == 1
    else:
        raise ValueError
    assert layer_ir["structure"]["mask_mode"] in ("fsb", "none")
    sparse_flag = (layer_ir["structure"]["mask_mode"] == "fsb")

    N_block_num, M_block_num, K_block_num = get_block_num(N, M, K, cfg)

    bias_flag = layer_ir["structure"]["bias_flag"]
    relu_flag = layer_ir["structure"]["relu_flag"]

    if bias_flag and sparse_flag:
        assert len(layer_ir["param"]) == 4 # 按顺序分别是weight, bias, meta, fsb
        param_weight_idx    = 0
        param_bias_idx      = 1
        param_meta_idx      = 2
        param_fsb_idx       = 3
    elif not bias_flag and sparse_flag:
        assert len(layer_ir["param"]) == 3 # 按顺序分别是weight, meta, fsb
        param_weight_idx    = 0
        param_bias_idx      = None
        param_meta_idx      = 1
        param_fsb_idx       = 2
    elif bias_flag and not sparse_flag:
        assert len(layer_ir["param"]) == 2 # 按顺序分别是weight, bias
        param_weight_idx    = 0
        param_bias_idx      = 1
        param_meta_idx      = None
        param_fsb_idx       = None
    else: # not bias_flag and not sparse_flag
        assert len(layer_ir["param"]) == 1 # 按顺序分别是weight
        param_weight_idx    = 0
        param_bias_idx      = None
        param_meta_idx      = None
        param_fsb_idx       = None

    assert param_weight_idx == 0
    assert is_hbm_addr_legal(layer_ir["input"][0]["addr"], cfg)
    assert is_hbm_addr_legal(layer_ir["output"][0]["addr"], cfg)
    assert is_hbm_addr_legal(layer_ir["param"][param_weight_idx]["addr"], cfg)
    if param_bias_idx is not None:
        assert layer_ir["param"][param_bias_idx]["shape"] == [1, 1, N]
        assert is_hbm_addr_legal(layer_ir["param"][param_bias_idx]["addr"], cfg)
    if param_meta_idx is not None:
        assert layer_ir["param"][param_meta_idx]["shape"][:2] == [1, 1] # 最后一维代表长度，不固定，与数据有关
        assert is_hbm_addr_legal(layer_ir["param"][param_meta_idx]["addr"], cfg)
    if param_fsb_idx is not None:
        assert layer_ir["param"][param_fsb_idx]["shape"] == [1, K_block_num, N_block_num]
        assert is_hbm_addr_legal(layer_ir["param"][param_fsb_idx]["addr"], cfg)

    return N, M, K, N_block_num, M_block_num, K_block_num, bias_flag, relu_flag, sparse_flag, param_weight_idx, param_bias_idx, param_meta_idx, param_fsb_idx

def get_misc_layer_info(layer_ir, cfg: CFG):
    BATCH, M, K = layer_ir["input"][0]["shape"]
    __, __, K_block_num = get_block_num(K, M, K, cfg)
    assert len(layer_ir["output"]) == 1

    if layer_ir["type"] == "softmax":
        assert len(layer_ir["input"]) == 1
        assert "param" not in layer_ir.keys() # softmax没有param
        assert layer_ir["input"][0]["shape"] == layer_ir["output"][0]["shape"] # softmax形状一致
        assert BATCH == 32
        assert layer_ir["structure"]["dim"] == 2
        eltwise_flag = False
        operation_name = layer_ir["type"] # softmax

    elif layer_ir["type"] == "layernorm":
        assert len(layer_ir["input"]) == 1
        assert "param" in layer_ir.keys() # layernorm有param，但可以被量化解决，不考虑仿射变换的部分
        assert layer_ir["input"][0]["shape"] == layer_ir["output"][0]["shape"] # layernorm形状一致
        if layer_ir["structure"]["bias_flag"]:
            assert len(layer_ir["param"]) == 2 # scale, bias
            assert layer_ir["param"][0]["shape"] == [1, 1, layer_ir["input"][0]["shape"][2]]
            assert layer_ir["param"][1]["shape"] == [1, 1, layer_ir["input"][0]["shape"][2]]
        else:
            assert len(layer_ir["param"]) == 1 # scale
            assert layer_ir["param"][0]["shape"] == [1, 1, layer_ir["input"][0]["shape"][2]]
        assert BATCH == 1
        eltwise_flag = False
        if layer_ir["structure"]["RMS_flag"]:
            operation_name = "RMSlayernorm"
        else:
            operation_name = "layernorm"

    elif layer_ir["type"] == "eltwise":
        assert len(layer_ir["input"]) == 2
        assert "param" not in layer_ir.keys() # eltwise没有param
        assert layer_ir["input"][0]["shape"] == layer_ir["input"][1]["shape"] == layer_ir["output"][0]["shape"] # eltwise形状一致
        assert layer_ir["structure"]["eltwise_type"] in ("eltwise_add", "eltwise_mul")
        eltwise_flag = True
        operation_name = layer_ir["structure"]["eltwise_type"] # eltwise_add / eltwise_mul

    elif layer_ir["type"] == "silu":
        assert len(layer_ir["input"]) == 1
        assert "param" not in layer_ir.keys() # silu没有param
        assert layer_ir["input"][0]["shape"] == layer_ir["output"][0]["shape"] # silu形状一致
        assert BATCH == 1
        eltwise_flag = False
        operation_name = layer_ir["type"] # silu

    else:
        raise ValueError("Unknown misc layer type: {}".format(layer_ir["type"]))

    return BATCH, M, K, K_block_num, eltwise_flag, operation_name

def ceil(value, unit):
    return (value + unit - 1) // unit

def align_ceil(value, alignment):
    return ceil(value, alignment) * alignment

def tiling_to_list(start_id, end_id, split_unit):
    # return a list of tiling tuple: [(tiling_start_id, tiling_end_id)]
    # example: start_id = 1, end_id = 11, split_unit = 3
    # return [(1, 4), (4, 7), (7, 10), (10, 11)]
    assert end_id > start_id
    assert start_id >= 0
    assert split_unit >= 0
    
    tiling_start_end_id_list = list()
    tiling_start_id = None
    tiling_end_id = start_id
    for i in range(start_id, end_id + split_unit, split_unit):
        tiling_start_id = tiling_end_id
        tiling_end_id = min(i, end_id)
        if i > start_id:
            tiling_start_end_id_list.append((tiling_start_id, tiling_end_id, ))
    return tiling_start_end_id_list

def get_hbm_channel_info_by_buffer_name(
        buffer_name: str, 
        cfg: CFG,
    ):
    assert buffer_name in ("A buffer", "B buffer", "global buffer")
    if buffer_name in ("A buffer", "global buffer"):
        hbm_channel_num = cfg.A_BUFFER_HBM_CHANNEL_NUM # == GLOBAL_BUFFER_HBM_CHANNEL_NUM
        hbm_channel_start_id = cfg.A_BUFFER_HBM_CHANNEL_START_ID # == GLOBAL_HBM_CHANNEL_START_ID
    elif buffer_name == "B buffer":
        hbm_channel_num = cfg.B_BUFFER_HBM_CHANNEL_NUM
        hbm_channel_start_id = cfg.B_BUFFER_HBM_CHANNEL_ID
    else:
        raise ValueError("Unknown buffer_name: {}".format(buffer_name))
    return hbm_channel_num, hbm_channel_start_id

def get_dense_matrix_hbm_addr(
        addr_base, 
        K_block_num, 
        K_block_id, 
        M_or_N_block_id, 
        cfg: CFG,
    ):
    # dense矩阵，存在hbm_channel_num个HBM channel中
    # 获取第K_block_id个K_block的地址
    # 注意！未考虑batch，如有batch需要在函数外addr_base中加上batch的偏移！
    assert K_block_id <= K_block_num
    single_block_size = cfg.FSB_BLOCK_SIZE * cfg.FSB_BLOCK_SIZE
    addr = addr_base + (M_or_N_block_id * K_block_num + K_block_id) * single_block_size
    return addr

