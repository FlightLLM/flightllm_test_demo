import os
import numpy as np
from utils import tools
from utils.config_generator import CFG
from inst_gen import isa

# compile concat layer
def generate_layer_inst(
        layer_ir: dict(), 
        cfg: CFG, 
        layer_first_LD_wait: list(), 
        layer_last_ST_release: list(), 
    ):
    if cfg.DEBUG:
        print(f"Generate inst for {layer_ir['type']} layer {layer_ir['name']}")
    assert len(layer_ir["input"]) == 2
    assert len(layer_ir["output"]) == 1
    assert layer_ir["structure"]["dim"] == 1
    assert layer_ir["input"][0]["shape"][1] >= 1 # 第1个输入是KV cache
    assert layer_ir["input"][1]["shape"][1] == 1 # 第2个输入是本次产生的
    each_channel_LD_KV_cache_size_B = tools.ceil(np.prod(layer_ir["input"][0]["shape"]), cfg.A_BUFFER_HBM_CHANNEL_NUM)

    inst_list = list()
    for channel_id in range(cfg.A_BUFFER_HBM_CHANNEL_NUM):
        LD_inst = isa.generate_LD_inst(
            LD_wait                 = [],
            LD_release              = [],
            LD_1d_length            = each_channel_LD_KV_cache_size_B,
            LD_hbm_addr             = 0,
            LD_bank_addr            = 0,
            LD_target_bank_name     = "global buffer",
            LD_hbm_channel_id       = channel_id,
            LD_cross_hbm_channel    = False,
            LD_parallel_channel_num = cfg.A_BUFFER_HBM_CHANNEL_NUM,
        )
        inst_list.append(LD_inst)

    ST_inst = isa.generate_ST_inst(
        ST_wait                 = [],
        ST_release              = [],
        ST_1d_length            = 0,
        ST_hbm_addr             = 0,
        ST_bank_addr            = 0,
        ST_hbm_channel_id       = 0,
        ST_parallel_channel_num = 1,
    )
    inst_list.append(ST_inst)

    inst_list = isa.set_first_wait_last_release(
        inst_list       = inst_list,
        first_wait      = layer_first_LD_wait,
        last_release    = layer_last_ST_release,
        first_inst_type = "LD",
        last_inst_type  = "ST",
    )
    return inst_list
