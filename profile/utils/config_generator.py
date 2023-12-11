# -*-coding:utf-8-*-
import os
import yaml

class CFG():
    def __init__(self, config_dir) -> None:
        assert os.path.exists(config_dir)
        with open(config_dir, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        ## OUTPUT_CONFIG:
        self.OUTPUT_DIR = cfg["OUTPUT_DIR"]
        self.DEBUG = cfg["DEBUG"]
        ## ALGORITHM_CONFIG:
        # max length
        self.MAX_LEN = cfg["MAX_LEN"]
        # sparse param
        self.FSB_BLOCK_SIZE = cfg["FSB_BLOCK_SIZE"]
        self.MASK_LAYOUT_BLOCK_SIZE = cfg["MASK_LAYOUT_BLOCK_SIZE"]
        self.FSB_BLOCK_NUM_PER_MASK_LAYOUT_BLOCK = cfg["MASK_LAYOUT_BLOCK_SIZE"] // cfg["FSB_BLOCK_SIZE"]

        ## FREQUENCY_CONFIG:
        self.CLOCK_FREQ_MHZ = cfg["CLOCK_FREQ_MHZ"]
        self.CLOCK_TIME_MS = (1 / cfg["CLOCK_FREQ_MHZ"]) * 0.001

        ## PARALLEL_CONFIG:
        # SLR
        self.SLR_NUM = cfg["SLR_NUM"]
        # mm parallelism
        self.MM_PARALLEL_N = cfg["MM_PARALLEL_N"]
        self.MM_PARALLEL_K = cfg["MM_PARALLEL_K"]
        self.MM_PARALLEL_M = cfg["MM_PARALLEL_M"]
        self.MM_PARALLEL_TOTAL = cfg["MM_PARALLEL_N"] * cfg["MM_PARALLEL_K"] * cfg["MM_PARALLEL_M"]
        # mv parallelism
        self.MV_PARALLEL_N = cfg["MV_PARALLEL_N"]
        self.MV_PARALLEL_K = cfg["MV_PARALLEL_K"]
        self.MV_PARALLEL_M = cfg["MV_PARALLEL_M"]
        self.MV_PARALLEL_TOTAL = cfg["MV_PARALLEL_N"] * cfg["MV_PARALLEL_K"] * cfg["MV_PARALLEL_M"]
        # compute start
        self.MM_START_N_NUM = cfg["MM_START_N_NUM"]
        self.MM_START_M_NUM = cfg["MM_START_M_NUM"]
        self.MV_START_N_NUM = cfg["MV_START_N_NUM"]
        self.MV_START_M_NUM = cfg["MV_START_M_NUM"]
        # misc parallelism
        self.SOFTMAX_PARALLEL_K = cfg["SOFTMAX_PARALLEL_K"]
        self.LAYERNORM_PARALLEL_K = cfg["LAYERNORM_PARALLEL_K"]
        self.ELTWISE_PARALLEL_K = cfg["ELTWISE_PARALLEL_K"]
        self.SILU_PARALLEL_K = cfg["SILU_PARALLEL_K"]

        ## BUFFER_CONFIG
        # A buffer (multiple banks)
        self.A_BUFFER_DEPTH = cfg["A_BUFFER_DEPTH"]
        self.A_BUFFER_PER_BANK_WIDTH_B = cfg["A_BUFFER_PER_BANK_WIDTH_B"]
        self.A_BUFFER_MM_PER_CHANNEL_BANK_NUM = cfg["A_BUFFER_MM_PER_CHANNEL_BANK_NUM"]
        self.A_BUFFER_MV_PER_CHANNEL_BANK_NUM = cfg["A_BUFFER_MV_PER_CHANNEL_BANK_NUM"]
        self.A_BUFFER_MM_PER_CHANNEL_WIDTH_B = cfg["A_BUFFER_MM_PER_CHANNEL_WIDTH_B"]
        self.A_BUFFER_MV_PER_CHANNEL_WIDTH_B = cfg["A_BUFFER_MV_PER_CHANNEL_WIDTH_B"]
        self.A_BUFFER_HBM_CHANNEL_NUM = cfg["A_BUFFER_HBM_CHANNEL_NUM"]
        self.A_BUFFER_HBM_CHANNEL_START_ID = cfg["A_BUFFER_HBM_CHANNEL_START_ID"]
        # B buffer (only 1 bank)
        self.B_BUFFER_DEPTH = cfg["B_BUFFER_DEPTH"]
        self.B_BUFFER_MM_BANK_WIDTH_B = cfg["B_BUFFER_MM_BANK_WIDTH_B"]
        self.B_BUFFER_MV_BANK_WIDTH_B = cfg["B_BUFFER_MV_BANK_WIDTH_B"]
        self.B_BUFFER_HBM_CHANNEL_NUM = cfg["B_BUFFER_HBM_CHANNEL_NUM"]
        self.B_BUFFER_HBM_CHANNEL_ID = cfg["B_BUFFER_HBM_CHANNEL_ID"]
        # global buffer (multiple banks, same shape as A buffer)
        self.GLOBAL_BUFFER_DEPTH = cfg["GLOBAL_BUFFER_DEPTH"]
        self.GLOBAL_BUFFER_PER_BANK_WIDTH_B = cfg["GLOBAL_BUFFER_PER_BANK_WIDTH_B"]
        self.GLOBAL_BUFFER_MM_PER_CHANNEL_BANK_NUM = cfg["GLOBAL_BUFFER_MM_PER_CHANNEL_BANK_NUM"]
        self.GLOBAL_BUFFER_MM_PER_CHANNEL_WIDTH_B = cfg["GLOBAL_BUFFER_MM_PER_CHANNEL_WIDTH_B"]
        self.GLOBAL_BUFFER_MV_PER_CHANNEL_BANK_NUM = cfg["GLOBAL_BUFFER_MV_PER_CHANNEL_BANK_NUM"]
        self.GLOBAL_BUFFER_MV_PER_CHANNEL_WIDTH_B = cfg["GLOBAL_BUFFER_MV_PER_CHANNEL_WIDTH_B"]
        self.GLOBAL_BUFFER_MISC_PER_CHANNEL_BANK_NUM = cfg["GLOBAL_BUFFER_MISC_PER_CHANNEL_BANK_NUM"]
        self.GLOBAL_BUFFER_MISC_PER_CHANNEL_WIDTH_B = cfg["GLOBAL_BUFFER_MISC_PER_CHANNEL_WIDTH_B"]
        self.GLOBAL_BUFFER_HBM_CHANNEL_NUM = cfg["GLOBAL_BUFFER_HBM_CHANNEL_NUM"]
        self.GLOBAL_BUFFER_HBM_CHANNEL_START_ID = cfg["GLOBAL_BUFFER_HBM_CHANNEL_START_ID"]
        # meta buffer (only 1 bank)
        self.META_BUFFER_DEPTH = cfg["META_BUFFER_DEPTH"]
        self.META_BUFFER_BANK_WIDTH_B = cfg["META_BUFFER_BANK_WIDTH_B"]
        self.META_BUFFER_HBM_CHANNEL_NUM = cfg["META_BUFFER_HBM_CHANNEL_NUM"]
        self.META_BUFFER_HBM_CHANNEL_ID = cfg["META_BUFFER_HBM_CHANNEL_ID"]
        # FSB buffer (only 1 bank)
        self.FSB_BUFFER_DEPTH = cfg["FSB_BUFFER_DEPTH"]
        self.FSB_BUFFER_BANK_WIDTH_B = cfg["FSB_BUFFER_BANK_WIDTH_B"]
        self.FSB_BUFFER_HBM_CHANNEL_NUM = cfg["FSB_BUFFER_HBM_CHANNEL_NUM"]
        self.FSB_BUFFER_HBM_CHANNEL_ID = cfg["FSB_BUFFER_HBM_CHANNEL_ID"]
        # bias buffer (only 1 bank)
        self.BIAS_BUFFER_DEPTH = cfg["BIAS_BUFFER_DEPTH"]
        self.BIAS_BUFFER_BANK_WIDTH_B = cfg["BIAS_BUFFER_BANK_WIDTH_B"]
        self.BIAS_BUFFER_HBM_CHANNEL_NUM = cfg["BIAS_BUFFER_HBM_CHANNEL_NUM"]
        self.BIAS_BUFFER_HBM_CHANNEL_ID = cfg["BIAS_BUFFER_HBM_CHANNEL_ID"]
        # inst
        self.INST_HBM_CHANNEL_NUM = cfg["INST_HBM_CHANNEL_NUM"]
        self.INST_HBM_CHANNEL_ID = cfg["INST_HBM_CHANNEL_ID"]

        ## HBM_CONFIG:
        self.HBM_CHANNEL_SIZE_B = cfg["HBM_CHANNEL_SIZE_MB"] * 1024 * 1024
        self.HBM_TOTAL_SIZE_B = cfg["HBM_TOTAL_SIZE_GB"] * 1024 * 1024 * 1024
        self.HBM_CHANNEL_NUM = cfg["HBM_CHANNEL_NUM"]
        self.HBM_CHANNEL_BW_B_PER_SEC = cfg["HBM_BW_GB_PER_SEC"] * 1024 * 1024 * 1024 / cfg["HBM_CHANNEL_NUM"] # B/s
        self.HBM_BW_SAME_CHANNEL_EFFICIENCY = cfg["HBM_BW_SAME_CHANNEL_EFFICIENCY"]
        self.HBM_BW_CROSS_CHANNEL_EFFICIENCY = cfg["HBM_BW_CROSS_CHANNEL_EFFICIENCY"]


        assert self.MASK_LAYOUT_BLOCK_SIZE % self.FSB_BLOCK_SIZE == 0
        assert self.MM_PARALLEL_K <= self.FSB_BLOCK_SIZE
        assert self.FSB_BLOCK_SIZE % self.MM_PARALLEL_K == 0

        assert self.MAX_LEN >= self.FSB_BLOCK_SIZE
        assert self.MAX_LEN % self.FSB_BLOCK_SIZE == 0
        assert self.MAX_LEN >= self.MASK_LAYOUT_BLOCK_SIZE
        assert self.MAX_LEN % self.MASK_LAYOUT_BLOCK_SIZE == 0

        assert self.A_BUFFER_PER_BANK_WIDTH_B == self.MM_PARALLEL_K
        assert self.A_BUFFER_MM_PER_CHANNEL_WIDTH_B == self.A_BUFFER_PER_BANK_WIDTH_B * self.A_BUFFER_MM_PER_CHANNEL_BANK_NUM
        assert self.A_BUFFER_MM_PER_CHANNEL_WIDTH_B // self.A_BUFFER_MM_PER_CHANNEL_BANK_NUM == self.A_BUFFER_MV_PER_CHANNEL_WIDTH_B // self.A_BUFFER_MV_PER_CHANNEL_BANK_NUM

        assert self.B_BUFFER_MM_BANK_WIDTH_B == self.MM_PARALLEL_N * self.MM_PARALLEL_K
        assert self.B_BUFFER_HBM_CHANNEL_ID == self.META_BUFFER_HBM_CHANNEL_ID == self.FSB_BUFFER_HBM_CHANNEL_ID == self.BIAS_BUFFER_HBM_CHANNEL_ID
        assert self.A_BUFFER_HBM_CHANNEL_START_ID + self.A_BUFFER_HBM_CHANNEL_NUM == self.B_BUFFER_HBM_CHANNEL_ID 
        assert self.B_BUFFER_HBM_CHANNEL_ID + 1 == self.INST_HBM_CHANNEL_ID
        assert self.B_BUFFER_HBM_CHANNEL_NUM == 1

        assert self.GLOBAL_BUFFER_PER_BANK_WIDTH_B == self.A_BUFFER_PER_BANK_WIDTH_B
        assert self.GLOBAL_BUFFER_MM_PER_CHANNEL_BANK_NUM == self.A_BUFFER_MM_PER_CHANNEL_BANK_NUM
        assert self.GLOBAL_BUFFER_MM_PER_CHANNEL_WIDTH_B == self.A_BUFFER_MM_PER_CHANNEL_WIDTH_B
        assert self.GLOBAL_BUFFER_HBM_CHANNEL_NUM == self.A_BUFFER_HBM_CHANNEL_NUM
        assert self.GLOBAL_BUFFER_HBM_CHANNEL_START_ID == self.A_BUFFER_HBM_CHANNEL_START_ID
        assert self.GLOBAL_BUFFER_MM_PER_CHANNEL_WIDTH_B // self.GLOBAL_BUFFER_MM_PER_CHANNEL_BANK_NUM == self.GLOBAL_BUFFER_MV_PER_CHANNEL_WIDTH_B // self.GLOBAL_BUFFER_MV_PER_CHANNEL_BANK_NUM
        assert self.GLOBAL_BUFFER_MM_PER_CHANNEL_BANK_NUM * 16 == self.GLOBAL_BUFFER_MM_PER_CHANNEL_WIDTH_B
        assert self.GLOBAL_BUFFER_MV_PER_CHANNEL_BANK_NUM * 16 == self.GLOBAL_BUFFER_MV_PER_CHANNEL_WIDTH_B
        assert self.GLOBAL_BUFFER_MISC_PER_CHANNEL_BANK_NUM * 16 == self.GLOBAL_BUFFER_MISC_PER_CHANNEL_WIDTH_B

        assert self.META_BUFFER_DEPTH == self.B_BUFFER_DEPTH
        assert self.META_BUFFER_BANK_WIDTH_B * 2 == self.B_BUFFER_MM_BANK_WIDTH_B
        assert self.META_BUFFER_HBM_CHANNEL_NUM == 1

        assert self.FSB_BUFFER_HBM_CHANNEL_NUM == 1

        assert self.BIAS_BUFFER_BANK_WIDTH_B % 4 == 0 # bias is int32
        assert self.BIAS_BUFFER_HBM_CHANNEL_NUM == 1

        assert self.INST_HBM_CHANNEL_NUM == 1

        # print(f"Load config from {config_dir}")
