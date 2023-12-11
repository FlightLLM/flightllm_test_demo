# -*-coding:utf-8-*-
import numpy as np
import os
from inst_gen.isa import ALL_INST_TYPE, ALL_MISC_OP_TYPE
from utils.config_generator import CFG
        
class InstProfiler:
    def __init__(
            self, 
            cfg: CFG,
            model_inst_list: list,
            this_case_name: str,
        ):
        self.module_num = len(ALL_INST_TYPE)
        self.module_inst_cnt = dict()
        for module_name in ALL_INST_TYPE:
            self.module_inst_cnt[module_name] = 0
        self.total_inst_cnt = 0
        self.model_inst_list = model_inst_list # 存储每条指令的dict
        self.this_case_name = this_case_name
        self.correction_factor = 3.2
        self.cfg = cfg
        self.profiler_output_dir = os.path.join(cfg.OUTPUT_DIR, "profiler_output")

    def time_LD(self, mode, length_1d, loop_1d, loop_2d, loop_3d, zero_fill, cross_hbm_channel, parallel_channel_num, bandwidth_ratio):
        if mode != 0:
            assert loop_2d == 1 and loop_3d == 1
        total_data_B = length_1d * loop_1d * loop_2d * loop_3d

        if cross_hbm_channel:
            assert parallel_channel_num == 1
            bandwidth_eff = self.cfg.HBM_BW_CROSS_CHANNEL_EFFICIENCY * bandwidth_ratio
        else:
            assert parallel_channel_num in (1, self.cfg.A_BUFFER_HBM_CHANNEL_NUM, self.cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM) # 1 or 8
            bandwidth_eff = self.cfg.HBM_BW_SAME_CHANNEL_EFFICIENCY * bandwidth_ratio

        if zero_fill:
            # 补0则可以按bank满带宽写数
            assert bandwidth_ratio == 1
            assert not cross_hbm_channel
            bandwidth_B_s = self.cfg.A_BUFFER_MM_PER_CHANNEL_WIDTH_B * parallel_channel_num / self.cfg.CLOCK_TIME_MS * 1e3
        else:
            bandwidth_B_s = self.cfg.HBM_CHANNEL_BW_B_PER_SEC * parallel_channel_num * bandwidth_eff

        total_time_ms = total_data_B / bandwidth_B_s * 1e3 # ms
        return total_time_ms

    def time_ST(self, length_1d, loop_1d, cross_hbm_channel, parallel_channel_num):
        total_data_B = length_1d * loop_1d
        if cross_hbm_channel:
            assert parallel_channel_num == 1
            bandwidth_B_s = self.cfg.HBM_CHANNEL_BW_B_PER_SEC * self.cfg.HBM_BW_CROSS_CHANNEL_EFFICIENCY
        else:
            assert parallel_channel_num in (1, self.cfg.A_BUFFER_HBM_CHANNEL_NUM, self.cfg.GLOBAL_BUFFER_HBM_CHANNEL_NUM) # 1 or 8
            bandwidth_B_s = self.cfg.HBM_CHANNEL_BW_B_PER_SEC * self.cfg.HBM_BW_SAME_CHANNEL_EFFICIENCY * parallel_channel_num
        total_time_ms = total_data_B / bandwidth_B_s * 1e3 # ms
        return total_time_ms

    def time_MM(self, sparse_flag, K, sparse_ratio):
        computation = self.cfg.MM_START_M_NUM * K * self.cfg.FSB_BLOCK_SIZE * self.cfg.MM_START_N_NUM
        if sparse_flag:
            assert sparse_ratio < 1.0
            computation *= sparse_ratio
        else:
            assert sparse_ratio == 1.0
        total_cycle = computation / self.cfg.MM_PARALLEL_TOTAL
        total_time_ms = total_cycle * self.cfg.CLOCK_TIME_MS
        return total_time_ms

    def time_MV(self, sparse_flag, K, sparse_ratio):
        computation = self.cfg.MV_START_M_NUM * K * self.cfg.FSB_BLOCK_SIZE * self.cfg.MV_START_N_NUM
        if sparse_flag:
            assert sparse_ratio < 1.0
            computation *= sparse_ratio
        else:
            assert sparse_ratio == 1.0
        total_cycle = computation / self.cfg.MV_PARALLEL_TOTAL
        total_time_ms = total_cycle * self.cfg.CLOCK_TIME_MS
        return total_time_ms

    def time_MISC(self, operation_flag, K):
        misc_op_type = ALL_MISC_OP_TYPE[operation_flag]
        if misc_op_type in ("eltwise_add", "eltwise_mul"):
            total_cycle = K * self.cfg.FSB_BLOCK_SIZE / self.cfg.ELTWISE_PARALLEL_K
        elif misc_op_type == "softmax":
            total_cycle = K * self.cfg.FSB_BLOCK_SIZE / self.cfg.SOFTMAX_PARALLEL_K * 2 # 需要先过一遍所有数据再计算，因此需要乘2
        elif misc_op_type in ("layernorm", "RMSlayernorm"):
            total_cycle = K * self.cfg.FSB_BLOCK_SIZE / self.cfg.LAYERNORM_PARALLEL_K * 2 # 需要先过一遍所有数据再计算，因此需要乘2
        elif misc_op_type == "silu":
            total_cycle = K * self.cfg.FSB_BLOCK_SIZE / self.cfg.SILU_PARALLEL_K
        else:
            raise ValueError(f"Unknown misc op type {misc_op_type}")
        total_time_ms = total_cycle * self.cfg.CLOCK_TIME_MS
        return total_time_ms

    def time_SYS(self):
        total_cycle = 1
        total_time_ms = total_cycle * self.cfg.CLOCK_TIME_MS
        return total_time_ms

    def get_inst_time_ms(self, inst_dict):
        inst_type = inst_dict["TYPE"]
        inst_param = inst_dict["PARAM"]
        if inst_type == "LD":
            time = self.time_LD(
                inst_param["mode"],
                inst_param["1d_length"],
                inst_param["1d_loop"],
                inst_param["2d_loop"],
                inst_param["3d_loop"],
                inst_param["zero_fill"],
                inst_dict["NOTE"]["cross_hbm_channel"],
                inst_dict["NOTE"]["parallel_channel_num"],
                inst_dict["NOTE"]["bandwidth_ratio"],
            )
        elif inst_type == "ST":
            time = self.time_ST(
                inst_param["1d_length"],
                inst_param["1d_loop"],
                inst_dict["NOTE"]["cross_hbm_channel"],
                inst_dict["NOTE"]["parallel_channel_num"],
            )
        elif inst_type == "MM":
            time = self.time_MM(
                inst_param["sparse_flag"],
                inst_param["K"],
                inst_dict["NOTE"]["sparse_ratio"],
            )
        elif inst_type == "MV":
            time = self.time_MV(
                inst_param["sparse_flag"],
                inst_param["K"],
                inst_dict["NOTE"]["sparse_ratio"],
            )
        elif inst_type == "MISC":
            time = self.time_MISC(
                inst_param["operation_flag"],
                inst_param["K"],
            )
        else:
            assert inst_type == "SYS"
            time = self.time_SYS()
        return time

    def run(self, start_time = 0.):
        """
        输入是每层的指令列表，本层的起始时间，本层初始的wait参数
        输出是本层每条指令的时间起点，时长，结束时间，wait release参数
        """
        # 遍历现有的inst_dict, 存储指令的依赖以及每条指令的执行时间
        inst_wr_time = []
        inst_last_init = []
        self.module_num = len(ALL_INST_TYPE)
        for i in range(self.module_num):
            inst_wr_time.append([])
            inst_last_init.append(None)
        # 自动check
        pub_array = np.zeros(shape = (self.module_num, self.module_num), dtype = np.int64)
        rev_array = np.zeros(shape = (self.module_num, self.module_num), dtype = np.int64)
        # 遍历所有的指令
        for layer_inst_list in self.model_inst_list:
            for inst_dict in layer_inst_list:
                inst_type = inst_dict["TYPE"]
                self.module_inst_cnt[inst_type] += 1
                self.total_inst_cnt += 1
                assert inst_type in ALL_INST_TYPE
                inst_type_id = ALL_INST_TYPE.index(inst_type)

                # get wr and time
                wait_list = inst_dict["WAIT_LIST"]
                wait_id_list = list()
                for wait_name in wait_list:
                    wait_id_list.append(ALL_INST_TYPE.index(wait_name))
                
                release_list = inst_dict["RELEASE_LIST"]
                release_id_list = list()
                for release_name in release_list:
                    release_id_list.append(ALL_INST_TYPE.index(release_name))

                time = self.get_inst_time_ms(inst_dict) / self.correction_factor
                inst_wr_time[inst_type_id].append((wait_id_list, release_id_list, time))

                # record for wait release
                for wait_id in wait_id_list:
                    rev_array[wait_id][inst_type_id] += 1
                for release_id in release_id_list:
                    pub_array[inst_type_id][release_id] += 1

        # check for wait release number
        for i in range(self.module_num):
            for j in range(self.module_num):
                if pub_array[i][j] != rev_array[i][j] and i != 0 and j != 0 and \
                    (not (i == 2 and j == 1 and pub_array[i][j] == rev_array[i][j] + 1)) and \
                    (not (i == 2 and j == 1 and pub_array[i][j] == rev_array[i][j] - 1)):
                    raise Exception(f"{ALL_INST_TYPE[i]} -> {ALL_INST_TYPE[j]}, pub {pub_array[i][j]}, rev {rev_array[i][j]}")
        # 根据一个原则来执行如何通过单线程模拟不同模块之间的时间和依赖
        # 每一时刻选择起始时间最小的能执行的来执行
        wr_array = []
        wr_index = []
        for i in range(self.module_num):
            wr_array.append([])
            wr_index.append([])
            for j in range(self.module_num):
                wr_array[i].append([])
                wr_index[i].append(0)

        # if start_wait != None:
        #     wr_array[start_wait[0]][start_wait[1]].append(start_time)

        # 代表着当前每条指令的执行编号
        inst_index = []
        inst_hardware_time = []
        for i in range(self.module_num):
            inst_index.append(0)
            inst_hardware_time.append(start_time)
        # 用于各单元利用情况画图的信息
        self.inst_plot_left = []
        self.inst_plot_width = []
        for i in range(self.module_num):
            self.inst_plot_left.append([])
            self.inst_plot_width.append([])
        # 此原则是，找到五类指令中满足依赖关系的，开始时间最小的作为本次指令的执行时间开始
        # 并对wait和release列表进行更新
        while True:
            # 判断是否有满足要求的指令可以执行
            inst_start_time = []
            for i in range(self.module_num):
                index = inst_index[i]
                # 是否已经是完成的指令序列
                if index > (len(inst_wr_time[i]) - 1):
                    continue
                # 判断依赖关系是否满足
                if len(inst_wr_time[i][index][0]) == 0:
                    # 如果没有任何依赖，那么这个开始时间由硬件依赖决定，即上条指令的结束时间
                    inst_start_time.append((i, inst_hardware_time[i]))
                else:
                    # 如果有依赖，那么需要查询这个依赖在当前状态下是否被满足
                    for w in inst_wr_time[i][index][0]:
                        # 依赖被满足需要所有的wr_index小于完整的长度
                        if wr_index[w][i] >= len(wr_array[w][i]):
                            break
                    else:
                        start_time = max([wr_array[w][i][wr_index[w][i]] for w in inst_wr_time[i][index][0]])
                        start_time = max(inst_hardware_time[i], start_time)
                        inst_start_time.append((i, start_time))
            # 对当前的状态进行一些判断
            if len(inst_start_time) == 0:
                # 说明所有的指令都无法执行，那么必然要求两种条件
                # 一是所有的指令完全执行
                for i in range(self.module_num):
                    assert inst_index[i] == len(inst_wr_time[i])
                # 二是所有的依赖都已经清空
                wait_release_count = 0
                for i in range(self.module_num):
                    for j in range(self.module_num):
                        if wr_index[i][j] < len(wr_array[i][j]):
                            wait_release_count += (len(wr_array[i][j]) - wr_index[i][j])
                assert wait_release_count == 0
                break
            # 从所有开始时间中选择最小的一条指令
            inst_start_time = sorted(inst_start_time, key=lambda x:x[1])
            chosen_id = inst_start_time[0][0]
            start_time = inst_start_time[0][1]
            wait_id_list, release_id_list, time = inst_wr_time[chosen_id][inst_index[chosen_id]]
            end_time = start_time + time
            # 更新中间结果的值
            # wait
            for wait_id in wait_id_list:
                wr_index[wait_id][chosen_id] += 1
            # release
            for release_id in release_id_list:
                wr_array[chosen_id][release_id].append(end_time)
            # inst_index
            inst_index[chosen_id] += 1
            # hardware_time
            inst_hardware_time[chosen_id] = end_time
            # record time
            # check overlap
            if len(self.inst_plot_left[chosen_id]) > 0:
                assert self.inst_plot_left[chosen_id][-1] + self.inst_plot_width[chosen_id][-1] <= start_time
            self.inst_plot_left[chosen_id].append(start_time)
            self.inst_plot_width[chosen_id].append(time)
        # total time
        self.inst_total_time = max(inst_hardware_time)
        
        file_out_str = "inst_type,inst_cnt,part_time,part_time_ratio\n"
        for i in range(self.module_num):
            if len(self.inst_plot_width[i]) > 0:
                part_time = sum(self.inst_plot_width[i])
                file_out_str += "%s,%d,%f,%f\n" % (ALL_INST_TYPE[i], self.module_inst_cnt[ALL_INST_TYPE[i]], part_time, part_time / self.inst_total_time * 100)
                # print(time_table_str)
            else:
                file_out_str += "%s,%d,0,0\n" % (ALL_INST_TYPE[i], self.module_inst_cnt[ALL_INST_TYPE[i]])

        file_out_str += "TOTAL,%d,%f,100\n" % (self.total_inst_cnt, self.inst_total_time)
        csv_output_dir = os.path.join(self.profiler_output_dir, f"{self.this_case_name}.csv")
        with open(csv_output_dir, "w") as f:
            f.write(file_out_str)
        # print(f"Dump csv output to {csv_output_dir}")
        # print("|----------------------------------------------|")
        # print("| TOTAL |  %8d  |  %.2e ms  |   100%%  |" % (self.total_inst_cnt, self.inst_total_time))
        # print("================================================")
