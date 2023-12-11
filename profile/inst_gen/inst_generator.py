import yaml
import os
from inst_gen.layer_support import IR_attention_mm
from inst_gen.layer_support import IR_attention_mv
from inst_gen.layer_support import IR_linear_mm
from inst_gen.layer_support import IR_linear_mv
from inst_gen.layer_support import IR_misc
from inst_gen.layer_support import IR_concat
from inst_gen import isa
from utils.config_generator import CFG

ALL_HW_LAYER_TYPE = ("linear_mm", "linear_mv", "attention_mm", "attention_mv", "eltwise", "layernorm", "softmax", "concat", "silu", "input", "output")

class InstGenerator():
    def __init__(
            self, 
            cfg: CFG,
            model_ir_dir = None,
        ):
        self.ir_output_dir = os.path.join(cfg.OUTPUT_DIR, "ir_output")
        self.profiler_output_dir = os.path.join(cfg.OUTPUT_DIR, "profiler_output")
        self.cfg = cfg
        self.model_inst_list = list()
        assert os.path.exists(self.ir_output_dir)
        if not os.path.exists(self.profiler_output_dir):
            os.mkdir(self.profiler_output_dir)

        self.model_ir_dir = os.path.join(self.ir_output_dir, "IR.yaml") if model_ir_dir is None else model_ir_dir
        # print(f"Load IR from {self.model_ir_dir}")
        with open(self.model_ir_dir, "r") as f:
            self.model_ir = yaml.load(f, Loader=yaml.FullLoader)
        assert self.model_ir is not None
    
    def run(self):
        # 生成每层IR对应的指令

        """ 正式版需要删除这一段 """
        first_layer_id = 0
        last_layer_id = len(self.model_ir)
        for i in range(len(self.model_ir)):
            if self.model_ir[i]["type"] == "input":
                first_layer_id = i + 1
                break
        for i in range(len(self.model_ir)):
            reverse_i = len(self.model_ir) - i - 1
            if self.model_ir[reverse_i]["type"] == "output":
                last_layer_id = reverse_i
                break
        assert last_layer_id > first_layer_id

        now_layer_id = first_layer_id
        while now_layer_id < last_layer_id:
            layer_ir = self.model_ir[now_layer_id]
            next_layer_ir = self.model_ir[now_layer_id + 1] if now_layer_id < (last_layer_id - 1) else None # 如果还有下一层，取出下一层IR
            layer_first_LD_wait = [] if now_layer_id == first_layer_id else ["SYS"]
            layer_last_ST_release = ["SYS"]
            layer_last_SYS_wait = ["ST"]
            layer_last_SYS_release = [] if now_layer_id == (last_layer_id - 1) else ["LD"]
            # 按照类型，生成对应的指令
            assert layer_ir["type"] in ALL_HW_LAYER_TYPE, f"Layer type {layer_ir['type']} is not supported!"
            if layer_ir["type"] in ("attention_mm", "attention_mv"):
                if next_layer_ir is not None and next_layer_ir["type"] == "softmax" and layer_ir["output"][0]["name"] == next_layer_ir["input"][0]["name"]: # 如果下一层是相连的softmax层，则可以融合
                    fuse_layer_ir = next_layer_ir
                else:
                    fuse_layer_ir = None

                if fuse_layer_ir is not None:
                    now_layer_id += 1 # fuse layer 
                    layer_last_SYS_release = [] if now_layer_id == (last_layer_id - 1) else ["LD"] # 更新layer_last_SYS_release
                # fuse_layer_ir = None
                if layer_ir["type"] == "attention_mm":
                    layer_inst_list = IR_attention_mm.generate_layer_inst(layer_ir, fuse_layer_ir, self.cfg, layer_first_LD_wait, layer_last_ST_release)
                elif layer_ir["type"] == "attention_mv":
                    layer_inst_list = IR_attention_mv.generate_layer_inst(layer_ir, fuse_layer_ir, self.cfg, layer_first_LD_wait, layer_last_ST_release)
                else:
                    raise ValueError

            elif layer_ir["type"] in ("linear_mm", "linear_mv"):
                # silu和eltwise不要求K维度全算完，所以可以和MV融合
                if next_layer_ir is not None and next_layer_ir["type"] == "silu" and layer_ir["output"][0]["name"] == next_layer_ir["input"][0]["name"]: # 如果下一层是相连的silu层，则可以融合
                    fuse_layer_ir = next_layer_ir
                    fuse_layer_num = 1
                    fuse_layer_ir["fuse_layer_num"] = fuse_layer_num
                elif next_layer_ir is not None and next_layer_ir["type"] == "eltwise" and layer_ir["output"][0]["name"] in (next_layer_ir["input"][0]["name"], next_layer_ir["input"][1]["name"]): # 如果下一层是相连的eltwise层，则可以融合
                    fuse_layer_ir = next_layer_ir
                    fuse_layer_id = now_layer_id + 1 # 第一个可融合的层
                    fuse_layer_num = 0
                    while fuse_layer_id < last_layer_id: # 检测是否RopE，可连续融合
                        if self.model_ir[fuse_layer_id]["type"] == "eltwise":
                            if fuse_layer_num > 0: # 证明已经有连着的eltwise层了
                                assert "RoPE" in self.model_ir[fuse_layer_id]["name"]
                            fuse_layer_id += 1
                            fuse_layer_num += 1
                        else:
                            break
                    assert fuse_layer_num in (1, 3) # 1 for normal, 3 for Q, K RoPE
                    fuse_layer_ir["fuse_layer_num"] = fuse_layer_num
                else:
                    fuse_layer_ir = None
                    fuse_layer_num = 0

                if fuse_layer_ir is not None:
                    now_layer_id += fuse_layer_num # fuse layer 
                    layer_last_SYS_release = [] if now_layer_id == (last_layer_id - 1) else ["LD"] # 更新layer_last_SYS_release
                # fuse_layer_ir = None
                if layer_ir["type"] == "linear_mm":
                    layer_inst_list = IR_linear_mm.generate_layer_inst(layer_ir, fuse_layer_ir, self.cfg, layer_first_LD_wait, layer_last_ST_release)
                elif layer_ir["type"] == "linear_mv":
                    layer_inst_list = IR_linear_mv.generate_layer_inst(layer_ir, fuse_layer_ir, self.cfg, layer_first_LD_wait, layer_last_ST_release)
                else:
                    raise ValueError

            elif layer_ir["type"] in ("softmax", "layernorm", "eltwise", "silu"):
                layer_inst_list = IR_misc.generate_layer_inst(layer_ir, self.cfg, layer_first_LD_wait, layer_last_ST_release)

            elif layer_ir["type"] == "concat":
                layer_inst_list = IR_concat.generate_layer_inst(layer_ir, self.cfg, layer_first_LD_wait, layer_last_ST_release)

            else:
                print(f"CAUTION: Layer {layer_ir['name']} is not supported yet, layer type is {layer_ir['type']}")
                raise NotImplementedError
            # 增加SYS指令同步本层所有SLR
            layer_inst_list.append(isa.generate_SYS_inst(layer_last_SYS_wait, layer_last_SYS_release))

            self.model_inst_list.append(layer_inst_list)
            now_layer_id += 1
            
        return self.model_inst_list
