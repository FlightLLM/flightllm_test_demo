# VHK158 FPGA and GPU Profiling

## Requirements

- GPU: A100/V100
- CUDA 11.7 and above
- Pytorch 2.0.1 and above

## Introduction

Part 1: Profiling for FlightLLM on VHK158 FPGA   
All the hardware configurations are in config.yaml.  
The inputs and outputs of this profiling tool are in the `.compiler_output` folder.  
* **Input**: (in `.compiler_output/ir_output`)
    * Pre-generated intermediate representation (IR) of the LLaMA2-7B model for different token lengths (`.compiler_output/ir_output/*.yaml`)  
    * Sparse attention masks (`.compiler_output/ir_output/attention_mask/*.npy`)  
 
* **Output**: (in `.compiler_outpout/profiler_output`)
    * Profiling results of instruction elapsed time for different hardware modules (`.compiler_output/profiler_output/*.csv`)   


## Install

```bash
conda create -n fpga python==3.11 -y
conda activate fpga
# The required dependency packages will be automatically installed
pip install -r requirements.txt
```

## Usage

```bash
python run.py
```

## Test Cases
We provide six test cases for the LLaMA2-7B model that match the evaluations in figure 12 in our paper:
| Batch | Prefill token num | Decode token num |
| :---: | :---------------: | :--------------: |
|   1   |        128        |       512        |
|   1   |        128        |       1024       |
|   1   |        128        |       1536       |
|   1   |        512        |       512        |
|   1   |        512        |       1024       |
|   1   |        512        |       1536       |

For GPU, the profiling process for these 6 cases mat take about 20 minutes. For FlightLLM on VHK158, it may take about 20 minutes. We pre-generated the profiling data in `.compiler_output/profiler_output` folder in order to get the end to end FPGA profiling results quickly. If you want to regenerate these FPGA profiling data, you can execute the following code to delete this data and re-run `run.py`. In addition, if you want to test other cases with different prefill and decode lengths, you could modify the `test_case_list` in `run.py` and re-run it.  

```bash
rm -r .compiler_output/profiler_output
python run.py
```


## Output Example

```plaintext
Profiling for llama2-7B on GPU, backend: torch
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:12<00:00,  6.21s/it]
Profiling for llama2-7B on GPU, backend: vllm
INFO 12-11 16:02:22 llm_engine.py:73] Initializing an LLM engine with config: model='meta-llama/Llama-2-7b-hf', tokenizer='meta-llama/Llama-2-7b-hf', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=None, seed=0)
INFO 12-11 16:02:22 tokenizer.py:32] For some LLaMA V1 models, initializing the fast tokenizer may take a long time. To reduce the initialization time, consider using 'hf-internal-testing/llama-tokenizer' instead of the original tokenizer.
INFO 12-11 16:10:48 llm_engine.py:222] # GPU blocks: 7449, # CPU blocks: 512
Profiling for llama2-7B on VHK158 FPGA
+--------Case: llama2-7B, (Batch 1, Prefill 128, Decode  512)--------+
| Hardware                 |  GPU-naive  |   GPU-opt   | VHK158 FPGA |
| Prefill time (ms)        |      26.1   |      17.6   |      289.9  |
| Decode time (ms)         |   11316.9   |    5727.0   |     4946.4  |
| Total time (ms)          |   11343.0   |    5744.6   |     5236.3  |
| Avg throughput (token/s) |      45.2   |      89.4   |      103.5  |
| VHK158 speedup ratio     |     2.166   |     1.097   |      1.000  |
+--------Case: llama2-7B, (Batch 1, Prefill 128, Decode 1024)--------+
| Hardware                 |  GPU-naive  |   GPU-opt   | VHK158 FPGA |
| Prefill time (ms)        |      23.3   |      15.5   |      289.9  |
| Decode time (ms)         |   22627.5   |   11645.2   |    10318.6  |
| Total time (ms)          |   22650.9   |   11660.7   |    10608.4  |
| Avg throughput (token/s) |      45.3   |      87.9   |       99.2  |
| VHK158 speedup ratio     |     2.135   |     1.099   |      1.000  |
+--------Case: llama2-7B, (Batch 1, Prefill 128, Decode 1536)--------+
| Hardware                 |  GPU-naive  |   GPU-opt   | VHK158 FPGA |
| Prefill time (ms)        |      23.4   |      15.5   |      289.9  |
| Decode time (ms)         |   34196.5   |   17561.8   |    16103.1  |
| Total time (ms)          |   34219.9   |   17577.2   |    16393.0  |
| Avg throughput (token/s) |      44.9   |      87.5   |       95.4  |
| VHK158 speedup ratio     |     2.087   |     1.072   |      1.000  |
+--------Case: llama2-7B, (Batch 1, Prefill 512, Decode  512)--------+
| Hardware                 |  GPU-naive  |   GPU-opt   | VHK158 FPGA |
| Prefill time (ms)        |      50.7   |      40.4   |     1181.4  |
| Decode time (ms)         |   11393.2   |    5879.7   |     5269.0  |
| Total time (ms)          |   11443.9   |    5920.1   |     6450.5  |
| Avg throughput (token/s) |      44.9   |      87.1   |       97.2  |
| VHK158 speedup ratio     |     1.774   |     0.918   |      1.000  |
+--------Case: llama2-7B, (Batch 1, Prefill 512, Decode 1024)--------+
| Hardware                 |  GPU-naive  |   GPU-opt   | VHK158 FPGA |
| Prefill time (ms)        |      50.5   |      40.4   |     1181.4  |
| Decode time (ms)         |   23225.7   |   11899.8   |    10950.5  |
| Total time (ms)          |   23276.3   |   11940.2   |    12131.9  |
| Avg throughput (token/s) |      44.1   |      86.1   |       93.5  |
| VHK158 speedup ratio     |     1.919   |     0.984   |      1.000  |
+--------Case: llama2-7B, (Batch 1, Prefill 512, Decode 1536)--------+
| Hardware                 |  GPU-naive  |   GPU-opt   | VHK158 FPGA |
| Prefill time (ms)        |      51.3   |      40.7   |     1181.4  |
| Decode time (ms)         |   35129.2   |   17627.1   |    17044.3  |
| Total time (ms)          |   35180.6   |   17667.7   |    18225.7  |
| Avg throughput (token/s) |      43.7   |      87.1   |       90.1  |
| VHK158 speedup ratio     |     1.930   |     0.969   |      1.000  |
+--------------------------------------------------------------------+
```
