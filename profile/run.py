#-*-coding:utf-8-*-

import os
import time
import csv
# for fpga profile
from inst_gen.isa import ALL_INST_TYPE
from utils import config_generator
from inst_gen import inst_generator
from inst_gen import inst_profiler
# for gpu profile
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

test_case_list = [
    # batch, input_length, output_length
    (1, 128, 512),
    (1, 128, 1024),
    (1, 128, 1536),

    (1, 512, 512),
    (1, 512, 1024),
    (1, 512, 1536),
]


def align_token_length(stage, value):
    if stage == "prefill":
        unit = 128
    elif stage == "decode":
        unit = 16
    else:
        raise ValueError
    return ((value + unit - 1) // unit) * unit


def get_case_profile_time(cfg, model_name, stage_name, token_num):

    this_case_name = f"{model_name}_{stage_name}_token_{token_num}"
    assert os.path.exists(cfg.OUTPUT_DIR)
    csv_output_dir = os.path.join(cfg.OUTPUT_DIR, f"profiler_output/{this_case_name}.csv")

    if not os.path.isfile(csv_output_dir): # if there is csv file, then skip profiling and use the existing data
        model_ir_dir = os.path.join(cfg.OUTPUT_DIR, "ir_output", f"{this_case_name}.yaml")
        assert os.path.isfile(model_ir_dir), model_ir_dir

        inst_gen = inst_generator.InstGenerator(cfg, model_ir_dir)
        inst_list = inst_gen.run()

        profiler = inst_profiler.InstProfiler(cfg, inst_list, f"{model_name}_{stage_name}_token_{token_num}")
        profiler.run()

    # inst_type, inst_cnt, part_time, part_time_ratio
    with open(csv_output_dir, "r") as f:
        csv_reader = csv.DictReader(f)
        for (data_line) in csv_reader:
            if data_line["inst_type"] == "TOTAL": # total time for infenrece
                total_time_ms = float(data_line["part_time"])
            else:
                assert data_line["inst_type"] in ALL_INST_TYPE

    return total_time_ms


def profile_case_list_on_fpga(cfg, model_name, test_case_list):

    start_time = time.time()
    print(f"Profiling for llama2-7B on VHK158 FPGA")

    results = list()
    for batch, prefill_len, decode_len in test_case_list:
        # get case performance
        # print("----------------------------------------------------")
        # print(f"Profiling for llama2-7B: (Prefill {prefill_len}, Decode {decode_len})")
        assert batch == 1
        assert prefill_len + decode_len <= 2048
        # profile prefill time
        prefill_align_length = align_token_length("prefill", prefill_len)
        prefill_time_ms = get_case_profile_time(cfg, model_name, "prefill", prefill_align_length)

        # profile decode time for decode_len
        decode_time_ms = 0
        # get the align decode cases
        decode_time_ms_dict = dict()
        for i in range(decode_len):
            decode_align_length = align_token_length("decode", prefill_len + i + 1)
            decode_dict_key = str(decode_align_length)
            if decode_dict_key not in decode_time_ms_dict.keys():
                decode_time_ms_dict[decode_dict_key] = get_case_profile_time(cfg, model_name, "decode", decode_align_length)
            decode_time_ms += decode_time_ms_dict[decode_dict_key]

        total_time_ms = prefill_time_ms + decode_time_ms

        this_case_result = {
            "prefill_time_ms": prefill_time_ms,
            "decode_time_ms": decode_time_ms,
            "total_time_ms": total_time_ms,
            "decode_throughtput": decode_len / (decode_time_ms / 1000),
        }
        results.append(this_case_result)

        # print(f"(Prefill len, Decode len):  ({prefill_len}, {decode_len})")
        # print(f"Prefill time:            {prefill_time_ms:.3f} (ms)")
        # print(f"Decode  time:            {decode_time_ms:.3f} (ms)")
        # print(f"Total   time:            {total_time_ms:.3f} (ms)")
        # print(f"Decode avg throughput:   {decode_len / (decode_time_ms / 1000):.3f} (token/s)")
    # print("----------------------------------------------------")

    end_time = time.time()
    # print(f"Total time for profiling {len(test_case_list)} cases: {end_time - start_time:.3f} (s)")
    return results


def profile_case_list_on_gpu(model_name, backend, test_case_list):

    assert model_name == "llama2"
    assert backend in ("torch", "vllm")
    model_name = "meta-llama/Llama-2-7b-hf"
    torch.cuda.empty_cache()

    print(f"Profiling for llama2-7B on GPU, backend: {backend}")

    assert torch.cuda.is_available()
    device = torch.device("cuda:{}".format(0))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    if backend == 'torch':
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True, ignore_mismatched_sizes = True, max_position_embeddings = 1024*2 + 512).half().to(device)
    else:
        llm = LLM(model=model_name, trust_remote_code = True)

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    warmup, freq = 5, 10

    # print("batch  prefill size  decode size  Latency(ms)  Throughput")
    results = list()
    for batch, input_length, output_length in test_case_list:
        # prepare inputs
        input_ids = [[666] * input_length] * batch
        input_ids_t = torch.tensor(input_ids, device=device)
        max_length = input_length + output_length

        # params
        if backend == 'vllm':
            ignore_eos = True
            sampling_params_query = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=ignore_eos)
            sampling_params_total = SamplingParams(temperature=0.0, max_tokens=output_length+1, ignore_eos=ignore_eos)

        # warm up
        for _ in range(warmup):
            if backend == 'torch':
                logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+1, use_cache=True)
            else:
                outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params_query, use_tqdm=False)

        st.record()
        for _ in range(freq):
            if backend == 'torch':
                logits = model.generate(input_ids_t, num_beams=1, max_length=input_length+1, use_cache=True)
            else:
                outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params_query, use_tqdm=False)
        ed.record()
        ed.synchronize()
        query_latency = st.elapsed_time(ed) / freq

        st.record()
        for _ in range(freq):
            if backend == 'torch':
                logits = model.generate(input_ids_t, num_beams=1, min_length=input_length+output_length+1, max_length=input_length+output_length+1, use_cache=True)
            else:
                outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params_total, use_tqdm=False)
        ed.record()
        ed.synchronize()
        total_latency = st.elapsed_time(ed) / freq

        answer_lantency = total_latency - query_latency
        answer_token_output_latency = answer_lantency / output_length
        answer_tokens_per_second = (1000 / answer_token_output_latency) * batch

        this_case_result = {
            "prefill_time_ms": query_latency,
            "decode_time_ms": answer_lantency,
            "total_time_ms": total_latency,
            "decode_throughtput": answer_tokens_per_second,
        }
        results.append(this_case_result)

        # print(str(batch).ljust(len('batch')) + "  " +
        #         str(input_length).ljust(len('prefill size')) + "  " +
        #         str(output_length).ljust(len('decode size')) + "  " +
        #         "{:.3f}".format(total_latency).ljust(len('Latency(ms)')) + "  " +
        #         "{:.3f}".format(answer_tokens_per_second).ljust(len('Throughput'))) 
    return results


if __name__ == "__main__":
    model_name = "llama2"
    cfg = config_generator.CFG("./config.yaml")
    gpu_naive_results = profile_case_list_on_gpu(model_name, "torch", test_case_list)
    gpu_opt_results = profile_case_list_on_gpu(model_name, "vllm", test_case_list)

    fpga_results = profile_case_list_on_fpga(cfg, model_name, test_case_list)
    for test_case, gpu_naive_perf, gpu_opt_perf, fpga_perf in zip(test_case_list, gpu_naive_results, gpu_opt_results, fpga_results):
        print(f"+--------Case: llama2-7B, (Batch {test_case[0]:1d}, Prefill {test_case[1]:3d}, Decode {test_case[2]:4d})--------+")
        print(f"| Hardware                 |  GPU-naive  |   GPU-opt   | VHK158 FPGA |")
        print(f"| Prefill time (ms)        |   {gpu_naive_perf['prefill_time_ms']:7.1f}   |   {gpu_opt_perf['prefill_time_ms']:7.1f}   |    {fpga_perf['prefill_time_ms']:7.1f}  |")
        print(f"| Decode time (ms)         |   {gpu_naive_perf['decode_time_ms']:7.1f}   |   {gpu_opt_perf['decode_time_ms']:7.1f}   |    {fpga_perf['decode_time_ms']:7.1f}  |")
        print(f"| Total time (ms)          |   {gpu_naive_perf['total_time_ms']:7.1f}   |   {gpu_opt_perf['total_time_ms']:7.1f}   |    {fpga_perf['total_time_ms']:7.1f}  |")
        print(f"| Avg throughput (token/s) |   {gpu_naive_perf['decode_throughtput']:7.1f}   |   {gpu_opt_perf['decode_throughtput']:7.1f}   |    {fpga_perf['decode_throughtput']:7.1f}  |")
        print(f"| VHK158 speedup ratio     |   {(gpu_naive_perf['total_time_ms'] / fpga_perf['total_time_ms']):7.3f}   |   {(gpu_opt_perf['total_time_ms'] / fpga_perf['total_time_ms']):7.3f}   |    {1:7.3f}  |")
    print(f"+--------------------------------------------------------------------+")
