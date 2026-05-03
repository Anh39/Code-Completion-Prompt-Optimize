from typing import List, Tuple
import tqdm
import asyncio
import random
import time
import warnings
import numpy as np
import argparse
import os
import json
import requests
from dataclasses import asdict

from transformers import AutoTokenizer
from utils.prompt import construct_prompt
from utils.metrics import calculate_metrics
from utils.splitter import split_code_file_text_into_three_parts
from utils.data import load_data, get_sample_text
from request_vllm import async_request_openai_completions, RequestFuncInput, RequestFuncOutput
from launch_server import ServerContextManager

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen2.5-Coder-0.5B')
    parser.add_argument('--modes', nargs="+", type=str, default=["fim", "efim", "sfim"])
    parser.add_argument('--splits', nargs="+", type=str, default=["efim", "cceval", "function"])
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=65511)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--num-round', type=int, default=5)
    parser.add_argument('--num-user', type=int, default=16)
    parser.add_argument('--output-file', type=str, default="performance.json")
    args = parser.parse_args()
    return args
def _split_k_parts(text: str, k: int) -> list[str]:
    n = len(text)
    base = n // k
    extra = n % k
    parts = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        parts.append(text[start:start+size])
        start += size
    return parts
def _efim_split(model_id: str, prompt_mode: str, prompt_tot: str, num_round: int) -> list[str]:
    tot_prefix_len = len(prompt_tot)//2
    prefix_len = int(tot_prefix_len/(num_round+2))
    suffix = prompt_tot[tot_prefix_len:]
    prefix = prompt_tot[:prefix_len]
    prompts = []
    for round_idx in range(num_round):
        middle = prompt_tot[prefix_len:int(
            tot_prefix_len/(num_round+2)*(round_idx+2))]
        input_text = construct_prompt(prompt_mode, model_id, prefix, middle, suffix)
        prompts.append(input_text)
    return prompts
def _function_split(model_id: str, prompt_mode: str, prompt_tot: str, num_round: int) -> list[str]:
    prefix, middle, suffix = split_code_file_text_into_three_parts(prompt_tot, "python")
    parts = _split_k_parts(middle, num_round+1)
    text = parts.pop(0)
    prompts = []
    for part in parts:
        text += part
        input_text = construct_prompt(prompt_mode, model_id, prefix, text, suffix)
        prompts.append(input_text)
    return prompts
def _cceval_split(model_id: str, prompt_mode: str, prefix: str, middle: str, suffix: str, num_round: int) -> list[str]:
    parts = _split_k_parts(middle, num_round+1)
    text = parts.pop(0)
    prompts = []
    for part in parts:
        text += part
        input_text = construct_prompt(prompt_mode, model_id, prefix, text, suffix)
        prompts.append(input_text)
    return prompts
def get_vllm_metrics(host: str, port: int):
    time.sleep(5)
    resp = requests.get(f"http://{host}:{port}/metrics")
    lines = resp.text.splitlines()
    metrics = {
        "vllm:gpu_prefix_cache_queries_total": 0.0,
        "vllm:gpu_prefix_cache_hits_total": 0.0
    }
    for line in lines:
        if line.startswith("vllm:gpu_prefix_cache_queries_total"):
            metrics["vllm:gpu_prefix_cache_queries_total"] = float(line.split("}")[-1].strip())
        if line.startswith("vllm:gpu_prefix_cache_hits_total"):
            metrics["vllm:gpu_prefix_cache_hits_total"] = float(line.split("}")[-1].strip())
    metrics["cache_reuse_percent"] = 100 * metrics["vllm:gpu_prefix_cache_hits_total"] / metrics["vllm:gpu_prefix_cache_queries_total"] if metrics["vllm:gpu_prefix_cache_queries_total"] > 0 else 0.0 
    return metrics
def benchmark_each(args: argparse.Namespace, prompt_mode: str, split_mode: str):
    model_id = args.model_id
    num_round = args.num_round
    num_user = args.num_user
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    port = args.port
    host = args.host
    
    api_url = f'http://{host}:{port}/v1/completions'
    random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    eos_token_ids = [tokenizer.eos_token_id]
    if "llama3.1-8b-train" in model_id.lower() and "enhance" not in model_id.lower():
        # Since we use continual pretrained model of EFIm paper
        eos_token_ids = [128000]
    # elif "deepseek" in model_id.lower() and "enhance" in model_id.lower():
    #     eos_token_ids = [32027]
    async def user_request(prompt_mode: str, split_mode: str, prefix: str, middle: str, suffix: str, pbar: tqdm.tqdm):
        tot_input_len = 0
        total = prefix + middle + suffix
        if split_mode == "efim":
            prompts = _efim_split(model_id, prompt_mode, total, num_round)
        elif split_mode == "function":
            prompts = _function_split(model_id, prompt_mode, total, num_round)
        elif split_mode == "cceval":
            prompts = _cceval_split(model_id, prompt_mode, prefix, middle, suffix, num_round)
        else:
            raise Exception(f"Split mode not supported: {split_mode}")
        outputs = []
        for input_text in prompts:
            tot_input_len += len(tokenizer.encode(input_text))
            request_input = RequestFuncInput(input_text, api_url, len(input_text), max_tokens, model_id,
                                             temperature, top_p, top_k, eos_token_ids, True)
            output = await async_request_openai_completions(
                request_func_input=request_input,
                pbar=pbar)
            outputs.append(output)
        return tot_input_len , outputs

    async def benchmark(prompt_mode: str, split_mode: str):
        problems = load_data("cceval-python")
        problems_filter = []
        for problem in problems:
            input_len = len(tokenizer.encode(get_sample_text("cceval-python", problem)))
            if input_len > 2048 and input_len < 4096:
                problems_filter.append(problem)
        problems = random.sample(problems_filter, num_user)

        start_time = time.perf_counter()
        pbar = tqdm.tqdm(total=len(problems)*num_round)
        tasks = []
        for problem in problems:
            prefix = problem["prompt"]
            suffix = problem["right_context"]
            middle = problem["groundtruth"]
            tasks.append(asyncio.create_task(user_request(prompt_mode, split_mode, prefix, middle, suffix, pbar)))

        outputs = await asyncio.gather(*tasks)
        pbar.close()

        benchmark_duration = time.perf_counter() - start_time

        output_ret = []
        tot_input_len = 0
        for outputs_each in outputs:
            output_ret.extend(outputs_each[1])
            tot_input_len += outputs_each[0]
        
        metrics, actual_output_lens = calculate_metrics(
            outputs=output_ret, dur_s=benchmark_duration, tokenizer=tokenizer, tot_input_len=tot_input_len)

        print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
        print("{:<40} {:<10}".format("Total requests:", len(outputs)))
        print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                        benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
        print("{:<40} {:<10}".format("Total generated tokens:",
                                     metrics.total_output))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                        metrics.request_throughput))
        print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                        metrics.output_throughput))
        print("{:<40} {:<10.2f}".format(
            "Avg latency (ms):", metrics.avg_latency_ms))
        print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            "Mean TTFT (ms):", metrics.mean_ttft_ms))
        print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                        metrics.median_ttft_ms))
        print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
        print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                                   n=50,
                                   c='-'))
        print("{:<40} {:<10.2f}".format(
            "Mean TPOT (ms):", metrics.mean_tpot_ms))
        print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                        metrics.median_tpot_ms))
        print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
        print("{s:{c}^{n}}".format(s='Inter-token Latency', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
        print("{:<40} {:<10.2f}".format(
            "Median ITL (ms):", metrics.median_itl_ms))
        print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
        print("=" * 50)
        result = asdict(metrics)
        result["total"] = len(tasks)
        return result
    async def total_task():
        with ServerContextManager(model_id, model_id, "logs", max_model_length=4096+512):
            result = await benchmark(prompt_mode, split_mode)
            server_metrics = get_vllm_metrics(host, port)
            print(server_metrics)
            result.update(server_metrics)
            return result
    return asyncio.run(total_task())
    
if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    prompt_modes: list[str] = args.modes
    split_modes: list[str] = args.splits
    total_result = {}
    for split_mode in split_modes:
        split_result = {}
        for prompt_mode in prompt_modes:
            result = benchmark_each(args, prompt_mode, split_mode)
            split_result[prompt_mode] = result
        total_result[split_mode] = split_result
    with open(args.output_file, 'w', encoding='utf-8') as file:
        file.write(json.dumps(total_result, ensure_ascii=False))