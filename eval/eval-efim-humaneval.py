import argparse
import asyncio
import json
import os
import random

import tqdm
from transformers import AutoTokenizer

from utils.prompt import construct_prompt
from utils.data import load_data, load_data_humaneval
from utils.humaneval_utils.eval import evaluate_functional_correctness
from request_vllm import async_request_openai_completions, RequestFuncInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id',
        type=str,
        default='Qwen/Qwen2.5-Coder-0.5B',
    )
    parser.add_argument(
        '--request-model',
        type=str,
        default=None,
        help='Model name sent to the serving API. Defaults to --model_id.',
    )
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        type=str,
        default=['single-line', 'multi-line', 'span', 'span-light'])
    parser.add_argument('--modes', nargs='+', type=str, default=['fim', 'efim', 'sfim'])
    parser.add_argument('--num-samples', help='number of samples per task', type=int, default=1)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=65511)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    return args


def benchmark_each(args: argparse.Namespace):
    model_id = args.model_id
    request_model = args.request_model or model_id
    benchmark_names: list[str] = args.benchmarks
    modes: list[str] = args.modes
    num_sample_per_task = args.num_samples
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    port = args.port
    host = args.host
    output_dir = args.output_dir

    api_url = f'http://{host}:{port}/v1/completions'
    random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    eos_token_ids = [tokenizer.eos_token_id]
    if "llama3.1-8b-train" in model_id.lower() and "enhance" not in model_id.lower():
        # Since we use continual pretrained model of EFIm paper
        eos_token_ids = [128000]
    # elif "deepseek" in model_id.lower() and "enhance" in model_id.lower():
    #     eos_token_ids = [32027]
    async def benchmark(prompt_mode: str, benchmark_name: str):
        problems_list = load_data(f'humaneval-fim-{benchmark_name}')
        problems = {item['task_id']: item for item in problems_list}
        pbar = tqdm.tqdm(total=num_sample_per_task * len(problems))
        samples = []
        tasks = []
        for _ in range(num_sample_per_task):
            for task_id in problems:
                prompt = problems[task_id]['prompt']
                suffix = problems[task_id]['suffix']
                idx = random.randint(1, len(prompt) - 1)
                input_text = construct_prompt(prompt_mode, model_id, prompt[:idx], prompt[idx:], suffix)
                request_input = RequestFuncInput(
                    input_text,
                    api_url,
                    len(input_text),
                    max_tokens,
                    request_model,
                    temperature,
                    top_p,
                    top_k,
                    eos_token_ids,
                    False,
                )
                tasks.append(
                    asyncio.create_task(async_request_openai_completions(
                        request_func_input=request_input,
                        pbar=pbar)))
                samples.append(dict(task_id=task_id, prompt=input_text, completion=''))
        completions = await asyncio.gather(*tasks)
        for idx, sample_each in enumerate(samples):
            assert completions[idx].success
            sample_each['completion'] = completions[idx].generated_text
        sample_file = f'log/{prompt_mode}-{benchmark_name}.jsonl'
        with open(sample_file, 'w', encoding='utf-8') as file:
            file.write('\n'.join([json.dumps(sample, ensure_ascii=False) for sample in samples]))
        return sample_file
    
    async def total_task():
        for mode in modes:
            result = {
                'single-line': 0,
                'multi-line': 0,
                'span': 0,
                'span-light': 0
            }

            for benchmark_name in benchmark_names:
                sample_file = await benchmark(mode, benchmark_name)
                name_mapping = {
                    'single-line': 'SingleLineInfilling',
                    'multi-line': 'MultiLineInfilling',
                    'span': 'RandomSpanInfilling',
                    'span-light': 'RandomSpanInfillingLight'
                }
                data_file = f"data/HumanEval-{name_mapping[benchmark_name]}.jsonl.gz"
                completions = load_data_humaneval(sample_file)
                results = evaluate_functional_correctness(completions, data_file, ks=[num_sample_per_task], n_workers=24, timeout=3.0)
                k = f'pass@{num_sample_per_task}'
                print(f'{k}: {results["pass_at_k"][k] * 100}')
                result[benchmark_name] = results['pass_at_k'][k] * 100
            with open(os.path.join(output_dir, f'{mode}-humaneval.json'), 'w', encoding='utf-8') as file:
                file.write(json.dumps(result, ensure_ascii=False))
    os.makedirs('log', exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    asyncio.run(total_task())

if __name__ == '__main__':
    args = parse_args()
    benchmark_each(args)
