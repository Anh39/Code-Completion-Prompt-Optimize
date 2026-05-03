import argparse
import asyncio
import json
import os
import random

import tqdm
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer

from utils.prompt import construct_prompt
from utils.data import load_data
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
    parser.add_argument('--modes', nargs='+', type=str, default=['fim', 'efim', 'sfim'])
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=65511)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    return args


def cal_edit_sim(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        edit_sim += fuzz.ratio(pred, gt)
    return edit_sim / total


def benchmark_each(args: argparse.Namespace):
    model_id = args.model_id
    request_model = args.request_model or model_id
    modes: list[str] = args.modes
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
    async def benchmark(prompt_mode: str):
        problems = load_data('cceval-python')
        problems = [problem for problem in problems if len(tokenizer.encode(problem['prompt'] + problem['groundtruth'] + problem['right_context'])) < 2048]
        pbar = tqdm.tqdm(total=len(problems))
        samples = []
        tasks = []
        for problem in problems:
            prompt_tot = problem['prompt'] + problem['groundtruth'] + problem['right_context']
            len_groundtruth = random.randint(16, 128)
            begin_groundtruth = random.randint(32, len(prompt_tot) - 160)
            prompt = prompt_tot[:begin_groundtruth]
            suffix = prompt_tot[begin_groundtruth + len_groundtruth:]
            idx = random.randint(1, begin_groundtruth - 1)
            problem['groundtruth'] = prompt_tot[begin_groundtruth:begin_groundtruth + len_groundtruth]
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
            samples.append(dict(prompt=input_text, groundtruth=problem['groundtruth'], completion=''))

        completions = await asyncio.gather(*tasks)
        pbar.close()
        em = 0
        refs, hyps = [], []
        for idx, sample_each in enumerate(samples):
            assert completions[idx].success
            sample_each['completion'] = completions[idx].generated_text
            if sample_each['groundtruth'] == sample_each['completion']:
                em += 1
            refs.append(sample_each['groundtruth'])
            hyps.append(sample_each['completion'])
        es_score = cal_edit_sim(refs, hyps)
        em_score = em / len(samples) * 100
        print(f'ES: {es_score}')
        print(f'EM: {em_score}')
        with open(os.path.join(output_dir, f'{prompt_mode}-cceval-python.json'), 'w', encoding='utf-8') as file:
            result = {
                'ES': es_score,
                'EM': em_score
            }
            file.write(json.dumps(result, ensure_ascii=False))

    async def total_task():
        for mode in modes:
            await benchmark(mode)
    os.makedirs('log', exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    asyncio.run(total_task())


if __name__ == '__main__':
    args = parse_args()
    benchmark_each(args)
