import argparse
from typing import Optional
import os
import shutil
import json
from copy import deepcopy
from vllm import SamplingParams
from utils.vllm_wrapper import VllmWrapper, ModelArgs
from utils.data_loader import load_data_humaneval, read_jsonl_file
from utils.prompts import prepare_prompt_humaneval
from utils.humaneval_pass.eval import evaluate_functional_correctness
from utils.models import prepare_model

INFO_MAP = {
    "single-line": {
        "file": "SingleLineInfilling.jsonl",
        "gen_length": 64
    },
    "multi-line": {
        "file": "MultiLineInfilling.jsonl",
        "gen_length": 256
    },
    "span": {
        "file": "RandomSpanInfilling.jsonl",
        "gen_length": 256
    },
    "span-light": {
        "file": "RandomSpanInfillingLight.jsonl",
        "gen_length": 256
    }
}

def _generate(args):
    lora_path: str | None = args.lora_path
    output_dir: str = args.model_output_dir
    model = prepare_model(args)
    tasks: list[str] = args.tasks
    for task in tasks:
        gen_length = INFO_MAP[task]["gen_length"]
        data_file = INFO_MAP[task]["file"]
        sampling_params = SamplingParams(temperature=0, max_tokens=gen_length)
        dataset = load_data_humaneval(os.path.join(args.input_dir, data_file))
        prompts = prepare_prompt_humaneval(dataset)
        print(f"Running on {task} dataset")
        outputs = model.generate(prompts, sampling_params, lora_path)
        assert(len(outputs) == len(dataset))
        file_name = f"humaneval-fim-pass-{task}.jsonl"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as file:
            lines = []
            for i in range(len(outputs)):
                sample = deepcopy(dataset[i])
                sample["completion"] = outputs[i]
                lines.append(json.dumps(sample, ensure_ascii=False))
            file.write("\n".join(lines))

def generate(custom_args: Optional[argparse.Namespace] = None):
    """
    Args:
        custom_args: Optional pre-configured arguments
    """
    if custom_args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name_or_path", type=str, required=True)
        parser.add_argument("--tp", type=int, default=1)
        parser.add_argument("--distributed_executor_backend", type=str, default="mp")
        parser.add_argument("--max_seq_length", type=int, default=2048)
        parser.add_argument("--max_model_length", type=int, default=4096)
        parser.add_argument("--vram_utilization", type=float, default=0.1, help="vllm gpu vram utilization")
        parser.add_argument("--input_dir", type=str, default="data")
        
        parser.add_argument("--workers", "-workers", default = 64, type=int, help="")
        parser.add_argument("--model_output_dir", type=str, default="model_outputs/humaneval-fim")
        parser.add_argument("--eval_output_dir", type=str, default="model_outputs/humaneval-fim")
        parser.add_argument("--tasks", nargs="+", type=str, default=["single-line", "multi-line", "span-light", "span"])
        parser.add_argument("--lora_path", type=str, default=None, help="lora path")
        args = parser.parse_args()
    else:
        args = custom_args
    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)
    _generate(args)
    for task in args.tasks:
        completions = read_jsonl_file(os.path.join(args.model_output_dir, f"humaneval-fim-pass-{task}.jsonl"))
        eval_result = evaluate_functional_correctness(
            samples=completions,
            file_path=os.path.join(args.input_dir, INFO_MAP[task]["file"]),
            ks=[1],
            n_workers=args.workers,
            timeout=5.0
        )
        print(f"{task}:{eval_result['pass_at_k']}")
        with open(os.path.join(args.eval_output_dir, f"humaneval-fim-pass-{task}.json"), 'w', encoding='utf-8') as file:
            file.write(json.dumps(eval_result, ensure_ascii=False))
if __name__ == "__main__":
    generate()