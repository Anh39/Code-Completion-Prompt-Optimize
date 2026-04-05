import argparse
from typing import Optional
import os
import json
from copy import deepcopy
from vllm import SamplingParams
from utils.vllm_wrapper import VllmWrapper, ModelArgs
from utils.data_loader import load_data_safim
from utils.prompts import prepare_prompt_safim
from utils.models import prepare_model

INFO_MAP = {
    "api": {
        "file": "api_completion.jsonl.gz",
        "gen_length": 64
    },
    "block": {
        "file": "block_completion.jsonl.gz",
        "gen_length": 256
    },
    "block-v2": {
        "file": "block_completion_v2.jsonl.gz",
        "gen_length": 256
    },
    "control": {
        "file": "control_completion.jsonl.gz",
        "gen_length": 64
    },
    "control-fixed": {
        "file": "control_completion_fixed.jsonl.gz",
        "gen_length": 64
    }
}

def _generate(args):
    lora_path: str | None = args.lora_path
    output_dir: str = args.model_output_dir
    model = prepare_model(args)
    tasks: list[str] = args.tasks
    languages: list[str] = args.languages
    for task in tasks:
        gen_length = INFO_MAP[task]["gen_length"]
        data_file = INFO_MAP[task]["file"]
        sampling_params = SamplingParams(temperature=0, max_tokens=gen_length)
        dataset = load_data_safim(os.path.join(args.input_dir, data_file), languages)
        prompts = prepare_prompt_safim(dataset)
        print(f"Running on {task} dataset")
        outputs = model.generate(prompts, sampling_params, lora_path)
        assert(len(outputs) == len(dataset))
        file_name = f"safim-{task}.jsonl"
        with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as file:
            lines = []
            for output, data in zip(outputs, dataset):
                sample = deepcopy(data)
                sample["completion"] = output
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
        parser.add_argument("--model_output_dir", type=str, default="model_outputs/safim")
        parser.add_argument("--eval_output_dir", type=str, default="eval_outputs/safim")
        parser.add_argument("--tasks", nargs="+", type=str, default=["api", "block", "block-v2", "control", "control-fixed"])
        parser.add_argument("--lora_path", type=str, default=None, help="lora path")
        parser.add_argument("--languages", nargs="+", type=str, default=["cpp", "java", "python", "csharp"])
        args = parser.parse_args()
    else:
        args = custom_args
    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)
    _generate(args)
if __name__ == "__main__":
    generate()