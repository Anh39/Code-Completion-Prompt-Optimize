from transformers import AutoTokenizer
import argparse
from typing import Optional
import os, shutil
import json
from copy import deepcopy
from vllm import SamplingParams
from tqdm import tqdm
from utils.vllm_wrapper import VllmWrapper, ModelArgs
from utils.data_loader import load_data_cceval
from utils.prompts import prepare_prompt
from utils.models import prepare_model
from utils.eval_metric import compute_metric_stmt

def _generate(args):
    lora_path: str | None = args.lora_path
    output_dir: str = args.model_output_dir
    model = prepare_model(args)
    tasks: list[str] = args.tasks
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.gen_length)
    for task in tasks:
        task_prompt_file = args.input_file.replace('TASK', task)
        dataset = load_data_cceval(task_prompt_file)
        prompts = []
        print(f"Running on {task} task")
        print(f"Preparing data :")
        for sample in tqdm(dataset):
            prefix = sample["prompt"]
            suffix = sample["right_context"]
            cfc_cxt = sample["crossfile_context"]
            prompt = prepare_prompt(tokenizer, "codelm_right_cfc_left", prefix, suffix,
                                    gen_length=args.gen_length,
                                    max_seq_length=args.max_seq_length,
                                    right_context_length=args.right_context_length,
                                    cfc_seq_length=args.cfc_seq_length,
                                    crossfile_cxt=cfc_cxt)
            prompts.append(prompt)
        outputs = model.generate(prompts, sampling_params, lora_path)
        assert(len(outputs) == len(dataset))
        file_name = f"cclongeval-{task}.jsonl"
        with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as file:
            lines = []
            for output, data, prompt in zip(outputs, dataset, prompts):
                sample = deepcopy(data)
                pred = {
                    "task_id": sample["metadata"]["task_id"],
                    "pred": output.split("<|fim_pad|>")[0].split("<|file_sep|>")[0],
                    "task_type": task,
                    "inputs": prompt
                }
                lines.append(json.dumps(pred, ensure_ascii=False))
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
        parser.add_argument("--gen_length", type=int, default=50, help="max length of generated token sequence")
        parser.add_argument("--right_context_length", type=int, default=512)
        parser.add_argument("--cfc_seq_length", type=int, default=512, help="For model_type=codelm_cfc: Text sequence length corresponding to the retrieved nodes")        
        parser.add_argument("--vram_utilization", type=float, default=0.1, help="vllm gpu vram utilization")
        parser.add_argument("--input_file", type=str, default="data")
        parser.add_argument("--model_output_dir", type=str, default="model_outputs")
        parser.add_argument("--eval_output_dir", type=str, default="eval_outputs")
        parser.add_argument("--tasks", nargs="+", type=str, default=["line_completion", "api_completion", "function_completion", "chunk_completion"])
        parser.add_argument("--lora_path", type=str, default=None, help="lora path")

        parser.add_argument("--ts_lib", type=str, default="build/python-lang-parser.so", help="tree-sitter lib for tokenize code")
        args = parser.parse_args()
    else:
        args = custom_args
    setattr(args, "max_model_length", 9000)
    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)
    _generate(args)
    tmp_output = os.path.join(args.eval_output_dir, "details")
    os.makedirs(tmp_output, exist_ok=True)
    kwargs = {
        "prompt_file": args.input_file,
        "tasks": args.tasks,
        "dataset": "cceval",
        "output_dir": tmp_output,
        "ts_lib": args.ts_lib.replace("LANGUAGE", "python"),
        "language": "python"
    }
    class tmp:
        pass
    obj = tmp()
    for key, value in kwargs.items():
        setattr(obj, key, value)
    for task in obj.tasks:
        folder_path = f"{obj.output_dir}/{obj.dataset}/python/{task}"
        src_path = os.path.join(args.model_output_dir, f"cclongeval-{task}.jsonl")
        dst_path = os.path.join(folder_path, "prediction.jsonl")
        os.makedirs(folder_path, exist_ok=True)
        shutil.copy(src_path, dst_path)
    compute_metric_stmt(obj)
    shutil.copy(f"{obj.output_dir}/{obj.dataset}/results.json", os.path.join(args.eval_output_dir, f"cclongeval.json"))
if __name__ == "__main__":
    generate()