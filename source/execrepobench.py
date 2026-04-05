import argparse
import json
import tqdm
import vllm
import tempfile
import os
import subprocess
import collections
from copy import deepcopy
import utils.exec_repo_utils as utils
from transformers import AutoTokenizer
from utils.data_loader import load_data_cceval, read_jsonl_file
from utils.prompts import prepare_prompt_exec_repo_bench
from utils.models import prepare_model

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--distributed_executor_backend", type=str, default="mp")
    parser.add_argument("--max_model_length", type=int, default=2048)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--gen_length", type=int, default=50, help="max length of generated token sequence")
    parser.add_argument("--right_context_length", type=int, default=512)
    parser.add_argument("--cfc_seq_length", type=int, default=512, help="For model_type=codelm_cfc: Text sequence length corresponding to the retrieved nodes")
    parser.add_argument("--vram_utilization", type=float, default=0.1, help="vllm gpu vram utilization")
    parser.add_argument("--input_file", type=str, default="data/exec_repo_bench.jsonl")
    parser.add_argument("--model_output_dir", type=str, default="model_outputs/execrepo")
    parser.add_argument("--eval_output_dir", type=str, default="eval_outputs/execrepo")
    parser.add_argument("--lora_path", type=str, default=None, help="lora path")
    
    parser.add_argument("--context-order", type=str, default="close", choices=["far", "close"])
    parser.add_argument("--chunk_size", "-chunk_size", default = 10, type=int, help="")
    parser.add_argument("--verbose", "-verbose", action="store_true", help="")

    parser.add_argument("--workers", "-workers", default = 64, type=int, help="")
    parser.add_argument("--env_path", "-env_path", default="./envs/", type=str, help="")
    parser.add_argument("--repo_dir", "-repo_dir", default="./repos/", type=str, help="")
    
    args = parser.parse_args()
    return args

# def evaluate_correctness(obj, args):
#     repo_name = obj["repo_name"]
#     masked_file = obj["file_name"]
#     prefix_code = obj["prefix_code"]
#     middle_code = obj["generated_middle_code"] if "generated_middle_code" in obj else obj["middle_code"]
#     suffix_code = obj["suffix_code"]
#     code = prefix_code + middle_code + suffix_code
#     repo_dir = args.repo_dir
#     env_path = args.env_path
#     with tempfile.TemporaryDirectory() as executable_repo_root_path:
#         utils.copy_src_to_dest(repo_dir, executable_repo_root_path, repo_name)
#         masked_file = f"{executable_repo_root_path}/{masked_file}"
#         with open(masked_file, "w") as w:
#             w.write(code)
#         if args.verbose:
#             print(f"Executing {repo_name} ({masked_file})")
#         os.environ["PATH"] = f"{env_path}/repo_{repo_name}/bin:" + os.environ["PATH"]
#         os.chdir(os.path.join(executable_repo_root_path, repo_name))
#         timeout_seconds = 240
#         try:
#             results = subprocess.run(f"python evaluate_repo.py", shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout = timeout_seconds) #f"{executable_repo_root_path}/{repo_name}"
#         except subprocess.TimeoutExpired:
#             return 0.0, f"The command timed out after {timeout_seconds} seconds."
#         if results.returncode != 0:
#             return 0.0, results.stderr.decode("utf8")
#     return 1.0, ""


def evaluate_correctness(obj, env_dir, repo_root_path):
    repo_name = obj["repo_name"]
    masked_file = obj["file_name"]
    prefix_code = obj["prefix_code"]
    middle_code = obj["generated_middle_code"] if "generated_middle_code" in obj else obj["middle_code"]
    suffix_code = obj["suffix_code"]
    code = prefix_code + middle_code + suffix_code
    with tempfile.TemporaryDirectory() as executable_repo_root_path:
        utils.copy_src_to_dest(repo_root_path, executable_repo_root_path, repo_name)
        masked_file = f"{executable_repo_root_path}/{masked_file}"
        with open(masked_file, "w") as w:
            w.write(code)
        #print(f"Executing {repo_name} ({masked_file})")
        os.environ["PATH"] = f"{env_dir}/envs/repo_{repo_name}/bin:" + os.environ["PATH"]
        os.chdir(os.path.join(executable_repo_root_path, repo_name))
        timeout_seconds = 120
        try:
            results = subprocess.run(f"python evaluate_repo.py", shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout = timeout_seconds) #f"{executable_repo_root_path}/{repo_name}"
        except subprocess.TimeoutExpired as e:
            return 0.0, f"timeout: {timeout_seconds}s"
        if results.returncode != 0:
            return 0.0, results.stderr.decode()
    return 1.0, ""

def evaluate_objs_correctness(objs, worker_id, workers, args):
    for obj in tqdm.tqdm(objs, position=worker_id, desc=f"Worker {worker_id}/{workers}"):
        is_pass, stderr = evaluate_correctness(obj, args.env_path, args.repo_dir)
        obj["is_pass"] = is_pass
        obj["stderr"] = stderr
    return objs

def evaluate_es_score(obj):
    hypo = utils.remove_comments(obj["middle_code"], language = "python")
    ref = utils.remove_comments(obj["generated_middle_code"], language = "python")
    es_score = utils.cal_edit_sim([ref], [hypo])
    return es_score

def evaluate_all_correctness(objs, args):
    if objs is None:
        objs = read_jsonl_file(args.output_path)
    for obj in tqdm.tqdm(objs):
        obj["es"] = evaluate_es_score(obj)
    objs = utils.multi_tasks_from_objs(objs, workers = args.workers, task = evaluate_objs_correctness, chunk_size = args.chunk_size, args = args)
    results = {}
    results["pass_at_1"] = round(utils.get_avg_score(objs, "is_pass") * 100, 1)
    results["es"] = round(utils.get_avg_score(objs, "es"), 1)
    results_dict = collections.defaultdict(list)
    for obj in objs:
        results_dict[obj["fill_type"]].append(obj)
    for k in results_dict:
        results[f"{k}:es"] = round(utils.get_avg_score(results_dict[k], "es"), 1)
        results[f"{k}:pass@1"] = round(utils.get_avg_score(results_dict[k], "is_pass") * 100, 1)
    #
    keys = ['Random Span Completion', 'Random Single-line Completion', 'Random Multi-line Completion', 'grammar-based: expression', 'grammar-based: statement', 'grammar-based: block']
    all_results = []
    for k in keys:
        pass_at_1 = results[f"{k}:es"]
        es = results[f"{k}:pass@1"]
        all_results.append(str(pass_at_1))
        all_results.append(str(es))
    all_results.append(str(results["es"]))
    all_results.append(str(results["pass_at_1"]))
    #all_results = [f"\colornum{{{r}}}" for r in all_results]
    results["latex_str"] = " & ".join(all_results)
    return objs, results

def get_continue_prompt_with_suffix(file_name, context_code, prefix_code, suffix_code):
    suffix_code_lines = suffix_code.split("\n")
    suffix_code_lines = ["# " + line for line in suffix_code_lines]
    suffix_code = "\n".join(suffix_code_lines)
    suffix_code = f"## Suffix code of {file_name}\n{suffix_code}"
    prefix_code = f"## Prefix code of {file_name}\n{prefix_code}"
    return f"{context_code}\n{file_name}\n{suffix_code}\n{prefix_code}"

def generate_samples(args):
    test_data = load_data_cceval(args.input_file)
    objs = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print(f"generating {len(test_data)} prompts...")
    min_gen_tokens_len = 100000000
    max_gen_tokens_len = -1
    for obj in tqdm.tqdm(test_data):
        # Just for describe obj content
        # prefix_code = obj["prefix_code"]
        # suffix_code = obj["suffix_code"]
        # context_code_files = obj["context_code"]
        obj = prepare_prompt_exec_repo_bench(
            tokenizer, 
            obj,
            args.context_order,
            args.max_model_length,
            args.gen_length,
            args.max_seq_length
        )
        gen_tokens_len = len(tokenizer.tokenize(obj["middle_code"]))
        if max_gen_tokens_len < gen_tokens_len:
            max_gen_tokens_len = gen_tokens_len + 10
        if min_gen_tokens_len > gen_tokens_len:
            min_gen_tokens_len = gen_tokens_len
        objs.append(obj)
    # Original code typo error ?
    print(f"Suggusting min: {min_gen_tokens_len}, max: {max_gen_tokens_len} tokens")
    sampling_params = vllm.SamplingParams(
        temperature = 0.0, 
        top_p = 0.95,  # Should remove this
        max_tokens = args.gen_length
    )
    prompts = [obj["input"] for obj in objs]
    model = prepare_model(args)
    output_dir = args.model_output_dir
    lora_path = args.lora_path
    outputs = model.generate(prompts, sampling_params, lora_path)
    file_name = f"execrepo.jsonl"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        lines = []
        for output, data in zip(outputs, objs):
            sample = deepcopy(data)
            sample["generated_middle_code"] = output
            lines.append(json.dumps(sample, ensure_ascii=False))
        file.write("\n".join(lines))
    return read_jsonl_file(file_path)

def main():
    args = parse_args()
    print(args)
    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)
    generated_samples = None
    generated_samples = generate_samples(args)
    # generated_samples = read_jsonl_file(os.path.join(args.model_output_dir, f"execrepo.jsonl"))
    objs, results = evaluate_all_correctness(generated_samples, args)
    os.makedirs(os.path.join(args.eval_output_dir, "details"), exist_ok=True)
    with open(os.path.join(args.eval_output_dir, "details", "execrepo_objs.json"), 'w', encoding='utf-8') as file:
        file.write(json.dumps(objs, ensure_ascii=False))
    results.update(vars(args)) # Seem likely to log metadata
    with open(os.path.join(args.eval_output_dir, "execrepo.json"), 'w', encoding='utf-8') as file:
        file.write(json.dumps(results, ensure_ascii=False))
    print(results)

if __name__ == "__main__":
    main()
