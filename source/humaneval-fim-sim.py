import argparse
from typing import Optional
from collections import defaultdict
import os
import json
from copy import deepcopy
from vllm import SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.vllm_wrapper import VllmWrapper, ModelArgs
from utils.data_loader import load_data_humaneval
from utils.prompts import prepare_prompt
from utils.models import prepare_model
from rich import print
from rich.table import Table
from fuzzywuzzy import fuzz
import numpy as np

# From Qwen source code
def evaluate_results(results_file):
    language_results = defaultdict(lambda: {
        "count": 0,
        "exact_matches": 0,
        "edit_similarities": []
    })
    
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            model_gen = data.get("model_gen")
            canonical_solution = data.get("canonical_solution")
            language = data.get("language", "Unknown")
            
            # Process model generation
            if model_gen.startswith("\n"):
                response = model_gen.split("\n")[1]
            elif model_gen.startswith(" \n"):
                response = model_gen.split("\n")[1]
            else:
                response = model_gen.split("\n")[0]
            
            response = response.strip()
            canonical_solution = canonical_solution.strip()
            
            # Update statistics
            language_results[language]["count"] += 1
            if response == canonical_solution:
                language_results[language]["exact_matches"] += 1
            
            # Calculate edit similarity
            similarity = fuzz.ratio(response, canonical_solution)
            language_results[language]["edit_similarities"].append(similarity)
    
    # Calculate final metrics
    final_results = {}
    total_count = 0
    total_exact_matches = 0
    all_similarities = []

    
    for lang, stats in language_results.items():
        exact_match_rate = (stats["exact_matches"] / stats["count"]) * 100
        avg_similarity = np.mean(stats["edit_similarities"])
        
        final_results[lang] = {
            "count": stats["count"],
            "exact_match_rate": exact_match_rate,
            "average_edit_similarity": avg_similarity
        }
        
        total_count += stats["count"]
        total_exact_matches += stats["exact_matches"]
        all_similarities.extend(stats["edit_similarities"])
    
    # Add overall results
    final_results["overall"] = {
        "count": total_count,
        "exact_match_rate": (total_exact_matches / total_count) * 100,
        "average_edit_similarity": np.mean(all_similarities)
    }
    
    return final_results

# From Qwen source code
def print_results_table(results):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Language")
    table.add_column("Count")
    table.add_column("Exact Match Rate")
    table.add_column("Edit Similarity")
    
    # Add results for each language
    for lang, metrics in results.items():
        if lang != "overall":
            table.add_row(
                lang,
                str(metrics["count"]),
                f"{metrics['exact_match_rate']:.2f}%",
                f"{metrics['average_edit_similarity']:.2f}%"
            )
    
    # Add overall results
    table.add_row(
        "Overall",
        str(results["overall"]["count"]),
        f"{results['overall']['exact_match_rate']:.2f}%",
        f"{results['overall']['average_edit_similarity']:.2f}%",
        style="bold"
    )
    
    print(table)

def generate(args):
    lora_path: str | None = args.lora_path
    output_dir: str = args.model_output_dir
    model = prepare_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.gen_length)
    dataset = load_data_humaneval(os.path.join(args.input_dir, "fim_singleline.jsonl"))
    
    prompts = []
    for data in tqdm(dataset):
        prefix = data["prompt"]
        suffix = data["suffix"]
        prompt = prepare_prompt(tokenizer, "codelm_leftright_context", prefix, suffix,
                                gen_length=args.gen_length,
                                max_seq_length=args.max_seq_length,
                                right_context_length=args.right_context_length)
        prompts.append(prompt)
    outputs = model.generate(prompts, sampling_params, lora_path)
    file_name = f"humaneval-fim-languages.jsonl"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        lines = []
        for output, data in zip(outputs, dataset):
            sample = deepcopy(data)
            sample["model_gen"] = output
            lines.append(json.dumps(sample, ensure_ascii=False))
        file.write("\n".join(lines))
    return file_path

def evaluate(custom_args: Optional[argparse.Namespace] = None):
    """
    Args:
        custom_args: Optional pre-configured arguments
    """
    if custom_args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name_or_path", type=str, required=True)
        parser.add_argument("--tp", type=int, default=1)
        parser.add_argument("--distributed_executor_backend", type=str, default="mp")
        parser.add_argument("--gen_length", type=int, default=64)
        parser.add_argument("--max_seq_length", type=int, default=2048)
        parser.add_argument("--max_model_length", type=int, default=4096)
        parser.add_argument("--right_context_length", type=int, default=512)
        parser.add_argument("--vram_utilization", type=float, default=0.1, help="vllm gpu vram utilization")
        parser.add_argument("--input_dir", type=str, default="data")
        parser.add_argument("--model_output_dir", type=str, default="model_outputs/humaneval-fim")
        parser.add_argument("--eval_output_dir", type=str, default="eval_outputs/humaneval-fim")
        parser.add_argument("--lora_path", type=str, default=None, help="lora path")
        args = parser.parse_args()
    else:
        args = custom_args
    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)
    file_path = generate(args)
    results = evaluate_results(file_path)
    print_results_table(results)
    with open(os.path.join(args.eval_output_dir, "humaneval-fim-simmilarity.json"), 'w', encoding='utf-8') as file:
        file.write(json.dumps(results, ensure_ascii=False))
    return results
if __name__ == "__main__":
    evaluate()