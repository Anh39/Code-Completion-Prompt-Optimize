import math
import random
import asyncio
import argparse
import os
import shutil
import re
import gzip
import glob
import json
from transformers import AutoTokenizer, Qwen2TokenizerFast
from dataclasses import dataclass
from typing import Optional, Any, cast, TypeAlias, Callable, Iterable
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=str, default="../eval/data")
    parser.add_argument("--input_file", type=str, default="train_data.jsonl")
    parser.add_argument("--output_file", type=str, default="train_data_filtered.jsonl")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    return args
def load_input_data(path: str) -> list[dict]:
    result = []
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
        for line in lines:
            if line.strip():
                try:
                    result.append(json.loads(line))
                except:
                    pass
    return result
def read_data_jsonl_or_gzip(file_path: str) -> list[dict]:
    if file_path.endswith(".gz"):
        result: list[dict] = []
        with open(file_path, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                lines = fp.read().splitlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        result.append(json.loads(line))
        return result
    else:
        result: list[dict] = []
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()
            for line in lines:
                line = line.strip()
                if line:
                    result.append(json.loads(line))
        return result
def load_reference_data(dir: str) -> set[str]:
    dataset: set[str] = set()
    for sample in read_data_jsonl_or_gzip(os.path.join(dir, "cceval_python_line_completion.jsonl")):
        prefix = sample["prompt"]
        suffix = sample["right_context"]
        middle = sample["groundtruth"]
        dataset.add(prefix + middle + suffix)
    human_eval_files = [
        "HumanEval-MultiLineInfilling.jsonl.gz",
        "HumanEval-SingleLineInfilling.jsonl.gz",
        "HumanEval-RandomSpanInfilling.jsonl.gz",
        "HumanEval-RandomSpanInfillingLight.jsonl.gz",
    ]
    for file_name in human_eval_files:
        for sample in read_data_jsonl_or_gzip(os.path.join(dir, file_name)):
            prefix = sample["prompt"]
            suffix = sample["suffix"]
            middle = sample["canonical_solution"]
            dataset.add(prefix + middle + suffix)
    return dataset
def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    # normalize text
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    # calculate n-gram
    tokens = text.split()
    if len(tokens) < n:
        return set()
    return {
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    }
def build_ngram_index(texts: Iterable[str], n: int):
    ngrams = set()
    for text in texts:
        ngrams.update(word_ngrams(text, n))
    return ngrams
def main():
    args = parse_args()
    n: int = args.n
    dataset = load_input_data(args.input_file)
    reference_dataset = load_reference_data(args.reference_dir)
    eval_ngrams = build_ngram_index(reference_dataset, n)
    filtered_dataset = []
    metrics = {
        "total_length": 0,
        "total_count": 0,
        "content_length": 0,
        "min_content_length": 999_999,
        "max_content_length": 0,
        "min_total_length": 999_999,
        "max_total_length": 0
    }
    for sample in tqdm(dataset):
        sample_ngrams = build_ngram_index(sample["content"], n)
        reject = False
        for item in sample_ngrams:
            if item in eval_ngrams:
                reject = True
                break
        if not reject:
            filtered_dataset.append(sample)
            total_length = sample["total_length"]
            content_length = sample["total_length"]
            metrics["total_length"] += total_length
            metrics["total_count"] += 1
            metrics["content_length"] += content_length
            metrics["min_total_length"] = min(metrics["min_total_length"], total_length)
            metrics["max_total_length"] = max(metrics["max_total_length"], total_length)
            metrics["min_content_length"] = min(metrics["min_content_length"], content_length)
            metrics["max_content_length"] = max(metrics["max_content_length"], content_length)
    with open(args.output_file, 'w', encoding='utf-8') as out_file:
        lines = [json.dumps(sample, ensure_ascii=False) for sample in filtered_dataset]
        out_file.write("\n".join(lines))
    print(f"\nHoàn tất! Dữ liệu đã lưu tại {args.output_file}")
    print(f"\nHoàn tất! Dữ liệu đã lưu tại {args.output_file}")
    metrics["average_total_length"] = metrics["total_length"] / metrics["total_count"]
    metrics["average_content_length"] = metrics["content_length"] / metrics["total_count"]
    print("Dataset metrics:")
    for key, value in metrics.items():
        print(f" - {key}: {value}")
    with open("train_data_filtered_metric.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(metrics, ensure_ascii=False))
if __name__ == "__main__":
    main()
