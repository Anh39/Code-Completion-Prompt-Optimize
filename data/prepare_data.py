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
from typing import Optional, Any, cast, TypeAlias, Callable
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--source_dir", type=str, default="stackv2_100mb")
    parser.add_argument("--output_file", type=str, default="train_data.jsonl")
    parser.add_argument("--min_file_length", type=int, default=50)
    parser.add_argument("--min_content_length", type=int, default=128)
    parser.add_argument("--max_file_length", type=int, default=1_000_000)
    parser.add_argument("--max_samples_per_repo", type=int, default=1000)
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(MODEL_ID)
    source_dir: str = args.source_dir
    seed: int = args.seed
    rng = random.Random(seed)
    max_samples_per_repo: int = args.max_samples_per_repo
    max_seq_length: int = args.max_seq_len - 8
    min_content_length: int = args.min_content_length
    metrics = {
        "total_length": 0,
        "total_count": 0,
        "content_length": 0,
        "min_content_length": 999_999,
        "max_content_length": 0,
        "min_total_length": 999_999,
        "max_total_length": 0
    }
    def get_length(text: str) -> int:
        return len(tokenizer.encode(text))
    with open(args.output_file, 'w', encoding='utf-8') as out_file:
        repo_files_map = {}
        for repo_name in os.listdir(source_dir):
            all_files = glob.glob(f"{os.path.join(source_dir, repo_name)}/**/*.py", recursive=True)
            if len(all_files) > max_samples_per_repo:
                all_files = rng.choices(all_files, k=max_samples_per_repo)
            repo_files_map[repo_name] = all_files
        total_samples = sum(len(all_files) for all_files in repo_files_map.values())
        pbar = tqdm(total=total_samples)
        for repo_name, all_files in repo_files_map.items():
            for file_path in all_files:
                try:
                    repo_path = os.path.join(source_dir, repo_name) + "\\"
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                    if len(content) < args.min_file_length or len(content) > args.max_file_length:
                        continue
                    # Calculate intra file
                    repo_header_length = get_length(repo_name) + 2 # <|repo_name|>repo_name\n
                    content_ids = tokenizer.encode(content)
                    rel_path = file_path.replace(repo_path, '')
                    content_header_length = get_length(rel_path) + 2 # <|file_sep|>rel_path\n
                    content_length = len(content_ids)
                    total_length = repo_header_length + content_header_length + content_length
                    if total_length > max_seq_length:
                        max_content_length = max_seq_length-repo_header_length-content_header_length
                        content_ids = content_ids[:max_content_length]
                        content_length = len(content_ids) # Update content length after truncate
                        content = tokenizer.decode(content_ids)
                    if content_length < min_content_length:
                        continue
                    total_length = repo_header_length + content_header_length + content_length
                    record = {
                        "repo_name": repo_name,
                        "path": rel_path,
                        "content": content,
                        "content_length": content_length,
                        "total_length": total_length,
                    }
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics["total_length"] += total_length
                    metrics["total_count"] += 1
                    metrics["content_length"] += content_length
                    metrics["min_total_length"] = min(metrics["min_total_length"], total_length)
                    metrics["max_total_length"] = max(metrics["max_total_length"], total_length)
                    metrics["min_content_length"] = min(metrics["min_content_length"], content_length)
                    metrics["max_content_length"] = max(metrics["max_content_length"], content_length)
                except Exception as e:
                    print(f"Lỗi {file_path}: {e}")
                finally:
                    pbar.update(1)
        pbar.close()
    print(f"\nHoàn tất! Dữ liệu đã lưu tại {args.output_file}")
    metrics["average_total_length"] = metrics["total_length"] / metrics["total_count"] #type:ignore
    metrics["average_content_length"] = metrics["content_length"] / metrics["total_count"] #type:ignore
    print("Dataset metrics:")
    for key, value in metrics.items():
        print(f" - {key}: {value}")
    with open("train_data_metric.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(metrics, ensure_ascii=False))
if __name__ == "__main__":
    main()
