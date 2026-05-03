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
    parser.add_argument("--cfc_len", type=int, default=512)
    parser.add_argument("--source_dir", type=str, default="../stackv2_v5_currated_100mb")
    parser.add_argument("--output_file", type=str, default="train_data_repo_v4.jsonl")
    parser.add_argument("--min_file_length", type=int, default=50)
    parser.add_argument("--max_file_length", type=int, default=1_000_000)
    parser.add_argument("--top_k_files", type=int, default=5)
    parser.add_argument("--max_lines_each_file", type=int, default=25)
    parser.add_argument("--max_samples_per_repo", type=int, default=100)
    args = parser.parse_args()
    return args
def get_cross_file_context(get_length_func: Callable[[str], int], budget: int, top_k: int, max_lines: int, repo_path: str, current_file_path: str, all_files_in_dir: list[str]) -> list[dict[str, Any]]:
    contexts = []
    parent_dir = os.path.dirname(current_file_path)
    # This get all file that is descent of parent_dir
    neighbors = [f for f in all_files_in_dir if f != current_file_path and f.startswith(parent_dir)]
    for neighbor in neighbors[:top_k]:
        with open(neighbor, 'r', encoding='utf-8', errors='ignore') as file:
            rel_path = neighbor.replace(repo_path, "")
            lines = file.read().splitlines()
            snippet = "\n".join(lines[:max_lines]) + "\n"
            header_length = get_length_func(rel_path) + 2 # <|file_sep|>file_path\n
            length = get_length_func(snippet)
            budget = budget - (header_length + length)
            if (budget < 0):
                break
            contexts.append({
                "path": rel_path,
                "text": snippet,
                "length": length
            })
    return contexts
def main():
    args = parse_args()
    tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(MODEL_ID)
    source_dir: str = args.source_dir
    cfc_len: int = args.cfc_len
    seed: int = args.seed
    rng = random.Random(seed)
    max_samples_per_repo: int = args.max_samples_per_repo
    max_seq_length: int = args.max_seq_len - 8
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
                    total_length_without_cfc = repo_header_length + content_header_length + content_length
                    if total_length_without_cfc > max_seq_length:
                        max_content_length = max_seq_length-repo_header_length-content_header_length
                        content_ids = content_ids[:max_content_length]
                        content_length = len(content_ids) # Update content length after truncate
                        content = tokenizer.decode(content_ids)
                    total_length_without_cfc = repo_header_length + content_header_length + content_length
                    # Calculate cross file
                    cfc_budget = min(cfc_len, max_seq_length - total_length_without_cfc)
                    contexts = get_cross_file_context(
                        get_length_func=get_length,
                        budget=cfc_budget,
                        top_k=args.top_k_files,
                        max_lines=args.max_lines_each_file, 
                        repo_path=repo_path,
                        current_file_path=file_path, 
                        all_files_in_dir=all_files
                    )
                    record = {
                        "repo_name": repo_name,
                        "path": rel_path,
                        "cfc_contexts": contexts,
                        "content": content,
                        "content_length": content_length
                    }
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Lỗi {file_path}: {e}")
                finally:
                    pbar.update(1)
        pbar.close()
    print(f"\nHoàn tất! Dữ liệu đã lưu tại {args.output_file}")
    
if __name__ == "__main__":
    main()
