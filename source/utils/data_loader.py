import json
import os
import gzip
def _read_data_jsonl(file_path: str) -> list[dict]:
    result: list[dict] = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
        for line in lines:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result
def _read_data_jsonl_or_gzip(file_path: str) -> list[dict]:
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
        return _read_data_jsonl(file_path)
def load_data_humaneval(file_path: str) -> list[dict]:
    return _read_data_jsonl(file_path)
def load_data_safim(file_path: str, languages: list[str]) -> list[dict]:
    full = _read_data_jsonl_or_gzip(file_path)
    result: list[dict] = []
    for sample in full:
        if sample["lang"] in languages:
            result.append(sample)
    return result
def load_data_cceval(file_path: str) -> list[dict]:
    return _read_data_jsonl(file_path)
def load_data_repoeval(file_path: str) -> list[dict]:
    return _read_data_jsonl(file_path)
def load_data_execrepo(file_path: str) -> list[dict]:
    return _read_data_jsonl(file_path)
def read_jsonl_file(file_path: str) -> list[dict]:
    return _read_data_jsonl(file_path)