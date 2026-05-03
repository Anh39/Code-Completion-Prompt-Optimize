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
    return _read_data_jsonl_or_gzip(file_path)
def load_data_cceval(file_path: str) -> list[dict]:
    return _read_data_jsonl(file_path)
def load_data(benchmark_name: str) -> list[dict]:
    if benchmark_name == "humaneval-fim-single-line":
        return load_data_humaneval("data/HumanEval-SingleLineInfilling.jsonl.gz")
    elif benchmark_name == "humaneval-fim-multi-line":
        return load_data_humaneval("data/HumanEval-MultiLineInfilling.jsonl.gz")
    elif benchmark_name == "humaneval-fim-span":
        return load_data_humaneval("data/HumanEval-RandomSpanInfilling.jsonl.gz")
    elif benchmark_name == "humaneval-fim-span-light":
        return load_data_humaneval("data/HumanEval-RandomSpanInfillingLight.jsonl.gz")
    elif benchmark_name == "cceval-python":
        return load_data_humaneval("data/cceval_python_line_completion.jsonl")
    else:
        raise Exception(f"Benchmark not supported: {benchmark_name}")
def get_sample_text(benchmark: str, sample: dict) -> str:
    if "humaneval-fim-" in benchmark:
        prefix = sample["prompt"]
        suffix = sample["suffix"]
        middle = sample["canonical_solution"]
        return prefix + middle + suffix
    elif "cceval-" in benchmark:
        prefix = sample["prompt"]
        suffix = sample["right_context"]
        middle = sample["groundtruth"]
        return prefix + middle + suffix
    else:
        raise Exception(f"Benchmark not supported: {benchmark}")