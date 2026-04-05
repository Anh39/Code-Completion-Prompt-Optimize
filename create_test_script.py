import os, json
file_path = "/root/workspace/ExecRepoBench/exec_repo_bench.jsonl"
ouput_path = "/root/workspace/ExecRepoBench/test_set/test.jsonl"
os.makedirs(os.path.dirname(ouput_path), exist_ok=True)
with open(ouput_path, 'w', encoding='utf-8') as wfile:
    with open(file_path, 'r', encoding='utf-8') as rfile:
        lines = rfile.readlines()
        data = [json.loads(line) for line in lines if line.strip()]
        lines: list[str] = []
        for item in data:
            item["generated_middle_code"] = item["middle_code"]
            lines.append(json.dumps(item, ensure_ascii=False))
        wfile.write("\n".join(lines))