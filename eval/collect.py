import os
import json
import argparse
import math
import shutil
parser = argparse.ArgumentParser(description="Aggregate evaluation results")
parser.add_argument("--output-file", type=str, required=True,
                    help="Path to save final result.json")
parser.add_argument('--modes', nargs='+', type=str, default=["fim", "efim", "sfim"])

args = parser.parse_args()
total_result = {
    
}
def delete_files(folder_path: str):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # delete subfolder
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
def round(value: float) -> float:
    return int(value * 100.0) / 100.0
for mode in args.modes:
    result = {
        "EM": 0.0,
        "ES": 0.0,
        "S": 0.0,
        "R": 0.0 
    }
    with open(f"outputs/{mode}-cceval-python.json", 'r', encoding='utf-8') as file:
        data = json.loads(file.read())
        result["EM"] = round(data["EM"])
        result["ES"] = round(data["ES"])
    with open(f"outputs/{mode}-humaneval.json", 'r', encoding='utf-8') as file:
        data = json.loads(file.read())
        result["S"] = round(data["single-line"])
        result["R"] = round(data["span"])
    total_result[mode] = result
with open(args.output_file, 'w', encoding='utf-8') as file:
    file.write(json.dumps(total_result))
delete_files("outputs")