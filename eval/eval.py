import os
import signal
import json
import subprocess
import argparse
import time
from launch_server import ServerContextManager

def run_command(cmd: list[str]):
    print(cmd)
    subprocess.run(cmd, check=True)
    
def main():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--model_or_checkpoint", help="Model path or checkpoint")
    parser.add_argument("--output_file", help="Output file path")
    parser.add_argument(
        "--request_model",
        default=None,
        help="Optional request model (default depends on input)"
    )
    args = parser.parse_args()
    model_or_checkpoint = args.model_or_checkpoint
    adapter_config_path = os.path.join(model_or_checkpoint, "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        with open(adapter_config_path, encoding="utf-8") as file:
            config = json.load(file)
        model_id = config["base_model_name_or_path"]
        request_model = args.request_model
    else:
        model_id = model_or_checkpoint
        request_model = args.request_model or model_id
    with ServerContextManager(model_or_checkpoint, request_model, log_dir, max_model_length=2304):
        modes = ["fim", "sfim"]
        cceval_cmd = [
            "python", "eval-efim-cceval.py",
            f"--model_id", model_id,
            f"--request-model", request_model,
            f"--modes", *modes,
        ]
        humaneval_cmd = [
            "python", "eval-efim-humaneval.py",
            "--model_id", model_id,
            "--request-model", request_model,
            "--modes", *modes,
            "--benchmarks", "single-line", "span"
        ]
        collect_cmd = [
            "python", "collect.py",
            "--output-file", args.output_file,
            "--modes", *modes
        ]
        run_command(cceval_cmd)
        run_command(humaneval_cmd)
        run_command(collect_cmd)
        
if __name__ == "__main__":
    main()