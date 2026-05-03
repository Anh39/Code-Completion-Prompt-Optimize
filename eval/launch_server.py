import os
import subprocess
import requests
import time
import signal

PORT = 65511
HOST = "0.0.0.0"
VRAM_UTILIZATION = 0.9

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _is_server_ready(host: str, port: int) -> bool:
    try:
        url = f"http://{host}:{port}/v1/models"
        r = requests.get(url, timeout=1)
        return r.status_code == 200
    except Exception:
        return False

def serve(model_id: str, request_model: str, log_dir: str, timeout: float = 60.0 * 5, max_model_length: int = 2304) -> int:
    cmd = [
        "vllm", "serve", model_id,
        "--host", str(HOST),
        "--port", str(PORT),
        "--enforce-eager",
        "--disable-log-requests",
        "--trust-remote-code",
        "--gpu-memory-utilization", str(VRAM_UTILIZATION),
        "--max-model-len", str(max_model_length),
    ]
    if request_model != None and model_id != request_model:
        cmd.extend([
            "--enable-lora",
            "--max-loras", "1",
            "--max-lora-rank", "16",
            "--lora-modules", f"{request_model}={request_model}"
        ])
    log_file = open(os.path.join(log_dir, "vllm.txt"), 'w', encoding='utf-8')
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=os.environ.copy()
    )
    log_file.close()
    start_time = time.time()
    print(f"Starting vLLM server")
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code: {process.returncode}")
        if _is_server_ready(HOST, PORT):
            print(f"vLLM is ready on port {PORT}")
            return process.pid
        if time.time() - start_time > timeout:
            process.kill()
            raise TimeoutError("Timeout waiting for vLLM server")
        time.sleep(1)
        
class ServerContextManager:
    def __init__(self, model_id: str, request_model: str, log_dir: str, timeout: float = 60.0 * 5, max_model_length: int = 2304) -> None:
        self.model_id = model_id
        self.request_model = request_model
        self.log_dir = log_dir
        self.timeout = timeout
        self.max_model_length = max_model_length
        self.shutdown_grace = 15
    def __enter__(self):
        self.pid = serve(self.model_id, self.request_model, self.log_dir, self.timeout, self.max_model_length)
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            print("Clean up completed")
            return False
        deadline = time.time() + self.shutdown_grace
        while time.time() < deadline:
            try:
                os.kill(self.pid, 0)
            except ProcessLookupError:
                print("Clean up completed")
                return False
            time.sleep(1)
        try:
            # Process still exists
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            print("Clean up completed")
        return False