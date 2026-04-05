from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelArgs:
    model_name_or_path: str
    tensor_parallel_size: int
    distributed_executor_backend: Literal["mp"]
    enforce_eager: bool
    max_model_length: int
    vram_utilization: float
    lora_paths: list[str]
    max_lora_rank: int = 32
    

class VllmWrapper:
    def __init__(self, args: ModelArgs) -> None:
        lora_args = {}
        if len(args.lora_paths) > 0:
            self.lora_paths = list(set(args.lora_paths))
            lora_args["enable_lora"] = True
            lora_args["max_loras"] = len(self.lora_paths)
            lora_args["max_lora_rank"] = args.max_lora_rank
        else:
            self.lora_paths = []
        self.llm = LLM(
            model=args.model_name_or_path, 
            tensor_parallel_size=args.tensor_parallel_size, 
            trust_remote_code=True,
            distributed_executor_backend=args.distributed_executor_backend,
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_length,
            gpu_memory_utilization=args.vram_utilization,
            **lora_args
        )
        print("Model loaded")
    def generate(self, prompts: list[str], sampling_params: SamplingParams, lora_path: str | None) -> list[str]:
        lora_request = None
        if lora_path != None:
            if lora_path in self.lora_paths:
                lora_request = LoRARequest(lora_path, self.lora_paths.index(lora_path)+1, lora_path)
            else:
                raise KeyError(f"{lora_path} does not exists inside intialized lora list")
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True, lora_request=lora_request)
        result: list[str] = []
        for output in outputs:
            result.append(output.outputs[0].text)
        return result