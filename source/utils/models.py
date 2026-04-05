from .vllm_wrapper import VllmWrapper, ModelArgs
def prepare_model(args):
    lora_path: str | None = args.lora_path
    model_args = ModelArgs(
        model_name_or_path=args.model_name_or_path,
        tensor_parallel_size=args.tp,
        distributed_executor_backend=args.distributed_executor_backend,
        enforce_eager=True,
        max_model_length=args.max_model_length,
        vram_utilization=args.vram_utilization,
        lora_paths=[lora_path] if lora_path else [],
        max_lora_rank=32
    )
    model = VllmWrapper(model_args)
    return model