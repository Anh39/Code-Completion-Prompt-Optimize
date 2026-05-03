from .cache import function_cache
def _construct_prompt(prompt_mode: str, pre: str, mid: str, suf: str, prefix: str, middle: str, suffix: str) -> str:
    if prompt_mode == "fim":
        return pre + prefix + middle + suf + suffix + mid
    elif prompt_mode == "efim":
        return pre + prefix + suf + suffix + mid + middle
    elif prompt_mode == "sfim":
        fim_prompt = _construct_prompt("fim", pre, mid, suf, prefix, middle, suffix)
        return function_cache(pre, mid, suf, fim_prompt)
    else:
        raise Exception(f"Unsupport mode: {prompt_mode}")
def construct_prompt(prompt_mode: str, model_id: str, prefix: str, middle: str, suffix: str) -> str:
    if "qwen" in model_id.lower():
        pre = "<|fim_prefix|>"
        mid = "<|fim_middle|>"
        suf = "<|fim_suffix|>"
    elif "deepseek" in model_id.lower():
        # DeepSeek format is <｜fim▁begin｜>{prompt}<｜fim▁hole｜>{suffix}<｜fim▁end｜>
        # So if we use normal order, it would be <｜fim▁begin｜>{prompt}<｜fim▁end｜>{suffix}<｜fim▁hole｜>
        # So we swap mid and suf token
        pre = "<｜fim▁begin｜>"
        mid = "<｜fim▁end｜>"
        suf = "<｜fim▁hole｜>"
    elif "llama" in model_id.lower():
        pre = "<fim_prefix>"
        mid = "<fim_middle>"
        suf = "<fim_suffix>"
    else:
        raise Exception(f"Unsupported model for prompt construction: {model_id}")
    return _construct_prompt(prompt_mode, pre, mid, suf, prefix, middle, suffix)
