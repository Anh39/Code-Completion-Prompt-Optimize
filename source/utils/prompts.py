PRE = "<|fim_prefix|>"
MID = "<|fim_middle|>"
SUF = "<|fim_suffix|>"
from .cache import function_cache
from .cache_old import function_cache_old
def post_process(prompt: str) -> str:
    return prompt
def prepare_prompt_humaneval(samples: list[dict]) -> list[str]:
    result: list[str] = []
    for sample in samples:
        prefix = sample["prompt"]
        suffix = sample["suffix"]
        prompt = PRE + prefix + SUF + suffix + MID
        result.append(post_process(prompt))
    return result
def prepare_prompt_safim(samples: list[dict]) -> list[str]:
    result: list[str] = []
    for sample in samples:
        total: str = sample["prompt"]
        prefix, suffix = total.split("{{completion}}")
        prompt = PRE + prefix + SUF + suffix + MID
        result.append(post_process(prompt))
    return result

# Simplified of Qwen source code
def prepare_prompt(
        tokenizer, 
        model_type, 
        left_cxt, 
        right_cxt=None, 
        crossfile_cxt=None, 
        gen_length: int = 64,
        max_seq_length: int = 1024,
        right_context_length: int = 512,
        cfc_seq_length: int = 1024):   
    
    # 设置模型特定的tokens
    prefix_token = PRE
    middle_token = MID
    suffix_token = SUF
        
    if model_type == "codelm_leftright_context":
        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(max_seq_length - gen_length - right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:right_context_length])
        prompt = f'{prefix_token}{left_cxt_truncated}{suffix_token}{right_cxt_truncated}{middle_token}'
    elif model_type == "codelm_right_cfc_left":
        assert crossfile_cxt is not None
        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(max_seq_length - gen_length - right_context_length - cfc_seq_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:right_context_length])
        crossfile_cxt_truncated = tokenizer.decode(tokenizer.encode('\n\n' + crossfile_cxt)[:cfc_seq_length])
        prompt = f'{prefix_token}{left_cxt_truncated}{suffix_token}{right_cxt_truncated}{crossfile_cxt_truncated}{middle_token}'
    else:
        raise NotImplementedError
    return post_process(prompt)

def _truncate_prompt(prompt, max_num_tokens, tokenizer, side="right"):
    tokens = tokenizer.tokenize(prompt)
    num_tokens = len(tokens)
    if num_tokens > max_num_tokens:
        if side == 'left':
            prompt_tokens = tokens[num_tokens - max_num_tokens:]
        elif side == 'right':
            prompt_tokens = tokens[:max_num_tokens]
        prompt = tokenizer.convert_tokens_to_string(prompt_tokens)
        new_len = len(tokenizer.tokenize(prompt))
        if new_len > max_num_tokens:
            print(f'Number of tokens after truncation is greater than max tokens allowed {max_num_tokens}: {new_len} {num_tokens}')
    return post_process(prompt)

# Simplified of Qwen source code
QWEN_CODER_TEMPLAT="""
{}<|fim_prefix|>{}<|fim_suffix|>{}<|fim_middle|>
"""
def prepare_prompt_exec_repo_bench(tokenizer, obj: dict, context_order, max_tokens, max_generation_tokens, max_context_tokens):
    context_code_files, prefix_code, suffix_code = obj["context_code"], obj["prefix_code"], obj["suffix_code"]
    context_code = []
    context_code_file_names = []
    context_code_tokens = 0
    for file_name, file_code in context_code_files:
        cur_tokens = tokenizer.tokenize(file_code)
        if len(cur_tokens) + context_code_tokens < max_context_tokens:
            context_code_tokens += len(cur_tokens)
            context_code.append(f"##{file_name}##:\n{file_code}")
            context_code_file_names.append(file_name)
        else:
            _context_code = f"##{file_name}##:\n{file_code}"
            _context_code = _truncate_prompt(_context_code, max_num_tokens = max_context_tokens - context_code_tokens, tokenizer = tokenizer, side="right")
            _context_code = "\n".join(_context_code.split("\n")[:-1])
            context_code.append(_context_code)
            context_code_file_names.append(file_name)
            break
    if context_order == "close":
        context_code = context_code[::-1]
        context_code_file_names = context_code_file_names[::-1]
    max_in_file_tokens = max_tokens - max_generation_tokens - max_context_tokens
    prefix_code = _truncate_prompt(prefix_code, max_num_tokens = max_in_file_tokens // 2, tokenizer = tokenizer, side="left")
    suffix_code = _truncate_prompt(suffix_code, max_num_tokens = max_in_file_tokens // 2, tokenizer = tokenizer, side="right")
    repo_start = f"<|repo_name|>{obj['repo_name']}"
    repo_content = ""
    for c, f in zip(context_code, context_code_file_names):
        repo_content += f"\n<|file_sep|>{f}\n{c}"
    context_code = repo_start + repo_content + f"\n<|file_sep|>{obj['file_name']}\n"
    input_prompt = QWEN_CODER_TEMPLAT.format(context_code, prefix_code, suffix_code)
    obj["input"] = post_process(input_prompt)
    return obj
def prepare_prompt_exec_repo_bench_no(tokenizer, obj: dict, context_order, max_tokens, max_generation_tokens, max_context_tokens):
    context_code_files, prefix_code, suffix_code = obj["context_code"], obj["prefix_code"], obj["suffix_code"]
    context_code = []
    context_code_file_names = []
    context_code_tokens = 0
    for file_name, file_code in context_code_files:
        cur_tokens = tokenizer.tokenize(file_code)
        if len(cur_tokens) + context_code_tokens < max_context_tokens:
            context_code_tokens += len(cur_tokens)
            context_code.append(f"##{file_name}##:\n{file_code}")
            context_code_file_names.append(file_name)
        else:
            _context_code = f"##{file_name}##:\n{file_code}"
            _context_code = _truncate_prompt(_context_code, max_num_tokens = max_context_tokens - context_code_tokens, tokenizer = tokenizer, side="right")
            _context_code = "\n".join(_context_code.split("\n")[:-1])
            context_code.append(_context_code)
            context_code_file_names.append(file_name)
            break
    if context_order == "close":
        context_code = context_code[::-1]
        context_code_file_names = context_code_file_names[::-1]
    max_in_file_tokens = max_tokens - max_generation_tokens - max_context_tokens
    prefix_code = _truncate_prompt(prefix_code, max_num_tokens = max_in_file_tokens // 2, tokenizer = tokenizer, side="left")
    suffix_code = _truncate_prompt(suffix_code, max_num_tokens = max_in_file_tokens // 2, tokenizer = tokenizer, side="right")
    repo_start = f"<|repo_name|>{obj['repo_name']}"
    repo_content = ""
    for c, f in zip(context_code, context_code_file_names):
        repo_content += f"\n<|file_sep|>{f}\n{c}"
    repo_content = ""
    context_code = repo_start + repo_content + f"\n<|file_sep|>{obj['file_name']}\n"
    input_prompt = QWEN_CODER_TEMPLAT.format(context_code, prefix_code, suffix_code)
    obj["input"] = post_process(input_prompt)
    return obj
def prepare_prompt_exec_repo_bench_ex(tokenizer, obj: dict, context_order, max_tokens, max_generation_tokens, max_context_tokens, grammer_path: str):
    from .collapse import collapse_code
    context_code_files, prefix_code, suffix_code = obj["context_code"], obj["prefix_code"], obj["suffix_code"]
    context_code = []
    context_code_file_names = []
    context_code_tokens = 0
    metadata = []
    for file_name, file_code in context_code_files:
        cur_tokens = tokenizer.tokenize(file_code)
        if len(cur_tokens) + context_code_tokens < max_context_tokens:
            context_code_tokens += len(cur_tokens)
            collapsed_code = collapse_code(file_code, grammer_path)
            metadata.append({
                "original_len": len(cur_tokens),
                "collapsed_len": len(tokenizer.tokenize(collapsed_code))
            })
            context_code.append(f"##{file_name}##:\n{collapsed_code}")
            context_code_file_names.append(file_name)
        else:
            collapsed_code = collapse_code(file_code, grammer_path)
            collapsed_len = len(tokenizer.tokenize(collapsed_code))
            metadata.append({
                "original_len": len(cur_tokens),
                "collapsed_len": collapsed_len
            })
            _context_code = f"##{file_name}##:\n{file_code}"
            _context_code = _truncate_prompt(_context_code, max_num_tokens = max_context_tokens - context_code_tokens, tokenizer = tokenizer, side="right")
            _context_code = "\n".join(_context_code.split("\n")[:-1])
            context_code.append(_context_code)
            context_code_file_names.append(file_name)
            break
    if context_order == "close":
        context_code = context_code[::-1]
        context_code_file_names = context_code_file_names[::-1]
    max_in_file_tokens = max_tokens - max_generation_tokens - max_context_tokens
    prefix_code = _truncate_prompt(prefix_code, max_num_tokens = max_in_file_tokens // 2, tokenizer = tokenizer, side="left")
    suffix_code = _truncate_prompt(suffix_code, max_num_tokens = max_in_file_tokens // 2, tokenizer = tokenizer, side="right")
    repo_start = f"<|repo_name|>{obj['repo_name']}"
    repo_content = ""
    for c, f in zip(context_code, context_code_file_names):
        repo_content += f"\n<|file_sep|>{f}\n{c}"
    context_code = repo_start + repo_content + f"\n<|file_sep|>{obj['file_name']}\n"
    input_prompt = QWEN_CODER_TEMPLAT.format(context_code, prefix_code, suffix_code)
    obj["input"] = post_process(input_prompt)
    return obj, metadata