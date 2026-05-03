# from debug import DEBUG
DEBUG = True
START_MARKS = ["def ", "async def ", "function ", "async function "] #, "private ", "public ", "internal ", "protected "]
END_MARKS = ["@", "class ", "def ", "async def ", "function ", "async function "] #, "private ", "public ", "internal ", "protected "]

class PromptCache:
    def __init__(self, pre_token: str, mid_token: str, suf_token: str) -> None:
        self.pre_token = pre_token
        self.mid_token = mid_token
        self.suf_token = suf_token
        if DEBUG:
            import os
            os.makedirs("log", exist_ok=True)
    def _split(self, prompt: str) -> tuple[str, str, str]:
        _, rest = prompt.split(self.pre_token)
        prefix, rest = rest.split(self.suf_token)
        suffix, middle = rest.split(self.mid_token)
        return (prefix, middle, suffix)
    def _construct_prompt(self, prefix: str, middle: str, suffix: str) -> str:
        return self.pre_token + prefix + self.suf_token + suffix + self.mid_token + middle
    def _construct_function_prompt(self, prefix: str, f_prefix: str, middle: str, f_suffix: str, suffix: str) -> str:
        return self.pre_token + prefix + self.suf_token + suffix + self.pre_token + f_prefix + self.suf_token + f_suffix + self.mid_token + middle
    def _split_line(self, total: str, line_number: int) -> tuple[str, str, str]:
        lines = total.splitlines()
        previous_lines = lines[:line_number]
        current_line = lines[line_number]
        suffix_lines = lines[line_number+1:]
        return "\n".join(previous_lines) + "\n", current_line, "\n".join(suffix_lines) + "\n"
    def _detect_function_prefix(self, prefix: str) -> int | None:
        lines = prefix.splitlines()
        for i in range(len(lines)-1, -1, -1):
            line = lines[i]
            if any([line.lstrip().startswith(mark) for mark in START_MARKS]):
                return i
        return None
    def _detect_stop_suffix(self, suffix: str) -> int:
        lines = suffix.splitlines()
        target = len(lines)
        for i in range(len(lines)-1, -1, -1):
            line = lines[i]
            if any([line.lstrip().startswith(mark) for mark in END_MARKS]):
                target = i
        return target
    def function_cache(self, prompt: str) -> str:
        prompt = prompt.replace("\r\n", "\n")
        if DEBUG:
            with open("log/original.py", 'w', encoding='utf-8') as file:
                file.write(prompt)
        preffix, middle, suffix = self._split(prompt)
        
        total = preffix + middle + suffix
        position = len(preffix)
        function_line = self._detect_function_prefix(preffix) or 0
        end_line = self._detect_stop_suffix(suffix)
        
        previous_lines = preffix.splitlines()[:function_line]
        after_lines = suffix.splitlines()[end_line:]
        if len(previous_lines) == 0:
            previous = ""
        else:
            previous = "\n".join(previous_lines) + "\n"
        if len(after_lines) == 0:
            after = ""
        else:
            after = "\n".join(after_lines) + "\n"
        previous_in_function = total[len(previous):position]
        after_in_function = total[position:len(total)-len(after)-len(middle)]

        result = self._construct_function_prompt(
            previous,
            previous_in_function,
            middle,
            after_in_function,
            after
        )
        if DEBUG:
            with open("log/cache.py", 'w', encoding='utf-8') as file:
                file.write(result)
        return result
    
def function_cache(pre_token: str, mid_token: str, suf_token: str, prompt: str) -> str:
    cache = PromptCache(pre_token, mid_token, suf_token)
    return cache.function_cache(prompt)