DEBUG = False

START_MARKS = ["def ", "async def ", "function ", "async function "]
END_MARKS = ["@", "class ", "def ", "async def ", "function ", "async function "]

class PromptCache:
    def __init__(self, pre_token: str, mid_token: str, suf_token: str) -> None:
        self.pre_mark = pre_token
        self.mid_mark = mid_token
        self.suf_mark = suf_token

    def _split(self, prompt: str) -> tuple[str, str, str]:
        _, rest = prompt.split(self.pre_mark)
        prefix, rest = rest.split(self.suf_mark)
        suffix, middle = rest.split(self.mid_mark)
        return (prefix, middle, suffix)

    def _construct_function_prompt(
        self,
        prefix: str,
        function_prefix: str,
        middle: str,
        function_suffix: str,
        suffix: str,
    ) -> str:
        return (
            self.pre_mark
            + prefix
            + self.suf_mark
            + suffix
            + self.pre_mark
            + function_prefix
            + self.suf_mark
            + function_suffix
            + self.mid_mark
            + middle
        )

    def _detect_function_prefix(self, prefix: str) -> int | None:
        lines = prefix.splitlines()
        for index in range(len(lines) - 1, -1, -1):
            line = lines[index]
            if any(line.lstrip().startswith(mark) for mark in START_MARKS):
                return index
        return None

    def _detect_stop_suffix(self, suffix: str) -> int:
        lines = suffix.splitlines()
        target = len(lines)
        for index in range(len(lines) - 1, -1, -1):
            line = lines[index]
            if any(line.lstrip().startswith(mark) for mark in END_MARKS):
                target = index
        return target

    def function_cache(self, prompt: str) -> str:
        prompt = prompt.replace("\r\n", "\n")
        if DEBUG:
            with open("log/original.py", "w", encoding="utf-8") as file:
                file.write(prompt)

        prefix, middle, suffix = self._split(prompt)
        total = prefix + middle + suffix
        position = len(prefix)
        function_line = self._detect_function_prefix(prefix) or 0
        end_line = self._detect_stop_suffix(suffix)

        previous_lines = prefix.splitlines()[:function_line]
        after_lines = suffix.splitlines()[end_line:]
        previous = "" if not previous_lines else "\n".join(previous_lines) + "\n"
        after = "" if not after_lines else "\n".join(after_lines) + "\n"
        previous_in_function = total[len(previous):position]
        after_in_function = total[position:len(total) - len(after) - len(middle)]

        result = self._construct_function_prompt(
            previous,
            previous_in_function,
            middle,
            after_in_function,
            after,
        )
        if DEBUG:
            with open("log/cache.py", "w", encoding="utf-8") as file:
                file.write(result)
        return result


def function_cache(prompt: str, prefix_mark: str, middle_mark: str, suffix_mark: str) -> str:
    cache = PromptCache(prefix_mark, middle_mark, suffix_mark)
    return cache.function_cache(prompt)
