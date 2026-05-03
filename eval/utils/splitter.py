def _line_indent(line: str) -> int:
    stripped = line.lstrip(" \t")
    return len(line) - len(stripped)


def _is_function_header(stripped_line: str) -> bool:
    return stripped_line.startswith("def ") or stripped_line.startswith("async def ")


def _find_function_blocks(code: str):
    lines = code.splitlines(keepends=True)
    offsets = []
    cursor = 0

    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    blocks = []
    line_count = len(lines)
    i = 0

    while i < line_count:
        line = lines[i]
        stripped = line.lstrip(" \t")
        if not _is_function_header(stripped):
            i += 1
            continue

        indent = _line_indent(line)

        start_line = i
        while start_line > 0:
            previous_line = lines[start_line - 1]
            previous_stripped = previous_line.lstrip(" \t")
            if _line_indent(previous_line) != indent or not previous_stripped.startswith("@"):
                break
            start_line -= 1

        end_line = i + 1
        while end_line < line_count:
            candidate_line = lines[end_line]
            candidate_stripped = candidate_line.strip()

            if candidate_stripped == "":
                end_line += 1
                continue

            if _line_indent(candidate_line) <= indent:
                break

            end_line += 1

        start = offsets[start_line]
        end = offsets[end_line] if end_line < line_count else len(code)
        blocks.append((start, end))
        i += 1

    return blocks


def split_python_code_into_three_parts(code: str):
    """
    Split Python code text into (prefix, middle, suffix).

    The middle part is the complete function block whose span is closest
    to the center of the file. This uses character/line-based heuristics
    instead of Python parsing so it still works on invalid syntax.
    """
    blocks = _find_function_blocks(code)
    if not blocks:
        if not code:
            return "", "", ""

        split_point = len(code) // 2
        return code[:split_point], code[split_point:], ""

    file_midpoint = len(code) / 2
    interior_blocks = [
        block for block in blocks
        if block[0] > 0 and block[1] < len(code)
    ]
    candidate_blocks = interior_blocks if interior_blocks else blocks

    start, end = min(
        candidate_blocks,
        key=lambda block: abs(((block[0] + block[1]) / 2) - file_midpoint),
    )

    prefix = code[:start]
    middle = code[start:end]
    suffix = code[end:]
    return prefix, middle, suffix


def split_code_file_text_into_three_parts(code: str, language: str = "python"):
    """
    Split code file text into (prefix, middle, suffix).

    For Python, the middle part is chosen first and should be a complete
    function block. The prefix is everything before that function, and the
    suffix is everything after it.
    """
    normalized_language = language.strip().lower()
    if normalized_language != "python":
        raise ValueError("Only Python is supported.")

    return split_python_code_into_three_parts(code)
