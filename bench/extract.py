"""Turn a model's reply into a runnable program."""
import re
from typing import Optional

FENCE = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
OPEN_FENCE = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*)$", re.DOTALL | re.IGNORECASE)
MAIN_GUARD = re.compile(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:.*", re.DOTALL | re.MULTILINE)


def extract_code(response: str, entry_point: Optional[str] = None) -> str:
    """Pick the code block that defines the function we care about.

    Order: a block defining entry_point; else the first block containing `def `; else the first block;
    else an unterminated fence; else the raw reply. A trailing `if __name__ == "__main__":` demo is
    dropped, since demos (input(), prints) are not part of the solution.
    """
    blocks = FENCE.findall(response)
    if not blocks:
        m = OPEN_FENCE.search(response)
        blocks = [m.group(1)] if m else [response]

    chosen = None
    if entry_point:
        chosen = next((b for b in blocks if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", b)), None)
    if chosen is None:
        chosen = next((b for b in blocks if "def " in b), blocks[0])
    return MAIN_GUARD.sub("", chosen).rstrip() + "\n"


def build_program(task: dict, response: str) -> str:
    """Complete program = (context) + model code + the benchmark's tests."""
    code = extract_code(response, task.get("entry_point"))
    if task["source"] == "humaneval":
        entry = task["entry_point"]
        defines = re.search(rf"def\s+{re.escape(entry)}\s*\(", code)
        # The HumanEval prompt (imports, helper functions, a docstring-only stub) goes first.
        # A full redefinition overrides the stub; a bare function body continues the stub.
        glue = "\n\n" if defines else "\n"
        return f"{task['prompt']}{glue}{code}\n\n{task['tests']}\n\ncheck({entry})\n"
    return f"{task.get('setup', '')}\n{code}\n\n{task['tests']}\n"
