"""Exact-match grading for GSM8K (a number) and MMLU (a letter). No code is executed."""
import re
from typing import Optional

FINAL = re.compile(r"(?i:final answer)\s*[:\-]?\s*(.+)")
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
LETTER_AFTER_FINAL = re.compile(r"(?i:final answer)\s*[:\-]?\s*\(?([ABCD])\b")
LETTER_FALLBACK = re.compile(r"(?i:answer is|answer:)\s*\(?([ABCD])\b")


def _to_float(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract_number(response: str) -> Optional[float]:
    """The number after the last 'Final answer:'; if the model skipped that line, the last number in the reply."""
    finals = FINAL.findall(response)
    if finals:
        nums = NUMBER.findall(finals[-1])
        if nums:
            return _to_float(nums[0])
    nums = NUMBER.findall(response)
    return _to_float(nums[-1]) if nums else None


def extract_letter(response: str) -> Optional[str]:
    """A/B/C/D after 'Final answer:' (last occurrence); else 'the answer is X'. Never guesses from stray capitals."""
    for pattern in (LETTER_AFTER_FINAL, LETTER_FALLBACK):
        found = pattern.findall(response)
        if found:
            return found[-1].upper()
    return None


def grade_exact(task: dict, response: str) -> dict:
    if task["source"] == "gsm8k":
        got = extract_number(response)
        want = _to_float(str(task["answer"]))
        ok = got is not None and want is not None and abs(got - want) < 1e-6
        return {"status": "pass" if ok else "fail", "detail": "" if ok else f"expected {task['answer']}, got {got}"}
    if task["source"] == "mmlu":
        got = extract_letter(response)
        ok = got == task["answer"]
        return {"status": "pass" if ok else "fail", "detail": "" if ok else f"expected {task['answer']}, got {got}"}
    raise ValueError(f"no exact-match grader for source {task['source']!r}")
