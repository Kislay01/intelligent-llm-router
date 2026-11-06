"""Build data/bench/tasks.json from HumanEval and MBPP (code, graded by unit tests) and GSM8K and MMLU (exact match).

Re-running with more sources keeps the existing task ids (same seed), so generate.py only does the new ones.

Needs `pip install -r requirements-bench.txt` (the `datasets` library) and internet.
Column names follow the Hugging Face dataset cards; the normalisers fail loudly if they differ.

Usage: python bench/load_benchmarks.py [--mbpp-n 100] [--seed 42]
"""
import argparse
import json
import os
import random
import sys

sys.path.append(os.path.dirname(__file__))

from common import BENCH_DIR, TASKS_FILE

HUMANEVAL_INSTRUCTION = (
    "Complete this Python function. Respond with the complete function, including its signature, "
    "in a single ```python code block and nothing else.\n\n"
)
MBPP_INSTRUCTION = (
    "Write a Python solution for the task below. Respond with the complete code in a single "
    "```python code block and nothing else.\n\n"
)


GSM8K_INSTRUCTION = (
    "Solve the math word problem. Think step by step, then finish with a final line of the form "
    "'Final answer: <number>' (digits only, no units).\n\n"
)
MMLU_INSTRUCTION = (
    "Answer the multiple-choice question. You may reason briefly, then finish with a final line of the form "
    "'Final answer: <letter>' where <letter> is A, B, C or D.\n\n"
)


def _require(row: dict, columns: list, source: str) -> None:
    missing = [c for c in columns if c not in row]
    if missing:
        raise KeyError(f"{source}: expected columns {missing} not found; got {sorted(row)}")


def normalize_humaneval(rows: list) -> list:
    tasks = []
    for r in rows:
        _require(r, ["task_id", "prompt", "test", "entry_point"], "humaneval")
        tasks.append({
            "id": r["task_id"].lower(),            # "humaneval/0"
            "source": "humaneval",
            "question": HUMANEVAL_INSTRUCTION + r["prompt"],
            "prompt": r["prompt"],
            "tests": r["test"],
            "entry_point": r["entry_point"],
        })
    return tasks


def normalize_mbpp(rows: list, n: int, seed: int) -> list:
    for r in rows[:1]:
        _require(r, ["task_id", "prompt", "test_list", "test_imports"], "mbpp (sanitized)")
    rows = sorted(rows, key=lambda r: r["task_id"])
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    tasks = []
    for r in sorted(picked, key=lambda r: r["task_id"]):
        tests = "\n".join(r["test_list"])
        tasks.append({
            "id": f"mbpp/{r['task_id']}",
            "source": "mbpp",
            # The tests are shown (standard MBPP practice) so the model knows the function name.
            "question": MBPP_INSTRUCTION + f"{r['prompt']}\n\nYour code must pass these tests:\n{tests}",
            "setup": "\n".join(r["test_imports"]),
            "tests": tests,
            "entry_point": None,
        })
    return tasks


def normalize_gsm8k(rows: list, n: int, seed: int) -> list:
    for r in rows[:1]:
        _require(r, ["question", "answer"], "gsm8k")
    picked = random.Random(seed).sample(range(len(rows)), min(n, len(rows)))
    tasks = []
    for i in sorted(picked):
        answer = rows[i]["answer"].split("####")[-1].strip().replace(",", "")
        tasks.append({"id": f"gsm8k/{i}", "source": "gsm8k", "kind": "exact",
                      "question": GSM8K_INSTRUCTION + rows[i]["question"], "answer": answer})
    return tasks


def normalize_mmlu(rows: list, n: int, seed: int) -> list:
    for r in rows[:1]:
        _require(r, ["question", "choices", "answer", "subject"], "mmlu")
    picked = random.Random(seed).sample(range(len(rows)), min(n, len(rows)))
    tasks = []
    for i in sorted(picked):
        r = rows[i]
        options = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", r["choices"]))
        tasks.append({"id": f"mmlu/{i}", "source": "mmlu", "kind": "exact", "subject": r["subject"],
                      "question": MMLU_INSTRUCTION + f"{r['question']}\n{options}", "answer": "ABCD"[int(r["answer"])]})
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbpp-n", type=int, default=100)
    parser.add_argument("--exact-n", type=int, default=100, help="questions sampled from each of GSM8K and MMLU")
    parser.add_argument("--sources", nargs="+", default=["humaneval", "mbpp", "gsm8k", "mmlu"],
                        choices=["humaneval", "mbpp", "gsm8k", "mmlu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import concatenate_datasets, load_dataset  # imported late: only this script needs it

    tasks = []
    if "humaneval" in args.sources:
        tasks += normalize_humaneval(list(load_dataset("openai/openai_humaneval", split="test")))
    if "mbpp" in args.sources:
        mbpp = load_dataset("google-research-datasets/mbpp", "sanitized")
        tasks += normalize_mbpp(list(concatenate_datasets([mbpp[s] for s in mbpp])), args.mbpp_n, args.seed)
    if "gsm8k" in args.sources:
        tasks += normalize_gsm8k(list(load_dataset("openai/gsm8k", "main", split="test")), args.exact_n, args.seed)
    if "mmlu" in args.sources:
        tasks += normalize_mmlu(list(load_dataset("cais/mmlu", "all", split="test")), args.exact_n, args.seed)

    os.makedirs(BENCH_DIR, exist_ok=True)
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=1, ensure_ascii=False)
    counts = {src: sum(t["source"] == src for t in tasks) for src in args.sources}
    print(f"Saved {len(tasks)} tasks {counts} to {TASKS_FILE}")


if __name__ == "__main__":
    main()
