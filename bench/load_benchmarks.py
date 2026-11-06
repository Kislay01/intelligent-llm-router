"""Build data/bench/tasks.json from HumanEval and MBPP (code tasks with unit tests).

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbpp-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import concatenate_datasets, load_dataset  # imported late: only this script needs it

    he = load_dataset("openai/openai_humaneval", split="test")
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized")
    mbpp_rows = list(concatenate_datasets([mbpp[s] for s in mbpp]))

    tasks = normalize_humaneval(list(he)) + normalize_mbpp(mbpp_rows, args.mbpp_n, args.seed)
    os.makedirs(BENCH_DIR, exist_ok=True)
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=1, ensure_ascii=False)
    print(f"Saved {len(tasks)} tasks ({len(he)} HumanEval + {len(tasks) - len(he)} MBPP) to {TASKS_FILE}")


if __name__ == "__main__":
    main()
