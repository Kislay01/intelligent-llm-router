"""Audit failed answers: are they wrong answers, or did extraction/sandboxing break?

Error types that point at OUR plumbing rather than the model: SyntaxError / IndentationError (bad code
extraction), NameError for the target function (wrong block picked). AssertionError = a genuinely wrong answer.

Usage: python bench/inspect_failures.py [--show 5] [--model llama3.2:3b]
"""
import argparse
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(__file__))

from common import GENERATIONS_FILE, RESULTS_FILE, load_tasks, read_jsonl
from extract import extract_code


def error_type(detail: str, status: str) -> str:
    """Exception name from the last line of stderr, e.g. 'AssertionError'."""
    if status == "timeout":
        return "Timeout"
    lines = [l for l in (detail or "").strip().splitlines() if l.strip()]
    if not lines:
        return "NoMessage"
    last = lines[-1].split(":", 1)[0].strip()
    return last if last.isidentifier() else "Other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--type", default=None, help="only failures of this error type, e.g. NameError")
    args = parser.parse_args()

    tasks = {t["id"]: t for t in load_tasks()}
    gens = {(g["task_id"], g["model"]): g for g in read_jsonl(GENERATIONS_FILE)}
    failed = [r for r in read_jsonl(RESULTS_FILE) if not r["passed"] and (not args.model or r["model"] == args.model)]
    if args.type:
        failed = [r for r in failed if error_type(r["detail"], r["status"]) == args.type]
    print(f"{len(failed)} failed answers\n")

    # An odd number of ``` fences means the reply was cut off mid-code (the 1024-token cap).
    truncated = Counter(r["model"] for r in failed if gens[(r["task_id"], r["model"])]["response"].count("```") % 2)
    if truncated:
        print("Possibly truncated replies among these failures: " + ", ".join(f"{m}: {n}" for m, n in sorted(truncated.items())) + "\n")

    by_model = {}
    for r in failed:
        by_model.setdefault(r["model"], Counter())[error_type(r["detail"], r["status"])] += 1
    for model, counts in sorted(by_model.items()):
        print(f"{model:<20}" + ", ".join(f"{k}: {v}" for k, v in counts.most_common()))

    print(f"\nFirst {args.show} failures:")
    for r in failed[: args.show]:
        task, gen = tasks[r["task_id"]], gens[(r["task_id"], r["model"])]
        code = extract_code(gen["response"], task.get("entry_point")).splitlines()
        print(f"\n--- {r['task_id']} / {r['model']} / {error_type(r['detail'], r['status'])}")
        print("extracted code (first 12 lines):\n  " + "\n  ".join(code[:12]))
        print("stderr tail:\n  " + (r["detail"] or "(none)").strip().replace("\n", "\n  ")[-400:])


if __name__ == "__main__":
    main()
