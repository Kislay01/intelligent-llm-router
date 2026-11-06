"""Have each Ollama model answer every benchmark task. Resumable; safe to stop and re-run.

Models are the OUTER loop on purpose: Ollama keeps only some models in memory, so alternating
models per query forces slow reloads (and was a likely cause of the old read timeouts).
Failed calls are never stored as answers - they are retried on the next run.

Usage: python bench/generate.py [--models llama3.2:3b mistral:7b] [--limit 20]
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(__file__))

from common import GENERATIONS_FILE, MODELS, append_jsonl, load_tasks, read_jsonl
from ollama_client import query_model

# Greedy decoding so a re-run reproduces the same answers; capped length stops runaway output.
OPTIONS = {"temperature": 0, "seed": 0, "num_predict": 1024}


def pending(tasks: list, model: str, done: set) -> list:
    return [t for t in tasks if (t["id"], model) not in done]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--limit", type=int, default=None, help="at most N tasks per model this run")
    args = parser.parse_args()

    tasks = load_tasks()
    done = {(r["task_id"], r["model"]) for r in read_jsonl(GENERATIONS_FILE)}
    failures = []

    for model in args.models:
        todo = pending(tasks, model, done)
        if args.limit is not None:
            todo = todo[: args.limit]
        print(f"\n{model}: {len(todo)} tasks to generate ({len(tasks) - len(pending(tasks, model, done))} already done)")
        started = time.perf_counter()
        for n, task in enumerate(todo, start=1):
            t0 = time.perf_counter()
            try:
                response = query_model(model, task["question"], options=OPTIONS, timeout=300)
            except Exception as e:
                failures.append((model, task["id"], str(e)[:120]))
                print(f"  [{n}/{len(todo)}] {task['id']}: FAILED ({str(e)[:80]})")
                continue
            append_jsonl(GENERATIONS_FILE, {
                "task_id": task["id"], "source": task["source"], "model": model,
                "response": response, "gen_seconds": round(time.perf_counter() - t0, 2),
            })
            elapsed = time.perf_counter() - started
            eta = elapsed / n * (len(todo) - n)
            print(f"  [{n}/{len(todo)}] {task['id']} ({time.perf_counter() - t0:.1f}s, eta {eta / 60:.0f} min)")

    print(f"\nFailed calls this run: {len(failures)} (re-run to retry them)")
    for model, task_id, err in failures[:10]:
        print(f"  {model} {task_id}: {err}")


if __name__ == "__main__":
    main()
