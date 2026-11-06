"""Run every stored answer against its unit tests inside the sandbox. Resumable.

Usage:
  python bench/grade.py --selftest          # run this first: proves the sandbox works and isolates
  python bench/grade.py [--workers 2] [--regrade]
"""
import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(__file__))

from common import GENERATIONS_FILE, RESULTS_FILE, append_jsonl, load_tasks, read_jsonl
from extract import build_program
from runner import SandboxError, run_program, selftest


def grade_pairs(tasks, generations, done):
    by_id = {t["id"]: t for t in tasks}
    return [(by_id[g["task_id"]], g) for g in generations
            if g["task_id"] in by_id and (g["task_id"], g["model"]) not in done]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["docker", "local"], default="docker",
                        help="local runs model code directly on this machine: unsafe, tests only")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--regrade", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    if args.selftest:
        print(f"Sandbox self-test ({args.mode}):")
        try:
            ok = selftest(args.mode)
        except SandboxError as e:
            sys.exit(f"Sandbox unavailable: {e}")
        sys.exit(0 if ok else "Self-test FAILED: do not trust grading results.")

    if args.mode == "local":
        print("WARNING: local mode executes model-written code on this machine without isolation.")

    tasks, generations = load_tasks(), read_jsonl(GENERATIONS_FILE)
    done = set() if args.regrade else {(r["task_id"], r["model"]) for r in read_jsonl(RESULTS_FILE)}
    if args.regrade and os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    todo = grade_pairs(tasks, generations, done)
    print(f"{len(todo)} answers to grade ({len(done)} already graded)")

    lock, counter = threading.Lock(), {"n": 0}

    def work(item):
        task, gen = item
        result = run_program(build_program(task, gen["response"]), args.mode, args.timeout)
        with lock:
            append_jsonl(RESULTS_FILE, {"task_id": task["id"], "source": task["source"], "model": gen["model"],
                                        "status": result["status"], "passed": result["status"] == "pass",
                                        "detail": result["detail"]})
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                print(f"  graded {counter['n']}/{len(todo)}")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, todo))
    except SandboxError as e:
        sys.exit(f"Stopped: {e}\nResults so far are saved; fix Docker and re-run to continue.")
    print("Done. Next: python bench/report.py")


if __name__ == "__main__":
    main()
