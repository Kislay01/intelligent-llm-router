import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pricing import SERVED_MODELS  # noqa: E402

BENCH_DIR = "data/bench"
TASKS_FILE = f"{BENCH_DIR}/tasks.json"
GENERATIONS_FILE = f"{BENCH_DIR}/generations.jsonl"
RESULTS_FILE = f"{BENCH_DIR}/results.jsonl"
OUTCOMES_FILE = f"{BENCH_DIR}/outcomes.jsonl"
HEADROOM_FILE = f"{BENCH_DIR}/headroom.json"

MODELS = list(SERVED_MODELS)


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_tasks() -> list:
    with open(TASKS_FILE, encoding="utf-8") as f:
        return json.load(f)
