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
    """Rows of a JSONL file. A line cut off by a crash or power loss is skipped (its row is redone)."""
    if not os.path.exists(path):
        return []
    rows, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"Note: skipped {skipped} unreadable line(s) in {path}; those rows will be redone.")
    return rows


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # If the last write was cut off mid-line, start on a fresh line so the new row is not corrupted too.
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
        if needs_newline:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_tasks() -> list:
    with open(TASKS_FILE, encoding="utf-8") as f:
        return json.load(f)
