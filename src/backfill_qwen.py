"""Give every labeled query a score for all four served models.

The first 150 queries were judged against codellama:7b instead of qwen2.5-coder:7b, so qwen
has scores for only 24 queries. For each row missing a qwen score this script:
  1. generates qwen's answer with Ollama (and re-generates any served model's answer that is a
     stored "[ERROR: ...]" timeout string, which would otherwise be judged as a terrible answer),
  2. re-judges {llama3.2:3b, llama3.1:8b, qwen2.5-coder:7b, mistral:7b} in ONE Gemini call
     (same prompt and model order as label_pipeline.py, so scores are comparable),
  3. stores the new scores/winner and keeps the old ones as legacy_scores / legacy_winner.
The file is rewritten atomically after every row, so the script is safe to interrupt and re-run.

Usage: python src/backfill_qwen.py [--limit N]
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(__file__))

from gemini_client import ask_gemini
from label_pipeline import JUDGE_PROMPT_TEMPLATE, OLLAMA_MODELS, parse_judge_response
from ollama_client import query_model

RESULTS_FILE = "data/labeled_results.jsonl"
QWEN = "qwen2.5-coder:7b"
ERROR_PREFIX = "[ERROR"


def models_to_generate(row: dict) -> list:
    """Served models whose answer is missing (qwen on old rows) or is a stored error string."""
    return [
        m for m in OLLAMA_MODELS
        if m not in row["responses"]
        or row["responses"][m].startswith(ERROR_PREFIX)
        or m not in row["scores"]
    ]


def needs_backfill(row: dict) -> bool:
    return bool(models_to_generate(row))


def build_prompt(query: str, responses: dict) -> str:
    m = OLLAMA_MODELS
    return JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        m0=m[0], r0=responses[m[0]], m1=m[1], r1=responses[m[1]],
        m2=m[2], r2=responses[m[2]], m3=m[3], r3=responses[m[3]],
    )


def validate_scores(scores: dict) -> dict:
    """Judge output must score exactly the four models with integers in 1..10."""
    if set(scores) != set(OLLAMA_MODELS):
        raise ValueError(f"judge scored {sorted(scores)}, expected {sorted(OLLAMA_MODELS)}")
    for model, value in scores.items():
        if not isinstance(value, (int, float)) or not 1 <= value <= 10:
            raise ValueError(f"bad score for {model}: {value!r}")
    return scores


def rebuild_row(row: dict, new_responses: dict, judgment: dict) -> dict:
    """New row with the full four-model pool; the first-ever judgement is kept for audit."""
    scores = validate_scores(judgment["scores"])
    responses = {**row["responses"], **new_responses}
    new = dict(row)
    new.setdefault("legacy_scores", row["scores"])  # keep the ORIGINAL on repeated passes
    new.setdefault("legacy_winner", row["winner"])
    new["responses"] = {m: responses[m] for m in OLLAMA_MODELS}  # drops codellama's answer
    new["scores"] = scores
    new["winner"] = max(scores, key=scores.get)  # derive it ourselves, do not trust the judge's field
    new["backfilled"] = True
    return new


def judge_with_retry(prompt: str, attempts: int = 3) -> dict:
    for attempt in range(attempts):
        try:
            return parse_judge_response(ask_gemini(prompt))
        except Exception as e:  # rate limits, 503s and malformed JSON all just need a pause
            if attempt == attempts - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"  Gemini/parse error ({e}), retrying in {wait}s...")
            time.sleep(wait)


def load_rows(path: str = RESULTS_FILE) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_rows(rows: list, path: str = RESULTS_FILE) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic: an interrupt can never leave a half-written file


def generate_with_retry(model: str, query: str, attempts: int = 2) -> str:
    """Ollama call that must not return an error string (a timeout on first model load is common)."""
    for attempt in range(attempts):
        try:
            return query_model(model, query)
        except Exception as e:
            if attempt == attempts - 1:
                raise
            print(f"  Ollama error on {model} ({e}), retrying...")


def backfill_one(row: dict) -> dict:
    new_responses = {m: generate_with_retry(m, row["query"]) for m in models_to_generate(row)}
    prompt = build_prompt(row["query"], {**row["responses"], **new_responses})
    return rebuild_row(row, new_responses, judge_with_retry(prompt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process at most N rows this run")
    args = parser.parse_args()

    rows = load_rows()
    todo = [i for i, r in enumerate(rows) if needs_backfill(r)]
    print(f"{len(rows)} rows, {len(todo)} need a missing/errored answer regenerated and re-judged.")
    if args.limit is not None:
        todo = todo[: args.limit]

    for n, i in enumerate(todo, start=1):
        row = rows[i]
        print(f"[{n}/{len(todo)}] Query {row['id']} ({row['category']}): {row['query'][:60]}...")
        try:
            rows[i] = backfill_one(row)
        except Exception as e:
            print(f"  -> FAILED: {e}. Skipping, will retry on next run.")
            continue
        save_rows(rows)
        print(f"  -> old winner {row['winner']} | new scores {rows[i]['scores']} | winner {rows[i]['winner']}")
        time.sleep(4)

    remaining = sum(needs_backfill(r) for r in rows)
    print(f"\nDone. Rows still needing work: {remaining}")
    if remaining == 0:
        print("Next: python src/train_classifier.py && python src/run_evaluation.py")


if __name__ == "__main__":
    main()
