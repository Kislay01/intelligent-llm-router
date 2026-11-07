"""How good is the LLM judge, measured against unit-test ground truth?

Takes the tasks where the four models DISAGREE (some pass, some fail), asks the same Gemini judge and
the same prompt as the original labelling pipeline to score all four answers, then checks the scores
against pass/fail. Needs GEMINI_API_KEY in .env. Resumable.

Usage: python bench/judge_vs_truth.py [--limit 20]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(__file__))

from common import (BENCH_DIR, GENERATIONS_FILE, MODELS, OUTCOMES_FILE, append_jsonl, load_tasks,
                    read_jsonl)
from relabel import cost_aware_label

JUDGE_RUNS_FILE = f"{BENCH_DIR}/judge_runs.jsonl"
RESULT_FILE = f"{BENCH_DIR}/judge_vs_truth.json"


def mixed_tasks(outcomes: list) -> list:
    """Tasks where the choice of model changes correctness."""
    return [o for o in outcomes if 0 < sum(v["passed"] for v in o["outcomes"].values()) < len(o["outcomes"])]


def pair_accuracy(scores: dict, passed: dict) -> float:
    """Over (passing, failing) answer pairs: share where the judge scored the passing one higher (ties = 0.5)."""
    pairs = [(p, f) for p in scores if passed[p] for f in scores if not passed[f]]
    wins = [1.0 if scores[p] > scores[f] else 0.5 if scores[p] == scores[f] else 0.0 for p, f in pairs]
    return float(np.mean(wins))


def top_pick_pass(scores: dict, passed: dict) -> float:
    """Chance the judge's top-scored answer passes (ties broken at random)."""
    best = max(scores.values())
    top = [m for m, s in scores.items() if s == best]
    return float(np.mean([passed[m] for m in top]))


def bootstrap_mean(values, n_boot: int = 2000, seed: int = 42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def analyze(judged: list) -> dict:
    """judged: [{"task_id", "scores": {model: score}, "passed": {model: bool}}]"""
    models = list(judged[0]["scores"])
    pair = [pair_accuracy(j["scores"], j["passed"]) for j in judged]
    top = [top_pick_pass(j["scores"], j["passed"]) for j in judged]
    label = [float(j["passed"][cost_aware_label(j["scores"], 0)]) for j in judged]
    random_pick = [np.mean(list(j["passed"].values())) for j in judged]

    pass_scores = [j["scores"][m] for j in judged for m in models if j["passed"][m]]
    fail_scores = [j["scores"][m] for j in judged for m in models if not j["passed"][m]]
    win_share = {m: float(np.mean([top_pick_pass_for(j, m) for j in judged])) for m in models}
    pass_share = {m: float(np.mean([j["passed"][m] for j in judged])) for m in models}

    by_source = {}
    for src in sorted({j.get("source") for j in judged if j.get("source")}):
        sub = [j for j in judged if j.get("source") == src]
        by_source[src] = {
            "n": len(sub),
            "pairwise_accuracy": bootstrap_mean([pair_accuracy(j["scores"], j["passed"]) for j in sub]),
            "judge_top_pick_passes": bootstrap_mean([top_pick_pass(j["scores"], j["passed"]) for j in sub]),
            "random_model_passes": bootstrap_mean([np.mean(list(j["passed"].values())) for j in sub]),
        }

    return {
        "by_source": by_source,
        "n_tasks": len(judged),
        "pairwise_accuracy": bootstrap_mean(pair),           # 0.5 = no better than chance
        "judge_top_pick_passes": bootstrap_mean(top),
        "cost_aware_label_passes": bootstrap_mean(label),     # what a router trained on judge labels aims at
        "random_model_passes": bootstrap_mean(random_pick),
        "always_model_passes": {m: pass_share[m] for m in models},
        "mean_judge_score_passing": float(np.mean(pass_scores)),
        "mean_judge_score_failing": float(np.mean(fail_scores)),
        "judge_top_pick_share": win_share,
    }


def top_pick_pass_for(judged_row: dict, model: str) -> float:
    """Share of the judge's top slot this model gets (ties split evenly)."""
    scores = judged_row["scores"]
    best = max(scores.values())
    top = [m for m, s in scores.items() if s == best]
    return (1.0 / len(top)) if model in top else 0.0


def print_report(r: dict) -> None:
    f = lambda t: f"{t[0]*100:5.1f}% [{t[1]*100:.1f}, {t[2]*100:.1f}]"
    print(f"\n{r['n_tasks']} tasks where the four models disagree (some pass, some fail)\n")
    print(f"Pairwise accuracy (passing answer scored above failing one): {f(r['pairwise_accuracy'])}   (chance = 50%)")
    print(f"Judge's top-scored answer actually passes:                  {f(r['judge_top_pick_passes'])}")
    print(f"Cost-aware judge label points at a passing model:           {f(r['cost_aware_label_passes'])}")
    print(f"Picking a model at random passes:                           {f(r['random_model_passes'])}")
    print(f"\nMean judge score: passing answers {r['mean_judge_score_passing']:.2f}, "
          f"failing answers {r['mean_judge_score_failing']:.2f}")
    print(f"\n{'Model':<20}{'Judge ranks it #1':<20}{'Actually passes'}")
    for m in r["judge_top_pick_share"]:
        print(f"{m:<20}{r['judge_top_pick_share'][m]*100:>6.1f}%{'':<12}{r['always_model_passes'][m]*100:>6.1f}%")


def print_by_source(r: dict) -> None:
    if len(r["by_source"]) < 2:
        return
    f = lambda t: f"{t[0]*100:5.1f}% [{t[1]*100:4.1f}, {t[2]*100:4.1f}]"
    print(f"\n{'Benchmark':<12}{'n':>5}  {'Pairwise accuracy':<22}{'Top pick passes':<22}{'Random model passes'}")
    for src, v in r["by_source"].items():
        print(f"{src:<12}{v['n']:>5}  {f(v['pairwise_accuracy']):<22}{f(v['judge_top_pick_passes']):<22}{f(v['random_model_passes'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    from backfill_qwen import build_prompt, judge_with_retry, validate_scores  # same prompt/model order as labelling

    tasks = {t["id"]: t for t in load_tasks()}
    gens = {(g["task_id"], g["model"]): g["response"] for g in read_jsonl(GENERATIONS_FILE)}
    targets = mixed_tasks(read_jsonl(OUTCOMES_FILE))
    done = {r["task_id"] for r in read_jsonl(JUDGE_RUNS_FILE)}
    todo = [o for o in targets if o["id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(targets)} disagreement tasks, {len(done)} already judged, {len(todo)} to do this run")

    for n, o in enumerate(todo, start=1):
        responses = {m: gens[(o["id"], m)] for m in MODELS}
        try:
            scores = validate_scores(judge_with_retry(build_prompt(tasks[o["id"]]["question"], responses))["scores"])
        except Exception as e:
            print(f"  [{n}/{len(todo)}] {o['id']}: FAILED ({str(e)[:80]}); will retry next run")
            continue
        append_jsonl(JUDGE_RUNS_FILE, {"task_id": o["id"], "scores": scores})
        print(f"  [{n}/{len(todo)}] {o['id']}: {scores}")
        time.sleep(args.sleep)

    runs = {r["task_id"]: r["scores"] for r in read_jsonl(JUDGE_RUNS_FILE)}
    judged = [{"task_id": o["id"], "source": o["source"], "scores": runs[o["id"]],
               "passed": {m: o["outcomes"][m]["passed"] for m in MODELS}} for o in targets if o["id"] in runs]
    if not judged:
        sys.exit("Nothing judged yet.")
    result = analyze(judged)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print_report(result)
    print_by_source(result)
    if len(judged) < len(targets):
        print(f"\nNote: only {len(judged)} of {len(targets)} tasks judged so far; re-run to finish.")


if __name__ == "__main__":
    main()
