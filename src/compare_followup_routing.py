"""Compare three ways of routing a follow-up message ("explain that more", "now in Java").

Each case is a labeled dataset query (asked first, answered by its best model) plus a follow-up.
Strategies: none (route the raw follow-up), sticky (low confidence -> previous model),
rewrite (route a standalone rewrite made by llama3.2:3b). Needs Ollama running for `rewrite`.

Proxy for quality: the follow-up is assumed to need about what the first query needed, so a choice
is scored with the judge score the chosen model got on the base query. The classifier is trained
WITHOUT the base queries, otherwise `rewrite` would route a near-copy of a training example.

Usage: python src/compare_followup_routing.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(__file__))

from context import rewrite_query
from evaluation import BASELINE_MODEL, bootstrap, outcome_matrices
from pricing import SERVED_MODELS
from relabel import DEFAULT_TOLERANCE, get_label
from routing import MODES, resolve_route
from sklearn.linear_model import LogisticRegression

CASES_FILE = "data/followup_cases.json"
LABELED_FILE = "data/labeled_results.jsonl"
REWRITE_CACHE = "data/followup_rewrites.json"
RESULTS_FILE = "data/followup_comparison.json"


def load_inputs():
    with open(LABELED_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    return rows, cases


def build_history(row: dict) -> list:
    """The first turn of the conversation: the base query answered by its cost-aware best model."""
    model = get_label(row, DEFAULT_TOLERANCE)
    return [{"query": row["query"], "response": row["responses"][model], "routed_model": model}]


def train_held_out(rows, X, held_out_ids):
    keep = [i for i, r in enumerate(rows) if r["id"] not in held_out_ids]
    labels = [get_label(rows[i], DEFAULT_TOLERANCE) for i in keep]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[keep], labels)
    return clf


def make_predict(clf, embed):
    classes = [str(c) for c in clf.classes_]

    def predict(text):
        p = clf.predict_proba([embed(text)])[0]
        i = int(p.argmax())
        return classes[i], float(p[i]), dict(zip(classes, map(float, p)))
    return predict


def cached_rewriter(cache: dict, timings: list):
    def rewrite(history, query):
        key = history[-1]["query"] + " || " + query
        if key not in cache:
            start = time.perf_counter()
            cache[key] = rewrite_query(history, query, strict=True)
            timings.append(time.perf_counter() - start)
        return cache[key]
    return rewrite


def score_strategy(rows_by_id, cases, chosen, quality, cost, index):
    """Per-case quality and cost of the chosen models, plus agreement with the base query's best model."""
    idx = [index[c["base_id"]] for c in cases]
    cols = [SERVED_MODELS.index(m) for m in chosen]
    q = quality[idx, cols]
    c = cost[idx, cols]
    truth = [get_label(rows_by_id[case["base_id"]], DEFAULT_TOLERANCE) for case in cases]
    agree = float(np.mean([m == t for m, t in zip(chosen, truth)]))
    return q, c, agree


def main():
    from embed import embed_query

    rows, cases = load_inputs()
    rows_by_id = {r["id"]: r for r in rows}
    index = {r["id"]: i for i, r in enumerate(rows)}
    X = np.array([embed_query(r["query"]) for r in rows])
    quality, cost = outcome_matrices(rows)
    base = SERVED_MODELS.index(BASELINE_MODEL)

    clf = train_held_out(rows, X, {c["base_id"] for c in cases})
    predict = make_predict(clf, embed_query)

    cache = json.load(open(REWRITE_CACHE, encoding="utf-8")) if os.path.exists(REWRITE_CACHE) else {}
    timings = []
    rewrite = cached_rewriter(cache, timings)

    idx = [index[c["base_id"]] for c in cases]
    base_q, base_c = quality[idx, base], cost[idx, base]

    results, per_case = {}, {m: [] for m in MODES}
    for mode in MODES:
        decisions = [
            resolve_route(mode, c["followup"], build_history(rows_by_id[c["base_id"]]), predict, rewrite)
            for c in cases
        ]
        per_case[mode] = decisions
        chosen = [d.model for d in decisions]
        q, c_, agree = score_strategy(rows_by_id, cases, chosen, quality, cost, index)
        stats = bootstrap(q, c_, base_q, base_c)
        results[mode] = {
            "agreement_with_best_model": agree,
            "fallback_rate": float(np.mean([d.fallback_triggered for d in decisions])),
            **{k: list(v) for k, v in stats.items()},
        }
    json.dump(cache, open(REWRITE_CACHE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"\n{len(cases)} follow-up cases; classifier trained without their base queries.")
    print("Quality is a proxy: judge score of the chosen model on the base query.\n")
    header = f"{'Strategy':<10}{'Agree w/ best':<15}{'Fallback':<10}{'Quality (95% CI)':<22}{'Savings % vs 8b (95% CI)'}"
    print(header + "\n" + "-" * len(header))
    for mode in MODES:
        r = results[mode]
        q, s = r["quality"], r["savings_pct"]
        print(f"{mode:<10}{r['agreement_with_best_model']*100:>6.1f}%{'':<8}{r['fallback_rate']*100:>5.0f}%{'':<5}"
              f"{q[0]:.2f} [{q[1]:.2f},{q[2]:.2f}]   {s[0]:5.1f} [{s[1]:.1f},{s[2]:.1f}]")
    if timings:
        print(f"\nRewrite adds {np.mean(timings):.1f}s per follow-up on average ({len(timings)} fresh calls).")

    print("\nRewrites (check these by eye):")
    for case, d in zip(cases, per_case["rewrite"]):
        print(f"  [{case['base_id']:>3}] {case['followup']!r}\n        -> {d.routing_query!r}  => {d.model}")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": results,
                   "rewrites": [{"base_id": c["base_id"], "followup": c["followup"], "rewrite": d.routing_query}
                                for c, d in zip(cases, per_case["rewrite"])]},
                  f, indent=1, ensure_ascii=False)
    print(f"\nSaved {RESULTS_FILE}")


if __name__ == "__main__":
    main()
