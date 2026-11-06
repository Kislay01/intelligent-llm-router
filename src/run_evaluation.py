"""Repeated stratified k-fold evaluation of the router vs. fixed baselines.

Every strategy is scored on the same out-of-fold queries, so the numbers are comparable.
Usage: python src/run_evaluation.py [--tolerance 0] [--judge-labels] [--folds 5] [--repeats 3]
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.append(os.path.dirname(__file__))

from evaluation import BASELINE_MODEL, bootstrap, outcome_matrices, pick
from pricing import SERVED_MODELS
from relabel import DEFAULT_TOLERANCE, get_label
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

LABELED_FILE = "data/labeled_results.jsonl"
RESULTS_FILE = "data/evaluation_results.json"
CHART_FILE = "data/evaluation_chart.png"


def load_rows(path: str = LABELED_FILE) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cross_validate(rows, X, tolerance, folds=5, repeats=3, seed=42):
    """Out-of-fold router choices, shape (repeats, n), plus per-fold accuracy / macro-F1."""
    labels = np.array([SERVED_MODELS.index(get_label(r, tolerance)) for r in rows])
    # Stratify on category (not on the label) so folds are identical for any labelling scheme.
    strata = np.array([r["category"] for r in rows])
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)

    choices = np.zeros((repeats, len(rows)), dtype=int)
    accs, f1s = [], []
    for k, (tr, te) in enumerate(cv.split(X, strata)):
        if len(np.unique(labels[tr])) < 2:  # degenerate fold: nothing to learn, predict the only class
            pred = np.full(len(te), labels[tr][0])
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X[tr], labels[tr])
            pred = clf.predict(X[te])
        choices[k // folds, te] = pred
        accs.append(accuracy_score(labels[te], pred))
        f1s.append(f1_score(labels[te], pred, average="macro", zero_division=0))
    return choices, accs, f1s


def evaluate(rows, X, tolerance, folds=5, repeats=3) -> Dict[str, dict]:
    quality, cost = outcome_matrices(rows)
    n = len(rows)
    base = SERVED_MODELS.index(BASELINE_MODEL)
    base_q, base_c = quality[:, base], cost[:, base]

    choices, accs, f1s = cross_validate(rows, X, tolerance, folds, repeats)
    labels = np.array([SERVED_MODELS.index(get_label(r, tolerance)) for r in rows])

    # Router picks a model that was never scored on this query -> counted with the baseline model.
    unscored = float(np.mean([np.isnan(pick(quality, c)) for c in choices]))
    strategies = {
        "Router (cross-validated)": (
            np.mean([pick(quality, c, base) for c in choices], axis=0),
            np.mean([pick(cost, c, base) for c in choices], axis=0),
        ),
        "Oracle (label upper bound)": (pick(quality, labels), pick(cost, labels)),
        "Random (expected)": (np.nanmean(quality, axis=1), np.nanmean(cost, axis=1)),
    }
    for j, model in enumerate(SERVED_MODELS):
        if not np.isnan(quality[:, j]).any():  # skip models not scored on every query
            strategies[f"Always {model}"] = (quality[:, j], cost[:, j])

    results = {name: bootstrap(q, c, base_q, base_c) for name, (q, c) in strategies.items()}
    results["_classifier"] = {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "n_queries": n, "router_unscored_fraction": unscored,
        "models_not_fully_scored": [m for j, m in enumerate(SERVED_MODELS) if np.isnan(quality[:, j]).any()], "folds": folds, "repeats": repeats,
    }
    return results


def print_report(results: Dict[str, dict]) -> None:
    clf = results["_classifier"]
    print(f"\n{clf['n_queries']} queries, {clf['folds']}-fold x {clf['repeats']} repeats, "
          f"95% bootstrap CIs, baseline = Always {BASELINE_MODEL}\n")
    header = f"{'Strategy':<30}{'Quality':<20}{'Cost/query ($)':<26}{'Savings %':<22}{'Quality vs base %'}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        if name.startswith("_"):
            continue
        q, c, s, d = r["quality"], r["cost_per_query"], r["savings_pct"], r["quality_delta_pct"]
        print(f"{name:<30}{q[0]:.2f} [{q[1]:.2f},{q[2]:.2f}]  "
              f"{c[0]*1e6:.1f} [{c[1]*1e6:.1f},{c[2]*1e6:.1f}]e-6  "
              f"{s[0]:6.1f} [{s[1]:.1f},{s[2]:.1f}]  "
              f"{d[0]:+6.1f} [{d[1]:+.1f},{d[2]:+.1f}]")
    print(f"\nRouter chose a never-scored model on {clf['router_unscored_fraction']*100:.1f}% of "
          f"out-of-fold queries (counted as {BASELINE_MODEL}); not fully scored: "
          f"{clf['models_not_fully_scored']}")
    print(f"Classifier per-fold accuracy {clf['accuracy_mean']:.3f} +/- {clf['accuracy_std']:.3f}, "
          f"macro-F1 {clf['macro_f1_mean']:.3f} +/- {clf['macro_f1_std']:.3f}")


def save_chart(results: Dict[str, dict], path: str = CHART_FILE) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        if name.startswith("_"):
            continue
        x, y = r["cost_per_query"], r["quality"]
        ax.errorbar(x[0] * 1e6, y[0], xerr=[[ (x[0] - x[1]) * 1e6 ], [ (x[2] - x[0]) * 1e6 ]],
                    yerr=[[y[0] - y[1]], [y[2] - y[0]]], fmt="o", capsize=3, label=name)
    ax.set_xlabel("Cost per query (USD x 1e-6)")
    ax.set_ylabel("Mean judge score (1-10)")
    ax.set_title("Cost vs quality, out-of-fold, 95% CI")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--judge-labels", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    tolerance = None if args.judge_labels else args.tolerance

    from embed import embed_query  # imported late so the helpers above work without the model

    rows = load_rows()
    X = np.array([embed_query(r["query"]) for r in rows])
    results = evaluate(rows, X, tolerance, args.folds, args.repeats)
    print_report(results)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"tolerance": tolerance, **results}, f, indent=2)
    save_chart(results)
    print(f"\nSaved {RESULTS_FILE} and {CHART_FILE}")


if __name__ == "__main__":
    main()
