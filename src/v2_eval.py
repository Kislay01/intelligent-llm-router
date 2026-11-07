"""v2 evaluation. Every number is out-of-sample; the dial default and C are chosen on experiment A only.

A  repeated stratified CV on the 464 objective tasks (judged queries added to training)   -> picks C and the dial default
B  leave-one-benchmark-out: train on 3 benchmarks (+judged), test on the 4th              -> can it generalise to a new kind of task?
C  train on the benchmarks only, test on the judged queries                                -> do benchmark labels transfer to general queries?
D  CV on the judged queries, with and without the benchmark tasks in training              -> does the extra data help the real target?
E  learning curve on A                                                                     -> would more data help?

Usage: python src/v2_eval.py [--folds 5] [--repeats 3] [--judged-weight 0.5]
"""
import argparse
import dataclasses
import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.append(os.path.dirname(__file__))

from pricing import SERVED_MODELS
from v2_data import build_pool
from v2_metrics import (LLAMA8, QWEN, cv_splits, log_loss, pick_default_lam, predict_over_splits, report)
from v2_router import KnnPredictor, PassPredictor, V1StyleClassifier, judge_score_label

RESULTS_FILE = "data/v2_results.json"
# Normalised prices are 0.33 / 0.39 / 0.33 / 1.0, so the dial only starts to prefer a 3b/mistral over qwen
# at values near 1; the grid therefore runs well past 0.4.
LAMS = [round(x, 2) for x in np.arange(0, 0.41, 0.04)] + [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
C_GRID = (1.0, 10.0, 100.0)
QUALITY_BUDGET = 0.01
FRACTIONS = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def predictors(C: float) -> Dict[str, callable]:
    return {"LR": lambda: PassPredictor(C), "kNN": lambda: KnnPredictor(15), "v1-style": lambda: V1StyleClassifier()}


def probs_for(pool, splits, repeats, eval_idx, C):
    return {name: predict_over_splits(pool, splits, repeats, eval_idx, make) for name, make in predictors(C).items()}


def choose_C(pool, obj, judged, folds):
    losses = {}
    splits, reps = cv_splits(pool, obj, judged, folds, 1)
    for C in C_GRID:
        P = predict_over_splits(pool, splits, reps, obj, lambda: PassPredictor(C))
        losses[C] = log_loss(P[0], pool.passed[obj])
    return min(losses, key=losses.get), losses


def experiment_a(pool, obj, judged, folds, repeats, C):
    splits, reps = cv_splits(pool, obj, judged, folds, repeats)
    P = probs_for(pool, splits, reps, obj, C)
    first = report(pool, obj, P, LAMS, None)
    lam = pick_default_lam(first["frontier"]["LR"], QUALITY_BUDGET)
    out = report(pool, obj, P, LAMS, lam)
    P_nj = predict_over_splits(pool, *cv_splits(pool, obj, np.array([], dtype=int), folds, repeats), obj,
                               lambda: PassPredictor(C))
    out["without_judged_in_training"] = report(pool, obj, {"LR": P_nj}, LAMS, lam)["frontier"]["LR"]
    return out, lam


def experiment_b(pool, obj, judged, C, lam):
    splits, results = [], {}
    for source in sorted(set(pool.source[obj])):
        test = obj[pool.source[obj] == source]
        train = np.concatenate([obj[pool.source[obj] != source], judged])
        splits.append((train, test, 0))
    P = probs_for(pool, splits, 1, obj, C)
    results["all"] = report(pool, obj, P, LAMS, lam)
    for source in sorted(set(pool.source[obj])):
        sel = np.where(pool.source[obj] == source)[0]
        results[source] = report(pool, obj[sel], {k: v[:, sel] for k, v in P.items()}, LAMS, lam)
    return results


def experiment_c(pool, obj, judged, C, lam):
    P = probs_for(pool, [(obj, judged, 0)], 1, judged, C)
    return {v: report(pool, judged, P, LAMS, lam, value=v) for v in ("passed", "judge")}


def experiment_d(pool, obj, judged, folds, repeats, C, lam):
    out = {}
    for label, extra in (("with_benchmarks", obj), ("judged_only", np.array([], dtype=int))):
        splits, reps = cv_splits(pool, judged, extra, folds, repeats)
        P = probs_for(pool, splits, reps, judged, C)
        out[label] = {v: report(pool, judged, P, LAMS, lam, value=v) for v in ("passed", "judge")}
    # The faithful v1 recipe (labels from 1-10 judge scores, trained on the judged queries only), same folds.
    view = dataclasses.replace(pool, passed=pool.judge)
    splits, reps = cv_splits(view, judged, np.array([], dtype=int), folds, repeats)
    P1 = predict_over_splits(view, splits, reps, judged, lambda: V1StyleClassifier(label_fn=judge_score_label))
    for v in ("passed", "judge"):
        row = report(pool, judged, {"v1-style": P1}, LAMS, lam, value=v)["strategies"]["v1-style classifier"]
        out["judged_only"][v]["strategies"]["v1 recipe (judge-score labels)"] = row
        out["with_benchmarks"][v]["strategies"]["v1 recipe (judge-score labels)"] = row
    return out


def experiment_e(pool, obj, judged, folds, C, lam):
    curve = []
    for frac in FRACTIONS:
        splits, reps = cv_splits(pool, obj, judged, folds, 2, frac=frac)
        P = predict_over_splits(pool, splits, reps, obj, lambda: PassPredictor(C))
        r = report(pool, obj, {"LR": P}, [lam], lam)["frontier"]["LR"][0]
        curve.append({"fraction": frac, "pass_rate": r["value"], "delta_vs_qwen": r["vs_qwen"]["delta"],
                      "savings_vs_8b": r["vs_8b"]["savings"]})
    return curve


def fmt(ci, scale=1.0, digits=1, signed=False):
    sign = "+" if signed else ""
    return f"{ci[0]*scale:{sign}.{digits}f} [{ci[1]*scale:{sign}.{digits}f},{ci[2]*scale:{sign}.{digits}f}]"


def print_table(title, res, value="pass rate %", scale=100.0, digits=1):
    print(f"\n{title}  (n={res['n']})")
    head = (f"{'Strategy':<34}{value:<12}{'vs always-qwen (pts) [95% CI]':<30}{'vs always-8b (pts) [95% CI]':<30}"
            f"{'cost saved vs always-8b % [CI]'}")
    print(head + "\n" + "-" * len(head))
    for name, s in res["strategies"].items():
        print(f"{name:<34}{s['value']*scale:<12.{digits}f}{fmt(s['vs_qwen']['delta'], scale, digits, True):<30}"
              f"{fmt(s['vs_8b']['delta'], scale, digits, True):<30}{fmt(s['vs_8b']['savings'])}")


def print_frontier(title, res, name, value="judge", scale=1.0, digits=2):
    print(f"\n{title}: dial -> {value}, cost saved vs always-8b, and which model it picks")
    print(f"  {'dial':<6}{value:<8}{'vs 8b (pts) [95% CI]':<26}{'saved %':<10}" + "".join(f"{m.split(':')[0][:9]:<11}" for m in SERVED_MODELS))
    for e in res["frontier"][name]:
        print(f"  {e['lam']:<6.2f}{e['value']*scale:<8.{digits}f}{fmt(e['vs_8b']['delta'], scale, digits, True):<26}"
              f"{e['vs_8b']['savings'][0]:<10.1f}" + "".join(f"{e['share'][m]*100:<11.0f}" for m in SERVED_MODELS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--judged-weight", type=float, default=0.5)
    args = parser.parse_args()

    pool, calib, dropped = build_pool(args.judged_weight)
    obj, judged = pool.objective, pool.judged
    print(f"{len(obj)} objective tasks + {len(judged)} judge-labelled queries ({dropped} dropped: incomplete scores)")
    print(f"Judge threshold calibrated on ground truth: score >= {calib.threshold} counts as pass "
          f"(balanced accuracy {calib.balanced_accuracy:.3f}, accuracy {calib.accuracy:.3f}, {calib.n_pairs} answers)")

    C, losses = choose_C(pool, obj, judged, args.folds)
    print(f"Regularisation C={C} (out-of-fold log-loss: { {k: round(v, 4) for k, v in losses.items()} })")

    a, lam = experiment_a(pool, obj, judged, args.folds, args.repeats, C)
    print(f"Dial default = {lam} (largest value losing <= {QUALITY_BUDGET*100:.0f} point pass rate vs dial 0; chosen on A only)")
    print_table("A. Cross-validation on benchmark tasks", a)
    b = experiment_b(pool, obj, judged, C, lam)
    for key in ("all", *[k for k in b if k != "all"]):
        print_table(f"B. Leave-one-benchmark-out: {key}", b[key])
    c = experiment_c(pool, obj, judged, C, lam)
    print_table("C. Train on benchmarks only, test on judged queries (pass = judge >= threshold)", c["passed"])
    print_table("C. Same, judge score (1-10) of the chosen model", c["judge"], "judge", 1.0, 2)
    d = experiment_d(pool, obj, judged, args.folds, args.repeats, C, lam)
    print_table("D. CV on judged queries, benchmarks in training", d["with_benchmarks"]["passed"])
    print_table("D. CV on judged queries, judged only (v1 data)", d["judged_only"]["passed"])
    print_table("D. judge score, benchmarks in training", d["with_benchmarks"]["judge"], "judge", 1.0, 2)
    print_table("D. judge score, judged only", d["judged_only"]["judge"], "judge", 1.0, 2)
    for name in ("LR", "kNN"):
        print_frontier(f"D. judged queries, benchmarks in training, {name}", d["with_benchmarks"]["judge"], name)
    print_frontier("A. benchmark tasks, LR", a, "LR", "pass %", 100.0, 1)
    e = experiment_e(pool, obj, judged, args.folds, C, lam)
    print("\nE. Learning curve (training fraction -> pass rate, delta vs always-qwen in points)")
    for row in e:
        print(f"  {row['fraction']:.0%}: {row['pass_rate']*100:.1f}%  delta {fmt(row['delta_vs_qwen'], 100, 1, True)}")

    os.makedirs("data", exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"C": C, "default_lam": lam, "calibration": {"threshold": calib.threshold, "table": calib.table},
                   "A": a, "B": b, "C_experiment": c, "D": d, "E": e}, f, indent=1)
    print(f"\nSaved {RESULTS_FILE}")


if __name__ == "__main__":
    main()
