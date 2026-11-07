"""Splits, paired bootstrap and summaries for the v2 evaluation. Pure numpy: unit-testable without the embedder."""
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.append(os.path.dirname(__file__))

from pricing import SERVED_MODELS
from v2_data import Pool
from v2_router import cheapest_passing_label, choose

QWEN = SERVED_MODELS.index("qwen2.5-coder:7b")
LLAMA8 = SERVED_MODELS.index("llama3.1:8b")
Split = Tuple[np.ndarray, np.ndarray, int]   # (train row indices, test row indices, repeat number)


def cv_splits(pool: Pool, eval_idx: np.ndarray, extra_train: np.ndarray, folds: int = 5, repeats: int = 3,
              seed: int = 42, frac: float = 1.0) -> Tuple[List[Split], int]:
    """Repeated stratified k-fold over eval_idx. extra_train rows are added to every training set but never tested.
    frac < 1 subsamples each training set (learning curve)."""
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    rng = np.random.default_rng(seed)
    splits = []
    for k, (tr, te) in enumerate(cv.split(eval_idx, pool.category[eval_idx])):
        train = np.concatenate([eval_idx[tr], extra_train]).astype(int)
        if frac < 1:
            train = rng.choice(train, max(4, int(len(train) * frac)), replace=False)
        splits.append((train, eval_idx[te], k // folds))
    return splits, repeats


def predict_over_splits(pool: Pool, splits: Sequence[Split], repeats: int, eval_idx: np.ndarray,
                        make: Callable) -> np.ndarray:
    """P[r, i, j]: predictor trained without eval row i, probability model j passes. Shape (repeats, len(eval_idx), models)."""
    pos = np.full(len(pool.X), -1)
    pos[eval_idx] = np.arange(len(eval_idx))
    P = np.zeros((repeats, len(eval_idx), len(SERVED_MODELS)))
    for train, test, r in splits:
        predictor = make().fit(pool.X[train], pool.passed[train], pool.weight[train])
        P[r, pos[test]] = predictor.predict_proba(pool.X[test])
    return P


def log_loss(P: np.ndarray, passed: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(P, eps, 1 - eps)
    y = np.broadcast_to(passed, p.shape)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def chosen(matrix: np.ndarray, choices: np.ndarray) -> np.ndarray:
    """Per-query value of the chosen model, averaged over repeats. matrix (m, models), choices (R, m)."""
    rows = np.arange(matrix.shape[0])
    return np.mean([matrix[rows, c] for c in choices], axis=0)


def paired_bootstrap(value, cost, ref_value, ref_cost, n_boot: int = 2000, seed: int = 42) -> dict:
    """Point and 95% CI for (value - ref_value) and for cost savings vs the reference (%), resampling queries."""
    rng = np.random.default_rng(seed)
    n = len(value)
    idx = rng.integers(0, n, (n_boot, n))
    delta = value[idx].mean(1) - ref_value[idx].mean(1)
    savings = (1 - cost[idx].sum(1) / ref_cost[idx].sum(1)) * 100
    point = (value.mean() - ref_value.mean(), (1 - cost.sum() / ref_cost.sum()) * 100)
    out = {}
    for name, p, dist in (("delta", point[0], delta), ("savings", point[1], savings)):
        lo, hi = np.percentile(dist, [2.5, 97.5])
        out[name] = [float(p), float(lo), float(hi)]
    return out


def summarize(value: np.ndarray, cost: np.ndarray, qwen: tuple, llama8: tuple) -> dict:
    return {"value": float(value.mean()), "cost": float(cost.mean()),
            "vs_qwen": paired_bootstrap(value, cost, *qwen),
            "vs_8b": paired_bootstrap(value, cost, *llama8)}


def report(pool: Pool, rows: np.ndarray, P: Dict[str, np.ndarray], lams: Sequence[float],
           default_lam: Optional[float], value: str = "passed") -> dict:
    """Summaries for one evaluation setting.

    rows: pool row indices evaluated; P[name]: (R, len(rows), models) probabilities per predictor
    (name 'v1-style' is one-hot and always uses lam=0). value: 'passed' or 'judge'.
    """
    V = (pool.passed if value == "passed" else pool.judge)[rows]
    C = pool.cost[rows]
    qwen, llama8 = (V[:, QWEN], C[:, QWEN]), (V[:, LLAMA8], C[:, LLAMA8])
    out = {"n": int(len(rows)), "strategies": {}, "frontier": {}}
    for j, model in enumerate(SERVED_MODELS):
        out["strategies"][f"Always {model}"] = summarize(V[:, j], C[:, j], qwen, llama8)
    oracle = np.tile(cheapest_passing_label(pool.passed[rows]), (1, 1))
    out["strategies"]["Oracle (cheapest passing)"] = summarize(chosen(V, oracle), chosen(C, oracle), qwen, llama8)
    for name, probs in P.items():
        if name == "v1-style":
            c = np.array([choose(p, 0.0) for p in probs])
            out["strategies"]["v1-style classifier"] = summarize(chosen(V, c), chosen(C, c), qwen, llama8)
            continue
        out["frontier"][name] = []
        for lam in lams:
            c = np.array([choose(p, lam) for p in probs])
            entry = summarize(chosen(V, c), chosen(C, c), qwen, llama8)
            entry["lam"] = float(lam)
            entry["share"] = {m: float(np.mean(c == j)) for j, m in enumerate(SERVED_MODELS)}
            out["frontier"][name].append(entry)
            if default_lam is not None and abs(lam - default_lam) < 1e-9:
                out["strategies"][f"{name} dial={lam:.2f}"] = entry
    return out


def pick_default_lam(frontier: List[dict], quality_budget: float = 0.01) -> float:
    """Largest dial value whose out-of-fold pass rate is within `quality_budget` of the dial=0 point."""
    assert frontier[0]["lam"] == 0, "frontier must start at dial 0"
    best = frontier[0]["value"]
    ok = [e["lam"] for e in frontier if e["value"] >= best - quality_budget]
    return max(ok)
