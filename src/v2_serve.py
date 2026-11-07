"""Serving side of the v2 router: nearest-neighbour pass-rate predictor, cost/quality dial, explanations.

Why nearest neighbours: in src/v2_eval.py it was at least as good as logistic regression everywhere
(at dial 1.0: general queries 8.63 vs 8.64 judge; held-out benchmarks -1.7 vs -3.7 points vs always-qwen), and its
decision can be explained by the training queries that produced it.
"""
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(__file__))

from pricing import SERVED_MODELS
from v2_router import KnnPredictor, choose

ARTIFACT = "models/router_v2.pkl"
RESULTS_FILE = "data/v2_results.json"
# Judgement call, not a validated optimum: dial 0.08 gives about v1's cost on the judged queries
# (roughly half the cost of always-llama3.1:8b) with a judge score within noise of it.
DEFAULT_LAM = 0.08
# Past ~0.3 the measured curve is flat (llama3.1:8b is never chosen), so the slider stops there.
MAX_LAM = 0.30
DEFAULT_COST_WEIGHT = DEFAULT_LAM / MAX_LAM


def lam_from_cost_weight(cost_weight: Optional[float]) -> float:
    """cost_weight 0 = best quality, 1 = cheapest; linear on the dial range 0..MAX_LAM. None = default."""
    if cost_weight is None:
        return DEFAULT_LAM
    return float(min(max(cost_weight, 0.0), 1.0)) * MAX_LAM


class V2Router:
    def __init__(self, knn: KnnPredictor, snippets: List[str], sources: List[str], meta: Optional[dict] = None):
        self.knn, self.snippets, self.sources, self.meta = knn, snippets, sources, meta or {}

    def predict(self, vec: np.ndarray, lam: float) -> Tuple[str, float, Dict[str, float]]:
        """(chosen model, its predicted pass probability, predicted pass probability of every model)."""
        proba = self.knn.predict_proba(np.asarray(vec, dtype=float)[None])
        j = int(choose(proba, lam)[0])
        return SERVED_MODELS[j], float(proba[0, j]), {m: float(p) for m, p in zip(SERVED_MODELS, proba[0])}

    def explain(self, vec: np.ndarray, k: int = 3) -> List[dict]:
        """The k most similar training queries and which models passed them."""
        idx, sims = self.knn.neighbours(np.asarray(vec, dtype=float)[None], k)
        out = []
        for i, s in zip(idx[0], sims[0]):
            passed = self.knn.passed[i]
            out.append({
                "text": self.snippets[i], "source": self.sources[i], "similarity": float(s),
                "passed": {m: bool(p == 1) for m, p in zip(SERVED_MODELS, passed) if not np.isnan(p)},
            })
        return out


def load_router(path: str = ARTIFACT) -> Optional[V2Router]:
    if not os.path.exists(path):
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def load_curve(path: str = RESULTS_FILE) -> dict:
    """The measured dial -> (judge score, cost saved vs always-8b, model shares) curve shown next to the slider.
    Empty dict if the results file is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            judged = json.load(f)["D"]["with_benchmarks"]["judge"]
        frontier = judged["frontier"]["kNN"]
        always_8b = judged["strategies"]["Always llama3.1:8b"]["value"]
    except (OSError, KeyError, ValueError):
        return {}
    points = [{"lam": e["lam"], "cost_weight": e["lam"] / MAX_LAM, "judge": e["value"],
               "saved_vs_8b_pct": e["vs_8b"]["savings"][0], "share": e["share"]}
              for e in frontier if e["lam"] <= MAX_LAM + 1e-9]
    return {"points": points, "always_8b_judge": always_8b}
