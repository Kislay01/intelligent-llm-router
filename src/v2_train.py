"""Builds models/router_v2.pkl from the benchmark outcomes + judge-labelled queries (embeddings come from the cache).

Usage: python src/v2_train.py [--judged-weight 0.5] [--k 15]
"""
import argparse
import os
import pickle
import sys

sys.path.append(os.path.dirname(__file__))

from v2_data import build_pool
from v2_router import KnnPredictor
from v2_serve import ARTIFACT, V2Router

SNIPPET_CHARS = 240


def train(out: str = ARTIFACT, judged_weight: float = 0.5, k: int = 15) -> V2Router:
    pool, calibration, dropped = build_pool(judged_weight)
    knn = KnnPredictor(k).fit(pool.X, pool.passed, pool.weight)
    meta = {"n_objective": len(pool.objective), "n_judged": len(pool.judged), "judged_dropped": dropped,
            "judge_threshold": calibration.threshold, "judged_weight": judged_weight, "k": k}
    router = V2Router(knn, [t[:SNIPPET_CHARS] for t in pool.texts], [str(s) for s in pool.source], meta)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(router, f)
    return router


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged-weight", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()
    router = train(ARTIFACT, args.judged_weight, args.k)
    print(f"Saved {ARTIFACT}: {router.meta}")


if __name__ == "__main__":
    main()
