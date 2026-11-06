"""Pure evaluation helpers (no embedding model needed, so they are unit-testable)."""
from typing import Dict, List, Tuple

import numpy as np

from pricing import SERVED_MODELS, estimate_cost

BASELINE_MODEL = "llama3.1:8b"


def outcome_matrices(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """quality[i, j] and cost[i, j] for query i answered by SERVED_MODELS[j].

    NaN where that model was never scored on the query (e.g. qwen2.5-coder only has
    scores for the 24 supplementary coding queries).
    """
    quality = np.full((len(rows), len(SERVED_MODELS)), np.nan)
    cost = np.full_like(quality, np.nan)
    for i, row in enumerate(rows):
        for j, model in enumerate(SERVED_MODELS):
            if model in row["scores"] and model in row["responses"]:
                quality[i, j] = row["scores"][model]
                cost[i, j] = estimate_cost(row["query"], row["responses"][model], model)
    return quality, cost


def pick(matrix: np.ndarray, choice: np.ndarray, fallback_col: int | None = None) -> np.ndarray:
    """Per-query value of the chosen model column.

    If the chosen model has no score for a query and fallback_col is given, the fallback
    model's value is used instead (the router would have had to answer somehow).
    """
    values = matrix[np.arange(len(choice)), choice]
    if fallback_col is not None:
        values = np.where(np.isnan(values), matrix[:, fallback_col], values)
    return values


def bootstrap(
    quality: np.ndarray,
    cost: np.ndarray,
    base_quality: np.ndarray,
    base_cost: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Point estimate and 95% CI, resampling queries (paired with the baseline).

    savings_pct = 1 - total cost / baseline total cost; quality_delta_pct is relative to baseline.
    """
    rng = np.random.default_rng(seed)
    n = len(quality)

    def stats(idx):
        bq = base_quality[idx].mean()
        return (
            quality[idx].mean(),
            cost[idx].mean(),
            (quality[idx].mean() - bq) / bq * 100,
            (1 - cost[idx].sum() / base_cost[idx].sum()) * 100,
        )

    point = stats(np.arange(n))
    boots = np.array([stats(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    names = ["quality", "cost_per_query", "quality_delta_pct", "savings_pct"]
    return {k: (float(point[i]), float(lo[i]), float(hi[i])) for i, k in enumerate(names)}
