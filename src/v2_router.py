"""v2 router: one pass-probability predictor per served model, plus a cost/quality dial.

Route = argmax_j  P(model j passes) - lam * normalised_price_j.   lam=0 is "best quality";
larger lam trades quality for cost. The same decision rule is used for every predictor,
so predictors can be compared fairly.
"""
import os
import sys
from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(__file__))

from pricing import MODEL_PRICING_PER_1M_TOKENS, SERVED_MODELS
from relabel import cost_aware_label

PRICES = np.array([MODEL_PRICING_PER_1M_TOKENS[m] for m in SERVED_MODELS])
NORM_PRICES = PRICES / PRICES.max()


def choose(proba: np.ndarray, lam: float) -> np.ndarray:
    """Index of the chosen model per row. Exact ties go to the cheaper model."""
    utility = proba - lam * NORM_PRICES - 1e-9 * NORM_PRICES
    return utility.argmax(axis=1)


def _constant_proba(passed: np.ndarray, weight: np.ndarray) -> float:
    """Laplace-smoothed weighted pass rate, used when a model has only one outcome class in training."""
    return float((np.sum(weight * passed) + 1) / (np.sum(weight) + 2))


class PassPredictor:
    """Logistic regression per model on query embeddings -> P(pass)."""

    def __init__(self, C: float = 10.0):
        self.C = C
        self.models = []

    def fit(self, X: np.ndarray, passed: np.ndarray, weight: Optional[np.ndarray] = None):
        weight = np.ones(len(X)) if weight is None else weight
        self.models = []
        for j in range(passed.shape[1]):
            ok = ~np.isnan(passed[:, j])
            y = passed[ok, j].astype(int)
            if len(np.unique(y)) < 2:
                self.models.append(_constant_proba(y, weight[ok]))
                continue
            clf = LogisticRegression(C=self.C, max_iter=2000)
            clf.fit(X[ok], y, sample_weight=weight[ok])
            self.models.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cols = [np.full(len(X), m) if isinstance(m, float) else m.predict_proba(X)[:, 1] for m in self.models]
        return np.column_stack(cols)


class KnnPredictor:
    """Baseline: P(pass) = weighted pass rate among the k most similar training queries (cosine)."""

    def __init__(self, k: int = 15):
        self.k = k

    def fit(self, X: np.ndarray, passed: np.ndarray, weight: Optional[np.ndarray] = None):
        self.X = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.passed = passed
        self.weight = np.ones(len(X)) if weight is None else weight
        return self

    def neighbours(self, X: np.ndarray, k: Optional[int] = None):
        """(indices, similarities) of the k nearest training rows for each query row."""
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        sims = Xn @ self.X.T
        k = min(k or self.k, sims.shape[1])
        idx = np.argsort(-sims, axis=1)[:, :k]
        return idx, np.take_along_axis(sims, idx, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        idx, _ = self.neighbours(X)
        w = self.weight[idx][:, :, None]                       # (n, k, 1)
        y = self.passed[idx]                                   # (n, k, models)
        seen = ~np.isnan(y)
        prior = np.nanmean(self.passed, axis=0)
        return (np.nansum(w * y, axis=1) + prior) / ((w * seen).sum(axis=1) + 1)


def cheapest_passing_label(passed: np.ndarray) -> np.ndarray:
    """v1-style class label per row: cheapest model that passes; if none passes, the cheapest model."""
    order = np.argsort(PRICES, kind="stable")
    labels = np.empty(len(passed), dtype=int)
    for i, row in enumerate(passed):
        winners = [j for j in order if row[j] == 1]
        labels[i] = winners[0] if winners else order[0]
    return labels


def judge_score_label(scores: np.ndarray) -> np.ndarray:
    """The actual v1 label: cheapest model whose judge score equals the best score (tolerance 0)."""
    return np.array([SERVED_MODELS.index(cost_aware_label(dict(zip(SERVED_MODELS, row)), 0)) for row in scores])


class V1StyleClassifier:
    """Multiclass 'which model is the cheapest good one?'. No dial.

    label_fn=cheapest_passing_label: the v1 idea applied to binary pass/fail outcomes.
    label_fn=judge_score_label: the faithful v1 recipe; pass judge scores in place of `passed`.
    """

    def __init__(self, C: float = 1.0, label_fn=cheapest_passing_label):
        self.C = C
        self.label_fn = label_fn

    def fit(self, X: np.ndarray, passed: np.ndarray, weight: Optional[np.ndarray] = None):
        y = self.label_fn(passed)
        self.only = int(y[0]) if len(np.unique(y)) < 2 else None
        if self.only is None:
            self.clf = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced")
            self.clf.fit(X, y, sample_weight=weight)
        return self

    def choose(self, X: np.ndarray) -> np.ndarray:
        if self.only is not None:
            return np.full(len(X), self.only)
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """One-hot of the chosen model, so this plugs into the same evaluation code (use lam=0)."""
        return np.eye(len(SERVED_MODELS))[self.choose(X)]
