import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation import bootstrap, outcome_matrices, pick
from pricing import SERVED_MODELS
from run_evaluation import cross_validate, evaluate, load_rows


def _rows(n=40):
    cats = ["simple", "complex", "coding", "creative"]
    rows = []
    for i in range(n):
        scores = {m: 8 for m in SERVED_MODELS}
        scores["llama3.1:8b"] = 10
        rows.append({
            "id": i, "category": cats[i % 4], "query": f"q{i}",
            "responses": {m: "x" * 40 for m in SERVED_MODELS},
            "scores": scores, "winner": "llama3.1:8b",
        })
    return rows


def test_outcome_matrices_shape_and_values():
    q, c = outcome_matrices(_rows(4))
    assert q.shape == (4, len(SERVED_MODELS))
    assert q[0, SERVED_MODELS.index("llama3.1:8b")] == 10
    assert (c > 0).all()


def test_bootstrap_baseline_against_itself_is_zero():
    q = np.array([9.0, 8.0, 10.0, 7.0])
    c = np.array([1.0, 2.0, 1.0, 2.0])
    r = bootstrap(q, c, q, c, n_boot=100)
    assert r["savings_pct"][0] == 0
    assert r["quality_delta_pct"][0] == 0
    assert r["quality"][1] <= r["quality"][0] <= r["quality"][2]


def test_cross_validate_covers_every_query_each_repeat():
    rows = _rows()
    X = np.random.default_rng(0).normal(size=(len(rows), 8))
    choices, accs, f1s = cross_validate(rows, X, tolerance=0, folds=5, repeats=2)
    assert choices.shape == (2, len(rows))
    assert len(accs) == 10 and len(f1s) == 10


def test_evaluate_runs_and_oracle_beats_random_on_cost_aware_labels():
    rows = _rows()
    X = np.random.default_rng(0).normal(size=(len(rows), 8))
    res = evaluate(rows, X, tolerance=2, folds=5, repeats=1)
    assert res["Oracle (label upper bound)"]["savings_pct"][0] > 0
    assert "Always llama3.1:8b" in res


def test_real_data_baseline_models_fully_scored():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "labeled_results.jsonl")
    rows = load_rows(path)
    q, _ = outcome_matrices(rows)
    for m in ("llama3.1:8b", "llama3.2:3b", "mistral:7b"):
        assert not np.isnan(q[:, SERVED_MODELS.index(m)]).any()


def test_missing_scores_become_nan_and_fall_back():
    rows = _rows(8)
    del rows[0]["scores"]["qwen2.5-coder:7b"]
    q, _ = outcome_matrices(rows)
    j = SERVED_MODELS.index("qwen2.5-coder:7b")
    assert np.isnan(q[0, j])
    base = SERVED_MODELS.index("llama3.1:8b")
    choice = np.full(8, j)
    assert pick(q, choice, base)[0] == 10
