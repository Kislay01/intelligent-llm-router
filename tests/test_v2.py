import json
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pricing import SERVED_MODELS
from v2_data import build_pool, cached_embed, calibrate_threshold, raw_query
from v2_metrics import cv_splits, paired_bootstrap, pick_default_lam, predict_over_splits, report
from v2_router import (NORM_PRICES, KnnPredictor, PassPredictor, V1StyleClassifier, cheapest_passing_label,
                        choose, judge_score_label)

N_MODELS = len(SERVED_MODELS)
HAS_DATA = os.path.exists("data/bench/tasks.json") and os.path.exists("data/labeled_results.jsonl")


def fake_embed(batch):
    rng_vectors = []
    for t in batch:
        v = np.zeros(32)
        for w in t.lower().split():
            v[hash(w) % 32] += 1  # hash() varies per process, fine: only used within one test run
        rng_vectors.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(rng_vectors)


# ---- data layer -------------------------------------------------------------------------------

def test_raw_query_strips_instruction_wrapper_and_tests():
    he = {"source": "humaneval", "question": "Complete this...\n\ndef f(): pass", "prompt": "def f(): pass\n"}
    mbpp = {"source": "mbpp", "question": "Write a Python solution...\n\nAdd two numbers.\n\nYour code must pass these tests:\nassert add(1,2)==3"}
    gsm = {"source": "gsm8k", "question": "Solve the math word problem...\n\nTom has 3 apples."}
    assert raw_query(he) == "def f(): pass"
    assert raw_query(mbpp) == "Add two numbers."
    assert raw_query(gsm) == "Tom has 3 apples."


@pytest.mark.skipif(not HAS_DATA, reason="benchmark data not present")
def test_raw_query_leaves_no_wrapper_text_in_any_real_task():
    with open("data/bench/tasks.json", encoding="utf-8") as f:
        tasks = json.load(f)
    for t in tasks:
        text = raw_query(t)
        assert text, t["id"]
        for marker in ("Respond with the complete", "Final answer:", "Your code must pass", "Solve the math word problem",
                       "Answer the multiple-choice question"):
            assert marker not in text, (t["id"], marker)


def test_calibrate_threshold_recovers_known_cutoff():
    outcomes, runs = [], []
    rng = np.random.default_rng(0)
    for i in range(60):
        scores = {m: int(rng.integers(1, 11)) for m in SERVED_MODELS}
        outcomes.append({"id": f"t{i}", "outcomes": {m: {"passed": s >= 7} for m, s in scores.items()}})
        runs.append({"task_id": f"t{i}", "scores": scores})
    c = calibrate_threshold(runs, outcomes)
    assert c.threshold == 7 and c.accuracy == 1.0


def test_calibrate_threshold_rejects_empty_overlap():
    with pytest.raises(ValueError):
        calibrate_threshold([{"task_id": "x", "scores": {"llama3.2:3b": 5}}], [])


def test_cached_embed_computes_each_distinct_text_once(tmp_path):
    calls = []

    def embed(batch):
        calls.append(list(batch))
        return fake_embed(batch)

    path = str(tmp_path / "e.npz")
    a = cached_embed(["x y", "z", "x y"], path, embed)
    assert a.shape == (3, 32) and np.allclose(a[0], a[2])
    assert sum(len(c) for c in calls) == 2
    b = cached_embed(["z", "x y", "new text"], path, embed)
    assert sum(len(c) for c in calls) == 3                      # only "new text" was embedded
    assert np.allclose(b[0], a[1]) and np.allclose(b[1], a[0])  # order follows the request, not the cache


@pytest.mark.skipif(not HAS_DATA, reason="benchmark data not present")
def test_build_pool_on_real_data(tmp_path):
    pool, calib, dropped = build_pool(0.5, embed_fn=fake_embed, cache=str(tmp_path / "e.npz"))
    assert len(pool.objective) == 464 and len(pool.judged) + dropped == 174
    assert pool.X.shape == (len(pool.ids), 32)
    assert set(np.unique(pool.passed)) <= {0.0, 1.0}
    assert (pool.weight[pool.objective] == 1).all() and (pool.weight[pool.judged] == 0.5).all()
    assert np.isnan(pool.judge[pool.objective]).all() and not np.isnan(pool.judge[pool.judged]).any()
    # judged "passed" must be exactly the calibrated threshold applied to the judge score
    assert ((pool.judge[pool.judged] >= calib.threshold) == pool.passed[pool.judged].astype(bool)).all()


# ---- router -----------------------------------------------------------------------------------

def test_choose_lam_zero_takes_most_likely_and_ties_go_to_cheaper():
    P = np.array([[0.2, 0.9, 0.5, 0.1], [0.5, 0.5, 0.5, 0.5]])
    assert choose(P, 0.0).tolist() == [1, int(np.argmin(NORM_PRICES))]


def test_choose_large_dial_prefers_cheap_models():
    P = np.array([[0.70, 0.80, 0.75, 0.70]])
    assert choose(P, 0.0)[0] == 1
    assert choose(P, 1.0)[0] in (0, 3)       # cheapest tier


def test_pass_predictor_learns_signal_and_handles_constant_and_nan_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    passed = np.zeros((300, N_MODELS))
    passed[:, 0] = (X[:, 0] > 0)
    passed[:, 1] = 1.0                                   # always passes -> constant column
    passed[:, 2] = (X[:, 1] > 0)
    passed[:150, 3] = np.nan                             # half unscored
    passed[150:, 3] = (X[150:, 2] > 0)
    P = PassPredictor(C=10).fit(X, passed).predict_proba(X)
    assert P.shape == (300, N_MODELS) and not np.isnan(P).any()
    assert np.mean((P[:, 0] > 0.5) == passed[:, 0]) > 0.9
    assert (P[:, 1] > 0.9).all() and (P[:, 1] < 1.0).all()   # smoothed, never exactly 1
    assert np.mean((P[150:, 3] > 0.5) == passed[150:, 3]) > 0.9   # trained only on the scored half


def test_knn_neighbours_and_probabilities():
    X = np.array([[1.0, 0], [0.99, 0.1], [0, 1.0], [0.1, 0.99]])
    passed = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 1.0, 0, 0]])
    knn = KnnPredictor(k=2).fit(X, passed)
    idx, sims = knn.neighbours(np.array([[1.0, 0.05]]))
    assert set(idx[0]) == {0, 1} and (sims > 0.99).all()
    P = knn.predict_proba(np.array([[1.0, 0.05]]))
    assert P[0, 0] > 0.6 and P[0, 1] < 0.3 and ((P >= 0) & (P <= 1)).all()


def test_cheapest_passing_label_and_v1_style():
    passed = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
    labels = cheapest_passing_label(passed)
    assert labels[0] == 0                    # ties on price resolved by order, cheapest overall
    assert labels[1] == SERVED_MODELS.index("qwen2.5-coder:7b")   # 0.07 < 0.18
    assert labels[2] == 0                    # nothing passes -> cheapest
    assert labels[3] == SERVED_MODELS.index("qwen2.5-coder:7b")
    X = np.random.default_rng(0).normal(size=(4, 3))
    assert V1StyleClassifier().fit(X, passed).predict_proba(X).shape == (4, N_MODELS)


# ---- metrics ----------------------------------------------------------------------------------

def _pool_like(n=60):
    from v2_data import Pool
    rng = np.random.default_rng(1)
    source = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    return Pool([str(i) for i in range(n)], ["t"] * n, source, source.copy(),
                (rng.random((n, N_MODELS)) > 0.4).astype(float), rng.random((n, N_MODELS)) * 1e-5,
                np.full((n, N_MODELS), np.nan), np.ones(n), rng.normal(size=(n, 6)))


def test_cv_splits_cover_each_eval_row_once_per_repeat_and_never_test_extras():
    pool = _pool_like()
    eval_idx, extra = np.arange(0, 40), np.arange(40, 60)
    splits, reps = cv_splits(pool, eval_idx, extra, folds=4, repeats=2)
    for r in range(reps):
        tested = np.concatenate([te for _, te, rr in splits if rr == r])
        assert sorted(tested) == list(eval_idx)
    for tr, te, _ in splits:
        assert not set(tr) & set(te)                       # no leakage
        assert set(extra) <= set(tr) and not set(extra) & set(te)
    small, _ = cv_splits(pool, eval_idx, extra, folds=4, repeats=1, frac=0.25)
    assert all(len(tr) < len(splits[0][0]) for tr, _, _ in small)


def test_predict_over_splits_trains_without_the_test_rows():
    pool = _pool_like()
    seen = []

    class Spy:
        def fit(self, X, passed, weight):
            self.n = len(X)
            seen.append(X)
            return self

        def predict_proba(self, X):
            return np.full((len(X), N_MODELS), 0.5)

    eval_idx = np.arange(40)
    splits, reps = cv_splits(pool, eval_idx, np.arange(40, 60), folds=4, repeats=1)
    P = predict_over_splits(pool, splits, reps, eval_idx, Spy)
    assert P.shape == (1, 40, N_MODELS) and (P == 0.5).all()
    for (tr, te, _), X in zip(splits, seen):
        assert len(X) == len(tr)


def test_paired_bootstrap_identical_and_clearly_better():
    v = np.random.default_rng(0).random(200)
    same = paired_bootstrap(v, v, v, v)
    assert same["delta"] == [0.0, 0.0, 0.0] and same["savings"][0] == 0.0
    better = paired_bootstrap(np.ones(200), np.ones(200), np.zeros(200) + 0.5, np.ones(200) * 2)
    assert better["delta"][1] > 0 and better["savings"][0] == 50.0


def test_pick_default_lam_largest_within_budget_and_requires_dial_zero_first():
    frontier = [{"lam": 0.0, "value": 0.80}, {"lam": 0.1, "value": 0.795}, {"lam": 0.2, "value": 0.70},
                {"lam": 0.3, "value": 0.805}]
    assert pick_default_lam(frontier, 0.01) == 0.3
    assert pick_default_lam(frontier[:3], 0.01) == 0.1
    with pytest.raises(AssertionError):
        pick_default_lam(frontier[1:], 0.01)


def test_report_structure_and_oracle_dominates():
    pool = _pool_like()
    rows = np.arange(len(pool.X))
    P = {"LR": np.random.default_rng(2).random((2, len(rows), N_MODELS)), "v1-style": np.eye(N_MODELS)[
        np.random.default_rng(3).integers(0, N_MODELS, (1, len(rows)))]}
    out = report(pool, rows, P, [0.0, 0.5], 0.5)
    assert out["n"] == len(rows)
    for key in ("Always llama3.1:8b", "Oracle (cheapest passing)", "v1-style classifier", "LR dial=0.50"):
        assert key in out["strategies"]
    assert out["strategies"]["Oracle (cheapest passing)"]["value"] >= max(
        s["value"] for k, s in out["strategies"].items() if k.startswith("Always"))
    assert [e["lam"] for e in out["frontier"]["LR"]] == [0.0, 0.5]
    assert abs(sum(out["frontier"]["LR"][0]["share"].values()) - 1) < 1e-9


def test_judge_score_label_matches_v1_cost_aware_label():
    from relabel import cost_aware_label
    scores = np.array([[10, 10, 10, 10], [6, 9, 9, 7], [5, 5, 8, 8], [4, 4, 4, 4]], dtype=float)
    expected = [SERVED_MODELS.index(cost_aware_label(dict(zip(SERVED_MODELS, r)), 0)) for r in scores]
    assert judge_score_label(scores).tolist() == expected
    assert judge_score_label(scores)[1] == SERVED_MODELS.index("qwen2.5-coder:7b")  # ties at 9: 0.07 < 0.18
    clf = V1StyleClassifier(label_fn=judge_score_label).fit(np.random.default_rng(0).normal(size=(4, 3)), scores)
    assert clf.choose(np.zeros((2, 3))).shape == (2,)
