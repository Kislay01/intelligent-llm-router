import json
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient

import api
import db
from pricing import SERVED_MODELS
from v2_router import KnnPredictor
from v2_serve import DEFAULT_COST_WEIGHT, DEFAULT_LAM, MAX_LAM, V2Router, lam_from_cost_weight, load_curve

QUERY_VEC = np.eye(384)[0]


def tiny_router() -> V2Router:
    """Ten identical training queries: llama3.1:8b passes 9/10, llama3.2:3b 8/10, the others 5/10."""
    X = np.tile(QUERY_VEC, (10, 1))
    passed = np.zeros((10, 4))
    passed[:9, 1] = 1
    passed[:8, 0] = 1
    passed[:5, 2] = 1
    passed[:5, 3] = 1
    return V2Router(KnnPredictor(k=10).fit(X, passed), [f"training query {i}" for i in range(10)],
                    ["judged"] * 5 + ["mbpp"] * 5)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


@pytest.fixture
def v2(monkeypatch):
    router = tiny_router()
    monkeypatch.setattr(api, "v2", router)
    monkeypatch.setattr(api, "embed_query", lambda text: QUERY_VEC)
    return router


client = TestClient(api.app)


@patch("api.query_model", return_value="answer")
def test_v2_route_returns_explanation_and_probabilities(_q, v2):
    data = client.post("/route", json={"query": "hello"}).json()
    assert data["router_version"] == "v2"
    assert set(data["all_probabilities"]) == set(SERVED_MODELS)
    assert len(data["explanation"]) == 3
    first = data["explanation"][0]
    assert set(first) == {"text", "source", "similarity", "passed"} and set(first["passed"]) <= set(SERVED_MODELS)
    assert data["cost_weight"] == pytest.approx(DEFAULT_COST_WEIGHT)


@patch("api.query_model", return_value="answer")
def test_cost_weight_moves_the_choice_from_quality_to_cheap(_q, v2):
    best = client.post("/route", json={"query": "hello", "cost_weight": 0}).json()
    cheap = client.post("/route", json={"query": "hello", "cost_weight": 1}).json()
    assert best["routed_model"] == "llama3.1:8b" and cheap["routed_model"] == "llama3.2:3b"
    assert cheap["cost_weight"] == 1


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_cost_weight_out_of_range_is_rejected(bad, v2):
    assert client.post("/route", json={"query": "hello", "cost_weight": bad}).status_code == 422


@patch("api.query_model", return_value="answer")
def test_v1_fallback_has_no_explanation(_q, monkeypatch):
    monkeypatch.setattr(api, "v2", None)
    data = client.post("/route", json={"query": "hello", "cost_weight": 0.5}).json()
    assert data["router_version"] == "v1" and data["explanation"] is None and data["cost_weight"] is None


def test_dial_curve_endpoint(v2, monkeypatch):
    monkeypatch.setattr(api, "load_curve", lambda: {"points": [{"cost_weight": 0.5}], "always_8b_judge": 8.99})
    data = client.get("/dial-curve").json()
    assert data["router_version"] == "v2" and data["points"] == [{"cost_weight": 0.5}] and data["always_8b_judge"] == 8.99
    monkeypatch.setattr(api, "v2", None)
    assert client.get("/dial-curve").json()["router_version"] == "v1"


def test_lam_mapping():
    assert lam_from_cost_weight(None) == DEFAULT_LAM
    assert lam_from_cost_weight(0) == 0 and lam_from_cost_weight(1) == pytest.approx(MAX_LAM)
    assert lam_from_cost_weight(5) == pytest.approx(MAX_LAM) and lam_from_cost_weight(-1) == 0
    assert lam_from_cost_weight(DEFAULT_COST_WEIGHT) == pytest.approx(DEFAULT_LAM)


def test_load_curve_reads_measured_frontier_and_tolerates_bad_files(tmp_path):
    def entry(lam):
        return {"lam": lam, "value": 8.8, "vs_8b": {"savings": [50.0, 45.0, 55.0]}, "share": {}}
    good = {"D": {"with_benchmarks": {"judge": {"frontier": {"kNN": [entry(0.0), entry(0.28), entry(1.0)]},
                                                  "strategies": {"Always llama3.1:8b": {"value": 8.99}}}}}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(good))
    curve = load_curve(str(path))
    assert [p["lam"] for p in curve["points"]] == [0.0, 0.28]       # the flat tail (dial > 0.3) is dropped
    assert curve["always_8b_judge"] == 8.99 and curve["points"][1]["cost_weight"] == pytest.approx(0.28 / MAX_LAM)
    path.write_text("not json")
    assert load_curve(str(path)) == {}
    assert load_curve(str(tmp_path / "missing.json")) == {}


def test_explain_reports_only_scored_models_and_orders_by_similarity():
    router = tiny_router()
    router.knn.passed[0, 3] = np.nan
    router.knn.X[0] = np.eye(384)[1]                                  # make row 0 the worst match
    out = router.explain(QUERY_VEC, k=10)
    assert out[-1]["text"] == "training query 0" and "mistral:7b" not in out[-1]["passed"]
    assert all(a["similarity"] >= b["similarity"] for a, b in zip(out, out[1:]))


def test_train_writes_loadable_artifact(tmp_path, monkeypatch):
    import v2_train
    from v2_data import Pool
    rng = np.random.default_rng(0)
    n = 30
    source = np.array(["mbpp"] * 20 + ["judged"] * 10)
    pool = Pool([str(i) for i in range(n)], [f"text {i} " * 100 for i in range(n)], source, source.copy(),
                (rng.random((n, 4)) > 0.5).astype(float), rng.random((n, 4)), np.full((n, 4), np.nan),
                np.ones(n), rng.normal(size=(n, 16)))
    from v2_data import Calibration
    monkeypatch.setattr(v2_train, "build_pool", lambda w: (pool, Calibration(8, 0.9, 0.9, 10, {}), 2))
    out = str(tmp_path / "m" / "router_v2.pkl")
    v2_train.train(out, 0.5, 5)
    router = v2_serve_load(out)
    assert router.meta["n_judged"] == 10 and router.meta["judged_dropped"] == 2 and router.meta["k"] == 5
    assert all(len(t) <= v2_train.SNIPPET_CHARS for t in router.snippets)
    model, p, probs = router.predict(rng.normal(size=16), 0.0)
    assert model in SERVED_MODELS and 0 <= p <= 1 and set(probs) == set(SERVED_MODELS)


def v2_serve_load(path):
    from v2_serve import load_router
    return load_router(path)


@patch("api.query_model", return_value="answer")
def test_why_this_model_survives_reopening_the_chat(_q, v2):
    sent = client.post("/route", json={"query": "hello"}).json()
    stored = client.get(f"/sessions/{sent['session_id']}").json()["messages"][0]
    assert stored["explanation"] == sent["explanation"]
    assert stored["all_probabilities"] == pytest.approx(sent["all_probabilities"])


@patch("api.query_model", return_value="answer")
def test_chats_without_a_why_have_none(_q, monkeypatch):
    monkeypatch.setattr(api, "v2", None)
    sent = client.post("/route", json={"query": "hello"}).json()
    stored = client.get(f"/sessions/{sent['session_id']}").json()["messages"][0]
    assert stored["explanation"] is None and stored["all_probabilities"] is None


def test_old_database_without_why_column_is_migrated(tmp_path, monkeypatch):
    import sqlite3
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP, "
                 "query TEXT NOT NULL, routed_model TEXT NOT NULL, confidence REAL NOT NULL, "
                 "fallback_triggered INTEGER NOT NULL, latency_seconds REAL NOT NULL)")
    conn.execute("INSERT INTO requests (query, routed_model, confidence, fallback_triggered, latency_seconds) "
                 "VALUES ('old', 'llama3.2:3b', 0.5, 0, 1.0)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    sid = db.create_session("t")
    db.log_request("new", "llama3.1:8b", 0.9, False, 1.0, "r", sid, "new", why={"probabilities": {"a": 1}, "neighbours": []})
    msg = db.get_session(sid)["messages"][0]
    assert msg["all_probabilities"] == {"a": 1} and msg["explanation"] == []
