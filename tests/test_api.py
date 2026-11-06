import os
import sys
from unittest.mock import patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient

import api
import db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


client = TestClient(api.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("api.query_model")
def test_route_endpoint(mock_query_model):
    mock_query_model.return_value = "mocked response text"

    response = client.post("/route", json={"query": "What is the capital of France?"})

    assert response.status_code == 200
    data = response.json()
    assert "routed_model" in data
    assert data["response"] == "mocked response text"
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["fallback_triggered"], bool)
    assert isinstance(data["session_id"], int)


@patch("api.query_model")
def test_chat_history_is_stored_and_can_be_reopened(mock_query_model):
    mock_query_model.side_effect = ["answer one", "answer two"]

    first = client.post("/route", json={"query": "first question"}).json()
    second = client.post("/route", json={"query": "second question", "session_id": first["session_id"]}).json()
    assert second["session_id"] == first["session_id"]

    listing = client.get("/sessions").json()
    assert len(listing) == 1
    assert listing[0]["title"] == "first question"
    assert listing[0]["message_count"] == 2

    detail = client.get(f"/sessions/{first['session_id']}").json()
    assert [m["response"] for m in detail["messages"]] == ["answer one", "answer two"]
    assert mock_query_model.call_count == 2  # reopening a chat never re-runs a model


@patch("api.query_model")
def test_new_chats_get_new_sessions_and_delete_works(mock_query_model):
    mock_query_model.return_value = "x"
    a = client.post("/route", json={"query": "a"}).json()["session_id"]
    b = client.post("/route", json={"query": "b"}).json()["session_id"]
    assert a != b

    assert client.delete(f"/sessions/{a}").status_code == 200
    assert client.get(f"/sessions/{a}").status_code == 404
    assert client.delete(f"/sessions/{a}").status_code == 404
    assert [s["id"] for s in client.get("/sessions").json()] == [b]


@patch("api.query_model")
def test_unknown_session_is_rejected_before_calling_a_model(mock_query_model):
    response = client.post("/route", json={"query": "hi", "session_id": 999})
    assert response.status_code == 404
    mock_query_model.assert_not_called()


@patch("api.query_model", side_effect=RuntimeError("ollama down"))
def test_failed_model_call_leaves_no_empty_session(_mock):
    with pytest.raises(RuntimeError):
        client.post("/route", json={"query": "hi"})
    assert client.get("/sessions").json() == []
