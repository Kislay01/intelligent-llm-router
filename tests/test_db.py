import os
import sqlite3
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


def _log(session_id, query="q", response="r"):
    return db.log_request(query, "llama3.2:3b", 0.5, False, 1.0, response=response, session_id=session_id)


def test_session_roundtrip_keeps_response_text():
    sid = db.create_session("hello")
    _log(sid, "What is 2+2?", "4")
    session = db.get_session(sid)
    assert session["title"] == "hello"
    assert session["messages"][0]["response"] == "4"
    assert session["messages"][0]["fallback_triggered"] is False


def test_messages_come_back_in_order_and_list_counts_them():
    sid = db.create_session("t")
    _log(sid, "first")
    _log(sid, "second")
    assert [m["query"] for m in db.get_session(sid)["messages"]] == ["first", "second"]
    assert db.list_sessions()[0]["message_count"] == 2


def test_most_recently_used_session_is_listed_first():
    a, b = db.create_session("a"), db.create_session("b")
    with db._connect() as conn:  # make timestamps distinct without sleeping
        conn.execute("UPDATE sessions SET updated_at = '2020-01-01 00:00:00' WHERE id = ?", (b,))
    _log(a)
    assert [s["id"] for s in db.list_sessions()] == [a, b]


def test_delete_session_removes_messages_and_unknown_returns_false():
    sid = db.create_session("t")
    _log(sid)
    assert db.delete_session(sid) is True
    assert db.get_session(sid) is None
    assert db.get_stats()["total_requests"] == 0
    assert db.delete_session(sid) is False


def test_init_db_migrates_old_schema_without_losing_rows(tmp_path, monkeypatch):
    old = str(tmp_path / "old.db")
    conn = sqlite3.connect(old)
    conn.execute("""CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP, query TEXT NOT NULL, routed_model TEXT NOT NULL,
        confidence REAL NOT NULL, fallback_triggered INTEGER NOT NULL, latency_seconds REAL NOT NULL)""")
    conn.execute("INSERT INTO requests (query, routed_model, confidence, fallback_triggered, latency_seconds) "
                 "VALUES ('old', 'm', 0.9, 0, 1.0)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", old)
    db.init_db()
    db.init_db()  # idempotent
    assert db.get_stats()["total_requests"] == 1
    sid = db.create_session("new")
    _log(sid)
    assert db.get_session(sid)["messages"][0]["query"] == "q"


def test_deleting_a_chat_keeps_the_learning_log_without_responses():
    sid = db.create_session("t")
    _log(sid, "secret question", "secret answer")
    assert db.log_learning("secret question", "llama3.2:3b", 0.5, False) is True
    db.delete_session(sid)
    assert db.count_learning_log() == 1
    with db._connect() as conn:
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(learning_log)")}
    assert "response" not in columns and "session_id" not in columns


def test_learning_log_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("LEARNING_LOG", "off")
    assert db.log_learning("q", "m", 0.5, False) is False
    assert db.count_learning_log() == 0
