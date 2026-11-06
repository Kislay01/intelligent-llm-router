import os
import sqlite3
from contextlib import closing
from typing import List, Optional

# Overridable so Docker can keep the database on a mounted volume, separate from the baked-in dataset.
DB_PATH = os.getenv("ROUTER_DB_PATH", "data/router_logs.db")


def _connect() -> sqlite3.Connection:
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                routed_model TEXT NOT NULL,
                confidence REAL NOT NULL,
                fallback_triggered INTEGER NOT NULL,
                latency_seconds REAL NOT NULL
            )
        """)
        # Queries kept for improving the router. Deliberately independent of chat history:
        # deleting a chat does not delete from here, and no response text or session link is stored.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                routed_model TEXT NOT NULL,
                confidence REAL NOT NULL,
                fallback_triggered INTEGER NOT NULL
            )
        """)
        # Databases created before chat history existed lack these columns; add them in place.
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(requests)")}
        if "response" not in existing:
            conn.execute("ALTER TABLE requests ADD COLUMN response TEXT")
        if "session_id" not in existing:
            conn.execute("ALTER TABLE requests ADD COLUMN session_id INTEGER")
        conn.commit()


def create_session(title: str) -> int:
    with closing(_connect()) as conn:
        cur = conn.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
        conn.commit()
        return cur.lastrowid


def session_exists(session_id: int) -> bool:
    with closing(_connect()) as conn:
        return conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is not None


def log_request(query, routed_model, confidence, fallback_triggered, latency_seconds,
                response: str = "", session_id: Optional[int] = None) -> int:
    with closing(_connect()) as conn:
        cur = conn.execute(
            "INSERT INTO requests (query, routed_model, confidence, fallback_triggered, "
            "latency_seconds, response, session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query, routed_model, confidence, int(fallback_triggered), latency_seconds, response, session_id),
        )
        if session_id is not None:
            conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
        conn.commit()
        return cur.lastrowid


def log_learning(query, routed_model, confidence, fallback_triggered) -> bool:
    """Record a query for later relabelling. Set LEARNING_LOG=off to disable. Returns whether it wrote."""
    if os.getenv("LEARNING_LOG", "on").lower() == "off":
        return False
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO learning_log (query, routed_model, confidence, fallback_triggered) VALUES (?, ?, ?, ?)",
            (query, routed_model, confidence, int(fallback_triggered)),
        )
        conn.commit()
    return True


def count_learning_log() -> int:
    with closing(_connect()) as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM learning_log").fetchone()["c"]


def list_sessions(limit: int = 100) -> List[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute("""
            SELECT s.id, s.title, s.created_at, s.updated_at, COUNT(r.id) AS message_count
            FROM sessions s LEFT JOIN requests r ON r.session_id = s.id
            GROUP BY s.id ORDER BY s.updated_at DESC, s.id DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


def get_session(session_id: int) -> Optional[dict]:
    with closing(_connect()) as conn:
        session = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            return None
        messages = conn.execute("""
            SELECT id, timestamp, query, response, routed_model, confidence,
                   fallback_triggered, latency_seconds
            FROM requests WHERE session_id = ? ORDER BY id
        """, (session_id,)).fetchall()
        result = dict(session)
        result["messages"] = [{**dict(m), "fallback_triggered": bool(m["fallback_triggered"])} for m in messages]
        return result


def delete_session(session_id: int) -> bool:
    """Deletes the session and its messages (they also leave the usage stats).

    learning_log is intentionally left untouched.
    """
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM requests WHERE session_id = ?", (session_id,))
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cur.rowcount > 0


def get_stats():
    with closing(_connect()) as conn:
        total = conn.execute("SELECT COUNT(*) AS total FROM requests").fetchone()["total"]
        model_stats = [dict(row) for row in conn.execute("""
            SELECT routed_model, COUNT(*) AS count, AVG(latency_seconds) AS avg_latency
            FROM requests GROUP BY routed_model
        """)]
        fallback_count = conn.execute(
            "SELECT COUNT(*) AS c FROM requests WHERE fallback_triggered = 1"
        ).fetchone()["c"]
        recent = [dict(row) for row in conn.execute(
            "SELECT query, routed_model, confidence, timestamp FROM requests ORDER BY id DESC LIMIT 20"
        )]
    return {
        "total_requests": total,
        "fallback_count": fallback_count,
        "model_usage": model_stats,
        "recent_requests": recent,
    }
