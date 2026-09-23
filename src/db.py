import sqlite3

DB_PATH = "data/router_logs.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
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
    conn.commit()
    conn.close()


def log_request(query, routed_model, confidence, fallback_triggered, latency_seconds):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO requests (query, routed_model, confidence, fallback_triggered, latency_seconds) "
        "VALUES (?, ?, ?, ?, ?)",
        (query, routed_model, confidence, int(fallback_triggered), latency_seconds)
    )
    conn.commit()
    conn.close()


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as total FROM requests")
    total = cur.fetchone()["total"]

    cur.execute("""
        SELECT routed_model, COUNT(*) as count, AVG(latency_seconds) as avg_latency
        FROM requests GROUP BY routed_model
    """)
    model_stats = [dict(row) for row in cur.fetchall()]

    cur.execute("SELECT COUNT(*) as fallback_count FROM requests WHERE fallback_triggered = 1")
    fallback_count = cur.fetchone()["fallback_count"]

    cur.execute("SELECT query, routed_model, confidence, timestamp FROM requests ORDER BY id DESC LIMIT 20")
    recent = [dict(row) for row in cur.fetchall()]

    conn.close()
    return {
        "total_requests": total,
        "fallback_count": fallback_count,
        "model_usage": model_stats,
        "recent_requests": recent
    }