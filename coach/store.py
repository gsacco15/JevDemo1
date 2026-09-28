"""SQLite persistence with privacy defaults.

- Sessions (which contain conversation text) auto-expire after RETENTION_HOURS.
- Screenshots are never written to disk.
- Feedback keeps only the suggestion text + its judgment features, not the conversation.
- Everything for a user can be deleted with one call.
"""

import json
import os
import sqlite3
import threading
import time

from .config import settings

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        if settings.db_path != ":memory:":
            os.makedirs(os.path.dirname(settings.db_path) or ".", exist_ok=True)
        _conn = sqlite3.connect(settings.db_path, check_same_thread=False)
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, style TEXT, prefs TEXT, updated REAL);
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, user_id TEXT, created REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, session_id TEXT, candidate_id TEXT,
                kind TEXT, text TEXT, features TEXT, created REAL);
            """
        )
    return _conn


def purge_expired() -> None:
    cutoff = time.time() - settings.retention_hours * 3600
    with _lock:
        conn().execute("DELETE FROM sessions WHERE created < ?", (cutoff,))
        conn().commit()


def get_user(user_id: str) -> tuple[dict | None, dict]:
    with _lock:
        row = conn().execute("SELECT style, prefs FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        return None, {}
    return (json.loads(row[0]) if row[0] else None), (json.loads(row[1]) if row[1] else {})


def save_user(user_id: str, style: dict | None = None, prefs: dict | None = None) -> None:
    cur_style, cur_prefs = get_user(user_id)
    style = style if style is not None else cur_style
    prefs = prefs if prefs is not None else cur_prefs
    with _lock:
        conn().execute(
            "INSERT INTO users (id, style, prefs, updated) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET style = excluded.style, prefs = excluded.prefs, updated = excluded.updated",
            (user_id, json.dumps(style) if style else None, json.dumps(prefs), time.time()),
        )
        conn().commit()


def save_session(session_id: str, user_id: str, payload: dict) -> None:
    with _lock:
        conn().execute(
            "INSERT OR REPLACE INTO sessions (id, user_id, created, payload) VALUES (?, ?, ?, ?)",
            (session_id, user_id, time.time(), json.dumps(payload)),
        )
        conn().commit()


def get_session(session_id: str, user_id: str) -> dict | None:
    with _lock:
        row = conn().execute("SELECT payload FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)).fetchone()
    return json.loads(row[0]) if row else None


def delete_session(session_id: str, user_id: str) -> None:
    with _lock:
        conn().execute("DELETE FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
        conn().commit()


def add_feedback(user_id: str, session_id: str, candidate_id: str, kind: str, text: str, features: dict) -> None:
    with _lock:
        conn().execute(
            "INSERT INTO feedback (user_id, session_id, candidate_id, kind, text, features, created) VALUES (?,?,?,?,?,?,?)",
            (user_id, session_id, candidate_id, kind, text, json.dumps(features), time.time()),
        )
        conn().commit()


def feedback_counts(user_id: str) -> dict:
    with _lock:
        rows = conn().execute("SELECT kind, COUNT(*) FROM feedback WHERE user_id = ? GROUP BY kind", (user_id,)).fetchall()
    return dict(rows)


def delete_user(user_id: str) -> None:
    with _lock:
        for t, col in (("sessions", "user_id"), ("feedback", "user_id"), ("users", "id")):
            conn().execute(f"DELETE FROM {t} WHERE {col} = ?", (user_id,))
        conn().commit()
