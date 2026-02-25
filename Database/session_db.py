"""
JesterClaw — Session Database
Team Lapanic / EmolOrbit

SQLite-backed action log and session history.
Stored in Server-side/Database/jesterclaw.db
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("jesterclaw.db")

DB_PATH = Path(__file__).parent / "jesterclaw.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Called at startup."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            client_ip   TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            ended_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS action_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            client_ip   TEXT NOT NULL,
            action      TEXT NOT NULL,
            risk_level  TEXT NOT NULL,
            result      TEXT,
            timestamp   TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", DB_PATH)


def log_action(
    session_id: str,
    client_ip: str,
    action: str,
    risk_level: str,
    result: str,
):
    """Append one action log row."""
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO action_logs
               (session_id, client_ip, action, risk_level, result, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, client_ip, action, risk_level, result,
             datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("DB log_action failed: %s", e)


def register_session(session_id: str, client_ip: str):
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, client_ip, started_at) VALUES (?, ?, ?)",
            (session_id, client_ip, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("DB register_session failed: %s", e)


def close_session(session_id: str):
    try:
        conn = _get_conn()
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), session_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("DB close_session failed: %s", e)


# Auto-init on import
init_db()
