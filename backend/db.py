"""
SQLite-backed persistence for: uploaded document records, chat sessions,
and chat message history (so conversations survive a server restart).
"""
import os
import json
import sqlite3
import uuid
from datetime import datetime, timezone

DB_PATH = os.getenv("DB_PATH", "./study_assistant.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            added_at TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
            citations TEXT,
            created_at TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def create_document(name: str, type_: str) -> str:
    doc_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO documents (id, name, type, added_at) VALUES (?, ?, ?, ?)",
        (doc_id, name, type_, _now()),
    )
    conn.commit()
    conn.close()
    return doc_id


def list_documents_db():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM documents ORDER BY added_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_document_db(doc_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


def create_session() -> str:
    session_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute("INSERT INTO sessions (id, created_at) VALUES (?, ?)", (session_id, _now()))
    conn.commit()
    conn.close()
    return session_id


def add_message(session_id: str, role: str, content: str, citations=None) -> str:
    msg_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (id, session_id, role, content, citations, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (msg_id, session_id, role, content, json.dumps(citations or []), _now()),
    )
    conn.commit()
    conn.close()
    return msg_id


def get_history(session_id: str, limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["citations"] = json.loads(d["citations"] or "[]")
        out.append(d)
    return out
