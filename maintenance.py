"""Local, durable alert history and maintenance tickets."""
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from inference import DATA

DB_PATH = DATA / "maintenance.sqlite3"
STATUSES = ("Chờ kiểm tra", "Đang kiểm tra", "Đã xử lý")


def connection():
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def initialize():
    with closing(connection()) as db, db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                reference TEXT NOT NULL,
                created_at TEXT NOT NULL,
                score REAL NOT NULL,
                threshold REAL NOT NULL,
                target INTEGER CHECK (target IN (0, 1)),
                outside_train INTEGER NOT NULL CHECK (outside_train IN (0, 1)),
                features TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY,
                alert_id TEXT NOT NULL UNIQUE REFERENCES alerts(id),
                status TEXT NOT NULL,
                notes TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0
            );
        """)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_alerts(events):
    """Insert each distinct source row once, including across reruns and sessions."""
    rows = []
    created_at = utc_now()
    for event in events:
        identity = json.dumps(event["identity"], separators=(",", ":"))
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        rows.append((key, event["origin"], event["reference"], created_at, event["score"],
                     event["threshold"], event["target"], int(event["outside_train"]),
                     json.dumps(event["features"], ensure_ascii=False, allow_nan=False)))
    with closing(connection()) as db, db:
        db.executemany("""
            INSERT INTO alerts (id, origin, reference, created_at, score, threshold, target,
                                outside_train, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO NOTHING
        """, rows)
        return db.total_changes


def list_alerts():
    with closing(connection()) as db:
        return [dict(row) for row in db.execute("""
            SELECT a.*, t.id AS ticket_id, t.status, t.notes, t.updated_at, t.revision
            FROM alerts AS a
            LEFT JOIN tickets AS t ON a.id = t.alert_id
            ORDER BY a.created_at DESC, a.rowid DESC
        """)]


def create_ticket(alert_id):
    with closing(connection()) as db, db:
        db.execute("""
            INSERT INTO tickets (alert_id, status, notes, updated_at) VALUES (?, ?, ?, ?)
            ON CONFLICT (alert_id) DO NOTHING
        """, (alert_id, STATUSES[0], "", utc_now()))


def update_ticket(ticket_id, revision, status, notes):
    if status not in STATUSES:
        raise ValueError("Trạng thái phiếu không hợp lệ.")
    with closing(connection()) as db, db:
        result = db.execute("""
            UPDATE tickets SET status = ?, notes = ?, updated_at = ?, revision = revision + 1
            WHERE id = ? AND revision = ?
        """, (status, notes.strip(), utc_now(), ticket_id, revision))
        return result.rowcount == 1
