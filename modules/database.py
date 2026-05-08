"""
OpenVisionGuard — Dual-backend database module
===============================================
Primary  : PostgreSQL  (production, when POSTGRES_HOST is reachable)
Fallback : SQLite      (zero-config, data/openvisionguard.db)

All public functions are identical in both modes.
Callers never need to know which backend is active.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

# ── Try to import psycopg2; if missing, PG is simply unavailable ──────────────
try:
    import psycopg2
    import psycopg2.pool
    from psycopg2.extras import RealDictCursor
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

# ── State ─────────────────────────────────────────────────────────────────────
db_pool: Any = None          # psycopg2 pool  (PG mode)
_sqlite_path: str = ""       # file path      (SQLite mode)
_sqlite_lock = threading.Lock()
_backend: str = "none"       # "postgres" | "sqlite" | "none"


# ─────────────────────────────────────────────────────────────────────────────
#  Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    global db_pool, _sqlite_path, _backend

    # 1. Try PostgreSQL ────────────────────────────────────────────────────────
    if _PG_AVAILABLE:
        try:
            db_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20,
                user=os.getenv("POSTGRES_USER", "ovg_user"),
                password=os.getenv("POSTGRES_PASSWORD", "ovg_password"),
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DB", "openvisionguard"),
                connect_timeout=3,
            )
            _backend = "postgres"
            print("[DB] Connected to PostgreSQL ✓")
            _pg_create_tables()
            return
        except Exception as e:
            print(f"[DB] PostgreSQL unavailable ({e}) → falling back to SQLite")
            db_pool = None

    # 2. SQLite fallback ───────────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    _sqlite_path = os.path.join(data_dir, "openvisionguard.db")
    _backend = "sqlite"
    print(f"[DB] Using SQLite at {_sqlite_path} ✓")
    _sqlite_create_tables()


# ─────────────────────────────────────────────────────────────────────────────
#  Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
  global_id VARCHAR PRIMARY KEY,
  face_name VARCHAR,
  risk_level VARCHAR DEFAULT 'low',
  first_seen TIMESTAMP,
  last_seen TIMESTAMP,
  total_appearances INTEGER DEFAULT 0,
  metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id SERIAL PRIMARY KEY,
  global_id VARCHAR,
  camera_id VARCHAR,
  activity VARCHAR,
  location VARCHAR,
  timestamp TIMESTAMP,
  frame_path VARCHAR
);
CREATE TABLE IF NOT EXISTS alerts (
  alert_id VARCHAR PRIMARY KEY,
  severity VARCHAR,
  type VARCHAR,
  message TEXT,
  global_id VARCHAR,
  camera_id VARCHAR,
  timestamp TIMESTAMP,
  thumbnail_path VARCHAR,
  acknowledged BOOLEAN DEFAULT FALSE,
  metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS detections (
  id SERIAL PRIMARY KEY,
  object_id VARCHAR,
  material VARCHAR,
  confidence FLOAT,
  size VARCHAR,
  timestamp TIMESTAMP,
  latitude FLOAT,
  longitude FLOAT
);
CREATE TABLE IF NOT EXISTS person_logs (
  id SERIAL PRIMARY KEY,
  person_id VARCHAR,
  timestamp TIMESTAMP,
  position_x FLOAT,
  position_y FLOAT,
  speed FLOAT,
  zone VARCHAR,
  event_type VARCHAR
);
CREATE TABLE IF NOT EXISTS presence_logs (
  id SERIAL PRIMARY KEY,
  person_id VARCHAR,
  event_type VARCHAR,
  timestamp TIMESTAMP,
  session_duration FLOAT,
  camera_id VARCHAR DEFAULT 'CAM_01'
);
CREATE TABLE IF NOT EXISTS face_logs (
  id SERIAL PRIMARY KEY,
  person_id VARCHAR,
  face_name VARCHAR,
  camera_id VARCHAR,
  timestamp TIMESTAMP,
  confidence FLOAT,
  crop_path VARCHAR,
  match_status VARCHAR DEFAULT 'unknown'
);
"""

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
  global_id TEXT PRIMARY KEY,
  face_name TEXT,
  risk_level TEXT DEFAULT 'low',
  first_seen TEXT,
  last_seen TEXT,
  total_appearances INTEGER DEFAULT 0,
  metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  global_id TEXT,
  camera_id TEXT,
  activity TEXT,
  location TEXT,
  timestamp TEXT,
  frame_path TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
  alert_id TEXT PRIMARY KEY,
  severity TEXT,
  type TEXT,
  message TEXT,
  global_id TEXT,
  camera_id TEXT,
  timestamp TEXT,
  thumbnail_path TEXT,
  acknowledged INTEGER DEFAULT 0,
  metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS detections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  object_id TEXT,
  material TEXT,
  confidence REAL,
  size TEXT,
  timestamp TEXT,
  latitude REAL,
  longitude REAL
);
CREATE TABLE IF NOT EXISTS person_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id TEXT,
  timestamp TEXT,
  position_x REAL,
  position_y REAL,
  speed REAL,
  zone TEXT,
  event_type TEXT
);
CREATE TABLE IF NOT EXISTS presence_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id TEXT,
  event_type TEXT,
  timestamp TEXT,
  session_duration REAL,
  camera_id TEXT DEFAULT 'CAM_01'
);
CREATE TABLE IF NOT EXISTS face_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id TEXT,
  face_name TEXT,
  camera_id TEXT,
  timestamp TEXT,
  confidence REAL,
  crop_path TEXT,
  match_status TEXT DEFAULT 'unknown'
);
"""


def _pg_create_tables() -> None:
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            for stmt in _PG_SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
            # Migrations
            for stmt in [
                "ALTER TABLE alerts ADD COLUMN IF NOT EXISTS metadata_json TEXT;",
                "ALTER TABLE identities ADD COLUMN IF NOT EXISTS metadata_json TEXT;",
            ]:
                try:
                    cur.execute(stmt)
                except Exception:
                    pass
            conn.commit()
    finally:
        db_pool.putconn(conn)


def _sqlite_create_tables() -> None:
    with _sqlite_lock:
        con = sqlite3.connect(_sqlite_path)
        try:
            con.executescript(_SQLITE_SCHEMA)
            con.commit()
        finally:
            con.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Internal query helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def _sqlite_exec(sql: str, params: tuple = ()) -> None:
    """Execute a write statement against SQLite (INSERT/UPDATE/DELETE)."""
    if _backend != "sqlite":
        return
    with _sqlite_lock:
        con = sqlite3.connect(_sqlite_path)
        try:
            con.execute(sql, params)
            con.commit()
        except sqlite3.IntegrityError:
            pass   # duplicate PKs — ignore silently
        finally:
            con.close()


def _sqlite_query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute a SELECT and return list-of-dicts."""
    if _backend != "sqlite":
        return []
    with _sqlite_lock:
        con = sqlite3.connect(_sqlite_path)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — save_*
# ─────────────────────────────────────────────────────────────────────────────

def save_identity(identity: Dict[str, Any]) -> None:
    if _backend == "none":
        return
    now = _now()
    meta = json.dumps(identity.get("metadata", {}))
    gid  = identity["global_id"]

    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT global_id FROM identities WHERE global_id = %s", (gid,))
                if cur.fetchone():
                    cur.execute(
                        "UPDATE identities SET face_name=%s, risk_level=%s, last_seen=%s, "
                        "total_appearances=total_appearances+1, metadata_json=%s WHERE global_id=%s",
                        (identity.get("face_name"), identity.get("risk_level", "low"), now, meta, gid),
                    )
                else:
                    cur.execute(
                        "INSERT INTO identities (global_id,face_name,risk_level,first_seen,last_seen,total_appearances,metadata_json) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (gid, identity.get("face_name"), identity.get("risk_level","low"), now, now, 1, meta),
                    )
                conn.commit()
        finally:
            db_pool.putconn(conn)
        return

    # SQLite — use UPSERT
    _sqlite_exec(
        "INSERT INTO identities (global_id,face_name,risk_level,first_seen,last_seen,total_appearances,metadata_json) "
        "VALUES (?,?,?,?,?,1,?) "
        "ON CONFLICT(global_id) DO UPDATE SET "
        "  face_name=excluded.face_name, risk_level=excluded.risk_level, "
        "  last_seen=excluded.last_seen, total_appearances=total_appearances+1, "
        "  metadata_json=excluded.metadata_json",
        (gid, identity.get("face_name"), identity.get("risk_level","low"), now, now, meta),
    )


def save_event(event: Dict[str, Any]) -> None:
    if _backend == "none":
        return
    ts = event.get("timestamp") or _now()
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO events (global_id,camera_id,activity,location,timestamp,frame_path) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (event.get("global_id"), event.get("camera_id"), event.get("activity"),
                     event.get("location"), ts, event.get("frame_path")),
                )
                conn.commit()
        finally:
            db_pool.putconn(conn)
    else:
        _sqlite_exec(
            "INSERT INTO events (global_id,camera_id,activity,location,timestamp,frame_path) VALUES (?,?,?,?,?,?)",
            (event.get("global_id"), event.get("camera_id"), event.get("activity"),
             event.get("location"), ts, event.get("frame_path")),
        )


def save_alert(alert: Dict[str, Any]) -> None:
    if _backend == "none":
        return
    standard = {"alert_id","severity","type","message","global_id","camera_id","timestamp","thumbnail_path","acknowledged"}
    meta = json.dumps({k: v for k, v in alert.items() if k not in standard})
    ts   = alert.get("timestamp") or _now()

    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO alerts (alert_id,severity,type,message,global_id,camera_id,timestamp,thumbnail_path,acknowledged,metadata_json) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (alert["alert_id"], alert["severity"], alert["type"], alert["message"],
                     alert["global_id"], alert["camera_id"], ts,
                     alert.get("thumbnail_path"), alert.get("acknowledged", False), meta),
                )
                conn.commit()
        finally:
            db_pool.putconn(conn)
    else:
        _sqlite_exec(
            "INSERT OR IGNORE INTO alerts "
            "(alert_id,severity,type,message,global_id,camera_id,timestamp,thumbnail_path,acknowledged,metadata_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (alert["alert_id"], alert["severity"], alert["type"], alert["message"],
             alert.get("global_id"), alert.get("camera_id"), ts,
             alert.get("thumbnail_path"), 1 if alert.get("acknowledged") else 0, meta),
        )


def save_detection(det: Dict[str, Any]) -> None:
    if _backend == "none":
        return
    ts = det.get("timestamp") or _now()
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO detections (object_id,material,confidence,size,timestamp,latitude,longitude) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (str(det.get("object_id")), det.get("material"), det.get("confidence"),
                     det.get("size"), ts, det.get("latitude"), det.get("longitude")),
                )
                conn.commit()
        finally:
            db_pool.putconn(conn)
    else:
        _sqlite_exec(
            "INSERT INTO detections (object_id,material,confidence,size,timestamp,latitude,longitude) VALUES (?,?,?,?,?,?,?)",
            (str(det.get("object_id")), det.get("material"), det.get("confidence"),
             det.get("size"), ts, det.get("latitude"), det.get("longitude")),
        )


def save_person_log(log: Dict[str, Any]) -> None:
    if _backend == "none":
        return
    ts = log.get("timestamp") or _now()
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO person_logs (person_id,timestamp,position_x,position_y,speed,zone,event_type) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (str(log.get("person_id")), ts, log.get("position_x"), log.get("position_y"),
                     log.get("speed"), log.get("zone"), log.get("event_type")),
                )
                conn.commit()
        finally:
            db_pool.putconn(conn)
    else:
        _sqlite_exec(
            "INSERT INTO person_logs (person_id,timestamp,position_x,position_y,speed,zone,event_type) VALUES (?,?,?,?,?,?,?)",
            (str(log.get("person_id")), ts, log.get("position_x"), log.get("position_y"),
             log.get("speed"), log.get("zone"), log.get("event_type")),
        )


def save_presence_log(log: Dict[str, Any]) -> None:
    if _backend == "none":
        return
    ts = log.get("timestamp") or _now()
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO presence_logs (person_id,event_type,timestamp,session_duration,camera_id) "
                    "VALUES (%s,%s,%s,%s,%s)",
                    (str(log.get("person_id")), log.get("event_type"), ts,
                     log.get("session_duration", 0.0), log.get("camera_id","CAM_01")),
                )
                conn.commit()
        finally:
            db_pool.putconn(conn)
    else:
        _sqlite_exec(
            "INSERT INTO presence_logs (person_id,event_type,timestamp,session_duration,camera_id) VALUES (?,?,?,?,?)",
            (str(log.get("person_id")), log.get("event_type"), ts,
             log.get("session_duration", 0.0), log.get("camera_id","CAM_01")),
        )


def save_face_log(log: Dict[str, Any]) -> None:
    if _backend == "none":
        return
    ts = log.get("timestamp") or _now()
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO face_logs (person_id,face_name,camera_id,timestamp,confidence,crop_path,match_status) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (str(log.get("person_id")), log.get("face_name"), log.get("camera_id","CAM_01"),
                     ts, log.get("confidence",0.0), log.get("crop_path"), log.get("match_status","unknown")),
                )
                conn.commit()
        finally:
            db_pool.putconn(conn)
    else:
        _sqlite_exec(
            "INSERT INTO face_logs (person_id,face_name,camera_id,timestamp,confidence,crop_path,match_status) VALUES (?,?,?,?,?,?,?)",
            (str(log.get("person_id")), log.get("face_name"), log.get("camera_id","CAM_01"),
             ts, log.get("confidence",0.0), log.get("crop_path"), log.get("match_status","unknown")),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — get_*
# ─────────────────────────────────────────────────────────────────────────────

def _deserialize_alerts(rows: List[Dict]) -> List[Dict]:
    for alert in rows:
        if alert.get("metadata_json"):
            try:
                alert.update(json.loads(alert["metadata_json"]))
            except Exception:
                pass
    return rows


def get_identity_history(global_id: str) -> List[Dict[str, Any]]:
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM events WHERE global_id = %s ORDER BY timestamp DESC", (global_id,))
                return [dict(r) for r in cur.fetchall()]
        finally:
            db_pool.putconn(conn)
    return _sqlite_query("SELECT * FROM events WHERE global_id=? ORDER BY timestamp DESC", (global_id,))


def get_alerts(acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if acknowledged is None:
                    cur.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
                else:
                    cur.execute("SELECT * FROM alerts WHERE acknowledged=%s ORDER BY timestamp DESC", (acknowledged,))
                return _deserialize_alerts([dict(r) for r in cur.fetchall()])
        finally:
            db_pool.putconn(conn)

    if acknowledged is None:
        rows = _sqlite_query("SELECT * FROM alerts ORDER BY timestamp DESC")
    else:
        rows = _sqlite_query("SELECT * FROM alerts WHERE acknowledged=? ORDER BY timestamp DESC",
                             (1 if acknowledged else 0,))
    return _deserialize_alerts(rows)


def acknowledge_alert(alert_id: str) -> None:
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE alerts SET acknowledged=TRUE WHERE alert_id=%s", (alert_id,))
                conn.commit()
        finally:
            db_pool.putconn(conn)
    else:
        _sqlite_exec("UPDATE alerts SET acknowledged=1 WHERE alert_id=?", (alert_id,))


def get_face_logs(limit: int = 100) -> List[Dict[str, Any]]:
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM face_logs ORDER BY timestamp DESC LIMIT %s", (limit,))
                return [dict(r) for r in cur.fetchall()]
        finally:
            db_pool.putconn(conn)
    return _sqlite_query("SELECT * FROM face_logs ORDER BY timestamp DESC LIMIT ?", (limit,))


def get_face_logs_by_person(person_id: str) -> List[Dict[str, Any]]:
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM face_logs WHERE person_id=%s ORDER BY timestamp DESC", (person_id,))
                return [dict(r) for r in cur.fetchall()]
        finally:
            db_pool.putconn(conn)
    return _sqlite_query("SELECT * FROM face_logs WHERE person_id=? ORDER BY timestamp DESC", (person_id,))


def get_presence_logs(person_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if person_id:
                    cur.execute("SELECT * FROM presence_logs WHERE person_id=%s ORDER BY timestamp DESC LIMIT %s", (person_id, limit))
                else:
                    cur.execute("SELECT * FROM presence_logs ORDER BY timestamp DESC LIMIT %s", (limit,))
                return [dict(r) for r in cur.fetchall()]
        finally:
            db_pool.putconn(conn)
    if person_id:
        return _sqlite_query("SELECT * FROM presence_logs WHERE person_id=? ORDER BY timestamp DESC LIMIT ?", (person_id, limit))
    return _sqlite_query("SELECT * FROM presence_logs ORDER BY timestamp DESC LIMIT ?", (limit,))


def get_visit_history(person_id: str) -> List[Dict[str, Any]]:
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM presence_logs WHERE person_id=%s AND event_type IN ('entry','re-entry') ORDER BY timestamp ASC",
                    (person_id,),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            db_pool.putconn(conn)
    return _sqlite_query(
        "SELECT * FROM presence_logs WHERE person_id=? AND event_type IN ('entry','re-entry') ORDER BY timestamp ASC",
        (person_id,),
    )


def get_movement_logs(person_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    if _backend == "postgres":
        conn = db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT timestamp,position_x,position_y,speed,zone,event_type FROM person_logs "
                    "WHERE person_id=%s ORDER BY timestamp ASC LIMIT %s",
                    (person_id, limit),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            db_pool.putconn(conn)
    return _sqlite_query(
        "SELECT timestamp,position_x,position_y,speed,zone,event_type FROM person_logs "
        "WHERE person_id=? ORDER BY timestamp ASC LIMIT ?",
        (person_id, limit),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_connection():
    """Legacy helper — only works in Postgres mode."""
    if _backend == "postgres" and db_pool:
        return db_pool.getconn()
    raise Exception("Direct connection not supported in SQLite mode — use the module-level helpers.")


def release_connection(conn) -> None:
    if _backend == "postgres" and db_pool and conn:
        db_pool.putconn(conn)


def get_db_backend() -> str:
    """Returns 'postgres', 'sqlite', or 'none'."""
    return _backend
