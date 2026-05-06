import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import json
import datetime
from typing import List, Dict, Any, Optional
from config import config
import os

db_pool = None

def init_db():
    """Initializes the PostgreSQL database with identities, events, and alerts tables."""
    global db_pool
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,
            user=os.getenv("POSTGRES_USER", "ovg_user"),
            password=os.getenv("POSTGRES_PASSWORD", "ovg_password"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "openvisionguard")
        )
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return

    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            # identities table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS identities (
              global_id VARCHAR PRIMARY KEY,
              face_name VARCHAR,
              risk_level VARCHAR DEFAULT 'low',
              first_seen TIMESTAMP,
              last_seen TIMESTAMP,
              total_appearances INTEGER DEFAULT 0,
              metadata_json TEXT
            );
            ''')
            
            # events table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
              id SERIAL PRIMARY KEY,
              global_id VARCHAR,
              camera_id VARCHAR,
              activity VARCHAR,
              location VARCHAR,
              timestamp TIMESTAMP,
              frame_path VARCHAR
            );
            ''')
            
            # alerts table
            cursor.execute('''
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
            ''')
            
            # detections table
            cursor.execute('''
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
            ''')

            # person_logs table
            cursor.execute('''
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
            ''')

            # presence_logs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS presence_logs (
                id SERIAL PRIMARY KEY,
                person_id VARCHAR,
                event_type VARCHAR,
                timestamp TIMESTAMP,
                session_duration FLOAT,
                camera_id VARCHAR DEFAULT 'CAM_01'
            );
            ''')

            # face_logs table
            cursor.execute('''
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
            ''')
            conn.commit()
    finally:
        db_pool.putconn(conn)

def save_identity(identity: Dict[str, Any]):
    """Saves or updates an identity in the database."""
    if db_pool is None:
        return
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT global_id FROM identities WHERE global_id = %s", (identity['global_id'],))
            exists = cursor.fetchone()
            
            now = datetime.datetime.now().astimezone().isoformat()
            metadata_json = json.dumps(identity.get('metadata', {}))
            
            if exists:
                cursor.execute('''
                    UPDATE identities SET 
                        face_name = %s,
                        risk_level = %s,
                        last_seen = %s,
                        total_appearances = total_appearances + 1,
                        metadata_json = %s
                    WHERE global_id = %s
                ''', (
                    identity.get('face_name'),
                    identity.get('risk_level', 'low'),
                    now,
                    metadata_json,
                    identity['global_id']
                ))
            else:
                cursor.execute('''
                    INSERT INTO identities (global_id, face_name, risk_level, first_seen, last_seen, total_appearances, metadata_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (
                    identity['global_id'],
                    identity.get('face_name'),
                    identity.get('risk_level', 'low'),
                    now,
                    now,
                    1,
                    metadata_json
                ))
            conn.commit()
    finally:
        db_pool.putconn(conn)

def save_event(event: Dict[str, Any]):
    if db_pool is None:
        return
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO events (global_id, camera_id, activity, location, timestamp, frame_path)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (
                event.get('global_id'),
                event.get('camera_id'),
                event.get('activity'),
                event.get('location'),
                event.get('timestamp') or datetime.datetime.now().astimezone().isoformat(),
                event.get('frame_path')
            ))
            conn.commit()
    finally:
        db_pool.putconn(conn)

def save_alert(alert: Dict[str, Any]):
    if db_pool is None:
        return
    conn = db_pool.getconn()
    try:
        # Extract metadata (exclude standard fields to avoid redundancy)
        standard_keys = {'alert_id', 'severity', 'type', 'message', 'global_id', 'camera_id', 'timestamp', 'thumbnail_path', 'acknowledged'}
        metadata = {k: v for k, v in alert.items() if k not in standard_keys}
        
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO alerts (alert_id, severity, type, message, global_id, camera_id, timestamp, thumbnail_path, acknowledged, metadata_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                alert['alert_id'],
                alert['severity'],
                alert['type'],
                alert['message'],
                alert['global_id'],
                alert['camera_id'],
                alert['timestamp'] or datetime.datetime.now().astimezone().isoformat(),
                alert.get('thumbnail_path'),
                alert.get('acknowledged', False),
                json.dumps(metadata)
            ))
            conn.commit()
    finally:
        db_pool.putconn(conn)

def save_detection(det: Dict[str, Any]):
    if db_pool is None:
        return
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO detections (object_id, material, confidence, size, timestamp, latitude, longitude)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                str(det.get('object_id')),
                det.get('material'),
                det.get('confidence'),
                det.get('size'),
                det.get('timestamp') or datetime.datetime.now().astimezone().isoformat(),
                det.get('latitude'),
                det.get('longitude')
            ))
            conn.commit()
    finally:
        db_pool.putconn(conn)

def save_person_log(log: Dict[str, Any]):
    if db_pool is None:
        return
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO person_logs (person_id, timestamp, position_x, position_y, speed, zone, event_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                str(log.get('person_id')),
                log.get('timestamp') or datetime.datetime.now().astimezone().isoformat(),
                log.get('position_x'),
                log.get('position_y'),
                log.get('speed'),
                log.get('zone'),
                log.get('event_type')
            ))
            conn.commit()
    finally:
        db_pool.putconn(conn)

def save_presence_log(log: Dict[str, Any]):
    if db_pool is None:
        return
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO presence_logs (person_id, event_type, timestamp, session_duration, camera_id)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                str(log.get('person_id')),
                log.get('event_type'),
                log.get('timestamp') or datetime.datetime.now().astimezone().isoformat(),
                log.get('session_duration', 0.0),
                log.get('camera_id', 'CAM_01')
            ))
            conn.commit()
    finally:
        db_pool.putconn(conn)

def save_face_log(log: Dict[str, Any]):
    if db_pool is None:
        return
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO face_logs (person_id, face_name, camera_id, timestamp, confidence, crop_path, match_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                str(log.get('person_id')),
                log.get('face_name'),
                log.get('camera_id', 'CAM_01'),
                log.get('timestamp') or datetime.datetime.now().astimezone().isoformat(),
                log.get('confidence', 0.0),
                log.get('crop_path'),
                log.get('match_status', 'unknown')
            ))
            conn.commit()
    finally:
        db_pool.putconn(conn)

def get_identity_history(global_id: str) -> List[Dict[str, Any]]:
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM events WHERE global_id = %s ORDER BY timestamp DESC", (global_id,))
            return cursor.fetchall()
    finally:
        db_pool.putconn(conn)

def get_alerts(acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            if acknowledged is None:
                cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
            else:
                cursor.execute("SELECT * FROM alerts WHERE acknowledged = %s ORDER BY timestamp DESC", (acknowledged,))
            
            alerts = cursor.fetchall()
            # Deserialize metadata
            for alert in alerts:
                if alert.get('metadata_json'):
                    try:
                        meta = json.loads(alert['metadata_json'])
                        alert.update(meta)
                    except:
                        pass
            return alerts
    finally:
        db_pool.putconn(conn)

def acknowledge_alert(alert_id: str):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE alerts SET acknowledged = TRUE WHERE alert_id = %s", (alert_id,))
            conn.commit()
    finally:
        db_pool.putconn(conn)

def get_face_logs(limit: int = 100) -> List[Dict[str, Any]]:
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM face_logs ORDER BY timestamp DESC LIMIT %s", (limit,))
            return cursor.fetchall()
    finally:
        db_pool.putconn(conn)

def get_face_logs_by_person(person_id: str) -> List[Dict[str, Any]]:
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM face_logs WHERE person_id = %s ORDER BY timestamp DESC", (person_id,))
            return cursor.fetchall()
    finally:
        db_pool.putconn(conn)

def get_presence_logs(person_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            if person_id:
                cursor.execute("SELECT * FROM presence_logs WHERE person_id = %s ORDER BY timestamp DESC LIMIT %s", (person_id, limit))
            else:
                cursor.execute("SELECT * FROM presence_logs ORDER BY timestamp DESC LIMIT %s", (limit,))
            return cursor.fetchall()
    finally:
        db_pool.putconn(conn)

def get_visit_history(person_id: str) -> List[Dict[str, Any]]:
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM presence_logs WHERE person_id = %s AND event_type IN ('entry', 're-entry') ORDER BY timestamp ASC",
                (person_id,)
            )
            return cursor.fetchall()
    finally:
        db_pool.putconn(conn)

def get_movement_logs(person_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT timestamp, position_x, position_y, speed, zone, event_type FROM person_logs WHERE person_id = %s ORDER BY timestamp ASC LIMIT %s",
                (person_id, limit)
            )
            return cursor.fetchall()
    finally:
        db_pool.putconn(conn)

def get_connection():
    """Helper for legacy code or direct connection access."""
    if db_pool:
        return db_pool.getconn()
    raise Exception("Database connection pool not initialized")

def release_connection(conn):
    """Helper to return a connection to the pool."""
    if db_pool and conn:
        db_pool.putconn(conn)
