import sqlite3
import json
import datetime
from typing import List, Dict, Any, Optional
from config import config

def get_connection():
    return sqlite3.connect(config.db_path)

def init_db():
    """Initializes the SQLite database with identities, events, and alerts tables."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # identities table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS identities (
          global_id TEXT PRIMARY KEY,
          face_name TEXT,
          risk_level TEXT DEFAULT 'low',
          first_seen TIMESTAMP,
          last_seen TIMESTAMP,
          total_appearances INTEGER DEFAULT 0,
          metadata_json TEXT
        );
        ''')
        
        # events table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          global_id TEXT,
          camera_id TEXT,
          activity TEXT,
          location TEXT,
          timestamp TIMESTAMP,
          frame_path TEXT
        );
        ''')
        
        # alerts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
          alert_id TEXT PRIMARY KEY,
          severity TEXT,
          type TEXT,
          message TEXT,
          global_id TEXT,
          camera_id TEXT,
          timestamp TIMESTAMP,
          thumbnail_path TEXT,
          acknowledged BOOLEAN DEFAULT 0
        );
        ''')
        
        # detections table for real-time analytics
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_id TEXT,
            material TEXT,
            confidence FLOAT,
            size TEXT,
            timestamp TEXT,
            latitude FLOAT,
            longitude FLOAT
        );
        ''')

        # person_logs table for movement analytics
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS person_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            timestamp TEXT,
            position_x FLOAT,
            position_y FLOAT,
            speed FLOAT,
            zone TEXT,
            event_type TEXT
        );
        ''')

        # presence_logs table for entry/exit tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS presence_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            event_type TEXT,
            timestamp TEXT,
            session_duration FLOAT,
            camera_id TEXT DEFAULT 'CAM_01'
        );
        ''')

        # face_logs table for face recognition event history
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            face_name TEXT,
            camera_id TEXT,
            timestamp TEXT,
            confidence FLOAT,
            crop_path TEXT,
            match_status TEXT DEFAULT 'unknown'
        );
        ''')
        
        conn.commit()

def save_identity(identity: Dict[str, Any]):
    """Saves or updates an identity in the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT global_id FROM identities WHERE global_id = ?", (identity['global_id'],))
        exists = cursor.fetchone()
        
        now = datetime.datetime.now().astimezone().isoformat()
        metadata_json = json.dumps(identity.get('metadata', {}))
        
        if exists:
            cursor.execute('''
                UPDATE identities SET 
                    face_name = ?,
                    risk_level = ?,
                    last_seen = ?,
                    total_appearances = total_appearances + 1,
                    metadata_json = ?
                WHERE global_id = ?
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
                VALUES (?, ?, ?, ?, ?, ?, ?)
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

def save_event(event: Dict[str, Any]):
    """Saves a security event to the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (global_id, camera_id, activity, location, timestamp, frame_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            event.get('global_id'),
            event.get('camera_id'),
            event.get('activity'),
            event.get('location'),
            event.get('timestamp') or datetime.datetime.now().astimezone().isoformat(),
            event.get('frame_path')
        ))
        conn.commit()

def save_alert(alert: Dict[str, Any]):
    """Saves an alert to the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (alert_id, severity, type, message, global_id, camera_id, timestamp, thumbnail_path, acknowledged)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert['alert_id'],
            alert['severity'],
            alert['type'],
            alert['message'],
            alert['global_id'],
            alert['camera_id'],
            alert['timestamp'] or datetime.datetime.now().astimezone().isoformat(),
            alert.get('thumbnail_path'),
            alert.get('acknowledged', False)
        ))
        conn.commit()

def save_detection(det: Dict[str, Any]):
    """Saves a single detection event into the tracking database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections (object_id, material, confidence, size, timestamp, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?)
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

def save_person_log(log: Dict[str, Any]):
    """Saves a person tracking event into the person analytics database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO person_logs (person_id, timestamp, position_x, position_y, speed, zone, event_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
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

def get_identity_history(global_id: str) -> List[Dict[str, Any]]:
    """Retrieves all events for a specific global_id."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE global_id = ? ORDER BY timestamp DESC", (global_id,))
        return [dict(row) for row in cursor.fetchall()]

def get_alerts(acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Retrieves alerts, optionally filtered by acknowledgment status."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if acknowledged is None:
            cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
        else:
            cursor.execute("SELECT * FROM alerts WHERE acknowledged = ? ORDER BY timestamp DESC", (acknowledged,))
        return [dict(row) for row in cursor.fetchall()]

def acknowledge_alert(alert_id: str):
    """Marks an alert as acknowledged."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE alerts SET acknowledged = 1 WHERE alert_id = ?", (alert_id,))
        conn.commit()


def save_presence_log(log: Dict[str, Any]):
    """Saves a presence event (entry/exit/re-entry) to the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO presence_logs (person_id, event_type, timestamp, session_duration, camera_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            str(log.get('person_id')),
            log.get('event_type'),
            log.get('timestamp') or datetime.datetime.now().astimezone().isoformat(),
            log.get('session_duration', 0.0),
            log.get('camera_id', 'CAM_01')
        ))
        conn.commit()


def save_face_log(log: Dict[str, Any]):
    """Saves a face recognition event to the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO face_logs (person_id, face_name, camera_id, timestamp, confidence, crop_path, match_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
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


def get_face_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves all face recognition events."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM face_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_face_logs_by_person(person_id: str) -> List[Dict[str, Any]]:
    """Retrieves face recognition events for a specific person."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM face_logs WHERE person_id = ? ORDER BY timestamp DESC", (person_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_presence_logs(person_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves presence events, optionally filtered by person."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if person_id:
            cursor.execute("SELECT * FROM presence_logs WHERE person_id = ? ORDER BY timestamp DESC LIMIT ?", (person_id, limit))
        else:
            cursor.execute("SELECT * FROM presence_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_visit_history(person_id: str) -> List[Dict[str, Any]]:
    """Retrieves entry events for a person — used by frequency analysis."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM presence_logs WHERE person_id = ? AND event_type IN ('entry', 're-entry') ORDER BY timestamp ASC",
            (person_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

def get_movement_logs(person_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Retrieves chronological movement trajectory metrics for charting."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, position_x, position_y, speed, zone, event_type FROM person_logs WHERE person_id = ? ORDER BY timestamp ASC LIMIT ?",
            (person_id, limit)
        )
        return [dict(row) for row in cursor.fetchall()]
