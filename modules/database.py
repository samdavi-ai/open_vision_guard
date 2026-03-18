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
        
        conn.commit()

def save_identity(identity: Dict[str, Any]):
    """Saves or updates an identity in the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT global_id FROM identities WHERE global_id = ?", (identity['global_id'],))
        exists = cursor.fetchone()
        
        now = datetime.datetime.now().isoformat()
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
            event.get('timestamp') or datetime.datetime.now().isoformat(),
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
            alert['timestamp'] or datetime.datetime.now().isoformat(),
            alert.get('thumbnail_path'),
            alert.get('acknowledged', False)
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
