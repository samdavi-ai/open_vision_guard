import sqlite3
import datetime
import os

class DatabaseManager:
    def __init__(self, db_path="openvisionguard.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database with detections and alerts tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Detections table: Logs every recognized subject per frame or sequence
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        subject_id TEXT,
                        class_name TEXT,
                        confidence REAL
                    )
                ''')
                # Alerts table: Logs specific security events (Weapon, Falling, etc.)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        alert_type TEXT,
                        message TEXT
                    )
                ''')
                conn.commit()
            print(f"Database initialized at {self.db_path}")
        except Exception as e:
            print(f"Error initializing database: {e}")

    def log_detection(self, subject_id, class_name, confidence):
        """Logs a detection event to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO detections (timestamp, subject_id, class_name, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (datetime.datetime.now(), subject_id, class_name, confidence))
                conn.commit()
        except Exception as e:
            print(f"Error logging detection: {e}")

    def log_alert(self, alert_type, message):
        """Logs a security alert to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts (timestamp, alert_type, message)
                    VALUES (?, ?, ?)
                ''', (datetime.datetime.now(), alert_type, message))
                conn.commit()
        except Exception as e:
            print(f"Error logging alert: {e}")
