import uuid
import datetime
import cv2
import os
from typing import Dict, Any, List, Optional
from collections import defaultdict
from config import config
from modules import database


class AlertEngine:
    def __init__(self):
        # Deduplication: (global_id, type) -> last_alert_timestamp
        self._dedup_cache: Dict[tuple, float] = defaultdict(float)
        # Connected WebSocket clients for alert push
        self.alert_subscribers = []

        os.makedirs(config.thumbnails_dir, exist_ok=True)

    def create_alert(
        self,
        alert_type: str,
        global_id: str,
        camera_id: str,
        details: Any,
        frame: Optional[any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a context-aware alert with deduplication.
        Returns the alert dict or None if deduplicated.
        """
        now = datetime.datetime.now()
        now_ts = now.timestamp()

        # Deduplication check
        dedup_key = (global_id, alert_type)
        last_ts = self._dedup_cache.get(dedup_key, 0)
        if (now_ts - last_ts) < config.alert_dedup_window_seconds:
            return None  # Skip duplicate

        self._dedup_cache[dedup_key] = now_ts

        # Determine severity
        severity = self._get_severity(alert_type)

        # Build human-readable message
        message = self._build_message(alert_type, global_id, camera_id, details)

        # Save thumbnail if frame provided
        alert_id = str(uuid.uuid4())
        thumbnail_path = None
        if frame is not None:
            thumbnail_path = os.path.join(config.thumbnails_dir, f"{alert_id}.jpg")
            cv2.imwrite(thumbnail_path, frame)

        alert = {
            "alert_id": alert_id,
            "severity": severity,
            "type": alert_type,
            "message": message,
            "global_id": global_id,
            "camera_id": camera_id,
            "timestamp": now.isoformat(),
            "thumbnail_path": thumbnail_path,
            "acknowledged": False
        }

        # Persist to DB
        try:
            database.save_alert(alert)
        except Exception as e:
            print(f"Error saving alert to DB: {e}")

        # Push to WebSocket subscribers
        self._push_to_subscribers(alert)

        return alert

    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged."""
        database.acknowledge_alert(alert_id)

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts from the database."""
        return database.get_alerts()[:limit]

    def _get_severity(self, alert_type: str) -> str:
        severity_map = {
            "weapon": "critical",
            "fall": "high",
            "loitering": "medium",
            "zone_breach": "high",
            "unknown_person": "low",
            "object_left_behind": "critical",  # Suspicious package / bomb scenario
            "object_acquired": "high",          # Potential theft
            "object_swapped": "critical",       # Object exchange (drug deal, bomb swap)
        }
        return severity_map.get(alert_type, "medium")

    def _build_message(self, alert_type: str, global_id: str, camera_id: str, details: Any) -> str:
        """Build a human-readable alert message."""
        base = f"{global_id} at {camera_id}"

        if alert_type == "weapon":
            weapon_type = "weapon"
            if isinstance(details, dict):
                weapon_type = details.get("weapon_type", "weapon")
            return f"⚠ WEAPON: {base} — {weapon_type} detected nearby"

        elif alert_type == "fall":
            return f"⚠ FALL: {base} — Person appears to have fallen"

        elif alert_type == "loitering":
            return f"LOITERING: {base} — Person has been stationary for extended period"

        elif alert_type == "zone_breach":
            zone_name = ""
            if isinstance(details, dict):
                zone_name = details.get("zone_name", "restricted area")
            return f"🚨 ZONE BREACH: {base} — Motion detected in {zone_name}"

        elif alert_type == "unknown_person":
            return f"UNKNOWN: {base} — Unrecognized person detected"

        elif alert_type == "object_left_behind":
            obj = details.get("object", "object") if isinstance(details, dict) else "object"
            return f"🚨 SUSPICIOUS PACKAGE: {base} — {obj} left behind (possible threat)"

        elif alert_type == "object_acquired":
            obj = details.get("object", "object") if isinstance(details, dict) else "object"
            return f"⚠ OBJECT ACQUIRED: {base} — picked up {obj} (possible theft)"

        elif alert_type == "object_swapped":
            obj = details.get("object", "object") if isinstance(details, dict) else "object"
            return f"🚨 OBJECT SWAP: {base} — {obj} exchanged (suspicious activity)"

        return f"ALERT: {base} — {alert_type}"

    def _push_to_subscribers(self, alert: Dict[str, Any]):
        """Push alert to WebSocket subscribers (called by stream router)."""
        # Subscribers will be managed by the stream_router WebSocket handler
        for callback in self.alert_subscribers:
            try:
                callback(alert)
            except Exception:
                pass


# Singleton
alert_engine = AlertEngine()
