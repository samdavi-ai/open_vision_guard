import time
import datetime
from typing import Dict, Any, Optional, List
from modules import database

class PresenceTracker:
    def __init__(self):
        # person_id -> {entry_time, last_seen_time, is_present, total_dwell_time}
        self.active_presences: Dict[str, Dict[str, Any]] = {}
        self.exit_threshold = 5 

    def update(self, person_ids: List[str], current_time: float) -> List[Dict[str, Any]]:
        """
        Updates presence for all visible persons and returns entry events.
        Matches the interface expected by pipeline.py.
        """
        events = []
        now_iso = datetime.datetime.now().astimezone().isoformat()
        
        for pid in person_ids:
            if pid not in self.active_presences:
                self.active_presences[pid] = {
                    "entry_time": current_time,
                    "last_seen_time": current_time,
                    "is_present": True,
                    "total_dwell_seconds": 0.0
                }
                events.append({
                    "person_id": pid,
                    "type": "entry",
                    "timestamp": now_iso
                })
            else:
                self.active_presences[pid]["last_seen_time"] = current_time
                self.active_presences[pid]["is_present"] = True
                self.active_presences[pid]["total_dwell_seconds"] = current_time - self.active_presences[pid]["entry_time"]
        
        return events

    def get_presence_data(self, person_id: str) -> Optional[Dict[str, Any]]:
        return self.active_presences.get(person_id)

    def check_exits(self, currently_detected_ids: List[str], camera_id: str) -> List[Dict[str, Any]]:
        """
        Detects exits and returns exit events.
        """
        now = time.time()
        events = []
        to_remove = []
        
        for pid, info in self.active_presences.items():
            if pid not in currently_detected_ids:
                if now - info["last_seen_time"] > self.exit_threshold:
                    dwell_time = now - info["entry_time"]
                    exit_time_iso = datetime.datetime.now().astimezone().isoformat()
                    
                    events.append({
                        "person_id": pid,
                        "type": "exit",
                        "timestamp": exit_time_iso,
                        "session_duration": dwell_time
                    })
                    to_remove.append(pid)
        
        for pid in to_remove:
            del self.active_presences[pid]
            
        return events

presence_tracker = PresenceTracker()
