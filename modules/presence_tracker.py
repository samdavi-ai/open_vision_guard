"""
Presence Tracker Module
Tracks person entry/exit times, dwell duration, and generates presence events.
"""
import time
from typing import Dict, Any, List, Optional


class PresenceTracker:
    """Tracks when persons enter/exit the camera view and calculates dwell time."""

    EXIT_TIMEOUT = 5.0  # seconds — if not seen for this long, log an exit

    def __init__(self):
        # person_id → {entry_time, last_seen, is_present, sessions: [{entry, exit, duration}], total_dwell_ms}
        self._tracking: Dict[str, Dict[str, Any]] = {}

    def update(self, visible_person_ids: List[str], current_time: float) -> List[Dict[str, Any]]:
        """
        Update presence tracking with the set of currently visible persons.

        Args:
            visible_person_ids: List of person IDs currently detected in frame.
            current_time: Current timestamp.

        Returns:
            List of presence events: [{"type": "entry"|"exit"|"re-entry", "person_id", "timestamp", ...}]
        """
        events = []
        visible_set = set(visible_person_ids)

        # ── Handle visible persons ──
        for pid in visible_set:
            if pid not in self._tracking:
                # First ever appearance
                self._tracking[pid] = {
                    "entry_time": current_time,
                    "last_seen": current_time,
                    "is_present": True,
                    "sessions": [{"entry": current_time, "exit": None, "duration": 0}],
                    "total_dwell_ms": 0,
                }
                events.append({
                    "type": "entry",
                    "person_id": pid,
                    "timestamp": current_time,
                })
            else:
                tracker = self._tracking[pid]
                if not tracker["is_present"]:
                    # Re-entry after exit
                    tracker["is_present"] = True
                    tracker["sessions"].append({
                        "entry": current_time,
                        "exit": None,
                        "duration": 0,
                    })
                    events.append({
                        "type": "re-entry",
                        "person_id": pid,
                        "timestamp": current_time,
                    })
                tracker["last_seen"] = current_time

                # Update current session duration
                current_session = tracker["sessions"][-1]
                current_session["duration"] = current_time - current_session["entry"]

        # ── Check for exits (persons not seen recently) ──
        for pid, tracker in self._tracking.items():
            if tracker["is_present"] and pid not in visible_set:
                time_since_seen = current_time - tracker["last_seen"]
                if time_since_seen > self.EXIT_TIMEOUT:
                    tracker["is_present"] = False
                    # Finalize current session
                    current_session = tracker["sessions"][-1]
                    current_session["exit"] = tracker["last_seen"]
                    current_session["duration"] = tracker["last_seen"] - current_session["entry"]
                    # Update total dwell
                    tracker["total_dwell_ms"] = sum(
                        s["duration"] for s in tracker["sessions"] if s["duration"] > 0
                    )
                    events.append({
                        "type": "exit",
                        "person_id": pid,
                        "timestamp": tracker["last_seen"],
                        "session_duration": current_session["duration"],
                        "total_dwell": tracker["total_dwell_ms"],
                    })

        return events

    def get_presence_data(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get full presence data for a person."""
        tracker = self._tracking.get(person_id)
        if not tracker:
            return None

        # Calculate total dwell: sum completed sessions + live active session
        completed_dwell = sum(
            s["duration"] for s in tracker["sessions"]
            if s["exit"] is not None and s["duration"] > 0
        )
        active_dwell = 0.0
        if tracker["is_present"] and tracker["sessions"]:
            active = tracker["sessions"][-1]
            if active["exit"] is None:
                active_dwell = tracker["last_seen"] - active["entry"]
        total_dwell = completed_dwell + active_dwell

        return {
            "person_id": person_id,
            "entry_time": tracker["entry_time"],
            "last_seen": tracker["last_seen"],
            "is_present": tracker["is_present"],
            "session_count": len(tracker["sessions"]),
            "total_dwell_seconds": round(total_dwell, 1),
            "sessions": tracker["sessions"],
        }

    def get_all_present(self) -> List[str]:
        """Get list of all currently present person IDs."""
        return [pid for pid, t in self._tracking.items() if t["is_present"]]


# Singleton
presence_tracker = PresenceTracker()
