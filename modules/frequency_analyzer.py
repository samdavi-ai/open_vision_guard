"""
Frequency Analysis Module
Tracks visit frequency, detects patterns, and classifies visitors by frequency.
"""
import time
import datetime
from typing import Dict, Any, Optional, List
from modules import database


class FrequencyAnalyzer:
    """Tracks how often each person appears and classifies their visit frequency."""

    # Frequency classification thresholds
    FREQ_THRESHOLDS = {
        "regular": 10,      # 10+ visits
        "frequent": 5,      # 5-9 visits
        "occasional": 3,    # 3-4 visits
        "rare": 2,          # 2 visits
        "new": 1,           # first visit
    }

    # Session gap: if person not seen for >60 seconds, it counts as a new visit
    SESSION_GAP_SECONDS = 60.0

    def __init__(self):
        # person_id → {visit_count, visit_timestamps[], sessions[], total_dwell, last_seen}
        self._data: Dict[str, Dict[str, Any]] = {}

    def record_appearance(self, person_id: str, current_time: float):
        """Record that a person is currently visible."""
        if person_id not in self._data:
            self._data[person_id] = {
                "visit_count": 1,
                "visit_timestamps": [current_time],
                "total_dwell": 0.0,
                "last_seen": current_time,
                "session_start": current_time,
            }
            return

        data = self._data[person_id]
        gap = current_time - data["last_seen"]

        if gap > self.SESSION_GAP_SECONDS:
            # New visit/session
            data["visit_count"] += 1
            data["visit_timestamps"].append(current_time)
            # Finalize previous session dwell
            data["total_dwell"] += data["last_seen"] - data["session_start"]
            data["session_start"] = current_time

        data["last_seen"] = current_time

    def get_frequency_data(self, person_id: str) -> Dict[str, Any]:
        """Get full frequency analysis for a person."""
        data = self._data.get(person_id)
        if not data:
            return {
                "person_id": person_id,
                "visit_count": 0,
                "frequency_label": "new",
                "total_dwell_seconds": 0.0,
                "avg_session_duration": 0.0,
                "visit_timestamps": [],
                "unusual_timing": False,
            }

        visit_count = data["visit_count"]
        label = self._classify(visit_count)

        # Calculate dwell including active session
        total_dwell = data["total_dwell"]
        if data["last_seen"] > data["session_start"]:
            total_dwell += data["last_seen"] - data["session_start"]

        avg_session = total_dwell / max(visit_count, 1)

        # Check for unusual timing
        unusual = self._check_unusual_timing(data["visit_timestamps"])

        return {
            "person_id": person_id,
            "visit_count": visit_count,
            "frequency_label": label,
            "total_dwell_seconds": round(total_dwell, 1),
            "avg_session_duration": round(avg_session, 1),
            "visit_timestamps": data["visit_timestamps"][-20:],  # last 20 timestamps
            "unusual_timing": unusual,
        }

    def get_overview(self) -> Dict[str, Any]:
        """Get system-wide frequency statistics."""
        total_persons = len(self._data)
        label_counts = {"new": 0, "rare": 0, "occasional": 0, "frequent": 0, "regular": 0}

        for pid, data in self._data.items():
            label = self._classify(data["visit_count"])
            label_counts[label] = label_counts.get(label, 0) + 1

        frequent_visitors = []
        for pid, data in self._data.items():
            if data["visit_count"] >= self.FREQ_THRESHOLDS["occasional"]:
                frequent_visitors.append({
                    "person_id": pid,
                    "visit_count": data["visit_count"],
                    "label": self._classify(data["visit_count"]),
                })

        frequent_visitors.sort(key=lambda x: x["visit_count"], reverse=True)

        return {
            "total_persons_tracked": total_persons,
            "label_distribution": label_counts,
            "frequent_visitors": frequent_visitors[:20],
        }

    def _classify(self, visit_count: int) -> str:
        if visit_count >= self.FREQ_THRESHOLDS["regular"]:
            return "regular"
        elif visit_count >= self.FREQ_THRESHOLDS["frequent"]:
            return "frequent"
        elif visit_count >= self.FREQ_THRESHOLDS["occasional"]:
            return "occasional"
        elif visit_count >= self.FREQ_THRESHOLDS["rare"]:
            return "rare"
        return "new"

    def _check_unusual_timing(self, timestamps: List[float]) -> bool:
        """Check if visits occur at consistently odd hours (e.g., 11 PM - 5 AM)."""
        if len(timestamps) < 2:
            return False

        odd_hour_count = 0
        for ts in timestamps:
            hour = datetime.datetime.fromtimestamp(ts).hour
            if hour >= 23 or hour <= 5:
                odd_hour_count += 1

        return (odd_hour_count / len(timestamps)) > 0.5

    def is_frequent(self, person_id: str) -> bool:
        """Quick check if a person is a frequent visitor (3+ visits)."""
        data = self._data.get(person_id)
        if not data:
            return False
        return data["visit_count"] >= self.FREQ_THRESHOLDS["occasional"]


# Singleton
frequency_analyzer = FrequencyAnalyzer()
