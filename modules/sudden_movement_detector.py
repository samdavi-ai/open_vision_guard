"""
Sudden Movement Detector Module
Detects sudden velocity spikes — running, lunging, aggressive gestures, sudden stops.
Combines with weapon detection context for severity escalation.
"""
import time
from collections import deque
from typing import Dict, Any, Optional, List, Tuple


class SuddenMovementDetector:
    """Detects sudden acceleration/deceleration events per person."""

    VELOCITY_WINDOW = 1.0             # seconds — rolling window for velocity history
    ACCELERATION_MULTIPLIER = 3.0     # speed must jump 3x to trigger
    MIN_SPEED_FOR_SUDDEN = 30.0       # minimum speed to consider "sudden"
    SUDDEN_STOP_DECEL = 0.2           # speed drops to <20% of recent average
    COOLDOWN_SECONDS = 3.0            # minimum time between alerts for same person

    def __init__(self):
        # person_id → deque of {speed, t}
        self._velocity_history: Dict[str, deque] = {}
        # person_id → last_alert_time
        self._last_alert: Dict[str, float] = {}
        # person_id → current detection result
        self._results: Dict[str, Dict[str, Any]] = {}

    def update(self, person_id: str, speed: float, current_time: float,
               has_weapon: bool = False) -> Optional[Dict[str, Any]]:
        """
        Feed current speed sample. Returns detection event if sudden movement detected.

        Args:
            person_id: Person's global ID.
            speed: Current speed in px/sec.
            current_time: Current timestamp.
            has_weapon: Whether this person has a weapon nearby.

        Returns:
            None if no detection, or {
                "type": "sudden_run"|"lunge"|"aggressive_gesture"|"sudden_stop",
                "severity": "warning"|"high"|"critical",
                "speed": float,
                "acceleration_factor": float,
                "person_id": str
            }
        """
        if person_id not in self._velocity_history:
            self._velocity_history[person_id] = deque(maxlen=60)

        history = self._velocity_history[person_id]

        # Prune old entries outside window
        while history and (current_time - history[0]["t"]) > self.VELOCITY_WINDOW * 3:
            history.popleft()

        history.append({"speed": speed, "t": current_time})

        # Need at least 3 samples for comparison
        if len(history) < 3:
            self._results[person_id] = None
            return None

        # Check cooldown
        last_alert = self._last_alert.get(person_id, 0)
        if (current_time - last_alert) < self.COOLDOWN_SECONDS:
            return None

        # Calculate rolling average speed (excluding current sample)
        recent = [s["speed"] for s in list(history)[:-1]
                  if (current_time - s["t"]) <= self.VELOCITY_WINDOW]

        if not recent:
            return None

        avg_speed = sum(recent) / len(recent)

        # ── Sudden acceleration ──
        if avg_speed > 1.0 and speed > self.MIN_SPEED_FOR_SUDDEN:
            accel_factor = speed / max(avg_speed, 1.0)
            if accel_factor >= self.ACCELERATION_MULTIPLIER:
                movement_type = self._classify_acceleration(speed, accel_factor)
                severity = "high" if not has_weapon else "critical"

                result = {
                    "type": movement_type,
                    "severity": severity,
                    "speed": round(speed, 1),
                    "acceleration_factor": round(accel_factor, 2),
                    "person_id": person_id,
                }
                self._last_alert[person_id] = current_time
                self._results[person_id] = result
                return result

        # ── Sudden stop ──
        if avg_speed > self.MIN_SPEED_FOR_SUDDEN and speed < avg_speed * self.SUDDEN_STOP_DECEL:
            result = {
                "type": "sudden_stop",
                "severity": "warning",
                "speed": round(speed, 1),
                "acceleration_factor": round(speed / max(avg_speed, 1.0), 2),
                "person_id": person_id,
            }
            self._last_alert[person_id] = current_time
            self._results[person_id] = result
            return result

        self._results[person_id] = None
        return None

    def get_result(self, person_id: str) -> Optional[Dict[str, Any]]:
        return self._results.get(person_id)

    def _classify_acceleration(self, speed: float, factor: float) -> str:
        """Classify the type of sudden movement based on speed and acceleration."""
        if speed > 200:
            return "sudden_run"
        elif factor > 5.0:
            return "lunge"
        elif speed > 100:
            return "aggressive_gesture"
        else:
            return "sudden_run"


# Singleton
sudden_movement_detector = SuddenMovementDetector()
