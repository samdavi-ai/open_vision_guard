"""
Behaviour Analysis Module
Analyzes movement patterns over time — pacing, circle-walking, erratic direction changes,
prolonged stillness, following behaviour.
"""
import time
import math
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List, Tuple


class BehaviourAnalyzer:
    """Tracks and classifies behavioural patterns per person."""

    # Rolling buffer size (positions stored per person)
    BUFFER_SIZE = 60

    # Behaviour classification thresholds
    STILLNESS_SPEED_THRESHOLD = 5.0       # px/sec below which = stationary
    STILLNESS_DURATION_THRESHOLD = 8.0    # seconds to classify as prolonged_stillness
    PACING_MIN_REVERSALS = 3             # direction reversals in buffer
    PACING_MAX_DISPLACEMENT = 100.0      # max net displacement for pacing
    CIRCLE_MIN_ANGLE_SUM = 300.0         # degrees of cumulative turning
    ERRATIC_DIRECTION_CHANGES = 6        # rapid direction changes in short window
    RUNNING_SPEED_THRESHOLD = 150.0      # px/sec
    FOLLOWING_MAX_DISTANCE = 250.0       # max distance between follower and target
    FOLLOWING_MIN_FRAMES = 20            # minimum co-occurrence frames
    FOLLOWING_TRAJECTORY_SIM = 0.7       # cosine similarity threshold

    def __init__(self):
        # person_id → deque of {cx, cy, t, speed, direction_angle}
        self._buffers: Dict[str, deque] = {}
        # person_id → {label, score, last_update}
        self._results: Dict[str, Dict[str, Any]] = {}
        # person_id → stillness_start_time
        self._stillness_timers: Dict[str, float] = {}

    def update(self, person_id: str, cx: float, cy: float, speed: float, current_time: float) -> Dict[str, Any]:
        """
        Feed a new position sample for a person. Returns behaviour analysis result.

        Returns:
            {
                "behaviour_label": str,
                "behaviour_score": float (0-100),
                "trajectory": list of (x, y) tuples (last 60 positions)
            }
        """
        # Initialize buffer if needed
        if person_id not in self._buffers:
            self._buffers[person_id] = deque(maxlen=self.BUFFER_SIZE)
            self._stillness_timers[person_id] = 0.0

        buf = self._buffers[person_id]

        # Calculate direction angle from previous position
        direction_angle = 0.0
        if len(buf) > 0:
            prev = buf[-1]
            dx = cx - prev["cx"]
            dy = cy - prev["cy"]
            direction_angle = math.atan2(dy, dx)

        buf.append({
            "cx": cx, "cy": cy, "t": current_time,
            "speed": speed, "angle": direction_angle
        })

        # Need at least 5 samples to classify
        if len(buf) < 5:
            return {"behaviour_label": "observing", "behaviour_score": 0.0, "trajectory": [(p["cx"], p["cy"]) for p in buf]}

        # ── Classify behaviour ──
        label, score = self._classify(person_id, buf, speed, current_time)

        self._results[person_id] = {
            "behaviour_label": label,
            "behaviour_score": score,
            "last_update": current_time
        }

        trajectory = [(p["cx"], p["cy"]) for p in buf]
        return {"behaviour_label": label, "behaviour_score": score, "trajectory": trajectory}

    def check_following(self, all_persons: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
        """
        Check if any person is following another.
        all_persons: {person_id: (cx, cy)}

        Returns list of {follower_id, target_id, distance, similarity}
        """
        following_pairs = []
        person_ids = list(all_persons.keys())

        for i, pid_a in enumerate(person_ids):
            if pid_a not in self._buffers or len(self._buffers[pid_a]) < self.FOLLOWING_MIN_FRAMES:
                continue
            for pid_b in person_ids[i + 1:]:
                if pid_b not in self._buffers or len(self._buffers[pid_b]) < self.FOLLOWING_MIN_FRAMES:
                    continue

                # Check current distance
                ax, ay = all_persons[pid_a]
                bx, by = all_persons[pid_b]
                dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

                if dist > self.FOLLOWING_MAX_DISTANCE:
                    continue

                # Compare trajectory similarity (movement vectors)
                buf_a = list(self._buffers[pid_a])
                buf_b = list(self._buffers[pid_b])
                min_len = min(len(buf_a), len(buf_b), 20)

                if min_len < 5:
                    continue

                vecs_a = []
                vecs_b = []
                for j in range(1, min_len):
                    vecs_a.append((buf_a[-min_len + j]["cx"] - buf_a[-min_len + j - 1]["cx"],
                                   buf_a[-min_len + j]["cy"] - buf_a[-min_len + j - 1]["cy"]))
                    vecs_b.append((buf_b[-min_len + j]["cx"] - buf_b[-min_len + j - 1]["cx"],
                                   buf_b[-min_len + j]["cy"] - buf_b[-min_len + j - 1]["cy"]))

                # Cosine similarity of trajectory vectors
                flat_a = np.array([v for pair in vecs_a for v in pair])
                flat_b = np.array([v for pair in vecs_b for v in pair])

                norm_a = np.linalg.norm(flat_a)
                norm_b = np.linalg.norm(flat_b)

                if norm_a < 1e-6 or norm_b < 1e-6:
                    continue

                similarity = float(np.dot(flat_a, flat_b) / (norm_a * norm_b))

                if similarity > self.FOLLOWING_TRAJECTORY_SIM:
                    following_pairs.append({
                        "follower_id": pid_a,
                        "target_id": pid_b,
                        "distance": round(dist, 1),
                        "similarity": round(similarity, 3)
                    })

        return following_pairs

    def get_result(self, person_id: str) -> Optional[Dict[str, Any]]:
        return self._results.get(person_id)

    def _classify(self, person_id: str, buf: deque, current_speed: float,
                  current_time: float) -> Tuple[str, float]:
        """Classify the current behaviour and return (label, score)."""
        positions = list(buf)

        # ── 1. Running detection ──
        if current_speed > self.RUNNING_SPEED_THRESHOLD:
            return ("running", min(70.0 + (current_speed - self.RUNNING_SPEED_THRESHOLD) / 5.0, 95.0))

        # ── 2. Prolonged stillness ──
        recent_speeds = [p["speed"] for p in positions[-10:]]
        avg_recent_speed = sum(recent_speeds) / len(recent_speeds)

        if avg_recent_speed < self.STILLNESS_SPEED_THRESHOLD:
            if self._stillness_timers[person_id] == 0.0:
                self._stillness_timers[person_id] = current_time
            stillness_duration = current_time - self._stillness_timers[person_id]
            if stillness_duration > self.STILLNESS_DURATION_THRESHOLD:
                score = min(30.0 + stillness_duration * 2.0, 80.0)
                return ("prolonged_stillness", score)
        else:
            self._stillness_timers[person_id] = 0.0

        # ── 3. Pacing detection (back-and-forth) ──
        if len(positions) >= 15:
            reversals = self._count_reversals(positions)
            net_displacement = math.sqrt(
                (positions[-1]["cx"] - positions[0]["cx"]) ** 2 +
                (positions[-1]["cy"] - positions[0]["cy"]) ** 2
            )
            if reversals >= self.PACING_MIN_REVERSALS and net_displacement < self.PACING_MAX_DISPLACEMENT:
                score = min(40.0 + reversals * 5.0, 85.0)
                return ("pacing", score)

        # ── 4. Circle walking ──
        if len(positions) >= 20:
            total_turning = self._total_turning_angle(positions)
            if total_turning > self.CIRCLE_MIN_ANGLE_SUM:
                score = min(45.0 + (total_turning - self.CIRCLE_MIN_ANGLE_SUM) / 10.0, 85.0)
                return ("circle_walking", score)

        # ── 5. Erratic movement ──
        if len(positions) >= 10:
            direction_changes = self._count_direction_changes(positions[-10:])
            if direction_changes >= self.ERRATIC_DIRECTION_CHANGES:
                score = min(50.0 + direction_changes * 3.0, 90.0)
                return ("erratic_movement", score)

        # ── 6. Normal walking ──
        if avg_recent_speed > self.STILLNESS_SPEED_THRESHOLD:
            return ("normal_walking", 5.0)

        return ("normal_walking", 0.0)

    def _count_reversals(self, positions: list) -> int:
        """Count how many times movement direction reverses along x-axis."""
        reversals = 0
        prev_dx = 0
        for i in range(1, len(positions)):
            dx = positions[i]["cx"] - positions[i - 1]["cx"]
            if abs(dx) > 3:  # ignore tiny jitter
                if prev_dx != 0 and (dx > 0) != (prev_dx > 0):
                    reversals += 1
                prev_dx = dx
        return reversals

    def _total_turning_angle(self, positions: list) -> float:
        """Sum of absolute angular changes between consecutive movement vectors."""
        total = 0.0
        for i in range(2, len(positions)):
            angle1 = positions[i - 1]["angle"]
            angle2 = positions[i]["angle"]
            delta = abs(angle2 - angle1)
            if delta > math.pi:
                delta = 2 * math.pi - delta
            total += math.degrees(delta)
        return total

    def _count_direction_changes(self, positions: list) -> int:
        """Count significant angular direction changes in a sequence."""
        changes = 0
        for i in range(2, len(positions)):
            angle1 = positions[i - 1]["angle"]
            angle2 = positions[i]["angle"]
            delta = abs(angle2 - angle1)
            if delta > math.pi:
                delta = 2 * math.pi - delta
            if math.degrees(delta) > 45:  # more than 45° change
                changes += 1
        return changes


# Singleton
behaviour_analyzer = BehaviourAnalyzer()
