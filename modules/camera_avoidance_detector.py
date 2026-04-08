"""
Camera Avoidance Detector Module
Detects people who may be actively avoiding the camera — face hidden, edge hugging,
sudden turn-aways.
"""
import time
import math
from collections import deque
from typing import Dict, Any, Optional, List, Tuple


class CameraAvoidanceDetector:
    """Detects camera avoidance behaviours per person."""

    HISTORY_SIZE = 60                     # frames of history to track
    FACE_VISIBILITY_WINDOW = 30           # frames to evaluate face visibility ratio
    EDGE_MARGIN_RATIO = 0.1              # 10% of frame edge = "hugging" zone
    TURN_AWAY_ANGLE_THRESHOLD = 90.0     # degrees — sudden turn when facing camera
    LOW_FACE_RATIO_THRESHOLD = 0.2       # face visible < 20% of time = possible avoidance

    def __init__(self):
        # person_id → deque of {face_visible: bool, bbox, t, direction_angle}
        self._history: Dict[str, deque] = {}
        # person_id → {avoidance_score, avoidance_behaviours}
        self._results: Dict[str, Dict[str, Any]] = {}

    def update(self, person_id: str, bbox: Tuple[int, int, int, int],
               face_visible: bool, direction_angle: float,
               frame_width: int, frame_height: int,
               current_time: float) -> Dict[str, Any]:
        """
        Update avoidance tracking for a person.

        Args:
            person_id: Person's global ID.
            bbox: (x1, y1, x2, y2) bounding box.
            face_visible: Whether face was detected/recognized this frame.
            direction_angle: Movement direction angle in radians.
            frame_width, frame_height: Frame dimensions.
            current_time: Current timestamp.

        Returns:
            {"avoidance_score": float (0-100), "avoidance_behaviours": list}
        """
        if person_id not in self._history:
            self._history[person_id] = deque(maxlen=self.HISTORY_SIZE)

        self._history[person_id].append({
            "face_visible": face_visible,
            "bbox": bbox,
            "t": current_time,
            "angle": direction_angle,
        })

        history = list(self._history[person_id])

        if len(history) < 5:
            return {"avoidance_score": 0.0, "avoidance_behaviours": []}

        score = 0.0
        behaviours = []

        # ── 1. Face visibility ratio ──
        recent = history[-min(len(history), self.FACE_VISIBILITY_WINDOW):]
        face_count = sum(1 for h in recent if h["face_visible"])
        face_ratio = face_count / len(recent)

        if face_ratio < self.LOW_FACE_RATIO_THRESHOLD:
            score += 40.0
            behaviours.append("face_hidden")
        elif face_ratio < 0.4:
            score += 20.0
            behaviours.append("face_rarely_visible")

        # ── 2. Edge/blind-spot hugging ──
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        edge_x = self.EDGE_MARGIN_RATIO * frame_width
        edge_y = self.EDGE_MARGIN_RATIO * frame_height

        at_edge = (cx < edge_x or cx > frame_width - edge_x or
                   cy < edge_y or cy > frame_height - edge_y)

        if at_edge:
            # Check if consistently at edge
            edge_frames = 0
            for h in recent:
                hx1, hy1, hx2, hy2 = h["bbox"]
                hcx = (hx1 + hx2) / 2
                hcy = (hy1 + hy2) / 2
                if (hcx < edge_x or hcx > frame_width - edge_x or
                        hcy < edge_y or hcy > frame_height - edge_y):
                    edge_frames += 1

            edge_ratio = edge_frames / len(recent)
            if edge_ratio > 0.6:
                score += 30.0
                behaviours.append("edge_hugging")
            elif edge_ratio > 0.3:
                score += 15.0
                behaviours.append("near_edge")

        # ── 3. Sudden turn-away detection ──
        if len(history) >= 3:
            recent_angles = [h["angle"] for h in history[-5:]]
            for i in range(1, len(recent_angles)):
                delta = abs(recent_angles[i] - recent_angles[i - 1])
                if delta > math.pi:
                    delta = 2 * math.pi - delta
                if math.degrees(delta) > self.TURN_AWAY_ANGLE_THRESHOLD:
                    # Check if face was visible before and not after
                    if (i < len(recent) and i > 0 and
                            recent[-(len(recent_angles) - i + 1)]["face_visible"] and
                            not face_visible):
                        score += 25.0
                        behaviours.append("sudden_turn_away")
                        break

        # Clamp
        score = min(score, 100.0)

        result = {
            "avoidance_score": round(score, 1),
            "avoidance_behaviours": behaviours,
        }
        self._results[person_id] = result
        return result

    def get_result(self, person_id: str) -> Optional[Dict[str, Any]]:
        return self._results.get(person_id)


# Singleton
camera_avoidance_detector = CameraAvoidanceDetector()
