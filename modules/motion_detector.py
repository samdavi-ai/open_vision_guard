import cv2
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from config import config


class MotionDetector:
    def __init__(self, history: int = 500, var_threshold: int = 16, detect_shadows: bool = True):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.min_area = 500

        # Loitering tracking: global_id -> {first_seen_time, last_bbox_center}
        self.loitering_tracker: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "first_seen_time": None,
            "last_center": None
        })

    def detect_motion(self, frame: np.ndarray, zones: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Detect motion in frame. If zones are provided, check for zone breaches.
        zones: [{"name": "Zone A", "x1": 100, "y1": 100, "x2": 400, "y2": 400}]
        Returns: {motion_detected, active_zones, motion_mask}
        """
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        active_zones = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            motion_detected = True

            if zones:
                (cx, cy, cw, ch) = cv2.boundingRect(contour)
                motion_center = (cx + cw // 2, cy + ch // 2)

                for zone in zones:
                    if (zone["x1"] <= motion_center[0] <= zone["x2"] and
                            zone["y1"] <= motion_center[1] <= zone["y2"]):
                        if zone["name"] not in active_zones:
                            active_zones.append(zone["name"])

        return {
            "motion_detected": motion_detected,
            "active_zones": active_zones,
            "motion_mask": fg_mask
        }

    def check_loitering(self, global_id: str, bbox: tuple, timestamp: Optional[float] = None) -> bool:
        """
        Check if a person (global_id) has been in roughly the same location
        for longer than loitering_threshold_seconds.
        """
        if timestamp is None:
            timestamp = time.time()

        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        tracker = self.loitering_tracker[global_id]

        if tracker["first_seen_time"] is None:
            tracker["first_seen_time"] = timestamp
            tracker["last_center"] = center
            return False

        # Check if person moved significantly (> 50px)
        last_center = tracker["last_center"]
        distance = ((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2) ** 0.5

        if distance > 50:
            # Reset timer - person moved
            tracker["first_seen_time"] = timestamp
            tracker["last_center"] = center
            return False

        tracker["last_center"] = center

        elapsed = timestamp - tracker["first_seen_time"]
        if elapsed > config.loitering_threshold_seconds:
            return True

        return False

    def annotate_motion(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, str]:
        """Legacy-compatible method for drawing motion on frame."""
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = frame.copy()
        motion_detected = False
        total_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            motion_detected = True
            total_area += area
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

        anomaly = "None"
        frame_area = frame.shape[0] * frame.shape[1]
        if motion_detected:
            if total_area > frame_area * 0.4:
                anomaly = "Massive Motion / Disturbance"
            elif total_area > frame_area * 0.15:
                anomaly = "Moderate Motion"

        return annotated, motion_detected, anomaly
