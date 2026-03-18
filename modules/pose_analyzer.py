from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
from config import config


class PoseAnalyzer:
    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        self.model = YOLO(model_path)
        # Rolling buffer: global_id -> list of keypoint sets (last 3 frames)
        self.keypoint_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)

    def analyze_pose(self, frame: np.ndarray, bbox: tuple) -> Dict[str, Any]:
        """
        Analyze pose for a person within the given bbox region.
        Returns: {activity, keypoints, fall_detected, confidence}
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {"activity": "unknown", "keypoints": None, "fall_detected": False, "confidence": 0.0}

        results = self.model(crop, classes=[0], verbose=False)
        result = results[0]

        keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None

        activity = "standing"
        fall_detected = False
        confidence = 0.0

        if keypoints is not None and len(keypoints) > 0:
            kp = keypoints[0]
            activity, fall_detected, confidence = self._classify_activity(kp)

        return {
            "activity": activity,
            "keypoints": keypoints,
            "fall_detected": fall_detected,
            "confidence": confidence
        }

    def _classify_activity(self, kp: np.ndarray) -> tuple:
        """
        Heuristic activity classification from COCO keypoints.
        KP Indices: 0:nose, 5:l_shoulder, 6:r_shoulder, 11:l_hip, 12:r_hip, 15:l_ankle, 16:r_ankle
        """
        if len(kp) < 17:
            return "unknown", False, 0.0

        nose = kp[0]
        l_shoulder, r_shoulder = kp[5], kp[6]
        l_hip, r_hip = kp[11], kp[12]
        l_ankle, r_ankle = kp[15], kp[16]

        # Fall detection: head Y > hip Y (in image coords, Y increases downward)
        hip_y = (l_hip[1] + r_hip[1]) / 2
        nose_conf = nose[2]
        hip_conf = (l_hip[2] + r_hip[2]) / 2

        if nose_conf > config.fall_confidence_threshold and hip_conf > config.fall_confidence_threshold:
            if nose[1] > hip_y:
                return "falling", True, float(min(nose_conf, hip_conf))

        # Sitting: ankles close to hips vertically (compressed skeleton)
        ankle_y = (l_ankle[1] + r_ankle[1]) / 2
        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        body_height = abs(ankle_y - shoulder_y)
        body_width = abs(l_shoulder[0] - r_shoulder[0])

        if body_height > 0 and body_width > 0:
            aspect_ratio = body_height / body_width
            if aspect_ratio < 1.2:
                return "sitting", False, float(hip_conf)

        # Crouching: hips close to ankles
        hip_to_ankle = abs(hip_y - ankle_y)
        shoulder_to_hip = abs(shoulder_y - hip_y)
        if shoulder_to_hip > 0 and hip_to_ankle / shoulder_to_hip < 0.5:
            return "crouching", False, float(hip_conf)

        return "standing", False, float(hip_conf)

    def detect_pose_full_frame(self, frame: np.ndarray):
        """
        Run pose detection on full frame (used for drawing).
        Returns (annotated_frame, keypoints_data).
        """
        results = self.model(frame, classes=[0], verbose=False)
        result = results[0]
        annotated_frame = result.plot()
        keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
        return annotated_frame, keypoints
