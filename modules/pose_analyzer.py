import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, List, Optional
from collections import defaultdict
from config import config

class PoseAnalyzer:
    def __init__(self, model_path: str = None):
        # We drop YOLOv8 Pose and use MediaPipe for lighter weight
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0 is fastest/lightest, 1 is balanced, 2 is heavy
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.available = True
        except AttributeError:
            print("[PoseAnalyzer] Warning: MediaPipe solutions API not found. Pose analysis disabled.")
            self.available = False
        self.keypoint_buffers: Dict[str, List[Any]] = defaultdict(list)

    def analyze_pose(self, frame: np.ndarray, bbox: tuple) -> Dict[str, Any]:
        """
        Analyze pose for a person within the given bbox region using MediaPipe.
        """
        if not getattr(self, 'available', False):
            return {"activity": "unknown", "keypoints": None, "fall_detected": False, "confidence": 0.0}

        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {"activity": "unknown", "keypoints": None, "fall_detected": False, "confidence": 0.0}

        # MediaPipe needs RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(crop_rgb)

        activity = "standing"
        fall_detected = False
        confidence = 0.0
        keypoints = None

        if results.pose_landmarks:
            keypoints = results.pose_landmarks.landmark
            activity, fall_detected, confidence = self._classify_activity(keypoints, crop.shape)

        return {
            "activity": activity,
            "keypoints": keypoints,
            "fall_detected": fall_detected,
            "confidence": confidence
        }

    def _classify_activity(self, landmarks, shape) -> tuple:
        """
        Heuristic activity classification from MediaPipe landmarks.
        """
        h, w, _ = shape
        
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Convert normalized coordinates to pixel coordinates
        nose_y = nose.y * h
        hip_y = ((l_hip.y + r_hip.y) / 2) * h
        ankle_y = ((l_ankle.y + r_ankle.y) / 2) * h
        shoulder_y = ((l_shoulder.y + r_shoulder.y) / 2) * h
        shoulder_width = abs(l_shoulder.x - r_shoulder.x) * w

        confidence = min(nose.visibility, l_hip.visibility, r_hip.visibility)

        # Fall detection
        if confidence > config.fall_confidence_threshold:
            if nose_y > hip_y:
                return "falling", True, float(confidence)

        # Sitting
        body_height = abs(ankle_y - shoulder_y)
        if body_height > 0 and shoulder_width > 0:
            aspect_ratio = body_height / shoulder_width
            if aspect_ratio < 1.2:
                return "sitting", False, float(confidence)

        # Crouching
        hip_to_ankle = abs(hip_y - ankle_y)
        shoulder_to_hip = abs(shoulder_y - hip_y)
        if shoulder_to_hip > 0 and hip_to_ankle / shoulder_to_hip < 0.5:
            return "crouching", False, float(confidence)

        return "standing", False, float(confidence)

    def detect_pose_full_frame(self, frame: np.ndarray):
        """
        Run pose detection on full frame for drawing.
        """
        if not getattr(self, 'available', False):
            return frame.copy(), None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
        return annotated_frame, results.pose_landmarks

# Singleton instance
pose_analyzer = PoseAnalyzer()
