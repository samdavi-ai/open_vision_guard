"""
PoseAnalyzer — MediaPipe 0.10+ Tasks API (Pose Landmarker).

MediaPipe ≥0.10 dropped mp.solutions.pose.  This module uses the new
Tasks API (mp.tasks.vision.PoseLandmarker) which downloads a lightweight
.task bundle on first use and caches it in ~/.cache/mediapipe/.

Fall detection heuristic:
  If nose_y > hip_y the upper body has inverted — person is on the ground.

Pose labels returned: standing | sitting | crouching | falling
"""

import os
import threading
from collections import defaultdict
from typing import Any, Dict, List

import cv2
import numpy as np

from config import config

# ── Pose landmark indices (MediaPipe 33-point skeleton) ──────────────────────
_NOSE         = 0
_L_SHOULDER   = 11
_R_SHOULDER   = 12
_L_HIP        = 23
_R_HIP        = 24
_L_ANKLE      = 27
_R_ANKLE      = 28

# ── Model bundle URL (tiny model, ~5 MB) ─────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_CACHE_DIR  = os.path.expanduser("~/.cache/mediapipe")
_MODEL_PATH = os.path.join(_CACHE_DIR, "pose_landmarker_lite.task")


def _ensure_model() -> str:
    """Download the model bundle once and cache it locally."""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(_CACHE_DIR, exist_ok=True)
    try:
        import urllib.request
        print("[PoseAnalyzer] Downloading pose_landmarker_lite.task (~5 MB)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[PoseAnalyzer] Model downloaded.")
        return _MODEL_PATH
    except Exception as e:
        print(f"[PoseAnalyzer] Model download failed: {e}")
        return ""


class PoseAnalyzer:
    """
    Thread-safe pose analyzer using MediaPipe Pose Landmarker (Tasks API).
    Falls back gracefully if the model cannot be loaded.
    """

    def __init__(self) -> None:
        self.available   = False
        self._landmarker = None
        self._lock       = threading.Lock()
        self.keypoint_buffers: Dict[str, List[Any]] = defaultdict(list)
        self._init()

    def _init(self) -> None:
        try:
            import mediapipe as mp
            model_path = _ensure_model()
            if not model_path:
                print("[PoseAnalyzer] No model — pose analysis disabled.")
                return

            BaseOptions   = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode     = mp.tasks.vision.RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
            self.available   = True
            print("[PoseAnalyzer] Ready (MediaPipe Tasks API, pose_landmarker_lite).")
        except Exception as e:
            print(f"[PoseAnalyzer] Init failed — pose analysis disabled: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_pose(self, frame: np.ndarray, bbox: tuple) -> Dict[str, Any]:
        """
        Analyze pose for the person crop defined by bbox.
        Returns: {activity, fall_detected, confidence, keypoints}
        """
        _null = {"activity": "unknown", "fall_detected": False, "confidence": 0.0, "keypoints": None}
        if not self.available or self._landmarker is None:
            return _null

        x1, y1, x2, y2 = map(int, bbox)
        h_f, w_f = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) < 20 or (y2 - y1) < 40:
            return _null

        try:
            import mediapipe as mp
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            with self._lock:
                result = self._landmarker.detect(mp_img)

            if not result.pose_landmarks:
                return _null

            lms = result.pose_landmarks[0]   # first (and only) pose
            activity, fall, conf = self._classify(lms, crop.shape)
            return {"activity": activity, "fall_detected": fall,
                    "confidence": conf, "keypoints": lms}

        except Exception as e:
            print(f"[PoseAnalyzer] analyze_pose error: {e}")
            return _null

    # ── Internals ─────────────────────────────────────────────────────────────

    def _lm(self, lms, idx):
        """Return (x_norm, y_norm, visibility) for a landmark index."""
        lm = lms[idx]
        return lm.x, lm.y, getattr(lm, "visibility", 1.0)

    def _classify(self, lms, shape) -> tuple:
        h, w = shape[:2]

        _, nose_y,     nose_v    = self._lm(lms, _NOSE)
        _, ls_y,       ls_v      = self._lm(lms, _L_SHOULDER)
        _, rs_y,       rs_v      = self._lm(lms, _R_SHOULDER)
        lh_x, lh_y,   lh_v      = self._lm(lms, _L_HIP)
        rh_x, rh_y,   rh_v      = self._lm(lms, _R_HIP)
        _, la_y,       _         = self._lm(lms, _L_ANKLE)
        _, ra_y,       _         = self._lm(lms, _R_ANKLE)
        ls_x,  _,      _         = self._lm(lms, _L_SHOULDER)
        rs_x,  _,      _         = self._lm(lms, _R_SHOULDER)

        conf  = float(min(nose_v, lh_v, rh_v))
        nose_py     = nose_y * h
        hip_py      = ((lh_y + rh_y) / 2) * h
        shoulder_py = ((ls_y + rs_y) / 2) * h
        ankle_py    = ((la_y + ra_y) / 2) * h
        shoulder_w  = abs(ls_x - rs_x) * w

        # ── Fall: nose below hips ─────────────────────────────────────────────
        if conf >= config.fall_confidence_threshold and nose_py > hip_py:
            return "falling", True, conf

        # ── Sitting: body height ≈ shoulder width ──────────────────────────
        body_h = abs(ankle_py - shoulder_py)
        if body_h > 0 and shoulder_w > 0 and body_h / shoulder_w < 1.2:
            return "sitting", False, conf

        # ── Crouching: hips very close to ankles ─────────────────────────────
        hip_to_ankle   = abs(hip_py   - ankle_py)
        shoulder_to_hip = abs(shoulder_py - hip_py)
        if shoulder_to_hip > 0 and hip_to_ankle / max(1, shoulder_to_hip) < 0.5:
            return "crouching", False, conf

        return "standing", False, conf


# ── Singleton ─────────────────────────────────────────────────────────────────
pose_analyzer = PoseAnalyzer()
