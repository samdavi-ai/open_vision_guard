"""Motion gating for low-power edge inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class MotionDecision:
    should_process: bool
    motion_ratio: float
    mask: np.ndarray
    reason: str


class MotionGate:
    """OpenCV MOG2 gate to avoid detector calls on static frames."""

    def __init__(
        self,
        history: int = 400,
        var_threshold: int = 24,
        min_motion_ratio: float = 0.004,
        warmup_frames: int = 8,
        resize_width: int = 320,
        process_every: int = 2,
    ) -> None:
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )
        self.min_motion_ratio = min_motion_ratio
        self.warmup_frames = warmup_frames
        self.resize_width = resize_width
        self.process_every = max(1, process_every)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _prepare(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if w <= self.resize_width:
            return frame, 1.0
        scale = self.resize_width / float(w)
        resized = cv2.resize(frame, (self.resize_width, int(h * scale)))
        return resized, scale

    def update(self, frame: np.ndarray, frame_index: int) -> MotionDecision:
        small, _ = self._prepare(frame)
        mask = self.bg.apply(small)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        motion_pixels = int(cv2.countNonZero(mask))
        motion_ratio = motion_pixels / float(mask.shape[0] * mask.shape[1])

        if frame_index < self.warmup_frames:
            return MotionDecision(True, motion_ratio, mask, "warmup")
        if frame_index % self.process_every != 0:
            return MotionDecision(False, motion_ratio, mask, "frame_skip")
        if motion_ratio >= self.min_motion_ratio:
            return MotionDecision(True, motion_ratio, mask, "scheduled_motion")
        return MotionDecision(False, motion_ratio, mask, "static")
