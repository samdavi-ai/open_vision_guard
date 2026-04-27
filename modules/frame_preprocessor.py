"""
frame_preprocessor.py — Low-light and contrast enhancement for OpenVisionGuard.

Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) applied to the
luminance (L) channel in LAB color space. This is far superior to simple
histogram equalization because it limits noise amplification and preserves
regional contrast rather than applying a global curve.

Why this fixes missed detections:
  - CCTV cameras often produce washed-out or underexposed frames at night.
  - YOLO's confidence drops sharply on dark, low-contrast persons.
  - CLAHE lifts local contrast so edges and silhouettes become distinct enough
    for the detector to fire with sufficient confidence.
"""
from __future__ import annotations

import cv2
import numpy as np
from config import config


class FramePreprocessor:
    """Singleton that enhances frames before YOLO inference."""

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
        )

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE-based contrast enhancement.

        Steps:
          1. Convert BGR → LAB (separates luminance from colour)
          2. Apply CLAHE to L channel only (keeps colours natural)
          3. Merge channels and convert back to BGR
          4. Lightly denoise to reduce grain introduced by enhancement

        Returns a new array; the input is not mutated.
        """
        if not config.preprocessing_enabled:
            return frame

        # Work on a copy so the caller's frame is never mutated
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Enhance luminance only
        l_enhanced = self._clahe.apply(l_channel)

        # Light Gaussian blur on L to reduce noise amplification
        l_enhanced = cv2.GaussianBlur(l_enhanced, (3, 3), sigmaX=0.5)

        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr


# Module-level singleton — created once, reused every frame
frame_preprocessor = FramePreprocessor()
