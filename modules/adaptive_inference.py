"""
adaptive_inference.py — Dynamic AI Inference Rate Controller for OpenVisionGuard.

Key improvements over original:
  - Zero-motion gate: skips AI on truly static scenes after initial warmup.
  - Reuses preprocessor thumbnail to avoid duplicate resize/grayscale work.
  - should_run_inference() consolidates the interval + motion gate into one call.

IMPORTANT: The zero-motion gate has a warmup period of `_WARMUP_FRAMES` calls
to feed() before it activates, so the system always runs AI on the first few
frames and avoids a cold-start deadlock.
"""
from __future__ import annotations

import time
import cv2
import numpy as np
from typing import Any, Dict, List, Optional

from config import config

_WARMUP_FRAMES = 10  # Feed AI for at least this many cycles before gating


class AdaptiveInferenceController:
    """
    Dynamically adjusts AI inference interval and optionally gates inference
    on scene motion energy.
    """

    def __init__(self) -> None:
        self._activity_score: float = 0.5
        self._prev_gray: Optional[np.ndarray] = None
        self._ema_alpha = config.activity_score_ema_alpha
        self._last_energy: float = 1.0      # start HIGH so gate is open on boot
        self._last_inference_t: float = 0.0
        self._feed_count: int = 0           # warmup counter

    # ──────────────────────────────────────────────── public API ─────────

    def feed(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> None:
        """Update activity score after an AI cycle."""
        if not config.adaptive_inference_enabled:
            return

        self._feed_count += 1

        # Signal 1: person count (0 → 0.0, 5+ → 1.0)
        n_people = sum(1 for d in detections if not d.get("is_object", False))
        count_signal = min(1.0, n_people / 5.0)

        # Signal 2: average speed
        speeds = [
            float(d.get("speed", 0.0))
            for d in detections
            if not d.get("is_object", False) and isinstance(d.get("speed"), (int, float))
        ]
        speed_signal = min(1.0, (sum(speeds) / len(speeds)) / 150.0) if speeds else 0.0

        # Signal 3: frame-diff energy (reuses preprocessor thumb when available)
        energy = self._frame_diff_energy(frame)
        self._last_energy = energy

        raw_score = count_signal * 0.35 + speed_signal * 0.30 + energy * 0.35
        self._activity_score = (
            self._ema_alpha * raw_score
            + (1.0 - self._ema_alpha) * self._activity_score
        )

    def should_run_inference(self) -> bool:
        """
        Returns True when the AI thread should process a new frame.

        Logic:
          1. Interval gate  — enough wall-clock time since last AI run.
          2. Warmup bypass  — for the first _WARMUP_FRAMES cycles always allow.
          3. Zero-motion gate (after warmup) — skip if scene is truly static.
        """
        now = time.monotonic()
        if now - self._last_inference_t < self.get_interval():
            return False

        # Always run during warmup so bboxes appear immediately on stream start
        if self._feed_count < _WARMUP_FRAMES:
            self._last_inference_t = now
            return True

        # Zero-motion gate: only active after warmup
        if (
            config.zero_motion_gate_enabled
            and self._last_energy < config.zero_motion_energy_threshold
        ):
            # Scene is static — reset timer without running inference
            # (we still need to refresh should_run_inference at next call)
            return False

        self._last_inference_t = now
        return True

    def get_interval(self) -> float:
        """Return the recommended inference interval (seconds)."""
        if not config.adaptive_inference_enabled:
            return config.ai_interval_default_s
        score = max(0.0, min(1.0, self._activity_score))
        return config.ai_interval_max_s - score * (
            config.ai_interval_max_s - config.ai_interval_min_s
        )

    @property
    def activity_score(self) -> float:
        return self._activity_score

    @property
    def last_energy(self) -> float:
        return self._last_energy

    # ─────────────────────────────────────────────── internals ───────────

    def _frame_diff_energy(self, frame: np.ndarray) -> float:
        """
        Normalised frame-to-frame pixel difference (0 = no change, 1 = huge).
        Reuses frame_preprocessor's cached thumbnail when available.
        """
        try:
            from modules.frame_preprocessor import frame_preprocessor
            gray = frame_preprocessor.last_gray_thumb
        except Exception:
            gray = None

        if gray is None:
            small = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return 1.0  # treat first frame as high-energy (gate open)

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        return min(1.0, float(diff.mean()) / 40.0)


# Module-level singleton
adaptive_controller = AdaptiveInferenceController()
