"""
adaptive_inference.py — Dynamic AI Inference Rate Controller for OpenVisionGuard.

PURPOSE
-------
In a surveillance system, the scene is often static for long periods (empty
hallway, quiet parking lot) punctuated by bursts of activity (people entering,
running, congregating).  Running YOLO at a fixed 8-10 FPS wastes GPU/CPU cycles
during quiet moments and may miss fast action during busy moments.

The AdaptiveInferenceController solves this by mapping a real-time **activity
score** to an inference interval:

    Low activity  (0.0 - 0.3) -> 5 FPS  (200 ms interval)  -- saves power
    Normal        (0.3 - 0.7) -> 10 FPS (100 ms interval)  -- balanced
    High activity (0.7 - 1.0) -> 15 FPS (66 ms interval)   -- max responsiveness

ACTIVITY SCORING
----------------
The score is a 0-1 float computed from three signals:
  1. **Detection count**: more tracked people = higher activity
  2. **Average speed**: faster objects = higher urgency
  3. **Frame diff energy**: raw pixel change between consecutive frames
     (catches new entrances, sudden movements, lighting changes)

All three signals are normalised and combined with equal weight, then smoothed
with an EMA to avoid the inference rate oscillating erratically.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, List, Optional

from config import config


class AdaptiveInferenceController:
    """
    Dynamically adjusts AI inference interval based on scene activity.

    Usage in the stream worker:

        controller = AdaptiveInferenceController()

        # After each AI cycle:
        controller.feed(detections, frame)

        # Before deciding whether to run AI:
        interval = controller.get_interval()
        if now - last_ai_time > interval:
            run_ai()
    """

    def __init__(self) -> None:
        self._activity_score: float = 0.5         # start neutral
        self._prev_gray: Optional[np.ndarray] = None
        self._ema_alpha = config.activity_score_ema_alpha

    # ────────────────────────────────────────────────────── public API ───────

    def feed(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray,
    ) -> None:
        """
        Update the activity score after an AI cycle.

        Parameters
        ----------
        detections : list
            Current confirmed detections (from pipeline result).
        frame : np.ndarray
            The raw BGR frame that was just processed.
        """
        if not config.adaptive_inference_enabled:
            return

        # Signal 1: detection count (normalised: 0 people=0, 5+ people=1)
        n_people = sum(1 for d in detections if not d.get("is_object", False))
        count_signal = min(1.0, n_people / 5.0)

        # Signal 2: average speed of person detections
        speeds = []
        for d in detections:
            if not d.get("is_object", False):
                speed = d.get("speed", 0.0)
                if isinstance(speed, (int, float)):
                    speeds.append(float(speed))
        avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
        # Normalise: 0 px/s=0, 150+ px/s=1
        speed_signal = min(1.0, avg_speed / 150.0)

        # Signal 3: frame-difference energy (cheap motion detector)
        diff_signal = self._frame_diff_energy(frame)

        # Combine with equal weights
        raw_score = (count_signal * 0.35 + speed_signal * 0.30 + diff_signal * 0.35)

        # Smooth with EMA
        self._activity_score = (
            self._ema_alpha * raw_score
            + (1.0 - self._ema_alpha) * self._activity_score
        )

    def get_interval(self) -> float:
        """
        Return the recommended inference interval (seconds).

        Maps activity_score linearly between ai_interval_max_s (low activity)
        and ai_interval_min_s (high activity).
        """
        if not config.adaptive_inference_enabled:
            return config.ai_interval_default_s

        # Linear interpolation: high score -> short interval (fast AI)
        score = max(0.0, min(1.0, self._activity_score))
        interval = (
            config.ai_interval_max_s
            - score * (config.ai_interval_max_s - config.ai_interval_min_s)
        )
        return interval

    @property
    def activity_score(self) -> float:
        """Current smoothed activity score (0-1)."""
        return self._activity_score

    # ────────────────────────────────────────────────────── internals ────────

    def _frame_diff_energy(self, frame: np.ndarray) -> float:
        """
        Compute normalised frame-to-frame pixel difference.

        Uses a small grayscale + resize for speed (< 0.5 ms per frame).
        Returns 0.0 for no change, 1.0 for massive change.
        """
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray

        # Mean pixel change normalised to 0-1 (threshold at 40 = big change)
        energy = float(diff.mean()) / 40.0
        return min(1.0, energy)


# Module-level singleton
adaptive_controller = AdaptiveInferenceController()
