"""
production_guard.py — Production Hardening Systems for OpenVisionGuard.

Contains five subsystems for real-world multi-camera deployment:

1. AdaptiveThresholdCalibrator  — per-camera rolling confidence auto-tuning
2. LatencyGuard                 — disables expensive features when latency spikes
3. SceneProfiler               — per-camera lighting/density/motion profiling
4. FalsePositiveMemory          — suppresses recurring false detection patterns
5. EdgeCaseDetector             — handles lighting changes, camera shake, noise
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import config


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ADAPTIVE THRESHOLD CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveThresholdCalibrator:
    """
    Automatically adjusts person confidence threshold per camera using
    rolling statistics from the last N detections.

    Logic:
      - If recent detections have high average confidence → tighten threshold
        (fewer false positives in easy scenes).
      - If recent detections have low average confidence → loosen threshold
        (avoid missing people in hard scenes like low-light, rain).
      - Threshold is clamped between [min_conf, max_conf] to prevent extremes.

    The calibrated threshold is read by the pipeline via get_threshold(camera_id).
    """

    def __init__(self) -> None:
        self._window_size = config.adaptive_calib_window
        self._histories: Dict[str, collections.deque] = {}
        self._thresholds: Dict[str, float] = {}

    def feed(self, camera_id: str, detections: List[Dict[str, Any]]) -> None:
        """Record confidence values from this frame's detections."""
        if not config.adaptive_calibration_enabled:
            return

        if camera_id not in self._histories:
            self._histories[camera_id] = collections.deque(maxlen=self._window_size)

        for det in detections:
            if not det.get("is_object", False):
                conf = det.get("confidence", 0.5)
                self._histories[camera_id].append(conf)

        self._recalculate(camera_id)

    def get_threshold(self, camera_id: str) -> float:
        """Return the calibrated confidence threshold for this camera."""
        if not config.adaptive_calibration_enabled:
            return config.person_conf_threshold
        return self._thresholds.get(camera_id, config.person_conf_threshold)

    def _recalculate(self, camera_id: str) -> None:
        history = self._histories.get(camera_id)
        if not history or len(history) < 20:
            return

        avg_conf = sum(history) / len(history)
        std_conf = (sum((c - avg_conf) ** 2 for c in history) / len(history)) ** 0.5

        # Target: threshold = avg - 1.5 * std (captures ~93% of valid detections)
        target = avg_conf - 1.5 * std_conf

        # Clamp
        threshold = max(
            config.adaptive_calib_min_conf,
            min(config.adaptive_calib_max_conf, target),
        )

        # Smooth with EMA to avoid jumps
        prev = self._thresholds.get(camera_id, config.person_conf_threshold)
        self._thresholds[camera_id] = 0.1 * threshold + 0.9 * prev


# ═══════════════════════════════════════════════════════════════════════════════
#  2. LATENCY GUARD SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class LatencyGuard:
    """
    Monitors per-frame processing latency and dynamically disables expensive
    features when the system falls behind the real-time target.

    Priority order for disabling (from first to last):
      1. Multi-scale 960px pass          (saves ~40ms)
      2. Small object re-inference        (saves ~20ms)
      3. ROI re-detection on track loss   (saves ~15ms)

    Features are re-enabled once latency drops below the safe threshold
    for a sustained period (hysteresis prevents oscillation).
    """

    def __init__(self) -> None:
        self._latencies: collections.deque = collections.deque(
            maxlen=config.latency_guard_window
        )
        self.multiscale_allowed = True
        self.reinference_allowed = True
        self.redetection_allowed = True
        self._suppress_count = 0

    def record(self, latency_ms: float) -> None:
        """Record one frame's processing latency."""
        if not config.latency_guard_enabled:
            return
        self._latencies.append(latency_ms)
        self._evaluate()

    def _evaluate(self) -> None:
        if len(self._latencies) < 5:
            return

        avg = sum(self._latencies) / len(self._latencies)
        target = config.latency_guard_target_ms
        critical = config.latency_guard_critical_ms

        if avg > critical:
            # Critical: disable all expensive features
            self.multiscale_allowed = False
            self.reinference_allowed = False
            self.redetection_allowed = False
            self._suppress_count = 30  # keep suppressed for 30 frames
        elif avg > target:
            # Warning: disable most expensive feature first
            self.multiscale_allowed = False
            self.reinference_allowed = True
            self.redetection_allowed = True
            self._suppress_count = 15
        else:
            # OK: re-enable after hysteresis
            if self._suppress_count > 0:
                self._suppress_count -= 1
            else:
                self.multiscale_allowed = True
                self.reinference_allowed = True
                self.redetection_allowed = True

    @property
    def status(self) -> str:
        if not self.multiscale_allowed and not self.reinference_allowed:
            return "CRITICAL"
        elif not self.multiscale_allowed:
            return "DEGRADED"
        return "NORMAL"


# ═══════════════════════════════════════════════════════════════════════════════
#  3. SCENE PROFILER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SceneProfile:
    """Per-camera scene characteristics, updated continuously."""
    avg_brightness: float = 128.0       # 0-255 mean brightness
    avg_density: float = 0.0            # avg number of persons per frame
    motion_energy: float = 0.0          # avg frame-diff energy (0-1)
    is_low_light: bool = False          # brightness < 60
    is_crowded: bool = False            # density > 5
    is_static: bool = True              # motion_energy < 0.05
    frame_count: int = 0


class SceneProfiler:
    """
    Builds per-camera scene profiles from rolling statistics.

    The profile describes lighting conditions, crowd density, and motion
    patterns so the pipeline can adapt detection parameters per-camera:
      - Low light → increase CLAHE strength, lower confidence threshold
      - Crowded → increase NMS IoU, enable aggressive dedup
      - Static → reduce inference frequency to save compute
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, SceneProfile] = {}
        self._prev_grays: Dict[str, np.ndarray] = {}

    def update(
        self,
        camera_id: str,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> None:
        """Update the scene profile for this camera."""
        if not config.scene_profiling_enabled:
            return

        profile = self._profiles.setdefault(camera_id, SceneProfile())
        profile.frame_count += 1
        alpha = 0.05  # Slow EMA for stable profiles

        # Brightness
        gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        profile.avg_brightness = alpha * brightness + (1 - alpha) * profile.avg_brightness
        profile.is_low_light = profile.avg_brightness < 60

        # Density
        n_people = sum(1 for d in detections if not d.get("is_object", False))
        profile.avg_density = alpha * n_people + (1 - alpha) * profile.avg_density
        profile.is_crowded = profile.avg_density > 5

        # Motion energy
        if camera_id in self._prev_grays:
            diff = cv2.absdiff(gray, self._prev_grays[camera_id])
            energy = float(diff.mean()) / 40.0
            energy = min(1.0, energy)
            profile.motion_energy = alpha * energy + (1 - alpha) * profile.motion_energy
            profile.is_static = profile.motion_energy < 0.05
        self._prev_grays[camera_id] = gray

    def get_profile(self, camera_id: str) -> SceneProfile:
        return self._profiles.get(camera_id, SceneProfile())

    def get_clahe_boost(self, camera_id: str) -> float:
        """Return a multiplier for CLAHE clip_limit based on scene brightness."""
        profile = self.get_profile(camera_id)
        if profile.is_low_light:
            return 1.5  # Stronger contrast enhancement in dark scenes
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  4. FALSE POSITIVE MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

class FalsePositiveMemory:
    """
    Tracks recurring false detections and suppresses repeated patterns.

    How it works:
      - The frame is divided into a grid of spatial cells.
      - Every detection that is confirmed-then-quickly-dropped (lived < N frames)
        is counted as a probable false positive in that cell.
      - If a cell accumulates > threshold false detections in the time window,
        future detections in that cell require higher confidence.

    Common false-positive sources this catches:
      - Mannequins, posters, TVs showing people
      - Reflections in glass
      - Shadows shaped like humans
    """

    def __init__(self) -> None:
        # (camera_id, cell_x, cell_y) -> list of timestamps
        self._fp_counts: Dict[Tuple[str, int, int], collections.deque] = {}
        self._grid_cols = 16
        self._grid_rows = 12

    def record_short_lived_track(
        self, camera_id: str, bbox: Tuple[int, int, int, int], timestamp: float
    ) -> None:
        """Record a detection that confirmed but was dropped quickly."""
        if not config.fp_memory_enabled:
            return

        cx = int(((bbox[0] + bbox[2]) / 2.0) / max(1, bbox[2] - bbox[0] + 200) * self._grid_cols)
        cy = int(((bbox[1] + bbox[3]) / 2.0) / max(1, bbox[3] - bbox[1] + 200) * self._grid_rows)
        cx = min(self._grid_cols - 1, max(0, cx))
        cy = min(self._grid_rows - 1, max(0, cy))

        key = (camera_id, cx, cy)
        if key not in self._fp_counts:
            self._fp_counts[key] = collections.deque(maxlen=50)
        self._fp_counts[key].append(timestamp)

    def get_confidence_boost(
        self, camera_id: str, bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Return extra confidence required for detections in this cell.
        0.0 = no boost needed, 0.1+ = require higher confidence.
        """
        if not config.fp_memory_enabled:
            return 0.0

        cx = int(((bbox[0] + bbox[2]) / 2.0) / max(1, bbox[2] - bbox[0] + 200) * self._grid_cols)
        cy = int(((bbox[1] + bbox[3]) / 2.0) / max(1, bbox[3] - bbox[1] + 200) * self._grid_rows)
        cx = min(self._grid_cols - 1, max(0, cx))
        cy = min(self._grid_rows - 1, max(0, cy))

        key = (camera_id, cx, cy)
        history = self._fp_counts.get(key)
        if not history:
            return 0.0

        now = time.time()
        window = config.fp_memory_window_s
        recent = sum(1 for t in history if now - t < window)

        if recent >= config.fp_memory_trigger_count:
            return config.fp_memory_conf_boost
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  5. EDGE CASE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeCaseDetector:
    """
    Detects abnormal conditions and adjusts detection behavior:

    1. Sudden lighting change: brightness delta > 30 between frames
       → Temporarily increase hold frames and lower thresholds to avoid
         mass track loss during the transition.

    2. Camera shake: excessive frame-to-frame motion with no real objects
       → Suppress new detections for a few frames until stabilized.

    3. Image noise: very high pixel variance with low useful content
       → Boost CLAHE and lower detection frequency temporarily.
    """

    def __init__(self) -> None:
        self._prev_brightness: Dict[str, float] = {}
        self._prev_motion: Dict[str, float] = {}
        self._shake_cooldown: Dict[str, int] = {}
        self._lighting_event: Dict[str, int] = {}

    def check(
        self,
        camera_id: str,
        frame: np.ndarray,
        n_detections: int,
    ) -> Dict[str, Any]:
        """
        Analyse the frame for edge cases. Returns an event dict:
          lighting_change: bool
          camera_shake: bool
          high_noise: bool
          suppress_new_detections: bool
          hold_boost: int (extra hold frames to add temporarily)
        """
        result = {
            "lighting_change": False,
            "camera_shake": False,
            "high_noise": False,
            "suppress_new_detections": False,
            "hold_boost": 0,
        }

        if not config.edge_case_detection_enabled:
            return result

        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        noise_level = float(gray.std())

        # 1. Sudden lighting change
        prev_b = self._prev_brightness.get(camera_id, brightness)
        delta_b = abs(brightness - prev_b)
        self._prev_brightness[camera_id] = brightness

        if delta_b > config.edge_case_brightness_delta:
            result["lighting_change"] = True
            result["hold_boost"] = 5
            self._lighting_event[camera_id] = 10  # 10 frames of protection

        # Ongoing lighting event cooldown
        if self._lighting_event.get(camera_id, 0) > 0:
            result["hold_boost"] = max(result["hold_boost"], 3)
            self._lighting_event[camera_id] -= 1

        # 2. Camera shake: high motion energy but zero detections
        diff_energy = 0.0
        if camera_id in self._prev_brightness:
            # Use simple pixel diff as proxy
            diff_energy = delta_b / 255.0
        if camera_id not in self._prev_motion:
            self._prev_motion[camera_id] = 0.0

        prev_gray_path = f"_edge_{camera_id}"
        # Approximate shake: large brightness change + high noise + no detections
        if delta_b > 15 and n_detections == 0 and noise_level > 50:
            result["camera_shake"] = True
            result["suppress_new_detections"] = True
            self._shake_cooldown[camera_id] = 5

        if self._shake_cooldown.get(camera_id, 0) > 0:
            result["suppress_new_detections"] = True
            self._shake_cooldown[camera_id] -= 1

        # 3. High noise: very high stddev with low brightness
        if noise_level > 60 and brightness < 40:
            result["high_noise"] = True

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

threshold_calibrator = AdaptiveThresholdCalibrator()
latency_guard = LatencyGuard()
scene_profiler = SceneProfiler()
fp_memory = FalsePositiveMemory()
edge_detector = EdgeCaseDetector()
