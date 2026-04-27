"""Temporal behavior analysis for tracked people."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .types import Detection, TrackBehavior


@dataclass
class TrackState:
    first_seen: float
    last_seen: float
    samples: Deque[Tuple[float, float, float, float]] = field(default_factory=lambda: deque(maxlen=90))
    stationary_since: Optional[float] = None


class TemporalBehaviorAnalyzer:
    """Detects loitering, running, pacing, and abnormal motion from ByteTrack IDs."""

    def __init__(
        self,
        history_size: int = 90,
        loitering_seconds: float = 30.0,
        loitering_radius_px: float = 45.0,
        running_height_per_s: float = 2.4,
        abnormal_accel_px_s2: float = 900.0,
    ) -> None:
        self.history_size = history_size
        self.loitering_seconds = loitering_seconds
        self.loitering_radius_px = loitering_radius_px
        self.running_height_per_s = running_height_per_s
        self.abnormal_accel_px_s2 = abnormal_accel_px_s2
        self.states: Dict[int, TrackState] = {}

    def update(self, detections: List[Detection], timestamp: float) -> Dict[int, TrackBehavior]:
        behaviors: Dict[int, TrackBehavior] = {}
        for det in detections:
            if not det.is_person or det.track_id is None:
                continue
            track_id = int(det.track_id)
            cx, cy = det.center
            state = self.states.get(track_id)
            if state is None:
                state = TrackState(first_seen=timestamp, last_seen=timestamp, samples=deque(maxlen=self.history_size))
                self.states[track_id] = state

            speed_px_s = self._speed(state, cx, cy, timestamp)
            state.samples.append((cx, cy, timestamp, det.height))
            state.last_seen = timestamp
            behaviors[track_id] = self._classify(track_id, state, speed_px_s)
        self._cleanup(timestamp)
        return behaviors

    @staticmethod
    def _speed(state: TrackState, cx: float, cy: float, timestamp: float) -> float:
        if not state.samples:
            return 0.0
        px, py, pt, _ = state.samples[-1]
        dt = max(1e-3, timestamp - pt)
        return float(np.hypot(cx - px, cy - py) / dt)

    def _classify(self, track_id: int, state: TrackState, speed_px_s: float) -> TrackBehavior:
        samples = list(state.samples)
        dwell = state.last_seen - state.first_seen
        if len(samples) < 4:
            return TrackBehavior(track_id, "warming_up", 0.0, speed_px_s, dwell)

        heights = [max(1.0, s[3]) for s in samples[-10:]]
        median_height = float(np.median(heights))
        normalized_speed = speed_px_s / median_height
        reasons: List[str] = []
        label = "normal"
        score = 0.0

        if normalized_speed >= self.running_height_per_s:
            label = "running"
            score = min(85.0, 40.0 + normalized_speed * 15.0)
            reasons.append(f"speed={normalized_speed:.2f}_body_lengths_per_s")

        abnormal_score = self._abnormal_motion_score(samples)
        if abnormal_score > score:
            label = "sudden_abnormal_motion"
            score = abnormal_score
            reasons.append("large_acceleration_or_direction_change")

        pacing_score = self._pacing_score(samples)
        if pacing_score > score:
            label = "pacing"
            score = pacing_score
            reasons.append("repeated_back_and_forth_path")

        loitering_score = self._loitering_score(state)
        if loitering_score > score:
            label = "loitering"
            score = loitering_score
            reasons.append("stationary_dwell_time")

        return TrackBehavior(track_id, label, float(score), speed_px_s, dwell, reasons)

    def _loitering_score(self, state: TrackState) -> float:
        samples = list(state.samples)
        if len(samples) < 10:
            return 0.0
        points = np.array([(s[0], s[1]) for s in samples[-min(len(samples), 45):]], dtype=np.float32)
        center = points.mean(axis=0)
        radius = float(np.linalg.norm(points - center, axis=1).mean())

        if radius <= self.loitering_radius_px:
            if state.stationary_since is None:
                state.stationary_since = samples[-1][2]
            duration = state.last_seen - state.stationary_since
            if duration >= self.loitering_seconds:
                return min(95.0, 45.0 + duration)
            return min(35.0, duration / self.loitering_seconds * 35.0)

        state.stationary_since = None
        return 0.0

    @staticmethod
    def _pacing_score(samples: List[Tuple[float, float, float, float]]) -> float:
        if len(samples) < 16:
            return 0.0
        points = np.array([(s[0], s[1]) for s in samples[-40:]], dtype=np.float32)
        deltas = np.diff(points, axis=0)
        path = float(np.linalg.norm(deltas, axis=1).sum())
        displacement = float(np.linalg.norm(points[-1] - points[0]))
        if path < 120.0 or displacement <= 1.0:
            return 0.0

        axis = 0 if np.std(points[:, 0]) >= np.std(points[:, 1]) else 1
        signs = np.sign(deltas[:, axis])
        signs = signs[np.abs(signs) > 0]
        reversals = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
        path_ratio = path / max(displacement, 1.0)
        if reversals >= 3 and path_ratio >= 2.0:
            return min(70.0, 30.0 + reversals * 6.0 + path_ratio * 4.0)
        return 0.0

    def _abnormal_motion_score(self, samples: List[Tuple[float, float, float, float]]) -> float:
        if len(samples) < 6:
            return 0.0
        points = np.array([(s[0], s[1]) for s in samples[-12:]], dtype=np.float32)
        times = np.array([s[2] for s in samples[-12:]], dtype=np.float32)
        deltas = np.diff(points, axis=0)
        dts = np.maximum(np.diff(times), 1e-3)
        speeds = np.linalg.norm(deltas, axis=1) / dts
        if len(speeds) < 3:
            return 0.0
        accels = np.abs(np.diff(speeds)) / np.maximum(dts[1:], 1e-3)
        max_accel = float(np.max(accels))
        if max_accel >= self.abnormal_accel_px_s2:
            return min(75.0, 35.0 + max_accel / self.abnormal_accel_px_s2 * 20.0)
        return 0.0

    def _cleanup(self, timestamp: float, stale_seconds: float = 15.0) -> None:
        stale = [tid for tid, state in self.states.items() if timestamp - state.last_seen > stale_seconds]
        for tid in stale:
            del self.states[tid]
