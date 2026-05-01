"""
detection_memory.py — Detection Memory Layer for OpenVisionGuard.

Stores last-known detections between AI inference pulses so bounding boxes
never disappear from the display loop. Uses real-timestamp velocity instead
of a hardcoded frame-count assumption.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple


class DetectionMemory:
    """
    TTL-based in-memory store for last-known detections.

    Parameters
    ----------
    ttl_frames : int
        Video frames a detection survives without a fresh AI update.
    interpolate : bool
        Shift box positions by estimated velocity each tick.
    video_fps : float
        Expected display FPS — used to convert TTL frames to seconds.
    """

    def __init__(
        self,
        ttl_frames: int = 9,
        interpolate: bool = True,
        video_fps: float = 25.0,
    ) -> None:
        self._ttl = ttl_frames
        self._interpolate = interpolate
        self._ttl_seconds = ttl_frames / max(1.0, video_fps)
        # global_id → {det, expire_at, velocity, last_ts}
        self._store: Dict[str, Dict[str, Any]] = {}

    # ─────────────────────────────────────────────────── public API ──────

    def update(self, detections: List[Dict[str, Any]]) -> None:
        """Ingest a fresh batch of confirmed detections from the pipeline."""
        now = time.monotonic()
        seen_ids = set()

        for det in detections:
            gid = det.get("global_id")
            if not gid:
                continue
            seen_ids.add(gid)

            prev = self._store.get(gid)
            velocity = (
                self._compute_velocity(prev, det, now) if prev else (0.0, 0.0, 0.0, 0.0)
            )

            self._store[gid] = {
                "det": det,
                "expire_at": now + self._ttl_seconds,
                "velocity": velocity,
                "last_ts": now,
            }

        # Prune stale entries for tracks no longer seen by YOLO
        for gid in list(self._store.keys()):
            if gid not in seen_ids:
                if now >= self._store[gid]["expire_at"]:
                    del self._store[gid]

    def get_active(self) -> List[Dict[str, Any]]:
        """Return all live detections, applying position interpolation."""
        now = time.monotonic()
        results = []
        for gid, entry in list(self._store.items()):
            if now >= entry["expire_at"]:
                del self._store[gid]
                continue
            det = dict(entry["det"])  # shallow copy — never mutate store
            if self._interpolate and entry["velocity"] != (0.0, 0.0, 0.0, 0.0):
                det = self._shift_bbox(det, entry["velocity"])
            results.append(det)
        return results

    def tick(self) -> None:
        """
        Called every video frame between AI pulses. Removes expired entries.
        With time-based TTL this is lightweight — just one comparison per track.
        """
        now = time.monotonic()
        for gid in list(self._store.keys()):
            if now >= self._store[gid]["expire_at"]:
                del self._store[gid]

    def clear(self, gid: str) -> None:
        """Explicitly remove a specific identity (e.g. on confirmed exit)."""
        self._store.pop(gid, None)

    # ─────────────────────────────────────────────────── helpers ─────────

    @staticmethod
    def _compute_velocity(
        prev: Dict[str, Any],
        current: Dict[str, Any],
        now: float,
    ) -> Tuple[float, float, float, float]:
        """
        Estimate per-frame velocity using real elapsed time.

        Previously this used a hardcoded `frames_elapsed = max(1, 3)` which
        broke prediction whenever adaptive inference changed the AI cadence.
        Now we measure actual seconds elapsed and divide by the display FPS
        assumption (25 FPS) to get per-video-frame pixel shift.
        """
        pb = prev["det"].get("bbox", [0, 0, 0, 0])
        cb = current.get("bbox", [0, 0, 0, 0])
        if len(pb) < 4 or len(cb) < 4:
            return (0.0, 0.0, 0.0, 0.0)

        elapsed_s = max(0.033, now - prev.get("last_ts", now - 0.1))
        # Convert pixel delta over elapsed seconds → pixels per video frame
        frames_elapsed = max(1.0, elapsed_s * 25.0)  # 25 = display FPS assumption

        return (
            (cb[0] - pb[0]) / frames_elapsed,
            (cb[1] - pb[1]) / frames_elapsed,
            (cb[2] - pb[2]) / frames_elapsed,
            (cb[3] - pb[3]) / frames_elapsed,
        )

    @staticmethod
    def _shift_bbox(det: Dict[str, Any], vel: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Apply one frame of velocity to the bounding box."""
        x1, y1, x2, y2 = det["bbox"]
        det["bbox"] = [
            max(0, int(x1 + vel[0])),
            max(0, int(y1 + vel[1])),
            max(0, int(x2 + vel[2])),
            max(0, int(y2 + vel[3])),
        ]
        return det


# Module-level singleton
detection_memory = DetectionMemory(ttl_frames=9, interpolate=True)
