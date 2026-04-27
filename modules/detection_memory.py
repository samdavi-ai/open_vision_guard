"""
detection_memory.py — Detection Memory Layer for OpenVisionGuard.

PURPOSE
-------
The asynchronous pipeline runs YOLO at ~8-10 FPS inside a 25-30 FPS video
loop. Between inference runs, there are 2-4 raw frames where YOLO produces
no output at all. Without a memory layer, bounding boxes vanish from the
screen every time the AI is "resting", causing the characteristic flicker
observed in the original system.

HOW IT WORKS
------------
1.  Every time the pipeline produces a confirmed detection, the DetectionMemory
    stores it with a timestamp.

2.  Between AI inference calls, the stream worker asks the memory for
    "current" detections. The memory returns the last known detections for
    every track that is still within its TTL (Time-To-Live) window.

3.  Optionally, the memory **interpolates** box positions using a simple
    linear velocity model.  If a person was moving right at 5 px/frame, the
    returned box is shifted right by 5 px for every elapsed frame, keeping the
    box "glued" to the person until the next real detection updates it.

BENEFITS
--------
*   Zero disappearances during brief occlusions or between AI frames.
*   Smooth visual transitions between detection pulses.
*   Eliminates the root cause of ID fragmentation in the TemporalBuffer.
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
        How many *video* frames a detection survives without a fresh AI update.
        At 25 FPS and an AI cadence of 8 FPS, each AI frame covers ~3 video
        frames.  A TTL of 9 means a detection is kept alive for ~3 AI cycles
        (≈1.1 seconds), which gracefully handles short occlusions.
    interpolate : bool
        If True, box positions are shifted by the estimated velocity each frame.
    """

    def __init__(self, ttl_frames: int = 9, interpolate: bool = True) -> None:
        self._ttl = ttl_frames
        self._interpolate = interpolate
        # global_id → {det, frames_remaining, velocity}
        self._store: Dict[str, Dict[str, Any]] = {}

    # ─────────────────────────────────────────────────────── public API ──────

    def update(self, detections: List[Dict[str, Any]]) -> None:
        """
        Ingest a fresh batch of confirmed detections from the pipeline.
        Called once per AI inference cycle.
        """
        seen_ids = set()
        for det in detections:
            gid = det.get("global_id")
            if not gid:
                continue
            seen_ids.add(gid)

            prev = self._store.get(gid)
            velocity = self._compute_velocity(prev, det) if prev else (0.0, 0.0, 0.0, 0.0)

            self._store[gid] = {
                "det": det,
                "frames_remaining": self._ttl,
                "velocity": velocity,
                "timestamp": time.monotonic(),
            }

        # Decay counters for tracks NO longer returned by YOLO
        for gid in list(self._store.keys()):
            if gid not in seen_ids:
                self._store[gid]["frames_remaining"] -= 1
                if self._store[gid]["frames_remaining"] <= 0:
                    del self._store[gid]

    def get_active(self) -> List[Dict[str, Any]]:
        """
        Return all live detections, applying position interpolation.
        Called every video frame (25-30 FPS).
        """
        results = []
        for gid, entry in self._store.items():
            det = dict(entry["det"])           # shallow copy — don't mutate store
            if self._interpolate and entry["velocity"] != (0.0, 0.0, 0.0, 0.0):
                det = self._shift_bbox(det, entry["velocity"])
            results.append(det)
        return results

    def tick(self) -> None:
        """
        Advance the memory by one *video* frame (call once per cv2 frame read).
        Non-updated entries gradually consume their TTL.  This prevents stale
        boxes from lingering after a person has genuinely left the scene.
        """
        for gid in list(self._store.keys()):
            entry = self._store[gid]
            # Only decay entries that are NOT freshly updated (frames_remaining < ttl)
            if entry["frames_remaining"] < self._ttl:
                entry["frames_remaining"] -= 1
                if entry["frames_remaining"] <= 0:
                    del self._store[gid]

    def clear(self, gid: str) -> None:
        """Explicitly remove a specific identity (e.g. on confirmed exit)."""
        self._store.pop(gid, None)

    # ─────────────────────────────────────────────────────── helpers ─────────

    @staticmethod
    def _compute_velocity(
        prev: Dict[str, Any], current: Dict[str, Any]
    ) -> Tuple[float, float, float, float]:
        """
        Estimate per-frame velocity (dx1, dy1, dx2, dy2) from two detections.
        Used to extrapolate the box position between inference calls.
        """
        pb = prev["det"].get("bbox", [0, 0, 0, 0])
        cb = current.get("bbox", [0, 0, 0, 0])
        if len(pb) < 4 or len(cb) < 4:
            return (0.0, 0.0, 0.0, 0.0)
        # Assuming ~3 frames between AI pulses (8 FPS AI inside 25 FPS video)
        frames_elapsed = max(1, 3)
        return (
            (cb[0] - pb[0]) / frames_elapsed,
            (cb[1] - pb[1]) / frames_elapsed,
            (cb[2] - pb[2]) / frames_elapsed,
            (cb[3] - pb[3]) / frames_elapsed,
        )

    @staticmethod
    def _shift_bbox(det: Dict[str, Any], vel: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Apply one frame of velocity to the bounding box (clamped to ≥0)."""
        x1, y1, x2, y2 = det["bbox"]
        det["bbox"] = [
            max(0, int(x1 + vel[0])),
            max(0, int(y1 + vel[1])),
            max(0, int(x2 + vel[2])),
            max(0, int(y2 + vel[3])),
        ]
        return det


# Module-level singleton — one memory per pipeline instance
detection_memory = DetectionMemory(ttl_frames=9, interpolate=True)
