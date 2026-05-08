"""
LuggageTracker — Stable luggage ownership tracking with false-positive prevention.

Key design decisions:
  - Luggage IDs are stable cluster centroids, NOT per-frame pixel positions.
    Objects within `cluster_radius` pixels are treated as the same item.
  - Ownership is committed only after `min_owner_frames` consecutive frames
    with the same nearest person (prevents 1-frame flickering ownership swaps).
  - Transfer alerts fire only after `min_ownership_s` seconds of confirmed ownership
    AND respect a `transfer_cooldown_s` per-item cooldown.
  - Abandonment check requires `abandon_threshold_s` seconds without a nearby person.
"""

import time
import math
from typing import Dict, List, Any, Optional, Tuple


class LuggageTracker:
    def __init__(self):
        # Stable luggage registry: stable_id -> {owner_id, type, centroid, status, ...}
        self._items: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1

        # Config
        self.cluster_radius      = 80    # px  — objects within this dist = same item
        self.link_distance       = 160   # px  — max person-to-item dist to claim ownership
        self.abandon_threshold_s = 12.0  # sec — how long unattended before "abandoned"
        self.min_ownership_s     = 3.0   # sec — must own item this long before transfer fires
        self.transfer_cooldown_s = 30.0  # sec — min gap between transfer alerts for same item
        self.min_owner_frames    = 5     # frames — consecutive frames needed to commit ownership

        # Per-item candidate tracking: stable_id -> (candidate_pid, frame_count)
        self._owner_candidates: Dict[str, Tuple[Optional[str], int]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    def _find_or_create_item(self, cx: float, cy: float, obj_type: str, now: float) -> str:
        """
        Find the existing luggage item nearest to (cx, cy) within cluster_radius,
        or create a new stable item entry and return its ID.
        """
        best_id  = None
        best_dist = float("inf")

        for sid, info in self._items.items():
            if info["type"] != obj_type:
                continue
            ic, jc = info["centroid"]
            d = math.hypot(cx - ic, cy - jc)
            if d < self.cluster_radius and d < best_dist:
                best_id, best_dist = sid, d

        if best_id:
            # Update centroid with exponential moving average (0.8 old + 0.2 new)
            oc, oj = self._items[best_id]["centroid"]
            self._items[best_id]["centroid"] = (oc * 0.8 + cx * 0.2, oj * 0.8 + cy * 0.2)
            return best_id

        # New item
        new_id = f"Luggage_{self._next_id:04d}"
        self._next_id += 1
        self._items[new_id] = {
            "type":           obj_type,
            "centroid":       (cx, cy),
            "owner_id":       None,
            "owner_since":    None,
            "status":         "unattended",
            "last_seen":      now,
            "abandoned_time": None,
            "last_transfer_alert": 0.0,
        }
        self._owner_candidates[new_id] = (None, 0)
        return new_id

    # ─────────────────────────────────────────────────────────────────────────
    def update(
        self,
        object_detections: List[Dict[str, Any]],
        person_positions:  Dict[str, tuple],
        current_time:      float,
    ) -> Dict[str, Any]:
        """
        Update luggage ownership for one frame.
        Returns {"events": [...]} with only genuine events.
        """
        events: List[Dict[str, Any]] = []
        seen_ids = set()

        for obj in object_detections:
            if obj["type"] not in ("backpack", "handbag", "suitcase"):
                continue

            cx, cy = obj["center"]
            sid = self._find_or_create_item(cx, cy, obj["type"], current_time)
            info = self._items[sid]
            info["last_seen"] = current_time
            seen_ids.add(sid)

            # ── Find nearest person within link_distance ───────────────────
            nearest_pid  = None
            nearest_dist = float("inf")
            for pid, (px, py) in person_positions.items():
                d = math.hypot(cx - px, cy - py)
                if d < self.link_distance and d < nearest_dist:
                    nearest_dist = d
                    nearest_pid  = pid

            # ── Candidate ownership accumulation ──────────────────────────
            prev_candidate, frame_count = self._owner_candidates.get(sid, (None, 0))

            if nearest_pid == prev_candidate:
                frame_count += 1
            else:
                # New candidate; reset counter
                prev_candidate = nearest_pid
                frame_count = 1

            self._owner_candidates[sid] = (prev_candidate, frame_count)

            # ── Commit ownership once candidate is stable ──────────────────
            if nearest_pid and frame_count >= self.min_owner_frames:
                committed_owner = nearest_pid

                if info["owner_id"] is None:
                    # First owner — just record, no alert
                    info["owner_id"]    = committed_owner
                    info["owner_since"] = current_time
                    info["status"]      = "carried"

                elif info["owner_id"] != committed_owner:
                    # Potential transfer — validate conditions
                    ownership_duration = current_time - (info["owner_since"] or current_time)
                    cooldown_elapsed   = current_time - info["last_transfer_alert"]

                    if (ownership_duration >= self.min_ownership_s
                            and cooldown_elapsed >= self.transfer_cooldown_s):
                        events.append({
                            "type":         "luggage_transferred",
                            "owner_id":     info["owner_id"],
                            "new_holder":   committed_owner,
                            "luggage_type": info["type"],
                            "luggage_id":   sid,
                        })
                        info["last_transfer_alert"] = current_time

                    # Update owner regardless (to stay current)
                    info["owner_id"]    = committed_owner
                    info["owner_since"] = current_time
                    info["status"]      = "carried"

            elif not nearest_pid:
                # No person nearby — check for abandonment
                if info["status"] == "carried":
                    info["status"]         = "unattended"
                    info["abandoned_time"] = current_time

                elif (info["status"] == "unattended"
                        and info["abandoned_time"] is not None
                        and current_time - info["abandoned_time"] >= self.abandon_threshold_s):
                    # Confirmed abandoned
                    info["status"] = "abandoned"
                    events.append({
                        "type":         "luggage_abandoned",
                        "owner_id":     info["owner_id"] or "unknown",
                        "luggage_type": info["type"],
                        "luggage_id":   sid,
                    })

        # ── Expire items not seen for > 60s ───────────────────────────────
        stale = [sid for sid, info in self._items.items()
                 if current_time - info["last_seen"] > 60.0]
        for sid in stale:
            self._items.pop(sid, None)
            self._owner_candidates.pop(sid, None)

        return {"events": events}

    # ─────────────────────────────────────────────────────────────────────────
    def get_person_luggage(self, person_id: str) -> Dict[str, Any]:
        """Returns all luggage currently linked to a person."""
        return {
            sid: {"type": info["type"], "status": info["status"]}
            for sid, info in self._items.items()
            if info["owner_id"] == person_id
        }


luggage_tracker = LuggageTracker()
