"""
Luggage Tracker Module
Tracks luggage objects (backpack, suitcase, handbag), maintains persistent ownership,
detects abandonment and theft.
"""
import time
import math
from typing import Dict, Any, List, Optional, Tuple


class LuggageTracker:
    """Persistent luggage ownership tracking with abandonment and theft detection."""

    # Luggage COCO class IDs
    LUGGAGE_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}

    # Thresholds
    OWNERSHIP_LINK_DISTANCE = 200        # px — max distance to link luggage to person
    ABANDON_DISTANCE = 200               # px — owner distance to consider abandonment
    ABANDON_TIME_THRESHOLD = 10.0        # seconds — duration before marking abandoned
    THEFT_DISTANCE = 100                 # px — distance for new person to "pick up"

    def __init__(self):
        # luggage_id → {owner_id, type, first_seen, last_pos, status, abandon_timer_start, last_update}
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._next_luggage_id = 1

    def update(self, object_detections: List[Dict[str, Any]],
               person_positions: Dict[str, Tuple[float, float]],
               current_time: float) -> Dict[str, Any]:
        """
        Update luggage tracking with current frame data.

        Args:
            object_detections: List of {type, center: (cx, cy), bbox, cls_id}
            person_positions: {person_id: (cx, cy)}
            current_time: Current timestamp

        Returns:
            {
                "luggage_items": {luggage_id: {owner_id, type, status, ...}},
                "events": [{"type": "abandoned"|"theft"|"transferred", ...}]
            }
        """
        events = []

        # Track which luggage items we've seen this frame
        seen_luggage_ids = set()

        for obj in object_detections:
            obj_cx, obj_cy = obj["center"]
            obj_type = obj["type"]

            # Try to match to existing luggage by proximity
            matched_lid = self._match_existing_luggage(obj_cx, obj_cy, obj_type)

            if matched_lid:
                seen_luggage_ids.add(matched_lid)
                luggage = self._registry[matched_lid]
                luggage["last_pos"] = (obj_cx, obj_cy)
                luggage["last_update"] = current_time
            else:
                # Create new luggage entry
                matched_lid = f"LUG_{self._next_luggage_id:03d}"
                self._next_luggage_id += 1
                seen_luggage_ids.add(matched_lid)

                # Find nearest person as initial owner
                nearest_pid, nearest_dist = self._find_nearest_person(
                    obj_cx, obj_cy, person_positions
                )

                self._registry[matched_lid] = {
                    "owner_id": nearest_pid,
                    "type": obj_type,
                    "first_seen": current_time,
                    "last_pos": (obj_cx, obj_cy),
                    "status": "carried" if nearest_pid else "unowned",
                    "abandon_timer_start": None,
                    "last_update": current_time,
                }

            # ── Check ownership status ──
            luggage = self._registry[matched_lid]
            owner_id = luggage["owner_id"]

            if owner_id and owner_id in person_positions:
                owner_cx, owner_cy = person_positions[owner_id]
                owner_dist = math.sqrt(
                    (obj_cx - owner_cx) ** 2 + (obj_cy - owner_cy) ** 2
                )

                if owner_dist < self.OWNERSHIP_LINK_DISTANCE * 0.5:
                    # Owner is close — carried
                    luggage["status"] = "carried"
                    luggage["abandon_timer_start"] = None
                elif owner_dist < self.ABANDON_DISTANCE:
                    # Owner nearby but not carrying
                    luggage["status"] = "nearby"
                    luggage["abandon_timer_start"] = None
                else:
                    # Owner is far — start/continue abandon timer
                    if luggage["abandon_timer_start"] is None:
                        luggage["abandon_timer_start"] = current_time
                    elif (current_time - luggage["abandon_timer_start"]) > self.ABANDON_TIME_THRESHOLD:
                        if luggage["status"] != "abandoned":
                            luggage["status"] = "abandoned"
                            events.append({
                                "type": "luggage_abandoned",
                                "luggage_id": matched_lid,
                                "luggage_type": obj_type,
                                "owner_id": owner_id,
                                "position": (obj_cx, obj_cy),
                                "time": current_time,
                            })

                # ── Check for theft: another person near the luggage ──
                for pid, (pcx, pcy) in person_positions.items():
                    if pid == owner_id:
                        continue
                    pickup_dist = math.sqrt(
                        (obj_cx - pcx) ** 2 + (obj_cy - pcy) ** 2
                    )
                    if pickup_dist < self.THEFT_DISTANCE and luggage["status"] in ("abandoned", "nearby"):
                        prev_owner = luggage["owner_id"]
                        luggage["owner_id"] = pid
                        luggage["status"] = "transferred"
                        luggage["abandon_timer_start"] = None
                        events.append({
                            "type": "luggage_theft",
                            "luggage_id": matched_lid,
                            "luggage_type": obj_type,
                            "previous_owner": prev_owner,
                            "new_holder": pid,
                            "position": (obj_cx, obj_cy),
                            "time": current_time,
                        })
                        break

            elif owner_id and owner_id not in person_positions:
                # Owner left the frame entirely
                if luggage["abandon_timer_start"] is None:
                    luggage["abandon_timer_start"] = current_time
                elif (current_time - luggage["abandon_timer_start"]) > self.ABANDON_TIME_THRESHOLD:
                    if luggage["status"] != "abandoned":
                        luggage["status"] = "abandoned"
                        events.append({
                            "type": "luggage_abandoned",
                            "luggage_id": matched_lid,
                            "luggage_type": obj_type,
                            "owner_id": owner_id,
                            "position": (obj_cx, obj_cy),
                            "time": current_time,
                        })

        return {
            "luggage_items": {lid: self._format_luggage(lid) for lid in self._registry},
            "events": events,
        }

    def get_person_luggage(self, person_id: str) -> Dict[str, Any]:
        """Get all luggage linked to a person."""
        result = {}
        for lid, data in self._registry.items():
            if data["owner_id"] == person_id:
                result[lid] = {
                    "type": data["type"],
                    "status": data["status"],
                }
        return result

    def _match_existing_luggage(self, cx: float, cy: float, obj_type: str) -> Optional[str]:
        """Try to match a detection to an existing luggage item by proximity and type."""
        best_lid = None
        best_dist = float("inf")

        for lid, data in self._registry.items():
            if data["type"] != obj_type:
                continue
            lx, ly = data["last_pos"]
            dist = math.sqrt((cx - lx) ** 2 + (cy - ly) ** 2)
            if dist < 80 and dist < best_dist:  # within 80px = same luggage
                best_dist = dist
                best_lid = lid

        return best_lid

    def _find_nearest_person(self, cx: float, cy: float,
                             person_positions: Dict[str, Tuple[float, float]]) -> Tuple[Optional[str], float]:
        """Find the nearest person to a position."""
        nearest_pid = None
        min_dist = float("inf")

        for pid, (pcx, pcy) in person_positions.items():
            dist = math.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
            if dist < min_dist and dist < self.OWNERSHIP_LINK_DISTANCE:
                min_dist = dist
                nearest_pid = pid

        return nearest_pid, min_dist

    def _format_luggage(self, lid: str) -> Dict[str, Any]:
        data = self._registry[lid]
        return {
            "luggage_id": lid,
            "owner_id": data["owner_id"],
            "type": data["type"],
            "status": data["status"],
            "position": data["last_pos"],
        }


# Singleton
luggage_tracker = LuggageTracker()
