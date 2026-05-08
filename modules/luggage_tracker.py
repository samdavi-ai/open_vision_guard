"""
Airport-Grade Luggage Intelligence Tracker
==========================================

Tracks every detected bag (backpack, handbag, suitcase, laptop, phone) from
first appearance through exit, maintaining a full event timeline suitable for
international airport security review.

Key concepts
------------
BagRecord
    Represents a single physical bag with a YOLO-stable track_id.  Maintains
    current ownership, state (CARRIED | PUT_DOWN | UNOWNED), a chronological
    put-down log, and a running ownership history.

State Machine (per bag)
    UNOWNED  ──► CARRIED(owner_id)   : first person enters carry-zone
    CARRIED  ──► PUT_DOWN            : owner walks away, bag stays behind
    PUT_DOWN ──► CARRIED(same owner) : owner returns → normal behaviour
    PUT_DOWN ──► CARRIED(new owner)  : stranger picks up → THEFT CANDIDATE 🚨
    PUT_DOWN  (duration > threshold) : UNATTENDED OBJECT alert 🚨

Events emitted
--------------
  bag_first_seen          — bag detected for the first time
  bag_carried             — bag enters CARRIED state
  bag_put_down            — bag placed on floor
  bag_picked_up_by_owner  — owner retrieved their bag
  bag_handover            — voluntary transfer (both persons present)
  bag_theft_suspect       — stranger picks up unattended/owner-absent bag
  bag_unattended_warning  — bag on floor > warn_threshold with no owner near
  bag_unattended_critical — bag on floor > critical_threshold (security alert)
  bag_exit_match          — person exits, bag status verified (same / missing / swapped)
"""

from __future__ import annotations

import datetime
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from config import config


# ─────────────────────────────────────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class BagState(str, Enum):
    UNOWNED  = "unowned"    # newly detected, no owner yet
    CARRIED  = "carried"    # actively held/carried by a person
    PUT_DOWN = "put_down"   # placed on floor, no person within carry-zone


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PutDownEvent:
    """One instance of the bag being placed on the floor."""
    put_down_time: float
    put_down_position: Tuple[float, float]        # (cx, cy) in frame pixels
    camera_id: str
    picked_up_time: Optional[float] = None
    picked_up_by: Optional[str] = None            # global_id who retrieved it
    duration_s: Optional[float] = None            # seconds on floor (filled on pick-up)


@dataclass
class OwnershipSpan:
    """Continuous period a specific person held this bag."""
    owner_id: str
    camera_id: str
    start_time: float
    end_time: Optional[float] = None
    end_reason: str = ""                          # "put_down" | "handover" | "exit"


@dataclass
class BagRecord:
    """Complete record of one physical bag through its entire journey."""
    bag_id: str                                   # "Bag_<track_id>"
    class_name: str                               # backpack | handbag | suitcase | laptop | cell phone
    class_id: int
    first_seen_time: float
    first_seen_camera: str
    first_owner_id: Optional[str] = None

    # Current state
    state: BagState = BagState.UNOWNED
    current_owner: Optional[str] = None
    last_position: Tuple[float, float] = (0.0, 0.0)
    last_seen_time: float = 0.0
    last_seen_camera: str = ""

    # History
    ownership_spans: List[OwnershipSpan] = field(default_factory=list)
    put_down_log: List[PutDownEvent] = field(default_factory=list)

    # Unattended alert tracking
    _warn_alerted: bool = False
    _critical_alerted: bool = False

    # ── Computed properties ─────────────────────────────────────────────────

    @property
    def put_down_count(self) -> int:
        return len(self.put_down_log)

    @property
    def current_put_down(self) -> Optional[PutDownEvent]:
        """The active (unresolved) put-down event, if any."""
        if self.put_down_log and self.put_down_log[-1].picked_up_time is None:
            return self.put_down_log[-1]
        return None

    @property
    def unattended_seconds(self) -> float:
        """Seconds the bag has been on the floor in the current put-down."""
        pd = self.current_put_down
        if pd is None:
            return 0.0
        return time.time() - pd.put_down_time

    @property
    def owner_count(self) -> int:
        """Distinct number of people who have held this bag."""
        return len({span.owner_id for span in self.ownership_spans})

    def summary(self) -> Dict[str, Any]:
        return {
            "bag_id": self.bag_id,
            "class": self.class_name,
            "state": self.state.value,
            "current_owner": self.current_owner,
            "first_owner": self.first_owner_id,
            "put_down_count": self.put_down_count,
            "unattended_seconds": round(self.unattended_seconds, 1),
            "owner_count": self.owner_count,
            "first_seen_camera": self.first_seen_camera,
            "last_seen_camera": self.last_seen_camera,
        }


@dataclass
class LuggageEvent:
    """An event emitted by the luggage tracker, consumed by the alert engine."""
    event_type: str
    bag_id: str
    bag_class: str
    camera_id: str
    timestamp: float
    owner_id: Optional[str] = None
    previous_owner_id: Optional[str] = None
    put_down_count: int = 0
    unattended_seconds: float = 0.0
    position: Tuple[float, float] = (0.0, 0.0)
    severity: str = "medium"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  Core tracker
# ─────────────────────────────────────────────────────────────────────────────

class AirportLuggageTracker:
    """
    Airport-grade luggage tracker.

    Call ``update()`` every frame with the list of detected objects and the
    dict of active person positions.  The tracker returns a list of
    LuggageEvent objects that should be forwarded to the alert engine.
    """

    def __init__(self) -> None:
        # track_id (str) → BagRecord
        self._bags: Dict[str, BagRecord] = {}

        # Thresholds (resolved lazily from config so live config edits work)
        self._carry_ratio: float      = getattr(config, "luggage_carry_distance_ratio", 0.60)
        self._putdown_px: float       = getattr(config, "luggage_putdown_distance_px", 80.0)
        self._warn_s: float           = getattr(config, "luggage_unattended_warn_s", 30.0)
        self._critical_s: float       = getattr(config, "luggage_unattended_critical_s", 60.0)
        self._handover_radius_px: float = getattr(config, "luggage_handover_radius_px", 120.0)
        self._stale_s: float          = getattr(config, "luggage_stale_after_s", 60.0)

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        object_detections: List[Dict[str, Any]],
        person_positions: Dict[str, Tuple[float, float]],   # global_id → (cx, cy)
        person_bboxes: Dict[str, Tuple[int, int, int, int]], # global_id → (x1,y1,x2,y2)
        camera_id: str,
        now: float,
    ) -> List[LuggageEvent]:
        """
        Main entry point.  Returns a (possibly empty) list of LuggageEvent.

        Parameters
        ----------
        object_detections
            List of dicts with keys: track_id, class_id, class_name, center, bbox, confidence.
        person_positions
            Map from global_id to (cx, cy) of every currently visible person.
        person_bboxes
            Map from global_id to (x1, y1, x2, y2) for height estimation.
        camera_id
            Identifier of the camera producing this frame.
        now
            Current unix timestamp (time.time()).
        """
        events: List[LuggageEvent] = []
        seen_track_ids: Set[str] = set()

        for obj in object_detections:
            track_id = obj.get("track_id")
            if track_id is None:
                continue

            track_id = str(track_id)
            seen_track_ids.add(track_id)
            bag_id = track_id   # already prefixed with "BAG-" from fusion layer
            cx, cy = obj["center"]
            class_id = obj["class_id"]
            class_name = obj["class_name"]

            # ── Get or create BagRecord ────────────────────────────────────
            bag = self._bags.get(track_id)
            if bag is None:
                bag = BagRecord(
                    bag_id=bag_id,
                    class_name=class_name,
                    class_id=class_id,
                    first_seen_time=now,
                    first_seen_camera=camera_id,
                    last_seen_time=now,
                    last_seen_camera=camera_id,
                    last_position=(cx, cy),
                )
                self._bags[track_id] = bag
                events.append(LuggageEvent(
                    event_type="bag_first_seen",
                    bag_id=bag_id,
                    bag_class=class_name,
                    camera_id=camera_id,
                    timestamp=now,
                    position=(cx, cy),
                    severity="low",
                    message=f"[Luggage] {bag_id} ({class_name}) detected for the first time on {camera_id}",
                ))
            else:
                bag.last_position = (cx, cy)
                bag.last_seen_time = now
                bag.last_seen_camera = camera_id

            # ── Find nearest person ────────────────────────────────────────
            nearest_owner, dist = self._nearest_person(cx, cy, person_positions, person_bboxes)

            # ── State machine ──────────────────────────────────────────────
            ev = self._transition(bag, nearest_owner, dist, cx, cy, camera_id, now,
                                  person_positions)
            if ev:
                events.extend(ev)

            # ── Unattended timer checks ────────────────────────────────────
            unattended_ev = self._check_unattended(bag, camera_id, now)
            if unattended_ev:
                events.append(unattended_ev)

        # ── Stale bag cleanup ──────────────────────────────────────────────
        self._cleanup(seen_track_ids, now)

        return events

    def get_bag(self, track_id: int) -> Optional[BagRecord]:
        return self._bags.get(track_id)

    def get_bags_for_person(self, global_id: str) -> List[BagRecord]:
        return [b for b in self._bags.values() if b.current_owner == global_id]

    def get_person_bag_summary(self, global_id: str) -> Dict[str, Any]:
        """Summary of all bags currently or ever owned by a person in this session."""
        owned_now = [b for b in self._bags.values() if b.current_owner == global_id]
        ever_owned = [
            b for b in self._bags.values()
            if any(sp.owner_id == global_id for sp in b.ownership_spans)
        ]
        return {
            "currently_carrying": [b.summary() for b in owned_now],
            "total_bags_interacted": len(ever_owned),
            "bags_put_down": sum(b.put_down_count for b in ever_owned),
            "bags_by_class": {
                b.bag_id: b.class_name for b in ever_owned
            },
        }

    def finalize_person_exit(
        self,
        global_id: str,
        camera_id: str,
        now: float,
    ) -> List[LuggageEvent]:
        """
        Call when a person is confirmed to have exited the scene.
        Closes any open ownership spans and generates exit-match events.
        """
        events: List[LuggageEvent] = []
        for bag in self._bags.values():
            if bag.current_owner == global_id:
                # Close the current span
                if bag.ownership_spans:
                    bag.ownership_spans[-1].end_time = now
                    bag.ownership_spans[-1].end_reason = "exit"
                bag.current_owner = None
                bag.state = BagState.PUT_DOWN   # still in scene without owner
                bag._warn_alerted = False
                bag._critical_alerted = False

                # Start a new put-down event
                pd = PutDownEvent(
                    put_down_time=now,
                    put_down_position=bag.last_position,
                    camera_id=camera_id,
                )
                bag.put_down_log.append(pd)

                events.append(LuggageEvent(
                    event_type="bag_owner_exited",
                    bag_id=bag.bag_id,
                    bag_class=bag.class_name,
                    camera_id=camera_id,
                    timestamp=now,
                    owner_id=global_id,
                    put_down_count=bag.put_down_count,
                    severity="medium",
                    message=(
                        f"[Luggage] {bag.bag_id} ({bag.class_name}) — owner {global_id} "
                        f"exited camera {camera_id}. Bag remains in scene."
                    ),
                ))
        return events

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _nearest_person(
        self,
        cx: float,
        cy: float,
        positions: Dict[str, Tuple[float, float]],
        bboxes: Dict[str, Tuple[int, int, int, int]],
    ) -> Tuple[Optional[str], float]:
        """Return (global_id, distance_px) of the person whose bbox contains
        (cx, cy) within the carry-zone, or the geometrically nearest person
        if no one is within the carry-ratio threshold."""
        best_id: Optional[str] = None
        best_dist: float = float("inf")

        for gid, (px, py) in positions.items():
            bbox = bboxes.get(gid)
            if bbox:
                x1, y1, x2, y2 = bbox
                person_h = max(1, y2 - y1)
                carry_px = person_h * self._carry_ratio
            else:
                carry_px = self._putdown_px

            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best_id = gid

        return best_id, best_dist

    def _carry_threshold_for(
        self,
        owner_id: Optional[str],
        bboxes: Dict[str, Tuple[int, int, int, int]],
    ) -> float:
        if owner_id and owner_id in bboxes:
            x1, y1, x2, y2 = bboxes[owner_id]
            return max(self._putdown_px, (y2 - y1) * self._carry_ratio)
        return self._putdown_px

    def _transition(
        self,
        bag: BagRecord,
        nearest_id: Optional[str],
        dist: float,
        cx: float,
        cy: float,
        camera_id: str,
        now: float,
        person_bboxes: Dict[str, Tuple[int, int, int, int]],
    ) -> List[LuggageEvent]:
        events: List[LuggageEvent] = []
        carry_threshold = self._carry_threshold_for(nearest_id, person_bboxes)
        person_in_zone = nearest_id is not None and dist <= carry_threshold

        # ── UNOWNED ────────────────────────────────────────────────────────
        if bag.state == BagState.UNOWNED:
            if person_in_zone:
                bag.state = BagState.CARRIED
                bag.current_owner = nearest_id
                bag.first_owner_id = bag.first_owner_id or nearest_id
                bag.ownership_spans.append(OwnershipSpan(
                    owner_id=nearest_id,
                    camera_id=camera_id,
                    start_time=now,
                ))
                events.append(LuggageEvent(
                    event_type="bag_carried",
                    bag_id=bag.bag_id,
                    bag_class=bag.class_name,
                    camera_id=camera_id,
                    timestamp=now,
                    owner_id=nearest_id,
                    position=(cx, cy),
                    severity="low",
                    message=(
                        f"[Luggage] {bag.bag_id} ({bag.class_name}) claimed by "
                        f"{nearest_id} on {camera_id}"
                    ),
                ))

        # ── CARRIED ───────────────────────────────────────────────────────
        elif bag.state == BagState.CARRIED:
            carry_ok = (
                nearest_id == bag.current_owner
                and dist <= carry_threshold
            )
            if carry_ok:
                # Still with the same owner — nothing to do
                pass
            elif person_in_zone and nearest_id != bag.current_owner:
                # Someone new within carry-zone while bag is supposedly carried
                # This happens on a direct hand-to-hand transfer
                prev_owner = bag.current_owner
                events.append(self._emit_handover(
                    bag, prev_owner, nearest_id, cx, cy, camera_id, now,
                    voluntary=True   # previous owner may still be near
                ))
            else:
                # Owner walked away — bag is now put down
                bag.state = BagState.PUT_DOWN
                bag.current_owner = None
                bag._warn_alerted = False
                bag._critical_alerted = False

                # Close ownership span
                if bag.ownership_spans:
                    bag.ownership_spans[-1].end_time = now
                    bag.ownership_spans[-1].end_reason = "put_down"

                pd = PutDownEvent(
                    put_down_time=now,
                    put_down_position=(cx, cy),
                    camera_id=camera_id,
                )
                bag.put_down_log.append(pd)
                events.append(LuggageEvent(
                    event_type="bag_put_down",
                    bag_id=bag.bag_id,
                    bag_class=bag.class_name,
                    camera_id=camera_id,
                    timestamp=now,
                    owner_id=bag.ownership_spans[-1].owner_id if bag.ownership_spans else None,
                    put_down_count=bag.put_down_count,
                    position=(cx, cy),
                    severity="low",
                    message=(
                        f"[Luggage] {bag.bag_id} ({bag.class_name}) placed on floor "
                        f"(put-down #{bag.put_down_count}) on {camera_id}"
                    ),
                ))

        # ── PUT_DOWN ──────────────────────────────────────────────────────
        elif bag.state == BagState.PUT_DOWN:
            if person_in_zone and nearest_id is not None:
                # Someone is retrieving the bag
                pd = bag.current_put_down
                if pd:
                    pd.picked_up_time = now
                    pd.picked_up_by = nearest_id
                    pd.duration_s = now - pd.put_down_time

                # Determine if this is the original owner or a stranger
                original_owners = {sp.owner_id for sp in bag.ownership_spans}
                is_original_owner = nearest_id in original_owners

                if is_original_owner:
                    bag.state = BagState.CARRIED
                    bag.current_owner = nearest_id
                    bag.ownership_spans.append(OwnershipSpan(
                        owner_id=nearest_id,
                        camera_id=camera_id,
                        start_time=now,
                    ))
                    events.append(LuggageEvent(
                        event_type="bag_picked_up_by_owner",
                        bag_id=bag.bag_id,
                        bag_class=bag.class_name,
                        camera_id=camera_id,
                        timestamp=now,
                        owner_id=nearest_id,
                        put_down_count=bag.put_down_count,
                        unattended_seconds=round(pd.duration_s or 0.0, 1),
                        position=(cx, cy),
                        severity="low",
                        message=(
                            f"[Luggage] {bag.bag_id} retrieved by owner {nearest_id} "
                            f"after {pd.duration_s:.1f}s on floor on {camera_id}"
                        ),
                    ))
                else:
                    # Stranger picks up an unattended bag — THEFT CANDIDATE
                    prev_owner = (
                        bag.ownership_spans[-1].owner_id
                        if bag.ownership_spans else None
                    )
                    events.append(self._emit_handover(
                        bag, prev_owner, nearest_id, cx, cy, camera_id, now,
                        voluntary=False
                    ))

        return events

    def _emit_handover(
        self,
        bag: BagRecord,
        from_id: Optional[str],
        to_id: str,
        cx: float,
        cy: float,
        camera_id: str,
        now: float,
        voluntary: bool,
    ) -> LuggageEvent:
        # Close previous ownership span
        if bag.ownership_spans:
            bag.ownership_spans[-1].end_time = now
            bag.ownership_spans[-1].end_reason = "handover" if voluntary else "theft"

        # Open new span
        bag.state = BagState.CARRIED
        bag.current_owner = to_id
        bag.ownership_spans.append(OwnershipSpan(
            owner_id=to_id,
            camera_id=camera_id,
            start_time=now,
        ))

        if voluntary:
            ev_type = "bag_handover"
            sev = "medium"
            msg = (
                f"[Luggage] {bag.bag_id} ({bag.class_name}) handed from "
                f"{from_id} → {to_id} on {camera_id}"
            )
        else:
            ev_type = "bag_theft_suspect"
            sev = "critical"
            msg = (
                f"🚨 [Luggage] THEFT SUSPECT: {bag.bag_id} ({bag.class_name}) "
                f"picked up by {to_id} — original owner {from_id} — "
                f"put-down count: {bag.put_down_count} on {camera_id}"
            )

        return LuggageEvent(
            event_type=ev_type,
            bag_id=bag.bag_id,
            bag_class=bag.class_name,
            camera_id=camera_id,
            timestamp=now,
            owner_id=to_id,
            previous_owner_id=from_id,
            put_down_count=bag.put_down_count,
            position=(cx, cy),
            severity=sev,
            message=msg,
            details={
                "voluntary": voluntary,
                "owner_count": bag.owner_count,
            },
        )

    def _check_unattended(
        self,
        bag: BagRecord,
        camera_id: str,
        now: float,
    ) -> Optional[LuggageEvent]:
        if bag.state != BagState.PUT_DOWN:
            return None
        pd = bag.current_put_down
        if pd is None:
            return None

        secs = now - pd.put_down_time

        if secs >= self._critical_s and not bag._critical_alerted:
            bag._critical_alerted = True
            return LuggageEvent(
                event_type="bag_unattended_critical",
                bag_id=bag.bag_id,
                bag_class=bag.class_name,
                camera_id=camera_id,
                timestamp=now,
                previous_owner_id=(
                    bag.ownership_spans[-1].owner_id if bag.ownership_spans else None
                ),
                put_down_count=bag.put_down_count,
                unattended_seconds=round(secs, 1),
                position=bag.last_position,
                severity="critical",
                message=(
                    f"🚨 SECURITY ALERT: {bag.bag_id} ({bag.class_name}) UNATTENDED "
                    f"for {secs:.0f}s on {camera_id}. Last owner: "
                    f"{bag.ownership_spans[-1].owner_id if bag.ownership_spans else 'unknown'}"
                ),
            )

        if secs >= self._warn_s and not bag._warn_alerted:
            bag._warn_alerted = True
            return LuggageEvent(
                event_type="bag_unattended_warning",
                bag_id=bag.bag_id,
                bag_class=bag.class_name,
                camera_id=camera_id,
                timestamp=now,
                previous_owner_id=(
                    bag.ownership_spans[-1].owner_id if bag.ownership_spans else None
                ),
                put_down_count=bag.put_down_count,
                unattended_seconds=round(secs, 1),
                position=bag.last_position,
                severity="high",
                message=(
                    f"⚠ UNATTENDED BAG: {bag.bag_id} ({bag.class_name}) on floor "
                    f"for {secs:.0f}s on {camera_id}. Last owner: "
                    f"{bag.ownership_spans[-1].owner_id if bag.ownership_spans else 'unknown'}"
                ),
            )

        return None

    def _cleanup(self, seen_track_ids: Set[str], now: float) -> None:
        stale = [
            tid for tid, bag in self._bags.items()
            if tid not in seen_track_ids and (now - bag.last_seen_time) > self._stale_s
        ]
        for tid in stale:
            del self._bags[tid]

    def get_all_bags(self):
        """Return a list of serialisable bag dicts for the REST API."""
        import time as _time
        now = _time.time()
        result = []
        for tid, bag in list(self._bags.items()):
            result.append({
                "bag_id":            bag.bag_id,
                "class_name":        bag.class_name,
                "state":             bag.state.value,
                "current_owner":     bag.current_owner_id,
                "first_owner_id":    bag.first_owner_id,
                "put_down_count":    bag.put_down_count,
                "owner_count":       len({s.owner_id for s in bag.ownership_spans}),
                "unattended_seconds": (now - bag.put_down_started_at)
                                      if bag.state.value == "put_down" and bag.put_down_started_at
                                      else 0.0,
                "camera_id":         bag.last_camera_id,
                "last_seen_ts":      bag.last_seen_time,
                "global_track_id":   getattr(bag, "global_track_id", ""),
                "last_position":     bag.last_position,
            })
        return result

# ─────────────────────────────────────────────────────────────────────────────
#  Singleton
# ─────────────────────────────────────────────────────────────────────────────

luggage_tracker = AirportLuggageTracker()
