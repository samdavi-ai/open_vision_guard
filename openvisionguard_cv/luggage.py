"""Lightweight owner association and abandoned-luggage detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import Detection, LuggageEvent


@dataclass
class LuggageState:
    luggage_id: str
    class_name: str
    center: Tuple[float, float]
    owner_id: Optional[int]
    last_seen: float
    abandoned_since: Optional[float] = None


class LuggageTracker:
    """Associates bags with nearby people and emits abandoned-item events."""

    def __init__(
        self,
        link_distance_px: float = 130.0,
        same_bag_distance_px: float = 60.0,
        abandoned_seconds: float = 25.0,
    ) -> None:
        self.link_distance_px = link_distance_px
        self.same_bag_distance_px = same_bag_distance_px
        self.abandoned_seconds = abandoned_seconds
        self.states: Dict[str, LuggageState] = {}
        self._next_id = 1

    def update(self, detections: List[Detection], timestamp: float) -> List[LuggageEvent]:
        people = [d for d in detections if d.is_person and d.track_id is not None]
        bags = [d for d in detections if d.is_bag]
        events: List[LuggageEvent] = []

        for bag in bags:
            center = bag.center
            luggage_id = self._match_existing(center) or self._new_id()
            owner_id, owner_dist = self._nearest_owner(center, people)
            state = self.states.get(luggage_id)
            if state is None:
                state = LuggageState(luggage_id, bag.class_name, center, owner_id, timestamp)
                self.states[luggage_id] = state

            if owner_id is not None and owner_dist <= self.link_distance_px:
                if state.owner_id is not None and owner_id != state.owner_id:
                    events.append(LuggageEvent(
                        "luggage_owner_changed",
                        luggage_id,
                        bag.class_name,
                        owner_id,
                        35.0,
                        f"{bag.class_name} ownership changed to track {owner_id}",
                    ))
                state.owner_id = owner_id
                state.abandoned_since = None
            else:
                if state.abandoned_since is None:
                    state.abandoned_since = timestamp
                abandoned_for = timestamp - state.abandoned_since
                if abandoned_for >= self.abandoned_seconds:
                    events.append(LuggageEvent(
                        "abandoned_luggage",
                        luggage_id,
                        bag.class_name,
                        state.owner_id,
                        min(95.0, 55.0 + abandoned_for),
                        f"{bag.class_name} {luggage_id} abandoned for {abandoned_for:.1f}s",
                    ))

            state.class_name = bag.class_name
            state.center = center
            state.last_seen = timestamp

        self._cleanup(timestamp)
        return events

    def _nearest_owner(self, center: Tuple[float, float], people: List[Detection]) -> Tuple[Optional[int], float]:
        best_id: Optional[int] = None
        best_dist = float("inf")
        for person in people:
            dist = float(np.hypot(center[0] - person.center[0], center[1] - person.center[1]))
            if dist < best_dist:
                best_id = int(person.track_id) if person.track_id is not None else None
                best_dist = dist
        return best_id, best_dist

    def _match_existing(self, center: Tuple[float, float]) -> Optional[str]:
        best_id: Optional[str] = None
        best_dist = float("inf")
        for luggage_id, state in self.states.items():
            dist = float(np.hypot(center[0] - state.center[0], center[1] - state.center[1]))
            if dist < best_dist:
                best_id, best_dist = luggage_id, dist
        return best_id if best_dist <= self.same_bag_distance_px else None

    def _new_id(self) -> str:
        luggage_id = f"bag_{self._next_id:04d}"
        self._next_id += 1
        return luggage_id

    def _cleanup(self, timestamp: float, stale_seconds: float = 60.0) -> None:
        stale = [lid for lid, state in self.states.items() if timestamp - state.last_seen > stale_seconds]
        for lid in stale:
            del self.states[lid]
