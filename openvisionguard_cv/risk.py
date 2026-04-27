"""Risk scoring for edge surveillance events."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .types import LuggageEvent, TrackBehavior


class RiskScorer:
    """Combines temporal behavior and object events into 0-100 risk scores."""

    def __init__(self, alert_threshold: float = 65.0, decay: float = 0.92) -> None:
        self.alert_threshold = alert_threshold
        self.decay = decay
        self._scores: Dict[int, float] = defaultdict(float)

    def update(
        self,
        behaviors: Dict[int, TrackBehavior],
        luggage_events: Iterable[LuggageEvent],
    ) -> tuple[Dict[int, float], List[str]]:
        alerts: List[str] = []

        for track_id in list(self._scores):
            self._scores[track_id] *= self.decay
            if self._scores[track_id] < 1.0:
                del self._scores[track_id]

        for track_id, behavior in behaviors.items():
            if behavior.label == "normal" or behavior.label == "warming_up":
                continue
            self._scores[track_id] = max(self._scores[track_id], behavior.score)
            if behavior.score >= self.alert_threshold:
                alerts.append(f"track {track_id}: {behavior.label} risk={behavior.score:.1f}")

        for event in luggage_events:
            if event.owner_id is None:
                continue
            self._scores[event.owner_id] = max(self._scores[event.owner_id], event.score)
            if event.score >= self.alert_threshold:
                alerts.append(f"track {event.owner_id}: {event.message} risk={event.score:.1f}")

        return dict(self._scores), alerts

    def snapshot(self) -> Dict[int, float]:
        return dict(self._scores)
