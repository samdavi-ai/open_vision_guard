"""Dataclasses used by the edge surveillance runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_person(self) -> bool:
        return self.class_id == 0

    @property
    def is_bag(self) -> bool:
        return self.class_id in {1, 2, 3}


@dataclass
class TrackBehavior:
    track_id: int
    label: str
    score: float
    speed_px_s: float
    dwell_s: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class LuggageEvent:
    event_type: str
    luggage_id: str
    class_name: str
    owner_id: Optional[int]
    score: float
    message: str


@dataclass
class EdgeFrameResult:
    frame_index: int
    processed: bool
    detections: List[Detection]
    behaviors: Dict[int, TrackBehavior]
    luggage_events: List[LuggageEvent]
    risk_scores: Dict[int, float]
    alerts: List[str]
    fps: float
