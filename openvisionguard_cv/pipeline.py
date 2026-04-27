"""Production edge pipeline: motion gate, YOLO+ByteTrack, behavior, luggage, risk."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from .behavior import TemporalBehaviorAnalyzer
from .detector import YoloEdgeDetector
from .luggage import LuggageTracker
from .motion import MotionGate
from .risk import RiskScorer
from .types import Detection, EdgeFrameResult


@dataclass
class EdgePipelineConfig:
    model_path: str = "yolov8n.pt"
    device: str = "cpu"
    imgsz: int = 416
    conf: float = 0.28
    iou: float = 0.45
    process_every: int = 2
    min_motion_ratio: float = 0.004
    loitering_seconds: float = 30.0
    abandoned_seconds: float = 25.0
    alert_threshold: float = 65.0
    class_agnostic_source: bool = False


class EdgeSurveillancePipeline:
    """Offline surveillance pipeline tuned for low-spec CPU deployments."""

    def __init__(self, config: EdgePipelineConfig) -> None:
        self.config = config
        self.detector = YoloEdgeDetector(
            model_path=config.model_path,
            device=config.device,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            class_agnostic_source=config.class_agnostic_source,
        )
        self.motion_gate = MotionGate(
            process_every=config.process_every,
            min_motion_ratio=config.min_motion_ratio,
        )
        self.behavior = TemporalBehaviorAnalyzer(loitering_seconds=config.loitering_seconds)
        self.luggage = LuggageTracker(abandoned_seconds=config.abandoned_seconds)
        self.risk = RiskScorer(alert_threshold=config.alert_threshold)
        self._last_detections: List[Detection] = []
        self._last_behaviors = {}
        self._frame_index = 0
        self._last_time = time.perf_counter()

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> EdgeFrameResult:
        if timestamp is None:
            timestamp = time.time()

        decision = self.motion_gate.update(frame, self._frame_index)
        processed = decision.should_process
        detections = self._last_detections

        if processed:
            detections = self.detector.track(frame)
            self._last_detections = detections
            behaviors = self.behavior.update(detections, timestamp) if detections else {}
            luggage_events = self.luggage.update(detections, timestamp) if detections else []
            risk_scores, alerts = self.risk.update(behaviors, luggage_events)
            self._last_behaviors = behaviors
        else:
            behaviors = self._last_behaviors
            luggage_events = []
            risk_scores = self.risk.snapshot()
            alerts = []

        now = time.perf_counter()
        fps = 1.0 / max(1e-6, now - self._last_time)
        self._last_time = now

        result = EdgeFrameResult(
            frame_index=self._frame_index,
            processed=processed,
            detections=detections,
            behaviors=behaviors,
            luggage_events=luggage_events,
            risk_scores=risk_scores,
            alerts=alerts,
            fps=fps,
        )
        self._frame_index += 1
        return result

    @staticmethod
    def draw(frame: np.ndarray, result: EdgeFrameResult) -> np.ndarray:
        annotated = frame.copy()
        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = (60, 220, 60) if det.is_person else (0, 180, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id is not None:
                label = f"id {det.track_id} {label}"
            if det.track_id in result.behaviors:
                behavior = result.behaviors[int(det.track_id)]
                if behavior.label not in {"normal", "warming_up"}:
                    label += f" {behavior.label}:{behavior.score:.0f}"
            cv2.putText(annotated, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        y = 24
        for alert in result.alerts[:4]:
            cv2.putText(annotated, alert, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
            y += 24
        cv2.putText(
            annotated,
            f"fps={result.fps:.1f} processed={int(result.processed)}",
            (10, annotated.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        return annotated


def load_edge_config(path: Union[str, Path]) -> EdgePipelineConfig:
    """Load the YAML runtime config."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data: Dict = yaml.safe_load(f) or {}
    return EdgePipelineConfig(**data)
