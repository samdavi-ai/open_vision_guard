"""YOLOv8 detector/tracker wrapper with surveillance-specific filtering."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from ultralytics import YOLO

from .constants import COCO_TO_SURVEILLANCE, SURVEILLANCE_CLASSES
from .types import Detection


class YoloEdgeDetector:
    """Runs YOLOv8n or exported YOLO models through Ultralytics tracking."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        imgsz: int = 416,
        conf: float = 0.28,
        iou: float = 0.45,
        tracker: str = "bytetrack.yaml",
        class_agnostic_source: bool = False,
    ) -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self.class_agnostic_source = class_agnostic_source or self._looks_like_coco_model()
        self.class_conf = {
            0: max(conf, 0.30),
            1: max(conf, 0.35),
            2: max(conf, 0.35),
            3: max(conf, 0.35),
        }

    def track(self, frame: np.ndarray) -> List[Detection]:
        source_classes: Optional[Iterable[int]] = None
        if self.class_agnostic_source:
            source_classes = list(COCO_TO_SURVEILLANCE.keys())

        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            tracker=self.tracker,
            classes=source_classes,
        )
        if not results:
            return []
        return self._parse_result(results[0], frame.shape[:2])

    def _parse_result(self, result, frame_shape: tuple[int, int]) -> List[Detection]:
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.int().cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        track_ids = (
            result.boxes.id.int().cpu().numpy()
            if result.boxes.id is not None
            else [None] * len(boxes)
        )
        detections: List[Detection] = []
        h, w = frame_shape

        for box, cls_id, conf, track_id in zip(boxes, cls_ids, confs, track_ids):
            cls_id_int = int(cls_id)
            if self.class_agnostic_source:
                if cls_id_int not in COCO_TO_SURVEILLANCE:
                    continue
                cls_id_int = COCO_TO_SURVEILLANCE[cls_id_int]

            if cls_id_int not in SURVEILLANCE_CLASSES:
                continue

            confidence = float(conf)
            if confidence < self.class_conf.get(cls_id_int, self.conf):
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w - 1), x2), min(float(h - 1), y2)
            if x2 <= x1 or y2 <= y1:
                continue

            det = Detection(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                class_id=cls_id_int,
                class_name=SURVEILLANCE_CLASSES[cls_id_int],
                track_id=int(track_id) if track_id is not None else None,
            )
            if self._is_plausible(det, frame_shape):
                detections.append(det)
        return detections

    def _looks_like_coco_model(self) -> bool:
        names = getattr(self.model, "names", {}) or {}
        if isinstance(names, list):
            names = {idx: name for idx, name in enumerate(names)}
        return (
            names.get(0) == "person"
            and names.get(24) == "backpack"
            and names.get(26) == "handbag"
            and names.get(28) == "suitcase"
        )

    @staticmethod
    def _is_plausible(det: Detection, frame_shape: tuple[int, int]) -> bool:
        h, w = frame_shape
        frame_area = max(1.0, float(h * w))
        area_ratio = det.area / frame_area
        aspect = det.width / max(det.height, 1.0)

        if det.is_person:
            if area_ratio < 0.0015:
                return False
            if aspect < 0.12 or aspect > 1.25:
                return False
            if det.height > h * 0.92 and det.width < w * 0.08:
                return False
        elif det.is_bag:
            if area_ratio < 0.0004:
                return False
            if aspect < 0.25 or aspect > 3.5:
                return False
        return True
