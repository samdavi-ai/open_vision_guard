import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
from config import config


class WeaponDetector:
    def __init__(self, model_path: str = "weights/weapon_yolov8.pt"):
        self.weapon_classes = []

        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            names = self.model.names
            for idx, name in names.items():
                if any(w in name.lower() for w in ['gun', 'knife', 'weapon', 'rifle', 'pistol']):
                    self.weapon_classes.append(idx)
            if not self.weapon_classes:
                self.weapon_classes = list(self.model.names.keys())
        else:
            print(f"Warning: Weapon model {model_path} not found. Falling back to yolov8n.pt (knife class only).")
            self.model = YOLO("yolov8n.pt")
            self.weapon_classes = [43]  # COCO 'knife'

    def detect_weapons(self, frame: np.ndarray, person_boxes: Optional[List] = None, person_ids: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        Detect weapons in the frame and link them to the nearest person.
        Returns: [{weapon_type, bbox, confidence, nearest_person_id}]
        """
        results = self.model(frame, classes=self.weapon_classes, conf=config.weapon_confidence_threshold, verbose=False)
        result = results[0]

        weapons = []
        if result.boxes is None or len(result.boxes) == 0:
            return weapons

        for box in result.boxes:
            w_xyxy = box.xyxy.cpu().numpy()[0]
            w_cls = self.model.names[int(box.cls)]
            w_conf = float(box.conf)

            w_center = [(w_xyxy[0] + w_xyxy[2]) / 2, (w_xyxy[1] + w_xyxy[3]) / 2]

            nearest_person_id = None
            min_dist = float('inf')

            if person_boxes is not None and person_ids is not None:
                for p_box, p_id in zip(person_boxes, person_ids):
                    p_center = [(p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2]
                    dist = ((w_center[0] - p_center[0]) ** 2 + (w_center[1] - p_center[1]) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_person_id = p_id

            weapons.append({
                "weapon_type": w_cls,
                "bbox": w_xyxy.tolist(),
                "confidence": w_conf,
                "nearest_person_id": nearest_person_id if min_dist < 200 else None
            })

        return weapons

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw weapon detections on frame."""
        results = self.model(frame, classes=self.weapon_classes, conf=config.weapon_confidence_threshold, verbose=False)
        return results[0].plot()
