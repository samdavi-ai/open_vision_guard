import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Any

class CameraAvoidanceDetector:
    def __init__(self, window_size: int = 60):
        # person_id -> deque of (face_visible: bool, is_edge: bool)
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, person_id: str, bbox: tuple, face_visible: bool, dir_angle: float, frame_w: int, frame_h: int, current_time: float) -> Dict[str, Any]:
        """
        Detects camera avoidance behaviors.
        Matches the interface expected by pipeline.py.
        """
        x1, y1, x2, y2 = bbox
        
        # Check if person is near edges (blind spot hugging)
        is_edge = (x1 < frame_w * 0.1) or (x2 > frame_w * 0.9) or (y1 < frame_h * 0.1) or (y2 > frame_h * 0.9)
        
        self.history[person_id].append((face_visible, is_edge))
        
        hist = list(self.history[person_id])
        if len(hist) < 20:
            return {"avoidance_score": 0.0, "avoidance_behaviours": []}
            
        face_visibility_ratio = sum(1 for v, e in hist if v) / len(hist)
        edge_hugging_ratio = sum(1 for v, e in hist if e) / len(hist)
        
        score = 0.0
        behaviours = []
        
        if face_visibility_ratio < 0.2:
            score += 40
            behaviours.append("face_consistently_hidden")
            
        if edge_hugging_ratio > 0.7:
            score += 30
            behaviours.append("edge_hugging_detected")
            
        return {
            "avoidance_score": min(score, 100.0),
            "avoidance_behaviours": behaviours
        }

camera_avoidance_detector = CameraAvoidanceDetector()
