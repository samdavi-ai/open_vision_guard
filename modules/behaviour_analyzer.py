import torch
import torchvision.models.video as video_models
import torchvision.transforms as transforms
import numpy as np
import cv2
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time

@dataclass
class TemporalState:
    """Track temporal state for a single person."""
    person_id: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    time_in_zone: float = 0.0
    frame_buffer: deque = field(default_factory=lambda: deque(maxlen=16)) # 16 frames for 3D CNN
    behavior_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_seen_time(self, current_time: float):
        if self.first_seen == 0.0:
            self.first_seen = current_time
        self.last_seen = current_time
        self.time_in_zone = current_time - self.first_seen

class BehaviourAnalyzer:
    """
    Spatio-Temporal Behavior Analyzer using lightweight 3D CNN (r3d_18) from Torchvision.
    """
    def __init__(self, fps: int = 30, device: str = 'mps'):
        self.fps = fps
        # Prioritize Apple Silicon Metal (MPS)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        print(f"[BehaviourAnalyzer] Using device: {self.device}")
        
        # Load lightweight ResNet3D for Action Recognition (Kinetics 400 weights)
        try:
            self.model = video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Failed to load r3d_18 action recognition model: {e}")
            self.model = None

        self.temporal_states: Dict[str, TemporalState] = {}
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])

        # Example kinetics-400 classes mapping (simplified)
        self.kinetics_classes = self._load_kinetics_classes()

    def _load_kinetics_classes(self) -> List[str]:
        # In a real scenario, this would load the full 400 classes txt file.
        # Returning a dummy list of 400 "unknown" except for known indices to avoid massive file size.
        classes = ["unknown"] * 400
        return classes

    def update(self, person_id: str, crop: np.ndarray, current_time: float) -> Dict[str, Any]:
        """
        Update behavior analysis by processing 16-frame buffers through a 3D CNN.
        """
        if person_id not in self.temporal_states:
            self.temporal_states[person_id] = TemporalState(person_id=person_id)
        
        state = self.temporal_states[person_id]
        state.update_seen_time(current_time)
        
        if crop is None or crop.size == 0:
            return {"behaviour_label": "unknown", "behaviour_score": 0.0}

        # Process frame
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor_frame = self.transform(crop_rgb)
        state.frame_buffer.append(tensor_frame)

        # We need exactly 16 frames to run the 3D CNN efficiently
        if len(state.frame_buffer) == 16 and self.model is not None:
            # Stack into (C, T, H, W)
            video_tensor = torch.stack(list(state.frame_buffer), dim=1)
            # Add batch dim (1, C, T, H, W)
            video_tensor = video_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self.model(video_tensor)
                probs = torch.nn.functional.softmax(preds, dim=1)
                score, class_idx = torch.max(probs, 1)

            # Map class_idx to label
            label_idx = class_idx.item()
            score_val = score.item() * 100.0
            
            # Since we use dummy classes for now, just output "detected_action"
            label = "action_detected" # self.kinetics_classes[label_idx]
            
            state.behavior_history.append(label)
            return {
                "behaviour_label": label,
                "behaviour_score": score_val,
                "time_in_zone": state.time_in_zone
            }

        return {
            "behaviour_label": "accumulating_frames",
            "behaviour_score": 0.0,
            "time_in_zone": state.time_in_zone
        }

    def cleanup_old_states(self, current_time: float, timeout_seconds: float = 300):
        to_delete = []
        for person_id, state in self.temporal_states.items():
            if current_time - state.last_seen > timeout_seconds:
                to_delete.append(person_id)
        for person_id in to_delete:
            del self.temporal_states[person_id]

behaviour_analyzer = BehaviourAnalyzer()
