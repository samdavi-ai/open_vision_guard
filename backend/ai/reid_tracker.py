import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

class ReIDTracker:
    def __init__(self, similarity_threshold=0.85, device='cpu'):
        self.device = device
        self.similarity_threshold = similarity_threshold
        
        # Load feature extractor model
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.global_subjects = {} # global_id -> {'features':[], 'avg_feature': tensor}
        self.session_locks = {}   # local_track_id -> global_id
        self.next_subject_id = 1
        
    def extract_feature(self, frame_crop):
        try:
            tensor = self.transforms(frame_crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(tensor).flatten()
                # Normalize the feature vector
                feat = feat / feat.norm(p=2, dim=0, keepdim=True)
            return feat
        except Exception as e:
            return None
            
    def match_feature(self, feature):
        best_match = None
        best_sim = -1
        
        for subj_id, data in self.global_subjects.items():
            avg_feat = data['avg_feature']
            sim = torch.nn.functional.cosine_similarity(feature.unsqueeze(0), avg_feat.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_match = subj_id
                
        if best_sim >= self.similarity_threshold:
            return best_match, best_sim
        return None, best_sim

    def register_feature(self, subj_id, feature):
        if subj_id not in self.global_subjects:
            self.global_subjects[subj_id] = {'features': [feature], 'avg_feature': feature}
        else:
            self.global_subjects[subj_id]['features'].append(feature)
            # Temporal averaging (running average of last 10 features)
            recent_feats = self.global_subjects[subj_id]['features'][-10:]
            avg_feat = torch.stack(recent_feats).mean(dim=0)
            avg_feat = avg_feat / avg_feat.norm(p=2, dim=0, keepdim=True)
            self.global_subjects[subj_id]['avg_feature'] = avg_feat

    def process_frame_tracking(self, frame, yolo_results):
        annotated_frame = frame.copy()
        
        # Checking if there are bounding boxes and tracking IDs
        if not yolo_results or yolo_results[0].boxes is None or yolo_results[0].boxes.id is None:
            return annotated_frame, []
            
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        track_ids = yolo_results[0].boxes.id.int().cpu().numpy()
        cls_ids = yolo_results[0].boxes.cls.int().cpu().numpy()
        
        alerts = []
        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            if cls_id != 0: # Only track person class in COCO (0)
                continue
                
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure valid box
            if x2 <= x1 or y2 <= y1:
                continue
                
            # If already locked, just periodically average features
            if track_id in self.session_locks:
                global_id = self.session_locks[track_id]
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if crop.size > 0:
                    feat = self.extract_feature(crop)
                    if feat is not None:
                        self.register_feature(global_id, feat)
            else:
                # Need to run Re-ID for this new local track
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if crop.size == 0:
                    continue
                    
                feat = self.extract_feature(crop)
                if feat is None:
                    continue
                    
                matched_id, sim = self.match_feature(feat)
                
                if matched_id is not None:
                    global_id = matched_id
                    alerts.append(f"Re-Identified {global_id} (sim: {sim:.2f})")
                else:
                    global_id = f"Subject_{self.next_subject_id}"
                    self.next_subject_id += 1
                    alerts.append(f"New Subject: {global_id}")
                    
                # Register feature & lock track
                self.register_feature(global_id, feat)
                self.session_locks[track_id] = global_id
                
            # Draw on frame
            global_id = self.session_locks.get(track_id, "Unknown")
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, global_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
        return annotated_frame, alerts
