import cv2
import numpy as np
import pickle
import os
from scipy.spatial.distance import cosine
from ai.feature_extractor import DeepFeatureExtractor

class ReIDTracker:
    def __init__(self, similarity_threshold=0.6, device='cpu'):
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.extractor = DeepFeatureExtractor(device=device)
        
        self.global_subjects = {} # global_id -> {'features':[], 'avg_feature': ndarray, 'class_name': str, 'name': str}
        self.session_locks = {}   # local_track_id -> global_id
        self.next_subject_ids = {} # class_name -> next_id
        
    def save_registry(self, path="registry.pkl"):
        """Saves the global subjects registry to a file."""
        try:
            data = {
                'global_subjects': self.global_subjects,
                'next_subject_ids': self.next_subject_ids
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Registry saved to {path} ({len(self.global_subjects)} subjects)")
        except Exception as e:
            print(f"Error saving registry: {e}")

    def load_registry(self, path="registry.pkl"):
        """Loads the global subjects registry from a file."""
        if not os.path.exists(path):
            print(f"No registry found at {path}, starting fresh.")
            return

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.global_subjects = data.get('global_subjects', {})
            # Migration: Ensure all subjects have 'name' and 'class_name' keys
            for sid, subj in self.global_subjects.items():
                if 'name' not in subj:
                    subj['name'] = None
                if 'class_name' not in subj:
                    # Try to infer class from sid if it's formatted like "Class_N"
                    if "_" in sid:
                        subj['class_name'] = sid.split("_")[0].lower()
                    else:
                        subj['class_name'] = 'person' # Fallback
            
            self.next_subject_ids = data.get('next_subject_ids', {})
            print(f"Registry loaded from {path} ({len(self.global_subjects)} subjects)")
        except Exception as e:
            print(f"Error loading registry: {e}")
        
    def extract_feature(self, frame_crop):
        try:
            # Use Neural Deep Feature Extractor instead of histograms
            feat = self.extractor.extract(frame_crop)
            return feat
        except Exception as e:
            print(f"Extraction error: {e}")
            return None
            
    def match_feature(self, feature):
        best_match = None
        best_sim = -1
        
        for subj_id, data in self.global_subjects.items():
            avg_feat = data['avg_feature']
            # Cosine distance returns 1 - cosine similarity
            # So similarity = 1 - distance
            sim = 1.0 - cosine(feature, avg_feat)
            
            if sim > best_sim:
                best_sim = sim
                best_match = subj_id
                
        if best_sim >= self.similarity_threshold:
            return best_match, best_sim
        return None, best_sim

    def register_feature(self, subj_id, feature, class_name=None):
        if subj_id not in self.global_subjects:
            self.global_subjects[subj_id] = {
                'features': [feature], 
                'avg_feature': feature,
                'class_name': class_name,
                'name': None
            }
        else:
            self.global_subjects[subj_id]['features'].append(feature)
            # Temporal averaging (running average of last 10 features to smooth lighting changes)
            recent_feats = self.global_subjects[subj_id]['features'][-10:]
            avg_feat = np.mean(recent_feats, axis=0)
            
            # Re-normalize
            norm = np.linalg.norm(avg_feat)
            if norm > 0:
                avg_feat = avg_feat / norm
                
            self.global_subjects[subj_id]['avg_feature'] = avg_feat

    def update_identity(self, global_id, name):
        """Updates the name of a global subject (e.g., when a face is recognized)."""
        if global_id in self.global_subjects:
            self.global_subjects[global_id]['name'] = name
            return True
        return False

    def get_display_name(self, global_id):
        """Returns the name if available, otherwise the global_id."""
        if global_id in self.global_subjects:
            subj = self.global_subjects[global_id]
            if subj['name'] and subj['name'] != "Unknown":
                return f"{subj['name']} ({global_id})"
        return global_id

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
            
            # Ensure valid box bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Stable Identity Locking
            if track_id in self.session_locks:
                global_id = self.session_locks[track_id]
                # Periodically extract & temporal average features to adapt to lighting
                feat = self.extract_feature(crop)
                if feat is not None:
                    self.register_feature(global_id, feat)
            else:
                # Need to run Re-ID for this new local track
                feat = self.extract_feature(crop)
                if feat is None:
                    continue
                    
                matched_id, sim = self.match_feature(feat)
                
                if matched_id is not None:
                    global_id = matched_id
                    alerts.append(f"Re-Identified {self.get_display_name(global_id)} (sim: {sim:.2f})")
                else:
                    # Generate new ID based on class
                    class_name = yolo_results[0].names[cls_id].capitalize()
                    next_id = self.next_subject_ids.get(class_name, 1)
                    global_id = f"{class_name}_{next_id}"
                    self.next_subject_ids[class_name] = next_id + 1
                    alerts.append(f"New Subject: {global_id}")
                    
                # Register feature & lock track
                self.register_feature(global_id, feat, class_name=yolo_results[0].names[cls_id])
                self.session_locks[track_id] = global_id
                
            # Draw on frame
            global_id = self.session_locks.get(track_id, "Unknown")
            display_name = self.get_display_name(global_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Add a slight background for readable text
            (w, h), _ = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1-h-10), (x1 + w + 10, y1), (255, 0, 0), -1)
            cv2.putText(annotated_frame, display_name, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return annotated_frame, alerts
