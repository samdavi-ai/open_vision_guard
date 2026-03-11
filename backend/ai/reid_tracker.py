import cv2
import numpy as np
from scipy.spatial.distance import cosine

class ReIDTracker:
    def __init__(self, similarity_threshold=0.85, device='cpu'):
        self.similarity_threshold = similarity_threshold
        # Note: device arg kept for compatibility with detect.py API, but histograms run efficiently on CPU via OpenCV
        
        self.global_subjects = {} # global_id -> {'features':[], 'avg_feature': ndarray}
        self.session_locks = {}   # local_track_id -> global_id
        self.next_subject_id = 1
        
    def extract_feature(self, frame_crop):
        try:
            # Advanced Feature Matching: 3D Color Histograms in HSV space
            hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
            
            # Using 8 bins per channel -> 8x8x8 = 512 dimensional feature vector
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            
            # Normalize and flatten
            cv2.normalize(hist, hist)
            feat = hist.flatten()
            
            # To avoid divide by zero later, add tiny epsilon
            norm = np.linalg.norm(feat)
            if norm == 0:
                return None
            return feat / norm
        except Exception as e:
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

    def register_feature(self, subj_id, feature):
        if subj_id not in self.global_subjects:
            self.global_subjects[subj_id] = {'features': [feature], 'avg_feature': feature}
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
            # Add a slight background for readable text
            cv2.rectangle(annotated_frame, (x1, y1-25), (x1 + 120, y1), (255, 0, 0), -1)
            cv2.putText(annotated_frame, global_id, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return annotated_frame, alerts
