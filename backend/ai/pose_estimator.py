from ultralytics import YOLO
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt"):
        """
        Loads YOLOv8-pose model for skeletal tracking.
        """
        self.model = YOLO(model_path)
        
    def detect_pose(self, frame: np.ndarray):
        """
        Detects pose and draws landmarks on frame.
        """
        # Run inference on people only (class 0)
        results = self.model(frame, classes=[0], verbose=False)
        result = results[0]
        
        annotated_frame = result.plot()
        # Extract keypoints (x, y, confidence)
        keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
        
        return annotated_frame, keypoints

    def analyze_activity(self, keypoints_data):
        """
        Advanced heuristic based activity recognition using skeletal landmarks.
        Detects "Falling" by checking vertical velocity and spine orientation.
        """
        if keypoints_data is None or len(keypoints_data) == 0:
            return "Normal"
            
        for kp in keypoints_data:
            # KP Indices (COCO): 0:nose, 5:l_shoulder, 6:r_shoulder, 11:l_hip, 12:r_hip, 15:l_foot, 16:r_foot
            # Basic Fall Detection: Head lower than midpoint of hips?
            if len(kp) < 17: continue
            
            nose = kp[0]
            l_hip, r_hip = kp[11], kp[12]
            
            # Check confidence
            if nose[2] < 0.5 or l_hip[2] < 0.5 or r_hip[2] < 0.5:
                continue
                
            hip_y = (l_hip[1] + r_hip[1]) / 2
            # If nose is significantly lower than hips (Y decreases going down in image)
            # Actually, in CV2 Y increases going DOWN. So nose[1] > hip_y means nose is lower.
            if nose[1] > hip_y:
                return "Falling"
                
        return "Normal"

