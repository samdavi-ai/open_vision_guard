import cv2
import numpy as np

class MotionDetector:
    def __init__(self, history=500, varThreshold=16, detectShadows=True):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
        self.min_area = 500  # Minimum area to be considered motion
        
    def detect_motion(self, frame: np.ndarray):
        """
        Detect motion and anomalies like sudden bursts of motion.
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Threat shadows as background
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        
        # Dilate mask to merge broken parts of an object
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated_frame = frame.copy()
        motion_detected = False
        anomaly = "None"
        total_motion_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
                
            motion_detected = True
            total_motion_area += area
            
            # Form bounding box around the motion
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated_frame, 'Motion Detected', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Basic anomaly detection based on sudden massive motion (e.g. crowd running, explosion)
        frame_area = frame.shape[0] * frame.shape[1]
        
        if motion_detected:
            if total_motion_area > (frame_area * 0.4): # Over 40% of the screen is moving
                anomaly = "Massive Motion / Disturbance"
            elif total_motion_area > (frame_area * 0.15):
                anomaly = "Moderate Motion"

        return annotated_frame, fg_mask, motion_detected, anomaly

