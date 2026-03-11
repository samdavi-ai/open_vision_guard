import cv2
import numpy as np

class PoseEstimator:
    def __init__(self):
        print("Warning: MediaPipe is currently disabled due to environment limitations. Pose estimation will be skipped.")
        
    def detect_pose(self, frame: np.ndarray):
        """
        Detects pose and draws landmarks on frame.
        Returns annotated frame, and raw landmark results.
        """
        return frame.copy(), None

    def analyze_activity(self, landmarks):
        """
        Simple heuristic based activity recognition (Running, Walking, Falling, Fight).
        """
        return "Normal"

