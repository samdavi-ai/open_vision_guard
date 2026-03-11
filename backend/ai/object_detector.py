from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load the YOLO model (nano version by default for speed)
        self.model = YOLO(model_path)
        
        # Define allowed classes to match prompt: person, car, motorcycle, bicycle, phone (cell phone), laptop, bag (handbag/backpack), animals (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
        self.allowed_classes = [
            0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, # Persons, vehicles, animals
            26, 27, # Handbag, tie (backpack is 24, handbag is 26)
            63, 67, # Laptop, cell phone
        ]

    def detect(self, frame: np.ndarray):
        """
        Run object detection on a single frame.
        """
        results = self.model(frame, classes=self.allowed_classes, verbose=False)
        return results[0] # Return the first result (single frame)

    def track(self, frame: np.ndarray, persist=True):
        """
        Run object tracking on a single frame using ByteTrack (default in YOLOv8 when tracking).
        """
        results = self.model.track(frame, classes=self.allowed_classes, persist=persist, tracker="bytetrack.yaml", verbose=False)
        return results[0]

    def draw_results(self, frame: np.ndarray, result):
        """
        Draw bounding boxes and labels on the frame.
        """
        annotated_frame = result.plot()
        return annotated_frame

# Example usage
if __name__ == "__main__":
    detector = ObjectDetector()
    # Dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame)
    print("Dummy detection complete")
