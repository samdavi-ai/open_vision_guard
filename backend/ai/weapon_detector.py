import os
import cv2
import numpy as np
from ultralytics import YOLO
import sys

class WeaponDetector:
    def __init__(self, model_path="weapon_yolov8.pt"):
        """
        Loads a custom YOLOv8 model trained on weapons (guns, knives).
        If the model does not exist, it falls back to yolov8n.pt and 
        looks for generic objects just to prevent crashing, but will print a warning.
        """
        self.weapon_classes = [] # We'll populate based on the model
        
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            # Find the indices of 'gun', 'knife', 'weapon', etc.
            names = self.model.names
            for idx, name in names.items():
                if any(w in name.lower() for w in ['gun', 'knife', 'weapon', 'rifle', 'pistol']):
                    self.weapon_classes.append(idx)
                    
            if not self.weapon_classes:
               # If the model has different names, we just allow all classes
               self.weapon_classes = list(self.model.names.keys())

        else:
            print(f"Warning: Custom weapon model {model_path} not found.")
            print("To enable real weapon detection, train a YOLOv8 model on a weapon dataset and save it as weapon_yolov8.pt.")
            print("Falling back to yolov8n.pt with generic object detection for demonstration.")
            self.model = YOLO("yolov8n.pt") # fallback
            # In COCO, class 43 is 'knife', class 74 is 'clock' (placeholder). 
            # We'll just use knife as it's the closest default. 
            self.weapon_classes = [43] # knife

    def detect_weapon(self, frame: np.ndarray):
        results = self.model(frame, classes=self.weapon_classes, verbose=False)
        result = results[0]
        
        annotated_frame = result.plot()
        
        weapons_found = []
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            conf = float(box.conf)
            weapons_found.append({"class": class_name, "confidence": conf})
            
        return annotated_frame, weapons_found
