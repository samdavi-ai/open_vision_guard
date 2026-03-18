try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

import cv2
import os
import numpy as np
from typing import Dict, List, Optional, Any
from config import config


class FaceRecognitionModule:
    def __init__(self):
        self.known_faces_dir = config.known_faces_dir
        os.makedirs(self.known_faces_dir, exist_ok=True)
        
        if not HAS_DEEPFACE:
            print("Warning: deepface library not installed. Face recognition disabled.")
            return

        print(f"Face Recognition Module initialized with DeepFace (Fallback).")

    def load_known_faces(self, faces_dir: str) -> Dict[str, Any]:
        """DeepFace handles its own database indexing during find()."""
        return {}

    def recognize_face(self, face_crop: np.ndarray) -> Optional[str]:
        """
        Takes a face crop (BGR), returns the recognized name or None using DeepFace.
        """
        if not HAS_DEEPFACE:
            return None

        if face_crop.shape[0] < config.min_face_height_px:
            return None

        try:
            # DeepFace.find expects a path or a numpy array (BGR)
            # It will look into the db_path for matches
            results = DeepFace.find(
                img_path=face_crop,
                db_path=self.known_faces_dir,
                model_name="VGG-Face",
                enforce_detection=False,
                silent=True
            )

            if len(results) > 0 and not results[0].empty:
                # results[0] is a pandas DataFrame
                best_match_path = results[0].iloc[0]['identity']
                # The directory name is the person's name
                name = os.path.basename(os.path.dirname(best_match_path))
                return name

        except Exception as e:
            # print(f"DeepFace error: {e}")
            pass

        return None

    def detect_and_recognize_frame(self, frame: np.ndarray):
        """
        Full frame face detection + recognition using DeepFace.
        Note: This is slower than per-crop recognition.
        """
        if not HAS_DEEPFACE:
            return frame.copy(), []

        try:
            # Detect and recognize all faces in one go
            results = DeepFace.find(
                img_path=frame,
                db_path=self.known_faces_dir,
                model_name="VGG-Face",
                enforce_detection=True,
                silent=True
            )
            
            annotated = frame.copy()
            names = []
            
            for df in results:
                if not df.empty:
                    match = df.iloc[0]
                    name = os.path.basename(os.path.dirname(match['identity']))
                    names.append(name)
                    
                    # Draw box if coordinates available
                    # DeepFace 'find' returns 'source_x', 'source_y', 'source_w', 'source_h' in the dataframe
                    x, y, w, h = int(match['source_x']), int(match['source_y']), int(match['source_w']), int(match['source_h'])
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(annotated, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return annotated, names
        except Exception:
            return frame.copy(), []
