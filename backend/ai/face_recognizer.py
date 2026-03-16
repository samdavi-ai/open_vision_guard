try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False

import cv2
import os
import numpy as np

class FaceRecognizer:
    def __init__(self, faces_dir="faces"):
        self.faces_dir = faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        
        self.last_face_locations = []
        self.last_face_names = []
        
        if not HAS_FACE_RECOGNITION:
            print("Warning: face_recognition module not installed. Face recognition will be disabled.")
            return

        # Create dir if not exists
        os.makedirs(self.faces_dir, exist_ok=True)
        self.load_known_faces()

    def load_known_faces(self):
        """
        Loads and encodes all faces from the `faces/` directory.
        Folder structure should be:
        faces/
          sam/
            sam1.jpg
          john/
            john1.jpg
        """
        if not HAS_FACE_RECOGNITION: return
        if not os.path.exists(self.faces_dir):
            print(f"Faces directory {self.faces_dir} not found.")
            return
            
        print(f"Indexing faces in {self.faces_dir}...")
        for root, dirs, files in os.walk(self.faces_dir):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(root, file)
                    # Use directory name as person name
                    name = os.path.basename(root)
                    
                    try:
                        # Load image and find encodings
                        image = face_recognition.load_image_file(path)
                        # Perform jittering for higher accuracy (multi-sample averaging)
                        encodings = face_recognition.face_encodings(image, num_jitters=10, model="large")
                        
                        if len(encodings) > 0:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(name)
                            print(f"  Encoded: {name} ({file}) [High Precision]")
                        else:
                            print(f"  Warning: No face found in {path}")
                    except Exception as e:
                        print(f"  Error loading face {path}: {e}")
                        
        print(f"Loaded {len(self.known_face_names)} faces from DB.")

    def detect_and_recognize(self, frame: np.ndarray):
        """
        Detect faces in frame and identify them.
        Returns the original frame annotated, and a list of detected names.
        """
        if not HAS_FACE_RECOGNITION:
            return frame.copy(), []
        
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            if len(self.known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        self.last_face_locations = face_locations
        self.last_face_names = face_names

        # Draw the results
        annotated_frame = frame.copy()
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(annotated_frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

        return annotated_frame, face_names
