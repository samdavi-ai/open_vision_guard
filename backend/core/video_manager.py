import cv2
import threading
import time
import base64
from typing import Dict, Any, List

# Import AI Modules
from ai.object_detector import ObjectDetector
from ai.face_recognizer import FaceRecognizer
from ai.pose_estimator import PoseEstimator
from ai.motion_detector import MotionDetector
from ai.weapon_detector import WeaponDetector
from ai.reid_tracker import ReIDTracker

class VideoManager:
    def __init__(self, ws_manager):
        self.streams = {} # camera_id: thread
        self.configs = {} # camera_id: config dict
        self.ws_manager = ws_manager
        
        # Load AI Models ONCE to save memory
        self.object_detector = ObjectDetector()
        self.face_recognizer = FaceRecognizer()
        self.pose_estimator = PoseEstimator()
        self.motion_detector = MotionDetector()
        self.weapon_detector = WeaponDetector()
        # Single global tracker for cross-camera subjects
        self.reid_tracker = ReIDTracker()

    def start_stream(self, camera_id: str, url: str, modules: List[str]):
        """
        Starts a background thread to process a specific camera stream.
        """
        if camera_id in self.streams:
            return # Already running
            
        self.configs[camera_id] = {"url": url, "modules": modules, "running": True}
        thread = threading.Thread(target=self._process_stream, args=(camera_id,))
        thread.daemon = True
        thread.start()
        self.streams[camera_id] = thread
        
    def stop_stream(self, camera_id: str):
        if camera_id in self.configs:
            self.configs[camera_id]["running"] = False
            
    def _process_stream(self, camera_id: str):
        config = self.configs[camera_id]
        url = config["url"]
        
        # Handle webcam vs RTSP / file
        if url.isdigit():
            cap = cv2.VideoCapture(int(url))
        else:
            cap = cv2.VideoCapture(url)
            
        print(f"Starting stream {camera_id} from {url}")
        
        while config["running"]:
            ret, frame = cap.read()
            if not ret:
                # If it's a file, loop it. Assume non-digit URLs are files/RTSP
                if not str(url).isdigit() and cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    
                if not ret:
                    print(f"Failed to read from {camera_id}. Retrying...")
                    time.sleep(1)
                    continue
                
            annotated_frame = frame.copy()
            alerts = []
            
            # 1. Object Detection & Tracking
            if "object_detector" in config["modules"]:
                results = self.object_detector.track(annotated_frame)
                annotated_frame = self.object_detector.draw_results(annotated_frame, results)

            # 1.5 Re-ID Tracking
            if "reid_tracking" in config["modules"]:
                # Require tracking IDs
                results = self.object_detector.track(annotated_frame)
                annotated_frame, reid_alerts = self.reid_tracker.process_frame_tracking(annotated_frame.copy(), [results])
                alerts.extend(reid_alerts)
                
            # 2. Face Recognition
            if "face_recognition" in config["modules"]:
                annotated_frame, face_names = self.face_recognizer.detect_and_recognize(annotated_frame)
                for name in face_names:
                    if name != "Unknown":
                        alerts.append(f"Recognized Face: {name}")
                        
            # 3. Pose & Activity (Fight)
            if "pose" in config["modules"]:
                annotated_frame, landmarks = self.pose_estimator.detect_pose(annotated_frame)
                activity = self.pose_estimator.analyze_activity(landmarks)
                if activity != "Normal":
                    alerts.append(f"Activity Detected: {activity}")
                    if activity == "Falling":
                        # Draw warning text
                        cv2.putText(annotated_frame, "FALL DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 4. Motion & Anomaly
            if "motion" in config["modules"]:
                annotated_frame, fg_mask, motion_detected, anomaly = self.motion_detector.detect_motion(annotated_frame)
                if motion_detected:
                     alerts.append(f"Motion Detected")
                if anomaly != "None":
                     alerts.append(f"Anomaly: {anomaly}")
                     
            # 5. Weapon Detection
            if "weapon" in config["modules"]:
                annotated_frame, weapons_found = self.weapon_detector.detect_weapon(annotated_frame)
                for w in weapons_found:
                    alerts.append(f"Weapon Alert: {w['class']} ({w['confidence']:.2f})")
                    cv2.putText(annotated_frame, f"WEAPON DETECTED: {w['class']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Encode frame to JPG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # We need to dispatch via asyncio event loop since FastAPI websockets are async
            import asyncio
            try:
                 loop = asyncio.get_event_loop()
            except RuntimeError:
                 loop = asyncio.new_event_loop()
                 asyncio.set_event_loop(loop)
                 
            # Send to websockets
            data = {
                "camera_id": camera_id,
                "frame": frame_base64,
                "alerts": alerts
            }
            
            # Normally we should push this to a queue and let async worker broadcast it
            # For simplicity, we directly broadcast here using create_task if in running loop
            import json
            # This is tricky because manager is in uvicorn loop.
            # Best approach: Call an HTTP webhook internally, or use a shared async queue
            # I will write the frame to a global dictionary so ws_router can pull it,
            # or use an async-safe callback.
            
            # Simple global storage for polling fallback if WS push fails
            global_frame_store[camera_id] = data

        cap.release()
        del self.streams[camera_id]
        print(f"Stream {camera_id} stopped.")

# Global store for frames (quick hack for polling/ws integration)
global_frame_store = {}
