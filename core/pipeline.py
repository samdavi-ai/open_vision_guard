import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

from config import config
from modules.embedding_engine import embedding_engine
from modules.face_recognition_module import FaceRecognitionModule
from modules.pose_analyzer import PoseAnalyzer
from modules.motion_detector import MotionDetector
from modules.weapon_detector import WeaponDetector
from modules.alert_engine import alert_engine
from modules import database


@dataclass
class PipelineResult:
    annotated_frame: np.ndarray = None
    identities: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    current_detections: List[Dict[str, Any]] = field(default_factory=list)


class Pipeline:
    def __init__(self, yolo_model_path: str = "yolov8n.pt", device: str = "cpu"):
        print("Loading pipeline models (lightweight mode)...")
        self.yolo_model = YOLO(yolo_model_path)  # nano: fastest model
        self.face_module = FaceRecognitionModule()
        self.pose_analyzer = PoseAnalyzer()
        self.motion_detector = MotionDetector()
        self.weapon_detector = WeaponDetector()
        self.device = device

        # Per-camera zone configs
        self.camera_zones: Dict[str, List[Dict]] = {}

        # Initialize database
        database.init_db()
        print("Pipeline ready.")

    def process_frame(self, frame: np.ndarray, camera_id: str = "CAM_01") -> PipelineResult:
        """
        Main per-frame processing pipeline:
        detect → embed → identify → analyze → alert
        """
        result = PipelineResult()
        orig_h, orig_w = frame.shape[:2]
        annotated_frame = frame.copy()
        alerts_list = []
        current_detections_list = []
        current_time = time.time()

        # Resize to 320px max for fastest inference (scale back coords afterwards)
        max_dim = 320
        scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
        if scale < 1.0:
            inf_w, inf_h = int(orig_w * scale), int(orig_h * scale)
            inf_frame = cv2.resize(frame, (inf_w, inf_h))
        else:
            inf_frame = frame
            scale = 1.0

        # 1. YOLOv8n Detection + Tracking (320px — fast on CPU)
        yolo_results = self.yolo_model.track(inf_frame, persist=True, verbose=False, imgsz=320)
        # annotated_frame = yolo_results[0].plot() # Remove cluttered default drawings

        # Extract person detections
        person_boxes = []
        person_ids = []

        if yolo_results[0].boxes is not None and yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            track_ids = yolo_results[0].boxes.id.int().cpu().numpy()
            cls_ids = yolo_results[0].boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                if cls_id != 0:  # Only persons (COCO class 0)
                    continue

                # Scale coords back to original frame size
                x1 = int(box[0] / scale); y1 = int(box[1] / scale)
                x2 = int(box[2] / scale); y2 = int(box[3] / scale)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # 2. Embedding → Persistent Identity
                global_id = embedding_engine.get_or_create_identity(crop)
                embedding_engine.update_identity_metadata(global_id, {
                    "last_seen_camera": camera_id,
                    "last_seen_time": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                })

                person_boxes.append(box)
                person_ids.append(global_id)

                # Build display name for the frontend
                display_name = global_id
                identity = embedding_engine.get_identity(global_id)
                if identity and identity['metadata'].get('face_name'):
                    display_name = f"{identity['metadata']['face_name']} ({global_id})"

                # Save detection for frontend interactive overlays
                current_detections_list.append({
                    "global_id": global_id,
                    "bbox": [x1, y1, x2, y2],
                    "display_name": display_name
                })
                            
                # 3. Face Recognition (if face visible)
                if config.face_recognition_enabled and (y2 - y1) > config.min_face_height_px:
                    name = self.face_module.recognize_face(crop)
                    if name:
                        embedding_engine.update_identity_metadata(global_id, {"face_name": name})

                # 4. Pose & Activity
                if config.pose_enabled:
                    pose_result = self.pose_analyzer.analyze_pose(frame, (x1, y1, x2, y2))
                    embedding_engine.update_identity_metadata(global_id, {"activity": pose_result["activity"]})

                    if pose_result["fall_detected"]:
                        alert = alert_engine.create_alert("fall", global_id, camera_id, pose_result, frame)
                        if alert:
                            alerts_list.append(alert)

                # 5. Loitering check
                if self.motion_detector.check_loitering(global_id, (x1, y1, x2, y2), current_time):
                    alert = alert_engine.create_alert("loitering", global_id, camera_id, {}, frame)
                    if alert:
                        alerts_list.append(alert)

                # 6. Save event to DB
                identity_data = embedding_engine.get_identity(global_id)
                database.save_identity({
                    "global_id": global_id,
                    "face_name": identity_data['metadata'].get('face_name') if identity_data else None,
                    "risk_level": identity_data['metadata'].get('risk_level', 'low') if identity_data else 'low',
                    "metadata": identity_data['metadata'] if identity_data else {}
                })

        # 7. Object/Bag Possession Tracking
        # COCO classes: backpack=24, handbag=26, suitcase=28
        OBJECT_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}
        if yolo_results[0].boxes is not None:
            all_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            all_cls = yolo_results[0].boxes.cls.int().cpu().numpy()

            object_detections = []
            for obj_box, obj_cls in zip(all_boxes, all_cls):
                if int(obj_cls) in OBJECT_CLASSES:
                    ox1, oy1, ox2, oy2 = map(int, obj_box)
                    obj_cx = (ox1 + ox2) / 2
                    obj_cy = (oy1 + oy2) / 2
                    object_detections.append({
                        "type": OBJECT_CLASSES[int(obj_cls)],
                        "center": (obj_cx, obj_cy),
                        "bbox": [ox1, oy1, ox2, oy2]
                    })

            # Link objects to nearest person
            for obj in object_detections:
                min_dist = float('inf')
                nearest_pid = None
                for pbox, pid in zip(person_boxes, person_ids):
                    px1, py1, px2, py2 = map(int, pbox)
                    pcx = (px1 + px2) / 2
                    pcy = (py1 + py2) / 2
                    dist = ((obj["center"][0] - pcx)**2 + (obj["center"][1] - pcy)**2)**0.5
                    if dist < min_dist and dist < 200:  # Max 200px linkage
                        min_dist = dist
                        nearest_pid = pid

                if nearest_pid:
                    identity = embedding_engine.get_identity(nearest_pid)
                    if identity:
                        prev_objects = set(identity['metadata'].get('carried_objects', []))
                        new_obj = obj["type"]
                        if new_obj not in prev_objects:
                            # Person acquired a new object
                            prev_objects.add(new_obj)
                            embedding_engine.update_identity_metadata(nearest_pid, {
                                "carried_objects": list(prev_objects)
                            })
                            if len(prev_objects) > 1:
                                alert = alert_engine.create_alert(
                                    "object_acquired", nearest_pid, camera_id,
                                    {"object": new_obj}, frame
                                )
                                if alert:
                                    alerts_list.append(alert)
                        else:
                            embedding_engine.update_identity_metadata(nearest_pid, {
                                "carried_objects": list(prev_objects)
                            })

            # Update detection list with carried objects
            for det in current_detections_list:
                gid = det["global_id"]
                identity = embedding_engine.get_identity(gid)
                if identity:
                    det["carried_objects"] = identity['metadata'].get('carried_objects', [])
                    det["activity"] = identity['metadata'].get('activity', 'unknown')
                    det["risk_level"] = identity['metadata'].get('risk_level', 'low')

        # 8. Weapon Detection (full frame)
        if config.weapon_detection_enabled:
            weapons = self.weapon_detector.detect_weapons(frame, person_boxes, person_ids)
            for w in weapons:
                pid = w.get("nearest_person_id", "Unknown")
                if pid:
                    embedding_engine.update_identity_metadata(pid, {"risk_level": "critical"})
                alert = alert_engine.create_alert("weapon", pid or "Unknown", camera_id, w, frame)
                if alert:
                    alerts_list.append(alert)

        # 9. Motion / Zone breach
        zones = self.camera_zones.get(camera_id, [])
        motion_result = self.motion_detector.detect_motion(frame, zones)
        if motion_result["active_zones"]:
            for zone_name in motion_result["active_zones"]:
                alert = alert_engine.create_alert(
                    "zone_breach", "Unknown", camera_id,
                    {"zone_name": zone_name}, frame
                )
                if alert:
                    alerts_list.append(alert)

        result.annotated_frame = annotated_frame
        result.identities = embedding_engine.get_all_identities()
        result.alerts = alerts_list
        result.current_detections = current_detections_list

        return result

