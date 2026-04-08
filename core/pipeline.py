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
from modules.geolocation import geolocation_engine
import math
import os
import uuid

from modules.behaviour_analyzer import behaviour_analyzer
from modules.risk_engine import risk_engine
from modules.luggage_tracker import luggage_tracker
from modules.presence_tracker import presence_tracker
from modules.sudden_movement_detector import sudden_movement_detector
from modules.camera_avoidance_detector import camera_avoidance_detector
from modules.frequency_analyzer import frequency_analyzer


@dataclass
class PipelineResult:
    annotated_frame: np.ndarray = None
    identities: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    current_detections: List[Dict[str, Any]] = field(default_factory=list)


class Pipeline:
    # COCO class names for all 80 classes
    COCO_NAMES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
        38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
        42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
        47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
        52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
        57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
        66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
        70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
        78: 'hair drier', 79: 'toothbrush'
    }

    # Category groupings for color coding on the frontend
    CATEGORY_MAP = {
        'vehicle': {1, 2, 3, 4, 5, 6, 7, 8},
        'animal': {14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        'accessory': {24, 25, 26, 27, 28},
        'sports': {29, 30, 31, 32, 33, 34, 35, 36, 37, 38},
        'food': {46, 47, 48, 49, 50, 51, 52, 53, 54, 55},
        'furniture': {13, 56, 57, 59, 60},
        'electronic': {62, 63, 64, 65, 66, 67},
        'kitchen': {39, 40, 41, 42, 43, 44, 45, 68, 69, 70, 71, 72},
        'other': {9, 10, 11, 12, 58, 61, 73, 74, 75, 76, 77, 78, 79},
    }

    @staticmethod
    def get_object_category(cls_id: int) -> str:
        for cat, ids in Pipeline.CATEGORY_MAP.items():
            if cls_id in ids:
                return cat
        return 'other'

    def __init__(self, yolo_model_path: str = "yolov8n.pt", device: str = "cpu"):
        print("Loading pipeline models (Ultra-Fast Nano mode)...")
        self.yolo_model = YOLO(yolo_model_path)  # nano: fastest for real-time dynamic feel
        self.face_module = FaceRecognitionModule()
        self.pose_analyzer = PoseAnalyzer()
        self.motion_detector = MotionDetector()
        self.weapon_detector = WeaponDetector()
        self.device = device

        # Per-camera zone configs
        self.camera_zones: Dict[str, List[Dict]] = {}

        # Movement tracking: global_id → {prev_center, prev_time}
        self._prev_centers: Dict[str, Dict[str, Any]] = {}

        # Counter for non-person objects without tracker IDs
        self._obj_counter = 0

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
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        current_location = geolocation_engine.get_current_location()

        # 1. YOLOv8n Detection + Tracking (ALL classes)
        # Using 640px for accuracy (standard) + lower conf to catch "every person".
        yolo_results = self.yolo_model.track(
            frame, 
            persist=True, 
            verbose=False, 
            imgsz=640,
            conf=0.20,
            iou=0.45,
            tracker="bytetrack.yaml"  # sometimes smoother for high-density crowds
        )

        # Extract all detections
        person_boxes = []
        person_ids = []

        if yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            cls_ids = yolo_results[0].boxes.cls.int().cpu().numpy()
            confs = yolo_results[0].boxes.conf.cpu().numpy()
            has_ids = yolo_results[0].boxes.id is not None
            track_ids = yolo_results[0].boxes.id.int().cpu().numpy() if has_ids else [None] * len(boxes)

            for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
                # Coords are already in original frame space (YOLO maps them back)
                x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                x2, y2 = min(orig_w, int(box[2])), min(orig_h, int(box[3]))

                if x2 <= x1 or y2 <= y1:
                    continue

                # --- Accuracy Filter: Discard unrealistic "ghost" boxes ---
                # (e.g. extremely thin full-height boxes often seen on video edges or reflections)
                bw, bh = x2 - x1, y2 - y1
                if cls_id == 0:  # Person specific
                    # A person should not be a vertical line (aspect ratio < 0.1)
                    # or cover 100% height while being < 10% width (typical edge ghost)
                    if (bw / bh < 0.1) or (bh > orig_h * 0.9 and bw < orig_w * 0.1):
                        continue
                
                cls_id_int = int(cls_id)
                class_name = self.COCO_NAMES.get(cls_id_int, f'class_{cls_id_int}')
                category = self.get_object_category(cls_id_int)

                # ───── NON-PERSON OBJECTS ─────
                if cls_id_int != 0:
                    self._obj_counter += 1
                    obj_id = f"Obj_{track_id}" if track_id is not None else f"Obj_{self._obj_counter}"
                    current_detections_list.append({
                        "global_id": obj_id,
                        "bbox": [x1, y1, x2, y2],
                        "display_name": class_name,
                        "is_object": True,
                        "object_class": class_name,
                        "object_category": category,
                        "confidence": round(float(conf), 2),
                        "timestamp": now_iso,
                        "latitude": current_location["latitude"],
                        "longitude": current_location["longitude"]
                    })
                    
                    database.save_detection({
                        "object_id": obj_id,
                        "material": class_name,
                        "confidence": float(conf),
                        "size": f"{int(x2-x1)}x{int(y2-y1)}",
                        "timestamp": now_iso,
                        "latitude": current_location["latitude"],
                        "longitude": current_location["longitude"]
                    })
                    continue

                # ───── PERSON PROCESSING (cls_id == 0) ─────
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # 2. Embedding → Persistent Identity
                global_id = embedding_engine.get_or_create_identity(crop)
                now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                embedding_engine.update_identity_metadata(global_id, {
                    "last_seen_camera": camera_id,
                    "last_seen_time": now_iso,
                    "exit_time": now_iso,  # continuously update exit time
                    "latitude": current_location["latitude"],
                    "longitude": current_location["longitude"]
                })

                # --- Movement direction & speed ---
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                prev = self._prev_centers.get(global_id)
                speed = 0.0
                direction = "stationary"
                if prev:
                    dx = cx - prev["cx"]
                    dy = cy - prev["cy"]
                    dt = max(current_time - prev["t"], 0.001)
                    speed = ((dx**2 + dy**2)**0.5) / dt
                    if abs(dx) < 3 and abs(dy) < 3:
                        direction = "stationary"
                    elif abs(dx) > abs(dy):
                        direction = "right" if dx > 0 else "left"
                    else:
                        direction = "away" if dy > 0 else "towards"
                    embedding_engine.update_identity_metadata(global_id, {
                        "movement_direction": direction,
                        "speed": round(speed, 1),
                        "latitude": current_location["latitude"],
                        "longitude": current_location["longitude"]
                    })
                self._prev_centers[global_id] = {"cx": cx, "cy": cy, "t": current_time}

                # --- Intelligence Modules ---
                
                # Frequency
                frequency_analyzer.record_appearance(global_id, current_time)
                
                # Behaviour
                behaviour_res = behaviour_analyzer.update(global_id, cx, cy, speed, current_time)
                embedding_engine.update_identity_metadata(global_id, {
                    "behaviour_label": behaviour_res["behaviour_label"],
                    "behaviour_score": behaviour_res["behaviour_score"]
                })
                
                # Sudden Movement
                sudden_res = sudden_movement_detector.update(global_id, speed, current_time)
                if sudden_res:
                    alert = alert_engine.create_alert("sudden_movement", global_id, camera_id, sudden_res, frame)
                    if alert: alerts_list.append(alert)

                # Store scaled int coords (same space as current_detections_list)
                person_boxes.append([x1, y1, x2, y2])
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
                    "display_name": display_name,
                    "is_object": False,
                    "object_class": "person",
                    "object_category": "person",
                    "timestamp": now_iso,
                    "latitude": current_location["latitude"],
                    "longitude": current_location["longitude"]
                })
                
                # Save detection to DB
                database.save_detection({
                    "object_id": global_id,
                    "material": "person",
                    "confidence": float(conf),
                    "size": f"{int(x2-x1)}x{int(y2-y1)}",
                    "timestamp": now_iso,
                    "latitude": current_location["latitude"],
                    "longitude": current_location["longitude"]
                })
                
                # Save person log
                event_type = "moving"
                if prev:
                    event_type = "idle" if direction == "stationary" else "moving"
                database.save_person_log({
                    "person_id": global_id,
                    "timestamp": now_iso,
                    "position_x": cx,
                    "position_y": cy,
                    "speed": speed if prev else 0.0,
                    "zone": "General", 
                    "event_type": event_type
                })
                            
                # 3. Face Recognition (if face visible)
                face_visible = False
                if config.face_recognition_enabled and (y2 - y1) > config.min_face_height_px:
                    face_visible = True
                    name = self.face_module.recognize_face(crop)
                    if name:
                        embedding_engine.update_identity_metadata(global_id, {"face_name": name})
                        
                    # Face Logging
                    crop_filename = f"{uuid.uuid4()}.jpg"
                    crop_dir = "data/face_crops"
                    os.makedirs(crop_dir, exist_ok=True)
                    crop_path = os.path.join(crop_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    database.save_face_log({
                        "person_id": global_id,
                        "face_name": name or "Unknown",
                        "camera_id": camera_id,
                        "timestamp": now_iso,
                        "crop_path": crop_path,
                        "match_status": "known" if name else "unknown"
                    })

                # Camera Avoidance
                cam_dx = (cx - prev["cx"]) if prev else 0.0
                cam_dy = (cy - prev["cy"]) if prev else 0.0
                dir_angle = math.atan2(cam_dy, cam_dx)
                avoidance_res = camera_avoidance_detector.update(
                    global_id, (x1, y1, x2, y2), face_visible, dir_angle, orig_w, orig_h, current_time
                )
                embedding_engine.update_identity_metadata(global_id, {
                    "avoidance_score": avoidance_res["avoidance_score"],
                    "avoidance_flags": avoidance_res["avoidance_behaviours"]
                })

                # 4. Pose & Activity
                if config.pose_enabled:
                    pose_result = self.pose_analyzer.analyze_pose(frame, (x1, y1, x2, y2))
                    embedding_engine.update_identity_metadata(global_id, {
                        "activity": pose_result["activity"],
                        "pose_detail": pose_result["activity"],
                    })

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

        # 7. Presence Tracking
        presence_events = presence_tracker.update(person_ids, current_time)
        for ev in presence_events:
            # Map field names for database compatibility
            db_log = {
                "person_id": ev.get("person_id"),
                "event_type": ev.get("type"),  # presence tracker uses 'type', DB expects 'event_type'
                "timestamp": ev.get("timestamp"),
                "session_duration": ev.get("session_duration", 0.0),
            }
            database.save_presence_log(db_log)

        # Update dwell time & frequency for ALL visible persons every frame
        for pid in person_ids:
            tr = presence_tracker.get_presence_data(pid)
            if tr:
                freq_data = frequency_analyzer.get_frequency_data(pid)
                embedding_engine.update_identity_metadata(pid, {
                    "dwell_time_seconds": tr["total_dwell_seconds"],
                    "visit_count": freq_data["visit_count"],
                    "frequency_label": freq_data["frequency_label"],
                    "is_present": tr["is_present"]
                })

        # 8. Luggage Tracking
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

            # Create person positions map
            person_positions = {}
            for pbox, pid in zip(person_boxes, person_ids):
                px1, py1, px2, py2 = map(int, pbox)
                person_positions[pid] = ((px1 + px2) / 2, (py1 + py2) / 2)

            luggage_result = luggage_tracker.update(object_detections, person_positions, current_time)
            
            # Update person metadata with their luggage
            for pid in person_ids:
                pes_luggage = luggage_tracker.get_person_luggage(pid)
                carried = [l["type"] for l in pes_luggage.values() if l["status"] == "carried"]
                embedding_engine.update_identity_metadata(pid, {
                    "carried_objects": carried,
                    "luggage_status": pes_luggage
                })

            for ev in luggage_result["events"]:
                alert = alert_engine.create_alert(ev["type"], ev.get("owner_id") or ev.get("new_holder") or "Unknown", camera_id, ev, frame)
                if alert: alerts_list.append(alert)

        # 9. Weapon Detection (full frame)
        if config.weapon_detection_enabled:
            weapons = self.weapon_detector.detect_weapons(frame, person_boxes, person_ids)
            for w in weapons:
                pid = w.get("nearest_person_id", "Unknown")
                if pid:
                    embedding_engine.update_identity_metadata(pid, {"risk_level": "critical"})
                alert = alert_engine.create_alert("weapon", pid or "Unknown", camera_id, w, frame)
                if alert:
                    alerts_list.append(alert)

        # 10. Risk Engine & Following Check
        person_positions = {}
        for pbox, pid in zip(person_boxes, person_ids):
            px1, py1, px2, py2 = map(int, pbox)
            person_positions[pid] = ((px1 + px2) / 2, (py1 + py2) / 2)
            
        following_pairs = behaviour_analyzer.check_following(person_positions)
        following_pids = set()
        for pair in following_pairs:
            following_pids.add(pair["follower_id"])
            alert = alert_engine.create_alert("following", pair["follower_id"], camera_id, pair, frame)
            if alert: alerts_list.append(alert)
        
        for pid in person_ids:
            identity_data = embedding_engine.get_identity(pid)
            if not identity_data: continue
            m = identity_data['metadata']
            
            signals = {
                "weapon_proximity": m.get("risk_level") == "critical",
                "loitering": m.get("activity") == "loitering",
                "unknown_face": m.get("face_name") is None,
                "high_frequency": frequency_analyzer.is_frequent(pid),
                "following_someone": pid in following_pids,
                "prolonged_stillness": m.get("behaviour_label") == "prolonged_stillness",
                "pacing": m.get("behaviour_label") == "pacing",
                "circle_walking": m.get("behaviour_label") == "circle_walking",
                "running": m.get("behaviour_label") == "running",
            }
            
            risk_res = risk_engine.compute_risk(
                pid, signals, 
                behaviour_score=m.get("behaviour_score", 0.0),
                avoidance_score=m.get("avoidance_score", 0.0)
            )
            
            embedding_engine.update_identity_metadata(pid, {
                "risk_score": risk_res["risk_score"],
                "risk_level": risk_res["risk_level"],
                "risk_factors": risk_res["risk_factors"]
            })
            
            if risk_engine.should_alert(pid, threshold=70.0):
                alert = alert_engine.create_alert("high_risk", pid, camera_id, risk_res, frame)
                if alert: alerts_list.append(alert)

        # 11. Motion / Zone breach
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

        # 12. FINAL — Enrich ALL detection dicts with complete metadata for frontend
        for det in current_detections_list:
            if det.get("is_object"):
                continue
            gid = det["global_id"]
            identity = embedding_engine.get_identity(gid)
            if identity:
                m = identity['metadata']
                det["carried_objects"] = m.get('carried_objects', [])
                det["activity"] = m.get('activity', 'unknown')
                det["risk_level"] = m.get('risk_level', 'low')
                det["risk_score"] = m.get('risk_score', 0)
                det["behaviour_label"] = m.get('behaviour_label', 'normal')
                det["behaviour_score"] = m.get('behaviour_score', 0.0)
                det["movement_direction"] = m.get('movement_direction', 'stationary')
                det["speed"] = m.get('speed', 0.0)
                det["pose_detail"] = m.get('pose_detail', 'unknown')
                det["entry_time"] = m.get('entry_time')
                det["exit_time"] = m.get('exit_time')
                det["object_log"] = m.get('object_log', [])
                det["face_name"] = m.get('face_name')
                det["avoidance_score"] = m.get('avoidance_score', 0.0)
                det["avoidance_flags"] = m.get('avoidance_flags', [])
                det["risk_factors"] = m.get('risk_factors', [])
                det["dwell_time_seconds"] = m.get('dwell_time_seconds', 0.0)
                det["frequency_label"] = m.get('frequency_label', 'new')
                det["visit_count"] = m.get('visit_count', 1)
                det["luggage_status"] = m.get('luggage_status', {})

        result.annotated_frame = annotated_frame
        result.identities = embedding_engine.get_all_identities()
        result.alerts = alerts_list
        result.current_detections = current_detections_list

        return result

