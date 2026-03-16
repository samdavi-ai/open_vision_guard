import argparse
import os
import cv2
import sys

# Ensure this script can find backend modules if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from ultralytics import YOLO
from ai.reid_tracker import ReIDTracker
from ai.face_recognizer import FaceRecognizer
from ai.pose_estimator import PoseEstimator
from ai.motion_detector import MotionDetector
from ai.weapon_detector import WeaponDetector
from database_manager import DatabaseManager

def process_source(source_path, output_path, model, tracker, face_rec, pose_est, motion_det, weapon_det, db_manager, modules):
    is_webcam = str(source_path) == "0"
    
    cap = cv2.VideoCapture(int(source_path) if is_webcam else source_path)
    if not cap.isOpened():
        print(f"Error: Could not open source {source_path}")
        return

    # Video writer setup
    out = None
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        import imageio
        writer = imageio.get_writer(
            output_path,
            format='FFMPEG',
            fps=fps,
            codec='libx264',
            quality=8, # High quality
            pixelformat='yuv420p',
            macro_block_size=1
        )
        out = True # Flag

    print(f"Processing {source_path} with modules: {modules}...")
    
    tracker.session_locks.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame = frame.copy()
        alerts = []

        # Run YOLO Object Detection & Tracking first if any dependent module is active
        # We now track all objects (no class limit) to enable unique identification for everything
        results = model.track(frame, persist=True, verbose=False)
        if "yolo" in modules:
            annotated_frame = results[0].plot()

        # 1. Custom ReID Tracker (Processes all classes now)
        if "reid" in modules and results:
            annotated_frame, reid_alerts = tracker.process_frame_tracking(annotated_frame, results)
            alerts.extend(reid_alerts)
            
            # 2. Face Recognition - Link to Person IDs
            if "face" in modules:
                face_frame, faces = face_rec.detect_and_recognize(frame)
                # annotated_frame = face_frame # We'll draw our own unified boxes if needed, but ReID does it
                
                # Logic to link faces to tracked person IDs using IoU
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
                    yolo_ids = results[0].boxes.id.int().cpu().numpy()
                    yolo_cls = results[0].boxes.cls.int().cpu().numpy()
                    
                    # Face detection returns locations as (top, right, bottom, left)
                    # We need to match these to YOLO person boxes
                    for face_loc, face_name in zip(face_rec.last_face_locations, face_rec.last_face_names):
                        if face_name == "Unknown": continue
                        
                        ftop, fright, fbottom, fleft = [v * 4 for v in face_loc] # Scaled back up
                        
                        best_iou = 0
                        best_track_id = None
                        
                        for box, tid, cid in zip(yolo_boxes, yolo_ids, yolo_cls):
                            if cid != 0: continue # Only persons have faces
                            
                            x1, y1, x2, y2 = box
                            # Intersection
                            ix1 = max(x1, fleft)
                            iy1 = max(y1, ftop)
                            ix2 = min(x2, fright)
                            iy2 = min(y2, fbottom)
                            
                            if ix2 > ix1 and iy2 > iy1:
                                area = (ix2 - ix1) * (iy2 - iy1)
                                if area > best_iou:
                                    best_iou = area
                                    best_track_id = tid
                        
                        if best_track_id is not None:
                            global_id = tracker.session_locks.get(best_track_id)
                            if global_id:
                                if tracker.update_identity(global_id, face_name):
                                    alerts.append(f"Identity Verified: {face_name} is {global_id}")

        # 3. Pose & Activity - Link to Person IDs
        if "pose" in modules and results:
            annotated_frame, keypoints = pose_est.detect_pose(annotated_frame)
            active_results = pose_est.model(frame, classes=[0], verbose=False)[0] # Need boxes to link
            
            if active_results.boxes is not None:
                pose_boxes = active_results.boxes.xyxy.cpu().numpy()
                for i, kp in enumerate(keypoints):
                    activity = pose_est.analyze_activity([kp])
                    if activity != "Normal":
                        # Match pose box to tracked person
                        pbox = pose_boxes[i]
                        best_tid = None
                        best_iou = 0
                        if results[0].boxes is not None and results[0].boxes.id is not None:
                            for box, tid, cid in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.int().cpu().numpy(), results[0].boxes.cls.int().cpu().numpy()):
                                if cid != 0: continue
                                # IoU
                                ix1, iy1 = max(pbox[0], box[0]), max(pbox[1], box[1])
                                ix2, iy2 = min(pbox[2], box[2]), min(pbox[3], box[3])
                                if ix2 > ix1 and iy2 > iy1:
                                    iou = (ix2 - ix1) * (iy2 - iy1)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_tid = tid
                        
                        id_str = tracker.get_display_name(tracker.session_locks.get(best_tid, "Unknown")) if best_tid else "Unknown"
                        alerts.append(f"Activity Alert: {id_str} is {activity}")

        # 4. Motion & Anomaly
        if "motion" in modules:
            annotated_frame, fg_mask, motion_detected, anomaly = motion_det.detect_motion(annotated_frame)
            if motion_detected:
                alerts.append("Motion Detected")
            if anomaly != "None":
                alerts.append(f"Anomaly: {anomaly}")

        # 5. Weapon Detection - Link to nearest Person
        if "weapon" in modules:
            annotated_frame, weapons_found = weapon_det.detect_weapon(annotated_frame)
            # Find boxes for weapons
            weapon_results = weapon_det.model(frame, classes=weapon_det.weapon_classes, verbose=False)[0]
            if weapon_results.boxes is not None:
                for w_box in weapon_results.boxes:
                    w_xyxy = w_box.xyxy.cpu().numpy()[0]
                    w_cls = weapon_det.model.names[int(w_box.cls)]
                    
                    # Find nearest tracked person
                    best_tid = None
                    min_dist = float('inf')
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        for box, tid, cid in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.int().cpu().numpy(), results[0].boxes.cls.int().cpu().numpy()):
                            if cid != 0: continue
                            # Distance between centers
                            w_center = [(w_xyxy[0] + w_xyxy[2])/2, (w_xyxy[1] + w_xyxy[3])/2]
                            p_center = [(box[0] + box[2])/2, (box[1] + box[3])/2]
                            dist = ((w_center[0] - p_center[0])**2 + (w_center[1] - p_center[1])**2)**0.5
                            if dist < min_dist:
                                min_dist = dist
                                best_tid = tid
                    
                    if best_tid and min_dist < 200: # Threshold for "holding"
                        id_str = tracker.get_display_name(tracker.session_locks.get(best_tid, "Unknown"))
                        alerts.append(f"WEAPON ALERT: {id_str} holding {w_cls}")
                    else:
                        alerts.append(f"WEAPON ALERT: Unattended {w_cls} detected")

        # Draw Clean Alerts
        y_offset = 40
        unique_alerts = []
        for a in alerts:
            if a not in unique_alerts:
                unique_alerts.append(a)
                
        for alert in unique_alerts:
            # Log to database
            db_manager.log_alert("Security Alert", alert)
            
            # Add black background for text
            (w, h), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated_frame, (10, y_offset - int(h) - 5), (10 + int(w) + 10, y_offset + 5), (0, 0, 0), -1)
            # Use bright orange for alerts
            cv2.putText(annotated_frame, alert, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            y_offset += 40

        if out:
            # ImageIO expects RGB instead of BGR (from cv2)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)

    cap.release()
    if out:
        writer.close()

def main():
    parser = argparse.ArgumentParser(description="Security-focused object detection & Re-ID")
    parser.add_argument("--input", required=True, help="Input source: '0' for webcam, or path to video file, or path to folder")
    parser.add_argument("--output", default="outputs", help="Output folder (default: outputs/)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or 0)")
    parser.add_argument("--model", default="yolov8m.pt", help="Path to YOLO model (default: yolov8m.pt)")
    parser.add_argument("--similarity_threshold", type=float, default=0.6, help="Re-ID similarity threshold (default: 0.6)")
    parser.add_argument("--modules", default="yolo,reid,face,pose,motion,weapon", help="Comma-separated list of modules to run")

    args = parser.parse_args()
    
    modules = [m.strip().lower() for m in args.modules.split(',')]

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    if args.input != "0" and not os.path.exists(args.input):
        print(f"Error: Input path {args.input} does not exist.")
        return

    print("Loading models...")
    model = YOLO(args.model)
    tracker = ReIDTracker(similarity_threshold=args.similarity_threshold, device=args.device)
    registry_path = os.path.join(args.output, "reid_registry.pkl")
    tracker.load_registry(registry_path)
    
    db_path = os.path.join(args.output, "security_log.db")
    db_manager = DatabaseManager(db_path)
    
    face_rec = FaceRecognizer()
    pose_est = PoseEstimator()
    motion_det = MotionDetector()
    weapon_det = WeaponDetector()

    # Process based on input type
    if os.path.isdir(args.input):
        # Batch process folder
        valid_exts = (".mp4", ".avi", ".mov", ".mkv")
        files = [f for f in os.listdir(args.input) if f.lower().endswith(valid_exts)]
        if not files:
            print(f"No valid video files found in {args.input}")
            return
            
        for file in files:
            in_path = os.path.join(args.input, file)
            out_path = os.path.join(args.output, f"annotated_{file}")
            process_source(in_path, out_path, model, tracker, face_rec, pose_est, motion_det, weapon_det, db_manager, modules)
    else:
        # Single file or webcam
        out_name = "annotated_live.mp4" if args.input == "0" else f"annotated_{os.path.basename(args.input)}"
        out_path = os.path.join(args.output, out_name)
        process_source(args.input, out_path, model, tracker, face_rec, pose_est, motion_det, weapon_det, db_manager, modules)

    print("Saving registry...")
    tracker.save_registry(registry_path)
    print("Done!")

if __name__ == "__main__":
    main()
