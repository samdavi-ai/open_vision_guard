import argparse
import os
import cv2
import sys

# Ensure this script can find backend modules if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from ultralytics import YOLO
from ai.reid_tracker import ReIDTracker

def process_source(source_path, output_path, model, tracker):
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
        if fps == 0:
            fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {source_path}...")
    
    # In live tracking, we clear session locks if we want distinct videos to share IDs but not tracks 
    # But for a single detect.py run over a folder, we might want to keep the same ReIDTracker instance!
    tracker.session_locks.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO with tracking enabled (BoT-SORT or ByteTrack built into Ultralytics)
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        
        # Pass to custom ReID Tracker for global subject matching
        annotated_frame, alerts = tracker.process_frame_tracking(frame, results)

        for alert in alerts:
            print(f"[RE-ID ALERT] {alert}")

        if out:
            out.write(annotated_frame)

        if is_webcam:
            cv2.imshow("Live Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
    if is_webcam:
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Security-focused object detection & Re-ID")
    parser.add_argument("--input", required=True, help="Input source: '0' for webcam, or path to video file, or path to folder")
    parser.add_argument("--output", default="outputs", help="Output folder (default: outputs/)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or 0)")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model (default: yolov8n.pt)")
    parser.add_argument("--similarity_threshold", type=float, default=0.85, help="Re-ID similarity threshold (default: 0.85)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    if args.input != "0" and not os.path.exists(args.input):
        print(f"Error: Input path {args.input} does not exist.")
        return

    print("Loading models...")
    model = YOLO(args.model)
    tracker = ReIDTracker(similarity_threshold=args.similarity_threshold, device=args.device)

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
            process_source(in_path, out_path, model, tracker)
    else:
        # Single file or webcam
        out_name = "annotated_live.mp4" if args.input == "0" else f"annotated_{os.path.basename(args.input)}"
        out_path = os.path.join(args.output, out_name)
        process_source(args.input, out_path, model, tracker)

    print("Done!")

if __name__ == "__main__":
    main()
