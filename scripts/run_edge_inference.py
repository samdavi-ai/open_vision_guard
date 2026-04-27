"""Run the OpenVisionGuard edge pipeline on a camera or video file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openvisionguard_cv.pipeline import EdgeSurveillancePipeline, load_edge_config


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenVisionGuard edge inference")
    parser.add_argument("--config", default="configs/openvisionguard_edge.yaml")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--output", default=None, help="Optional annotated video output")
    parser.add_argument("--display", action="store_true", help="Show a local OpenCV preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means run the full source")
    parser.add_argument("--stats", action="store_true", help="Print processing FPS and detection summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = EdgeSurveillancePipeline(load_edge_config(args.config))
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    start = time.perf_counter()
    frames = 0
    processed_frames = 0
    detections_total = 0
    alerts_total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipeline.process_frame(frame)
        annotated = pipeline.draw(frame, result)
        frames += 1
        processed_frames += int(result.processed)
        detections_total += len(result.detections)
        alerts_total += len(result.alerts)

        if writer:
            writer.write(annotated)
        if args.display:
            cv2.imshow("OpenVisionGuard Edge", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_frames and frames >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()
    if args.stats:
        elapsed = max(1e-6, time.perf_counter() - start)
        print(
            f"frames={frames} processed_frames={processed_frames} "
            f"avg_fps={frames / elapsed:.2f} detector_calls_per_frame={processed_frames / max(frames, 1):.3f} "
            f"detections_total={detections_total} alerts_total={alerts_total}"
        )


if __name__ == "__main__":
    main()
