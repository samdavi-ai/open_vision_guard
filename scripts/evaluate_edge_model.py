"""Evaluate accuracy, false positives, and runtime FPS for edge models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def benchmark(weights: str, source: str, imgsz: int, device: str, frames: int) -> float:
    model = YOLO(weights)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open benchmark source: {source}")
    count = 0
    start = time.perf_counter()
    while count < frames:
        ok, frame = cap.read()
        if not ok:
            break
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
        count += 1
    elapsed = max(1e-6, time.perf_counter() - start)
    cap.release()
    return count / elapsed


def false_positive_rate(weights: str, negative_dir: str, imgsz: int, device: str, conf: float) -> float:
    model = YOLO(weights)
    image_paths = [p for p in Path(negative_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not image_paths:
        return 0.0
    fp = 0
    for path in image_paths:
        result = model.predict(str(path), imgsz=imgsz, device=device, conf=conf, verbose=False)[0]
        if result.boxes is not None and len(result.boxes) > 0:
            fp += 1
    return fp / len(image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenVisionGuard edge detector")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/openvisionguard_data.yaml")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark-source", default=None)
    parser.add_argument("--benchmark-frames", type=int, default=200)
    parser.add_argument("--negative-dir", default=None, help="Low-light/clutter images that should have no target detections")
    parser.add_argument("--conf", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = YOLO(args.weights).val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
    print(f"mAP@0.5={metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95={metrics.box.map:.4f}")
    print(f"precision={metrics.box.mp:.4f}")
    print(f"recall={metrics.box.mr:.4f}")

    if args.benchmark_source:
        fps = benchmark(args.weights, args.benchmark_source, args.imgsz, args.device, args.benchmark_frames)
        print(f"fps={fps:.2f}")

    if args.negative_dir:
        fpr = false_positive_rate(args.weights, args.negative_dir, args.imgsz, args.device, args.conf)
        print(f"false_positive_rate_negative_set={fpr:.4f}")


if __name__ == "__main__":
    main()
