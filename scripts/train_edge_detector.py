"""Train the OpenVisionGuard edge detector with transfer learning.

The script supports practical distillation through teacher-generated pseudo labels
and hard-negative mining for low-light/clutter false-positive reduction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Iterator

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openvisionguard_cv.constants import COCO_TO_SURVEILLANCE


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def iter_images_and_video_frames(source: Path, sample_every: int = 15) -> Iterator[tuple[str, object]]:
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTS:
            frame = cv2.imread(str(path))
            if frame is not None:
                yield path.stem, frame
        elif path.suffix.lower() in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(path))
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % sample_every == 0:
                    yield f"{path.stem}_{index:06d}", frame
                index += 1
            cap.release()


def write_yolo_label(label_path: Path, rows: Iterable[tuple[int, float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cls_id, x, y, w, h in rows:
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def generate_teacher_pseudo_labels(
    teacher_weights: str,
    source_dir: Path,
    output_dir: Path,
    imgsz: int,
    conf: float,
    sample_every: int,
) -> None:
    teacher = YOLO(teacher_weights)
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in iter_images_and_video_frames(source_dir, sample_every=sample_every):
        result = teacher.predict(frame, imgsz=imgsz, conf=conf, classes=list(COCO_TO_SURVEILLANCE), verbose=False)[0]
        rows = []
        if result.boxes is not None:
            h, w = frame.shape[:2]
            for box, cls_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.int().cpu().numpy()):
                src_cls = int(cls_id)
                if src_cls not in COCO_TO_SURVEILLANCE:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                cx = ((x1 + x2) * 0.5) / w
                cy = ((y1 + y2) * 0.5) / h
                bw = max(0.0, x2 - x1) / w
                bh = max(0.0, y2 - y1) / h
                rows.append((COCO_TO_SURVEILLANCE[src_cls], cx, cy, bw, bh))
        cv2.imwrite(str(image_dir / f"{name}.jpg"), frame)
        write_yolo_label(label_dir / f"{name}.txt", rows)


def mine_hard_negatives(
    weights: str,
    source_dir: Path,
    output_dir: Path,
    imgsz: int,
    conf: float,
    sample_every: int,
) -> None:
    model = YOLO(weights)
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in iter_images_and_video_frames(source_dir, sample_every=sample_every):
        result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        has_detection = result.boxes is not None and len(result.boxes) > 0
        if not has_detection:
            continue
        cv2.imwrite(str(image_dir / f"{name}.jpg"), frame)
        write_yolo_label(label_dir / f"{name}.txt", [])


def train(args: argparse.Namespace) -> Path:
    if args.architecture_yaml:
        model = YOLO(args.architecture_yaml).load(args.model)
    else:
        model = YOLO(args.model)

    result = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        cos_lr=True,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=args.weight_decay,
        warmup_epochs=3,
        label_smoothing=args.label_smoothing,
        mosaic=1.0,
        close_mosaic=15,
        mixup=args.mixup,
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.45,
        degrees=2.0,
        translate=0.10,
        scale=0.55,
        shear=1.0,
        fliplr=0.5,
        erasing=0.25,
        cache=args.cache,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    return Path(result.save_dir) / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OpenVisionGuard edge YOLO detector")
    parser.add_argument("--data", default="configs/openvisionguard_data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained student weights")
    parser.add_argument("--architecture-yaml", default=None, help="Optional YOLOv8n-P2 YAML for small-object recall")
    parser.add_argument("--teacher", default="yolov8s.pt", help="Teacher weights used only for pseudo-label distillation")
    parser.add_argument("--distill-source", default=None, help="Unlabeled images/videos for teacher pseudo labels")
    parser.add_argument("--distill-output", default="data/distilled_surveillance")
    parser.add_argument("--hard-negative-source", default=None, help="Known empty/clutter/low-light footage")
    parser.add_argument("--hard-negative-output", default="data/hard_negatives")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.003)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0007)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup", type=float, default=0.12)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--project", default="runs/openvisionguard")
    parser.add_argument("--name", default="yolov8n_edge")
    parser.add_argument("--sample-every", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.distill_source:
        generate_teacher_pseudo_labels(
            teacher_weights=args.teacher,
            source_dir=Path(args.distill_source),
            output_dir=Path(args.distill_output),
            imgsz=args.imgsz,
            conf=0.55,
            sample_every=args.sample_every,
        )
        print(f"Pseudo labels written to {args.distill_output}. Add that folder to your train split before final training.")

    best_weights = train(args)
    print(f"Best student weights: {best_weights}")

    if args.hard_negative_source:
        mine_hard_negatives(
            weights=str(best_weights),
            source_dir=Path(args.hard_negative_source),
            output_dir=Path(args.hard_negative_output),
            imgsz=args.imgsz,
            conf=0.30,
            sample_every=args.sample_every,
        )
        print(f"Hard negatives written to {args.hard_negative_output}. Add them to train images/labels and run a short fine tune.")


if __name__ == "__main__":
    main()
