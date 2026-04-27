"""
evaluate.py — Evaluation Pipeline for OpenVisionGuard Detection Accuracy.

PURPOSE
-------
Compute Precision, Recall, F1, and mAP@50 for the detection pipeline by
comparing model predictions against ground-truth annotations.

USAGE
-----
1. Prepare a COCO-format JSON annotation file with person bounding boxes.
2. Point to a folder of images corresponding to those annotations.
3. Run:
     python evaluate.py --images data/eval/images --annotations data/eval/gt.json

DATASET CREATION STRATEGY
-------------------------
For production evaluation of a CCTV-focused system:

  1. Collect 500-1000 representative frames from deployed cameras.
     Include: daytime, nighttime, rain, crowded, empty, distant persons, etc.
  2. Annotate persons with bounding boxes using LabelImg or CVAT (COCO format).
  3. Split 80/20 for tuning vs held-out test.

This script works with COCO-format annotation JSON:
    {
        "images": [{"id": 1, "file_name": "frame_0001.jpg", "width": 1920, "height": 1080}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class EvalMetrics:
    """Accumulated evaluation metrics."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_iou: float = 0.0
    total_matched: int = 0
    per_image_ap: List[float] = field(default_factory=list)
    per_image_ap_multi: Dict[str, List[float]] = field(default_factory=dict)
    latencies_ms: List[float] = field(default_factory=list)
    scenario_results: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mean_iou(self) -> float:
        return self.total_iou / self.total_matched if self.total_matched > 0 else 0.0

    @property
    def map50(self) -> float:
        return sum(self.per_image_ap) / len(self.per_image_ap) if self.per_image_ap else 0.0

    @property
    def map50_95(self) -> float:
        """mAP averaged across IoU thresholds 0.50, 0.55, ..., 0.95."""
        if not self.per_image_ap_multi:
            return 0.0
        aps = []
        for iou_key, ap_list in self.per_image_ap_multi.items():
            if ap_list:
                aps.append(sum(ap_list) / len(ap_list))
        return sum(aps) / len(aps) if aps else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  OpenVisionGuard Detection Evaluation Results",
            f"{'='*60}",
            f"  Precision:        {self.precision:.4f}  ({self.precision:.1%})",
            f"  Recall:           {self.recall:.4f}  ({self.recall:.1%})",
            f"  F1 Score:         {self.f1:.4f}",
            f"  mAP@50:           {self.map50:.4f}",
            f"  mAP@50:95:        {self.map50_95:.4f}",
            f"  Mean IoU:         {self.mean_iou:.4f}",
            f"  Avg Latency:      {self.avg_latency_ms:.1f} ms",
            f"{'─'*60}",
            f"  True Positives:   {self.true_positives}",
            f"  False Positives:  {self.false_positives}",
            f"  False Negatives:  {self.false_negatives}",
            f"  Images Evaluated: {len(self.per_image_ap)}",
        ]
        # Per-scenario breakdown
        if self.scenario_results:
            lines.append(f"{'─'*60}")
            lines.append(f"  Per-Scenario Breakdown:")
            for scenario, counts in self.scenario_results.items():
                tp = counts.get('tp', 0)
                fp = counts.get('fp', 0)
                fn = counts.get('fn', 0)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                lines.append(f"    {scenario:20s}  P={p:.3f}  R={r:.3f}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def match_detections(
    predictions: List[Dict[str, Any]],
    gt_boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.50,
) -> Tuple[int, int, int, float, float]:
    """
    Match predicted detections to ground-truth boxes using greedy IoU matching.

    Returns (TP, FP, FN, total_iou, AP@iou_threshold).
    """
    if not gt_boxes:
        return 0, len(predictions), 0, 0.0, 0.0 if not predictions else 0.0
    if not predictions:
        return 0, 0, len(gt_boxes), 0.0, 0.0

    # Sort predictions by descending confidence
    preds_sorted = sorted(predictions, key=lambda d: d.get("confidence", 0), reverse=True)

    matched_gt = set()
    tp = 0
    fp = 0
    total_iou = 0.0

    # For AP computation
    precisions = []
    recalls = []

    for i, pred in enumerate(preds_sorted):
        pred_box = pred["bbox"]
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
            total_iou += best_iou
        else:
            fp += 1

        # Running precision/recall for AP
        precisions.append(tp / (tp + fp))
        recalls.append(tp / len(gt_boxes))

    fn = len(gt_boxes) - len(matched_gt)

    # Compute AP using 11-point interpolation
    ap = 0.0
    if precisions:
        for t in [i / 10.0 for i in range(11)]:
            p_at_r = max(
                [p for p, r in zip(precisions, recalls) if r >= t],
                default=0.0,
            )
            ap += p_at_r / 11.0

    return tp, fp, fn, total_iou, ap


def load_coco_annotations(json_path: str) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """
    Load COCO-format annotations and return {image_id: [(x1, y1, x2, y2), ...]}.
    Only loads category_id=1 (person).
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    gt_by_image: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") != 1:
            continue
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        gt_by_image.setdefault(img_id, []).append((int(x), int(y), int(x + w), int(y + h)))

    return gt_by_image


def classify_scenario(filename: str, frame: np.ndarray) -> str:
    """Classify an image into a scenario category for breakdown reporting."""
    gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    name_lower = filename.lower()

    if brightness < 60 or 'night' in name_lower or 'dark' in name_lower:
        return 'low_light'
    if 'crowd' in name_lower or 'dense' in name_lower:
        return 'crowded'
    if 'occlu' in name_lower:
        return 'occluded'
    if 'distant' in name_lower or 'far' in name_lower or 'small' in name_lower:
        return 'distant'
    return 'normal'


def split_dataset(
    annotations_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, str]:
    """
    Split a COCO annotation file into train/val/test splits.
    Returns {split_name: path_to_split_json}.
    """
    import random

    with open(annotations_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    for split_name, split_images in splits.items():
        img_ids = {img["id"] for img in split_images}
        split_anns = [a for a in annotations if a["image_id"] in img_ids]
        split_coco = {
            "images": split_images,
            "annotations": split_anns,
            "categories": categories,
        }
        path = os.path.join(output_dir, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(split_coco, f, indent=2)
        paths[split_name] = path
        print(f"  [Split] {split_name}: {len(split_images)} images, {len(split_anns)} annotations -> {path}")

    return paths


def evaluate(
    images_dir: str,
    annotations_path: str,
    iou_threshold: float = 0.50,
    compute_map5095: bool = True,
) -> EvalMetrics:
    """
    Run full evaluation: load images, run pipeline, compare to ground truth.
    Computes mAP@50 and optionally mAP@50:95.
    """
    from core.pipeline import Pipeline

    print("[Eval] Initializing pipeline...")
    pipeline = Pipeline()

    with open(annotations_path, "r") as f:
        coco = json.load(f)

    image_map = {img["id"]: img for img in coco.get("images", [])}
    gt_by_image = load_coco_annotations(annotations_path)

    # IoU thresholds for mAP@50:95
    iou_thresholds = [0.50]
    if compute_map5095:
        iou_thresholds = [round(0.50 + i * 0.05, 2) for i in range(10)]

    metrics = EvalMetrics()
    for iou_t in iou_thresholds:
        metrics.per_image_ap_multi[f"iou_{iou_t:.2f}"] = []

    total_images = len(image_map)
    print(f"[Eval] Evaluating {total_images} images")
    print(f"[Eval] IoU thresholds: {iou_thresholds}")

    for idx, (img_id, img_info) in enumerate(image_map.items()):
        filename = img_info["file_name"]
        img_path = os.path.join(images_dir, filename)

        if not os.path.exists(img_path):
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Classify scenario
        scenario = classify_scenario(filename, frame)

        # Run pipeline
        t0 = time.time()
        result = pipeline.process_frame(frame, camera_id="EVAL")
        latency = (time.time() - t0) * 1000
        metrics.latencies_ms.append(latency)

        predictions = [d for d in result.current_detections if not d.get("is_object", False)]
        gt_boxes = gt_by_image.get(img_id, [])

        # Primary evaluation at given iou_threshold
        tp, fp, fn, total_iou, ap = match_detections(predictions, gt_boxes, iou_threshold)
        metrics.true_positives += tp
        metrics.false_positives += fp
        metrics.false_negatives += fn
        metrics.total_iou += total_iou
        metrics.total_matched += tp
        metrics.per_image_ap.append(ap)

        # Per-scenario tracking
        if scenario not in metrics.scenario_results:
            metrics.scenario_results[scenario] = {'tp': 0, 'fp': 0, 'fn': 0}
        metrics.scenario_results[scenario]['tp'] += tp
        metrics.scenario_results[scenario]['fp'] += fp
        metrics.scenario_results[scenario]['fn'] += fn

        # Multi-threshold AP for mAP@50:95
        for iou_t in iou_thresholds:
            _, _, _, _, ap_t = match_detections(predictions, gt_boxes, iou_t)
            metrics.per_image_ap_multi[f"iou_{iou_t:.2f}"].append(ap_t)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
            print(f"  [{idx+1}/{total_images}] P={metrics.precision:.3f} R={metrics.recall:.3f} "
                  f"mAP50={metrics.map50:.3f} mAP50:95={metrics.map50_95:.3f} Lat={latency:.0f}ms")

    return metrics


def create_sample_annotation_template(output_path: str, images_dir: str):
    """Generate a starter COCO annotation JSON from a folder of images."""
    images = []
    for i, fn in enumerate(sorted(os.listdir(images_dir))):
        if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img = cv2.imread(os.path.join(images_dir, fn))
            if img is not None:
                h, w = img.shape[:2]
                images.append({"id": i + 1, "file_name": fn, "width": w, "height": h})

    template = {
        "images": images,
        "annotations": [],  # Fill manually with LabelImg/CVAT
        "categories": [{"id": 1, "name": "person"}],
    }

    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)

    print(f"[Eval] Template saved to {output_path} with {len(images)} images.")
    print("       Annotate bounding boxes using LabelImg or CVAT (COCO format).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVisionGuard Detection Evaluation")
    parser.add_argument("--images", type=str, default="data/eval/images",
                        help="Path to evaluation images directory")
    parser.add_argument("--annotations", type=str, default="data/eval/gt.json",
                        help="Path to COCO-format ground truth JSON")
    parser.add_argument("--iou", type=float, default=0.50,
                        help="IoU threshold for matching (default: 0.50)")
    parser.add_argument("--create-template", action="store_true",
                        help="Generate a COCO annotation template from images")
    parser.add_argument("--split", action="store_true",
                        help="Split the annotation file into train/val/test")
    args = parser.parse_args()

    if args.split:
        output_dir = os.path.join(os.path.dirname(args.annotations), "splits")
        split_dataset(args.annotations, output_dir)
    elif args.create_template:
        os.makedirs(os.path.dirname(args.annotations) or ".", exist_ok=True)
        create_sample_annotation_template(args.annotations, args.images)
    else:
        if not os.path.exists(args.annotations):
            print(f"[Error] Annotation file not found: {args.annotations}")
            print(f"        Create one with: python evaluate.py --images {args.images} --create-template")
            exit(1)

        results = evaluate(args.images, args.annotations, args.iou)
        print(results.summary())

        # Save results
        out_path = "data/eval/results.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "precision": results.precision,
                "recall": results.recall,
                "f1": results.f1,
                "map50": results.map50,
                "map50_95": results.map50_95,
                "mean_iou": results.mean_iou,
                "avg_latency_ms": results.avg_latency_ms,
                "true_positives": results.true_positives,
                "false_positives": results.false_positives,
                "false_negatives": results.false_negatives,
                "scenario_results": results.scenario_results,
            }, f, indent=2)
        print(f"[Eval] Results saved to {out_path}")
