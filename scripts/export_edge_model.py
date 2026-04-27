"""Prune, validate, and export OpenVisionGuard edge models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO


def structured_prune(weights: str, amount: float, output_path: Path) -> Path:
    model = YOLO(weights)
    pruned_layers = 0
    for module in model.model.modules():
        if isinstance(module, nn.Conv2d) and module.out_channels >= 16:
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
            pruned_layers += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"Applied structured channel pruning masks to {pruned_layers} Conv2d layers: {output_path}")
    return output_path


def fine_tune(weights: str, data: str, imgsz: int, epochs: int, batch: int, device: str, project: str) -> Path:
    model = YOLO(weights)
    result = model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        close_mosaic=0,
        mosaic=0.2,
        mixup=0.0,
        patience=max(5, epochs // 3),
        project=project,
        name="pruned_finetune",
        exist_ok=True,
    )
    return Path(result.save_dir) / "weights" / "best.pt"


def validate(weights: str, data: str, imgsz: int, device: str) -> None:
    metrics = YOLO(weights).val(data=data, imgsz=imgsz, device=device, verbose=False)
    print(
        "Validation: "
        f"mAP50={metrics.box.map50:.4f} "
        f"mAP50-95={metrics.box.map:.4f} "
        f"precision={metrics.box.mp:.4f} "
        f"recall={metrics.box.mr:.4f}"
    )


def export(weights: str, data: str, imgsz: int, device: str, formats: list[str]) -> None:
    model = YOLO(weights)
    if "onnx" in formats:
        model.export(format="onnx", imgsz=imgsz, device=device, opset=12, simplify=True, dynamic=False)
    if "tflite" in formats:
        model.export(format="tflite", imgsz=imgsz, int8=True, data=data)
    if "openvino" in formats:
        model.export(format="openvino", imgsz=imgsz, int8=True, data=data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize and export OpenVisionGuard edge detector")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/openvisionguard_data.yaml")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prune-amount", type=float, default=0.10)
    parser.add_argument("--fine-tune-epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/openvisionguard_export")
    parser.add_argument("--formats", default="onnx,tflite", help="Comma list: onnx,tflite,openvino")
    parser.add_argument("--skip-prune", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights
    if not args.skip_validate:
        print("Baseline validation")
        validate(weights, args.data, args.imgsz, args.device)

    if not args.skip_prune and args.prune_amount > 0:
        pruned = structured_prune(weights, args.prune_amount, Path(args.project) / "pruned.pt")
        if args.fine_tune_epochs > 0:
            weights = str(fine_tune(str(pruned), args.data, args.imgsz, args.fine_tune_epochs, args.batch, args.device, args.project))
        else:
            weights = str(pruned)
        if not args.skip_validate:
            print("Post-prune validation")
            validate(weights, args.data, args.imgsz, args.device)

    export(weights, args.data, args.imgsz, args.device, [f.strip() for f in args.formats.split(",")])
    print(f"Export complete from weights: {weights}")


if __name__ == "__main__":
    main()
