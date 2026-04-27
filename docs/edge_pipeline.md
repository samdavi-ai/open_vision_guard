# OpenVisionGuard Edge CV Pipeline

This implementation is built around YOLOv8 Nano because it has the best practical accuracy/latency tradeoff for offline Raspberry Pi and low-end CPU deployment. Use `yolov8n.pt` for the default model. Try `configs/yolov8n_p2_surveillance.yaml` only if validation shows small bags or distant people are the dominant miss case; the P2 head improves small-object recall but costs CPU.

## Dataset

Use YOLO format:

```text
data/openvisionguard/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

Classes:

```text
0 person
1 backpack
2 handbag
3 suitcase
```

For mAP near 0.90, the dataset matters more than model tricks. Include day/night CCTV, low-light, IR, rain, shadows, crowd density, partial occlusion, motion blur, camera compression artifacts, empty corridors, reflective glass, posters/mannequins, and cluttered stores or stations. Keep a dedicated low-light negative set with no target objects for false-positive measurement.

## Train

```bash
python scripts/train_edge_detector.py --data configs/openvisionguard_data.yaml --model yolov8n.pt --imgsz 416 --epochs 160 --batch 16 --device 0
```

For small-object recall:

```bash
python scripts/train_edge_detector.py --architecture-yaml configs/yolov8n_p2_surveillance.yaml --model yolov8n.pt --data configs/openvisionguard_data.yaml --imgsz 416
```

The training script enables transfer learning, cosine LR, AdamW, label smoothing, Mosaic, MixUp, HSV jitter, scaling, translation, and erasing. Ultralytics also applies its internal augmentation stack; add `albumentations` for richer blur/noise transforms in training environments.

## Distillation

Teacher models are used only offline to create pseudo labels. Do not deploy them on the edge device.

```bash
python scripts/train_edge_detector.py --distill-source data/unlabeled_cctv --teacher yolov8s.pt --model yolov8n.pt
```

This writes `data/distilled_surveillance/images` and `data/distilled_surveillance/labels`. Review samples, then merge high-quality pseudo labels into the training split and retrain/fine tune the student.

## Hard Negative Mining

Use low-light/clutter/empty footage where false alarms happen:

```bash
python scripts/train_edge_detector.py --hard-negative-source data/negative_low_light --model yolov8n.pt
```

The script writes empty-label training samples under `data/hard_negatives`. Add them to the train split and fine tune for 20-40 epochs to reduce false positives.

## Optimize And Export

```bash
python scripts/export_edge_model.py --weights runs/openvisionguard/yolov8n_edge/weights/best.pt --data configs/openvisionguard_data.yaml --imgsz 416 --device cpu --prune-amount 0.10 --fine-tune-epochs 20 --formats onnx,tflite
```

The export flow validates the baseline, applies structured Conv2d channel pruning masks, fine tunes the pruned model, validates again, then exports ONNX and INT8 TFLite. Accept the optimized model only if mAP drops by less than 2 percentage points and the FPS target is met.

## Evaluate

```bash
python scripts/evaluate_edge_model.py --weights runs/openvisionguard/yolov8n_edge/weights/best.pt --data configs/openvisionguard_data.yaml --benchmark-source test2vid.mp4 --negative-dir data/negative_low_light --device cpu
```

Track:

```text
mAP@0.5
mAP@0.5:0.95
precision
recall
FPS on target hardware
false-positive rate on low-light negatives
```

## Run Edge Inference

Edit `configs/openvisionguard_edge.yaml` to point at your trained `.pt`, `.onnx`, or `.tflite` model, then:

```bash
python scripts/run_edge_inference.py --source 0 --output outputs/edge_live.mp4
```

For a video:

```bash
python scripts/run_edge_inference.py --source test2vid.mp4 --output outputs/edge_test2.mp4
```

The runtime pipeline is:

```text
OpenCV MOG2 motion gate -> YOLOv8n detection -> ByteTrack IDs -> temporal behavior -> luggage ownership -> risk scoring
```

Default edge settings process every second frame at 416x416. On a Raspberry Pi, use 320x320 and `process_every: 3` if FPS is below 15, then recover accuracy through better training data and hard negatives before increasing model size.
