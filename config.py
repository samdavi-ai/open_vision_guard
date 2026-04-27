import os

class OpenVisionConfig:
    # ── Model Selection ────────────────────────────────────────────────────────
    # Use a heavier model for higher accuracy (yolov8s.pt, yolov8m.pt, yolov8l.pt)
    # yolov8n.pt = fastest / lowest accuracy
    # yolov8s.pt = good balance for edge devices with 4+ GB RAM
    # yolov8m.pt = high accuracy, requires GPU or fast CPU  ← UPGRADED
    yolo_model_path: str = "yolov8m.pt"   # Upgraded from yolov8s for ≥95% recall
    yolo_device: str = "cpu"               # "cpu" | "cuda" | "mps"
    yolo_imgsz: int = 640                  # Inference resolution: 320, 640 or 960

    # ── Confidence Hierarchy (primary / secondary / re-detection) ───────────
    # Primary: used for the main 640px ByteTrack pass
    person_conf_threshold: float = 0.25
    object_conf_threshold: float = 0.25
    luggage_conf_threshold: float = 0.20
    vehicle_conf_threshold: float = 0.30
    # Secondary: used for multi-scale 960px detect-only pass
    secondary_conf_threshold: float = 0.20
    # Re-detection: used for ROI re-detection on track loss (most permissive)
    redetection_conf_threshold_tier: float = 0.15

    # ── Bounding Box Geometry Filters ─────────────────────────────────────────
    # Min pixel dimensions before a person box is accepted
    person_min_width_px: int = 10
    person_min_height_px: int = 20
    person_max_area_ratio: float = 0.72    # fraction of frame area
    person_min_area_ratio: float = 0.000035
    person_min_aspect: float = 0.10        # w/h
    person_max_aspect: float = 1.45        # w/h

    # ── Temporal Consistency (anti-flicker + recall) ─────────────────────────
    # A detection must appear for N consecutive frames before it is confirmed
    temporal_confirm_frames: int = 2       # Lowered 3→2 for faster confirmation = higher recall
    # A detection holds for N frames after disappearing (prevents ID fragmentation)
    temporal_hold_frames: int = 10         # Raised 6→10: more frames to survive occlusion
    # Smooth bbox positions over a rolling window to remove jitter
    bbox_smoothing_alpha: float = 0.45     # EMA weight for new bbox (0=freeze, 1=raw)

    # ── Frame Preprocessing ───────────────────────────────────────────────────
    # CLAHE contrast enhancement — dramatically improves low-light detections
    preprocessing_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8          # NxN grid

    # ── Adaptive Inference Controller ─────────────────────────────────────────
    # Dynamically adjusts AI inference rate based on scene activity
    adaptive_inference_enabled: bool = True
    ai_interval_min_s: float = 0.066       # Max AI FPS ~15 (high activity)
    ai_interval_max_s: float = 0.200       # Min AI FPS ~5  (static scene)
    ai_interval_default_s: float = 0.120   # Default ~8 FPS
    activity_score_ema_alpha: float = 0.3   # Smoothing for activity score

    # ── Smart Multi-Scale Detection ─────────────────────────────────────────
    # 960px pass is now CONDITIONAL — only triggered when needed
    multiscale_enabled: bool = True          # master switch
    multiscale_imgsz: int = 960
    multiscale_nms_iou: float = 0.50
    # Trigger conditions (any one triggers the 960px pass):
    multiscale_trigger_small_area: float = 0.01   # bbox area/frame area below this
    multiscale_trigger_low_conf: float = 0.40     # any person conf below this
    multiscale_trigger_on_track_loss: bool = True  # any confirmed track in hold phase

    # ── Small Object / Region Re-inference ────────────────────────────────────
    # Targeted high-res re-inference crop for small/distant persons
    small_object_reinference: bool = True
    small_object_area_threshold: float = 0.008  # bbox area / frame area
    reinference_imgsz: int = 960                # Higher res for second pass
    reinference_max_regions: int = 4            # Max crop regions per frame

    # ── Re-Detection on Track Loss ────────────────────────────────────────────
    # When a track enters hold phase, run a targeted ROI re-detection
    redetection_on_loss_enabled: bool = True
    redetection_roi_expand_factor: float = 1.8   # Expand bbox by this factor for ROI
    redetection_conf_threshold: float = 0.20     # Lower threshold for re-detection
    redetection_imgsz: int = 960

    # ── Hard Negative Filtering ───────────────────────────────────────────────
    hard_negative_filtering: bool = True
    hnf_min_visibility_ratio: float = 0.05
    hnf_edge_margin_px: int = 3

    # ── Track Quality Scoring ─────────────────────────────────────────────────
    # Score confirmed tracks by confidence + persistence + motion consistency
    track_quality_scoring: bool = True
    track_quality_min_score: float = 0.25   # Below this: track removed early
    track_quality_conf_weight: float = 0.4  # Weight for avg confidence
    track_quality_persist_weight: float = 0.35  # Weight for persistence (frames seen)
    track_quality_motion_weight: float = 0.25   # Weight for motion consistency

    # ── Occlusion-Aware Hold Extension ───────────────────────────────────────
    # When a tracked person overlaps with another bbox at disappearance,
    # extend hold duration because they're likely occluded, not gone.
    occlusion_hold_enabled: bool = True
    occlusion_iou_threshold: float = 0.15    # Overlap with another bbox = occlusion
    occlusion_hold_bonus_frames: int = 8     # Extra hold frames when occluded

    # ── TensorRT / Export Optimization ───────────────────────────────────────
    tensorrt_enabled: bool = False
    tensorrt_model_path: str = "yolov8m.engine"

    # ══════════════════════════════════════════════════════════════════════════
    #  PRODUCTION HARDENING
    # ══════════════════════════════════════════════════════════════════════════

    # ── Adaptive Threshold Calibration ───────────────────────────────────────
    # Auto-adjust person confidence thresholds per camera based on rolling stats
    adaptive_calibration_enabled: bool = True
    adaptive_calib_window: int = 200         # Rolling window of recent detections
    adaptive_calib_min_conf: float = 0.15    # Floor: never go below this
    adaptive_calib_max_conf: float = 0.45    # Ceiling: never go above this

    # ── Latency Guard System ─────────────────────────────────────────────────
    # Monitor processing time and auto-disable expensive features when slow
    latency_guard_enabled: bool = True
    latency_guard_window: int = 30           # Rolling window of frame latencies
    latency_guard_target_ms: float = 200.0   # Warning threshold
    latency_guard_critical_ms: float = 350.0 # Critical: disable all extras

    # ── Scene Profiling ──────────────────────────────────────────────────────
    # Build per-camera environmental profiles for adaptive parameter tuning
    scene_profiling_enabled: bool = True

    # ── False Positive Memory ────────────────────────────────────────────────
    # Track and suppress recurring false detections in spatial cells
    fp_memory_enabled: bool = True
    fp_memory_window_s: float = 300.0        # 5-minute window for FP history
    fp_memory_trigger_count: int = 5         # FPs in window to trigger suppression
    fp_memory_conf_boost: float = 0.10       # Extra confidence required in hot cell

    # ── Edge Case Detection ──────────────────────────────────────────────────
    # Handle sudden lighting changes, camera shake, and noise
    edge_case_detection_enabled: bool = True
    edge_case_brightness_delta: float = 30.0 # Brightness change triggering event

    # ── Tracking / Re-ID ──────────────────────────────────────────────────────
    # After how many seconds of absence an identity is pruned
    track_stale_after_s: float = 30.0
    # IOU threshold for deduplication of overlapping person boxes
    dedup_iou_threshold: float = 0.65
    dedup_overlap_threshold: float = 0.78

    # ── Embedding / Face Recognition ──────────────────────────────────────────
    similarity_threshold: float = 0.82
    embedding_model: str = "mobilenet_v2"
    face_recognition_enabled: bool = True
    face_tolerance: float = 0.5
    min_face_height_px: int = 80

    # ── Pose ──────────────────────────────────────────────────────────────────
    pose_enabled: bool = True
    fall_confidence_threshold: float = 0.6

    # ── Weapon Detection ──────────────────────────────────────────────────────
    weapon_detection_enabled: bool = True
    weapon_confidence_threshold: float = 0.5

    # ── Motion / Loitering ────────────────────────────────────────────────────
    loitering_threshold_seconds: int = 30

    # ── Alerts ────────────────────────────────────────────────────────────────
    alert_dedup_window_seconds: int = 60

    # ── Storage ───────────────────────────────────────────────────────────────
    db_path: str = "data/openvisionguard.db"
    thumbnails_dir: str = "data/thumbnails"
    known_faces_dir: str = "data/known_faces"

    # ── Streaming ─────────────────────────────────────────────────────────────
    frame_jpeg_quality: int = 75
    max_concurrent_cameras: int = 8

# Global config instance
config = OpenVisionConfig()
