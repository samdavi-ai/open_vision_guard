import os

class OpenVisionConfig:
    # ── Model Selection ────────────────────────────────────────────────────────
    # yolov8s.pt  = fast + accurate, CPU-friendly (CURRENT)
    # yolov8m.pt  = higher accuracy, slow on CPU
    # yolov8s_openvino_model = Intel OpenVINO accelerated yolov8s (use for Intel CPUs)
    yolo_model_path: str = "yolov8s.pt"   # Faster CPU inference
    yolo_device: str = "cpu"               # "cpu" | "cuda" | "mps"
    yolo_imgsz: int = 640

    # ── Confidence Hierarchy (primary / secondary / re-detection) ───────────
    # Primary: used for the main 640px ByteTrack pass
    person_conf_threshold: float = 0.18
    primary_tracker_conf_threshold: float = 0.10
    object_conf_threshold: float = 0.25
    luggage_conf_threshold: float = 0.20
    vehicle_conf_threshold: float = 0.30
    # Secondary: used for multi-scale 960px detect-only pass
    secondary_conf_threshold: float = 0.12
    # Re-detection: used for ROI re-detection on track loss (most permissive)
    redetection_conf_threshold_tier: float = 0.10
    tracker_conf_min: float = 0.08
    tracker_conf_max: float = 0.45
    tracker_conf_base_offset: float = -0.05
    tracker_conf_low_light_delta: float = -0.02
    tracker_conf_crowded_delta: float = 0.02
    object_low_light_conf_delta: float = -0.03
    object_low_light_conf_min: float = 0.10

    # ── Bounding Box Geometry Filters ─────────────────────────────────────────
    # Min pixel dimensions before a person box is accepted
    person_min_width_px: int = 6
    person_min_height_px: int = 12
    person_max_area_ratio: float = 0.72    # fraction of frame area
    person_min_area_ratio: float = 0.00001
    person_min_aspect: float = 0.10        # w/h
    person_max_aspect: float = 1.45        # w/h
    person_edge_tall_height_ratio: float = 0.92
    person_edge_tall_width_ratio: float = 0.08

    # ── Object Geometry Filters ───────────────────────────────────────────────
    object_min_width_px: int = 5
    object_min_height_px: int = 5
    object_min_area_ratio: float = 0.000025
    object_max_area_ratio: float = 0.45
    object_min_aspect: float = 0.08
    object_max_aspect: float = 8.0

    # ── Temporal Consistency (anti-flicker + recall) ─────────────────────────
    # A detection must appear for N consecutive frames before it is confirmed
    temporal_confirm_frames: int = 1       # Confirm immediately to avoid missing brief detections
    # A detection holds for N frames after disappearing (prevents ID fragmentation)
    temporal_hold_frames: int = 10         # Raised 6→10: more frames to survive occlusion
    # Smooth bbox positions over a rolling window to remove jitter
    bbox_smoothing_alpha: float = 0.45     # EMA weight for new bbox (0=freeze, 1=raw)
    bbox_snap_distance_ratio: float = 0.85  # Snap bbox when center jumps too far

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
    multiscale_enabled: bool = True          # Enable secondary high-res pass when triggered
    multiscale_imgsz: int = 960
    multiscale_nms_iou: float = 0.50
    tracker_nms_iou: float = 0.55
    # Trigger conditions (any one triggers the 960px pass):
    multiscale_trigger_small_area: float = 0.01   # bbox area/frame area below this
    multiscale_trigger_low_conf: float = 0.40     # any person conf below this
    multiscale_trigger_on_track_loss: bool = True  # any confirmed track in hold phase

    # ── Small Object / Region Re-inference ────────────────────────────────────
    # Targeted high-res re-inference crop for small/distant persons
    small_object_reinference: bool = True
    small_object_area_threshold: float = 0.015  # Trigger high-res reinference for more distant people
    reinference_imgsz: int = 960                # Higher res for second pass
    reinference_max_regions: int = 8            # Check more candidate regions per frame
    reinference_conf_scale: float = 0.8
    reinference_pad_scale: float = 2.0
    reinference_min_pad_px: int = 80
    reinference_min_crop_size_px: int = 64

    # ── Re-Detection on Track Loss ────────────────────────────────────────────
    # When a track enters hold phase, run a targeted ROI re-detection
    redetection_on_loss_enabled: bool = True
    redetection_roi_expand_factor: float = 1.8   # Expand bbox by this factor for ROI
    redetection_conf_threshold: float = 0.20     # Lower threshold for re-detection
    redetection_imgsz: int = 960

    # ── Hard Negative Filtering ───────────────────────────────────────────────
    hard_negative_filtering: bool = False
    hnf_min_visibility_ratio: float = 0.05
    hnf_edge_margin_px: int = 3
    hnf_min_clipped_sides: int = 3
    hnf_min_aspect_ratio: float = 0.04
    hnf_top_band_ratio: float = 0.05
    hnf_small_height_ratio: float = 0.08

    # ── Track Quality Scoring ─────────────────────────────────────────────────
    # Score confirmed tracks by confidence + persistence + motion consistency
    track_quality_scoring: bool = False
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
    adaptive_calib_min_conf: float = 0.08    # Lower floor to recover low-confidence distant people
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
    face_recognition_enabled: bool = False
    face_tolerance: float = 0.5
    min_face_height_px: int = 80

    # ── Pose ──────────────────────────────────────────────────────────────────
    pose_enabled: bool = False
    fall_confidence_threshold: float = 0.6

    # ── Weapon Detection ──────────────────────────────────────────────────────
    weapon_detection_enabled: bool = False
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
    # Adaptive display resolution (scales down when activity is low)
    display_width_high_activity: int = 640   # px — busy scene
    display_width_low_activity: int = 480    # px — quiet scene
    jpeg_quality_high_activity: int = 60
    jpeg_quality_low_activity: int = 42
    activity_high_threshold: float = 0.55   # score above this = high-activity
    # Crop update: skip rebuild when detection IDs unchanged
    crop_update_interval: int = 5           # every N frames (minimum)
    # Zero-motion gate: skip AI submission when scene is static
    zero_motion_gate_enabled: bool = True
    zero_motion_energy_threshold: float = 0.008  # Very conservative: only skip truly frozen scenes
    # DB write throttle: min seconds between per-person DB flushes
    db_write_interval_s: float = 2.0
    # Detection memory: explicit AI cadence for velocity (0 = measure real time)
    detection_memory_ai_cadence_frames: int = 0  # 0 = auto from timestamps
    # Behaviour thresholds (moved from hardcoded)
    behaviour_running_speed_px: float = 150.0
    behaviour_walking_speed_px: float = 30.0
    behaviour_loitering_radius_px: float = 50.0
    behaviour_loitering_time_s: float = 30.0
    # CLAHE: only enhance when scene is dark (brightness below this)
    clahe_brightness_gate: float = 100.0   # 0-255; skip CLAHE above this

# Global config instance
config = OpenVisionConfig()
