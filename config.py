import os

class OpenVisionConfig:
    # ── Model Selection ────────────────────────────────────────────────────────
    # yolov8s.pt  = fast + accurate, CPU-friendly
    # yolov8m.pt  = higher accuracy, best balance for MPS (CURRENT)
    # yolov8l.pt  = highest accuracy, slower
    # yolov8s_openvino_model = Intel OpenVINO accelerated yolov8s (use for Intel CPUs)
    yolo_model_path: str = "yolov8m.pt"   # Upgraded: more accurate person detection on MPS
    yolo_device: str = "mps"               # "cpu" | "cuda" | "mps"
    yolo_imgsz: int = 640

    # ── Confidence Hierarchy (primary / secondary / re-detection) ───────────
    # Primary: used for the main 640px ByteTrack pass
    person_conf_threshold: float = 0.45   # Raised: cut ghost/low-conf detections causing box pile-ups
    primary_tracker_conf_threshold: float = 0.12
    object_conf_threshold: float = 0.35   # Lowered: detect partially hidden bags (backpack under arm, handbag at side)
    luggage_conf_threshold: float = 0.30  # Even more permissive: bags are key for theft detection
    vehicle_conf_threshold: float = 0.40  # Raised: avoid ghost vehicle alerts
    # Secondary: used for multi-scale 960px detect-only pass
    secondary_conf_threshold: float = 0.15
    # Re-detection: used for ROI re-detection on track loss (most permissive)
    redetection_conf_threshold_tier: float = 0.12
    tracker_conf_min: float = 0.08
    tracker_conf_max: float = 0.40
    tracker_conf_base_offset: float = -0.05
    tracker_conf_low_light_delta: float = -0.02
    tracker_conf_crowded_delta: float = 0.02
    object_low_light_conf_delta: float = -0.03
    object_low_light_conf_min: float = 0.12

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
    temporal_confirm_frames: int = 2       # 2 frames = fast to confirm, still filters single-frame ghosts
    # A detection holds for N frames after disappearing (prevents ID fragmentation)
    # With AI at ~100ms per camera and 4 cameras: cycle = 400ms = 10 frames at 25fps
    # Hold 20 frames (0.8s) = 2× the cycle, enough to bridge gaps without stale boxes
    temporal_hold_frames: int = 20
    # Smooth bbox positions over a rolling window to remove jitter
    bbox_smoothing_alpha: float = 0.65     # Higher = snappier box response (was 0.50)
    bbox_snap_distance_ratio: float = 0.85  # Snap bbox when center jumps too far

    # ── Frame Preprocessing ───────────────────────────────────────────────────
    # CLAHE contrast enhancement — dramatically improves low-light detections
    preprocessing_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8          # NxN grid

    # ── Adaptive Inference Controller ─────────────────────────────────────────
    # Dynamically adjusts AI inference rate based on scene activity
    adaptive_inference_enabled: bool = True
    ai_interval_min_s: float = 0.050       # Max AI FPS ~20 (high activity)
    ai_interval_max_s: float = 0.120       # Min AI FPS ~8  (static scene) — was 0.200 (5fps)
    ai_interval_default_s: float = 0.080   # Default ~12 FPS
    activity_score_ema_alpha: float = 0.3   # Smoothing for activity score

    # ── Smart Multi-Scale Detection ─────────────────────────────────────────
    # 960px pass is DISABLED for real-time multi-camera.
    # It adds ~400ms per inference. With 4 cameras sharing MPS lock:
    #   multiscale ON:  4 × 1500ms = 6s per-camera cycle (NOT real-time)
    #   multiscale OFF: 4 × 100ms  = 400ms per-camera cycle (real-time)
    multiscale_enabled: bool = False
    multiscale_imgsz: int = 960
    multiscale_nms_iou: float = 0.50
    tracker_nms_iou: float = 0.55
    multiscale_trigger_small_area: float = 0.02
    multiscale_trigger_low_conf: float = 0.50
    multiscale_trigger_on_track_loss: bool = True

    # ── Small Object / Region Re-inference ────────────────────────────────────
    # DISABLED: adds ~200ms per inference. Re-enable only for single-camera setups.
    small_object_reinference: bool = False
    small_object_area_threshold: float = 0.020
    reinference_imgsz: int = 960
    reinference_max_regions: int = 6
    reinference_conf_scale: float = 0.75
    reinference_pad_scale: float = 1.8
    reinference_min_pad_px: int = 60
    reinference_min_crop_size_px: int = 64

    # ── Re-Detection on Track Loss ────────────────────────────────────────────
    # DISABLED: adds ~150ms per inference. Re-enable only for single-camera setups.
    redetection_on_loss_enabled: bool = False
    redetection_roi_expand_factor: float = 2.0
    redetection_conf_threshold: float = 0.25
    redetection_imgsz: int = 960

    # ── Hard Negative Filtering ───────────────────────────────────────────────
    hard_negative_filtering: bool = True       # ENABLED: reject detections that match known FP patterns
    hnf_min_visibility_ratio: float = 0.05
    hnf_edge_margin_px: int = 3
    hnf_min_clipped_sides: int = 3
    hnf_min_aspect_ratio: float = 0.10         # Reject very flat detections (shadows/floor artifacts)
    hnf_top_band_ratio: float = 0.04
    hnf_small_height_ratio: float = 0.06

    # ── Track Quality Scoring ─────────────────────────────────────────────────
    # Score confirmed tracks by confidence + persistence + motion consistency
    track_quality_scoring: bool = True       # ENABLED: prune ghost tracks with low quality scores
    track_quality_min_score: float = 0.20   # Permissive enough to keep low-conf walking persons
    track_quality_conf_weight: float = 0.35  # Slightly less weight on conf to favor moving persons
    track_quality_persist_weight: float = 0.40  # Higher weight: a track that persists is likely real
    track_quality_motion_weight: float = 0.25   # Keep motion weight for catching stationary ghosts

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
    adaptive_calib_min_conf: float = 0.30    # Floor: don't go below 0.30 (prevents too many FPs)
    adaptive_calib_max_conf: float = 0.55    # Ceiling: limit max threshold to still catch fast persons

    # ── Latency Guard System ─────────────────────────────────────────────────
    # Monitor processing time and auto-disable expensive features when slow
    latency_guard_enabled: bool = True
    latency_guard_window: int = 30           # Rolling window of frame latencies
    latency_guard_target_ms: float = 600.0   # Raised: 4 cameras share MPS lock, expect 400-600ms
    latency_guard_critical_ms: float = 900.0 # Critical threshold raised to match 4-cam reality

    # ── Scene Profiling ──────────────────────────────────────────────────────
    # Build per-camera environmental profiles for adaptive parameter tuning
    scene_profiling_enabled: bool = True

    # ── False Positive Memory ────────────────────────────────────────────────
    # Track and suppress recurring false detections in spatial cells
    fp_memory_enabled: bool = True
    fp_memory_window_s: float = 180.0        # 3-minute window (tighter = faster adaptation)
    fp_memory_trigger_count: int = 3         # 3 FPs in window to trigger suppression (was 5)
    fp_memory_conf_boost: float = 0.15       # Extra 15% conf required in known-FP hotspot cells

    # ── Edge Case Detection ──────────────────────────────────────────────────
    # Handle sudden lighting changes, camera shake, and noise
    edge_case_detection_enabled: bool = True
    edge_case_brightness_delta: float = 30.0 # Brightness change triggering event

    # ── Tracking / Re-ID ──────────────────────────────────────────────────────
    # After how many seconds of absence an identity is pruned
    track_stale_after_s: float = 30.0
    # IOU threshold for deduplication of overlapping person boxes
    # Lowered from 0.65 → 0.45: merges boxes that share 45%+ area (was too loose, causing pile-ups)
    dedup_iou_threshold: float = 0.45
    # Overlap-over-smaller: drops a box if 60%+ contained within another (was 0.78, too permissive)
    dedup_overlap_threshold: float = 0.60

    # ── Embedding / Re-ID (OSNet-AIN + multi-gallery) ────────────────────────
    # Within-camera, same-view threshold: strict.
    similarity_threshold: float = 0.80
    embedding_model: str = "osnet_ain_x1_0"
    # Cross-camera threshold: more lenient (different angle/crop size).
    cross_camera_reid_threshold: float = 0.72
    # Cross-view threshold: most lenient — same camera but person turned around
    # (back view → front view). OSNet-AIN embedding drift on view change ~0.15.
    # Used only when spatio-temporal proximity strongly suggests same person.
    cross_view_reid_threshold: float = 0.62
    # Gallery: max appearance snapshots stored per identity (FIFO eviction).
    # Higher = more view angles stored → better cross-view matching.
    reid_max_gallery_size: int = 12
    # Min seconds between gallery updates for the same identity.
    reid_min_update_interval_s: float = 2.0
    # Spatio-temporal Re-ID: when a NEW track appears within this window
    # after a track loss, and within max_px of the last known position,
    # apply the cross_view threshold instead of minting a new identity.
    spatiotemporal_reid_window_s: float = 5.0
    spatiotemporal_reid_max_px:   float = 200.0
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
    # Per-alert-type deduplication overrides (seconds).
    # Sudden movement is a continuous classifier — without a long cooldown it
    # floods the feed every frame a person is running.
    alert_dedup_overrides: dict = {
        "sudden_movement":  300,   # 5 minutes — one alert per person per run episode
        "loitering":        120,   # 2 minutes between loitering re-alerts
        "camera_avoidance": 180,   # 3 minutes
        "high_risk":        240,   # 4 minutes
    }

    # ── Session / Baggage Tracking ────────────────────────────────────────────
    # Seconds of absence before a person is considered to have "exited" the scene
    # Raised to 20s for mall footage: people walk around corners, behind pillars, use elevators
    session_exit_timeout_s: float = 20.0
    # Confidence threshold for laptop and electronics detection
    laptop_conf_threshold: float = 0.60    # Raised from 0.35: laptops in surveillance context have many FPs

    # ── Storage ───────────────────────────────────────────────────────────────
    db_path: str = "data/openvisionguard.db"
    thumbnails_dir: str = "data/thumbnails"
    known_faces_dir: str = "data/known_faces"

    # ── Streaming ─────────────────────────────────────────────────────────────
    frame_jpeg_quality: int = 75
    max_concurrent_cameras: int = 8
    # Adaptive display resolution (scales down when activity is low)
    display_width_high_activity: int = 480   # px — busy scene (lower = faster encode)
    display_width_low_activity: int = 360    # px — quiet scene
    jpeg_quality_high_activity: int = 60
    jpeg_quality_low_activity: int = 42
    activity_high_threshold: float = 0.55   # score above this = high-activity
    # Crop update: skip rebuild when detection IDs unchanged
    crop_update_interval: int = 5           # every N frames (minimum)
    # Zero-motion gate: skip AI submission when scene is static
    zero_motion_gate_enabled: bool = False  # DISABLED: slow-walking people in malls trigger gate, causing detection gaps
    zero_motion_energy_threshold: float = 0.008  # (kept for reference, gate is off)
    # DB write throttle: min seconds between per-person DB flushes
    db_write_interval_s: float = 2.0
    # Detection memory: explicit AI cadence for velocity (0 = measure real time)
    detection_memory_ai_cadence_frames: int = 0  # 0 = auto from timestamps
    # Behaviour thresholds (moved from hardcoded)
    behaviour_running_speed_px: float = 150.0
    behaviour_walking_speed_px: float = 30.0
    behaviour_loitering_radius_px: float = 50.0
    # Minimum body-lengths-per-second to classify as "running" (triggers sudden_movement).
    # 2.4 = too sensitive (brisk mall walkers). 3.5 ≈ genuine jogging/sprinting.
    behaviour_running_body_lengths_per_s: float = 3.5
    behaviour_loitering_time_s: float = 30.0
    # CLAHE: only enhance when scene is dark (brightness below this)
    clahe_brightness_gate: float = 100.0   # 0-255; skip CLAHE above this

    # Heavy-feature stride: run multiscale/reinference/pose/face/redetection only
    # every N frames. Basic YOLO track still runs every frame for smooth boxes.
    # 1 = every frame (max quality, slowest)
    # 3 = every 3rd frame (good quality, ~3× faster for heavy ops)  ← default
    # 5 = every 5th frame (basic, fastest — use for 4+ cameras with all settings on)
    heavy_feature_stride: int = 3

    # ══════════════════════════════════════════════════════════════════════════
    #  AIRPORT LUGGAGE INTELLIGENCE
    # ══════════════════════════════════════════════════════════════════════════

    # ── Carry-zone geometry ───────────────────────────────────────────────────
    # A bag is "carried" when its centre is within (person_height × ratio) px
    # of the person's centre.  0.60 = 60% of body height.
    luggage_carry_distance_ratio: float = 0.60

    # Fallback absolute carry distance when no person bbox is available
    luggage_putdown_distance_px: float = 80.0

    # Max pixel distance to consider a direct hand-to-hand transfer (vs. pick-up
    # after owner walked away).  Should be ~1 arm-length in frame pixels.
    luggage_handover_radius_px: float = 120.0

    # ── Unattended bag timers (security critical for airports) ────────────────
    # Warning: bag on floor for this many seconds with no owner nearby
    luggage_unattended_warn_s: float = 30.0
    # Critical: full security alert — escalate to operator immediately
    luggage_unattended_critical_s: float = 60.0

    # ── Stale bag eviction ────────────────────────────────────────────────────
    # Remove a bag record after it hasn't been detected for this many seconds
    luggage_stale_after_s: float = 60.0

    # ── Session exit timeout ──────────────────────────────────────────────────
    # How many seconds a person must be absent before they are "confirmed exited"
    # and their session is closed + bag exit-match generated.
    # Airport value: 15s (corridors are wide, exits are clear-cut)
    luggage_session_exit_timeout_s: float = 15.0

    # ══════════════════════════════════════════════════════════════════════════
    #  SENSOR FUSION LAYER
    # ══════════════════════════════════════════════════════════════════════════

    # Cosine similarity threshold for gallery match (bag Re-ID)
    # 0.82 = strict match required (fewer false-positives, slightly more ID breaks)
    fusion_similarity_threshold: float = 0.82

    # World-unit radius for Kalman spatial pre-filter.
    # Only gallery entries within this distance of the predicted position are
    # compared by cosine similarity. Reduces false cross-bag matches.
    fusion_spatial_filter_radius: float = 150.0

    # Gallery TTL: remove bag embeddings not seen in this many seconds (5 min)
    fusion_gallery_ttl_s: float = 300.0

    # Kalman filter process noise covariance (lower = smoother, slower to adapt)
    fusion_kalman_noise_cov: float = 1e-4

    # Kalman filter measurement noise covariance
    fusion_kalman_meas_noise: float = 1e-2

    # OSNet-AIN embedding dimension (do not change unless swapping models)
    fusion_embedding_dim: int = 512

    # Log FusionMetrics every N frames (at DEBUG level)
    fusion_log_interval_frames: int = 100

# Global config instance
config = OpenVisionConfig()
