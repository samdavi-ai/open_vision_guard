"""Accuracy-improved live-surveillance pipeline for OpenVisionGuard.

Key improvements over the previous version:
  1. CLAHE frame preprocessing  — boosts low-light detection significantly.
  2. Config-driven thresholds   — all confidence and geometry limits are in config.py.
  3. Temporal confirmation      — detections must appear for N frames before being shown
                                   (eliminates single-frame ghost detections / flicker).
  4. Detection hold buffer      — a track is kept alive for N frames after disappearing
                                   (prevents ID splits during brief occlusions).
  5. EMA bbox smoothing         — bounding box positions are smoothed with an exponential
                                   moving average (removes jitter while staying responsive).
  6. Heavier model by default   — config.yolo_model_path defaults to yolov8s.pt.
"""

from __future__ import annotations

import datetime
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from config import config
from modules import database
from modules.alert_engine import alert_engine
from modules.frame_preprocessor import frame_preprocessor
from modules.geolocation import geolocation_engine
from modules.luggage_tracker import luggage_tracker
from modules.presence_tracker import presence_tracker
from modules.risk_engine import risk_engine

BBox = Tuple[int, int, int, int]


@dataclass
class PipelineResult:
    annotated_frame: Optional[np.ndarray] = None
    identities: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    current_detections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TrackState:
    global_id: str
    first_seen: float
    last_seen: float
    last_center: Tuple[float, float]
    samples: Deque[Tuple[float, float, float, float]] = field(default_factory=lambda: deque(maxlen=90))
    stationary_since: Optional[float] = None
    movement_direction: str = "stationary"
    speed_px_s: float = 0.0


@dataclass
class TemporalEntry:
    """
    Per-candidate tracking record used by TemporalBuffer.

    confirm_count      : how many consecutive frames this candidate has been seen
    hold_frames        : countdown of frames the detection is held after disappearing
    smooth_bbox        : EMA-smoothed bounding box (avoids jitter on-screen)
    confirmed          : True once confirm_count >= temporal_confirm_frames
    last_raw           : the most recent raw detection dict from YOLO
    velocity           : (dx1, dy1, dx2, dy2) per-frame pixel shift for extrapolation
    avg_confidence     : running average confidence for track quality scoring
    total_frames_seen  : total frames this track has been detected (persistence metric)
    motion_consistency : 0-1 score for how consistent the velocity direction is
    prev_velocity      : previous velocity for motion consistency computation
    is_occluded        : True if this track was occluded when it disappeared
    """
    confirm_count: int = 0
    hold_frames: int = 0
    smooth_bbox: Optional[List[float]] = None
    confirmed: bool = False
    last_raw: Optional[Dict[str, Any]] = None
    velocity: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    avg_confidence: float = 0.0
    total_frames_seen: int = 0
    motion_consistency: float = 1.0
    prev_velocity: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    is_occluded: bool = False


class TemporalBuffer:
    """
    Two-stage gate that prevents false positives from reaching the UI.

    Stage 1 — Confirmation:
      A raw YOLO detection must appear in at least `confirm_frames` consecutive
      frames before it is declared "confirmed" and forwarded downstream.  This
      eliminates single-frame ghost detections (the dominant source of flicker).

    Stage 2 — Hold:
      Once confirmed, a track is kept alive for `hold_frames` extra frames after
      it disappears from YOLO output.  This prevents the ID from fragmenting when
      a person is briefly occluded (walking behind a pillar, for example).

    Smoothing:
      Confirmed bounding boxes are updated using an Exponential Moving Average
      (EMA) so the displayed rectangle moves fluidly rather than jumping pixel-
      to-pixel between frames.
    """

    def __init__(
        self,
        confirm_frames: int = 3,
        hold_frames: int = 6,
        smoothing_alpha: float = 0.45,
    ) -> None:
        self._confirm_frames = confirm_frames
        self._hold_frames = hold_frames
        self._alpha = smoothing_alpha          # weight for new observation
        self._entries: Dict[str, TemporalEntry] = {}

    # ------------------------------------------------------------------ public

    def update(
        self, raw_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Feed raw YOLO detections for this frame.
        Returns the list of *confirmed + held* detections to send downstream.

        Includes:
          - Track quality scoring (confidence + persistence + motion consistency)
          - Occlusion-aware hold extension
          - Velocity extrapolation during hold
          - Weak track pruning
        """
        seen_ids: set = set()

        for det in raw_detections:
            gid = det["global_id"]
            seen_ids.add(gid)
            entry = self._entries.setdefault(gid, TemporalEntry())
            entry.last_raw = det
            entry.hold_frames = self._hold_frames    # reset hold-down counter
            entry.is_occluded = False                 # re-seen = not occluded

            # Track quality: update running confidence average and persistence
            conf = det.get("confidence", 0.5)
            entry.total_frames_seen += 1
            n = entry.total_frames_seen
            entry.avg_confidence = entry.avg_confidence + (conf - entry.avg_confidence) / n

            if entry.confirmed:
                # Compute velocity from previous smooth_bbox -> new bbox
                prev_bbox = entry.smooth_bbox
                new_bbox = det["bbox"]
                if prev_bbox is not None:
                    prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2.0
                    prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2.0
                    new_cx = (float(new_bbox[0]) + float(new_bbox[2])) / 2.0
                    new_cy = (float(new_bbox[1]) + float(new_bbox[3])) / 2.0
                    prev_w = max(1.0, prev_bbox[2] - prev_bbox[0])
                    prev_h = max(1.0, prev_bbox[3] - prev_bbox[1])
                    prev_diag = math.hypot(prev_w, prev_h)
                    jump_px = math.hypot(new_cx - prev_cx, new_cy - prev_cy)

                    if jump_px > (config.bbox_snap_distance_ratio * prev_diag):
                        # Re-anchor instantly on large jumps to avoid sticky boxes.
                        entry.smooth_bbox = [float(v) for v in new_bbox]
                        entry.velocity = (0.0, 0.0, 0.0, 0.0)
                        entry.prev_velocity = (0.0, 0.0, 0.0, 0.0)
                        continue

                    new_vel = (
                        float(new_bbox[0]) - prev_bbox[0],
                        float(new_bbox[1]) - prev_bbox[1],
                        float(new_bbox[2]) - prev_bbox[2],
                        float(new_bbox[3]) - prev_bbox[3],
                    )
                    # Motion consistency: dot product of normalised velocity vectors
                    entry.motion_consistency = self._velocity_consistency(
                        entry.prev_velocity, new_vel, entry.motion_consistency
                    )
                    entry.prev_velocity = entry.velocity
                    entry.velocity = new_vel
                # Update smoothed bbox using EMA
                entry.smooth_bbox = self._ema(entry.smooth_bbox, det["bbox"])
            else:
                entry.confirm_count += 1
                if entry.confirm_count >= self._confirm_frames:
                    entry.confirmed = True
                    entry.smooth_bbox = list(det["bbox"])  # seed with first confirmed

        # Decrement hold timers for tracks no longer seen by YOLO
        to_delete = []
        for gid, entry in self._entries.items():
            if gid not in seen_ids:
                if entry.confirmed:
                    # Occlusion detection: does this track overlap with any visible track?
                    if config.occlusion_hold_enabled and not entry.is_occluded:
                        if self._is_occluded_by_visible(gid, seen_ids):
                            entry.is_occluded = True
                            entry.hold_frames += config.occlusion_hold_bonus_frames

                    entry.hold_frames -= 1
                    if entry.hold_frames <= 0:
                        to_delete.append(gid)
                    elif entry.smooth_bbox is not None:
                        # Velocity extrapolation during hold
                        damping = 0.75
                        vx1, vy1, vx2, vy2 = entry.velocity
                        entry.smooth_bbox = [
                            entry.smooth_bbox[0] + vx1,
                            entry.smooth_bbox[1] + vy1,
                            entry.smooth_bbox[2] + vx2,
                            entry.smooth_bbox[3] + vy2,
                        ]
                        entry.velocity = (
                            vx1 * damping,
                            vy1 * damping,
                            vx2 * damping,
                            vy2 * damping,
                        )
                else:
                    # Candidate never confirmed — reset immediately
                    entry.confirm_count = 0
        for gid in to_delete:
            del self._entries[gid]

        # Track quality pruning: remove confirmed tracks with weak scores
        if config.track_quality_scoring:
            weak = []
            for gid, entry in self._entries.items():
                if entry.confirmed and entry.total_frames_seen >= 6:
                    score = self._track_quality_score(entry)
                    if score < config.track_quality_min_score:
                        weak.append(gid)
            for gid in weak:
                del self._entries[gid]

        # Build output: only confirmed tracks
        output = []
        for gid, entry in self._entries.items():
            if entry.confirmed and entry.last_raw is not None and entry.smooth_bbox is not None:
                patched = dict(entry.last_raw)
                # Replace bbox with smoothed version (int-snapped)
                sx1, sy1, sx2, sy2 = [max(0, int(v)) for v in entry.smooth_bbox]
                if sx2 <= sx1 or sy2 <= sy1:
                    patched["bbox"] = list(entry.last_raw.get("bbox", [sx1, sy1, sx2, sy2]))
                else:
                    patched["bbox"] = [sx1, sy1, sx2, sy2]
                output.append(patched)
        return output

    def reset(self, gid: str) -> None:
        """Remove a specific identity from the buffer (e.g. after exit event)."""
        self._entries.pop(gid, None)

    # ----------------------------------------------------------------- private

    def _ema(
        self, prev: Optional[List[float]], new: List[int]
    ) -> List[float]:
        """Exponential Moving Average: smooth = alpha * new + (1-alpha) * prev."""
        if prev is None:
            return [float(v) for v in new]
        a = self._alpha
        return [a * n + (1.0 - a) * p for n, p in zip(new, prev)]

    @staticmethod
    def _velocity_consistency(
        prev_vel: Tuple[float, float, float, float],
        new_vel: Tuple[float, float, float, float],
        current_consistency: float,
    ) -> float:
        """
        Compute how consistent the motion direction is between frames.

        Uses a normalised dot product of the center velocity vectors, then
        smooths with EMA (alpha=0.3).  Returns 0-1 where:
          1.0 = perfectly linear motion
          0.0 = erratic / random direction changes
        """
        # Use center velocity (average of x1,x2 and y1,y2)
        pcx = (prev_vel[0] + prev_vel[2]) / 2.0
        pcy = (prev_vel[1] + prev_vel[3]) / 2.0
        ncx = (new_vel[0] + new_vel[2]) / 2.0
        ncy = (new_vel[1] + new_vel[3]) / 2.0

        prev_mag = (pcx ** 2 + pcy ** 2) ** 0.5
        new_mag = (ncx ** 2 + ncy ** 2) ** 0.5

        if prev_mag < 0.5 or new_mag < 0.5:
            # Nearly stationary — no meaningful direction
            return current_consistency

        # Normalised dot product: cos(angle) in [-1, 1], remap to [0, 1]
        dot = (pcx * ncx + pcy * ncy) / (prev_mag * new_mag)
        raw = (dot + 1.0) / 2.0  # 0 = opposite, 1 = same direction

        # EMA smooth
        alpha = 0.3
        return alpha * raw + (1.0 - alpha) * current_consistency

    @staticmethod
    def _track_quality_score(entry: 'TemporalEntry') -> float:
        """
        Weighted quality score for a confirmed track.

        Components:
          - avg_confidence: how confident YOLO is on average (0-1)
          - persistence: how many frames the track has survived (normalised)
          - motion_consistency: how linear the movement is (0-1)
        """
        # Normalise persistence: 30+ frames = 1.0
        persistence = min(1.0, entry.total_frames_seen / 30.0)

        return (
            config.track_quality_conf_weight * entry.avg_confidence
            + config.track_quality_persist_weight * persistence
            + config.track_quality_motion_weight * entry.motion_consistency
        )

    def _is_occluded_by_visible(self, lost_gid: str, visible_ids: set) -> bool:
        """
        Check if a lost track's bbox overlaps with any currently visible track.

        If overlap > occlusion_iou_threshold, the track is likely behind another
        person (occluded), not gone from the scene.
        """
        lost_entry = self._entries.get(lost_gid)
        if lost_entry is None or lost_entry.smooth_bbox is None:
            return False

        lx1, ly1, lx2, ly2 = lost_entry.smooth_bbox

        for gid in visible_ids:
            vis_entry = self._entries.get(gid)
            if vis_entry is None or not vis_entry.confirmed or vis_entry.smooth_bbox is None:
                continue

            vx1, vy1, vx2, vy2 = vis_entry.smooth_bbox

            # Compute IoU
            ix1, iy1 = max(lx1, vx1), max(ly1, vy1)
            ix2, iy2 = min(lx2, vx2), min(ly2, vy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_l = max(1.0, (lx2 - lx1) * (ly2 - ly1))
            area_v = max(1.0, (vx2 - vx1) * (vy2 - vy1))
            iou = inter / (area_l + area_v - inter)

            if iou >= config.occlusion_iou_threshold:
                return True

        return False


class TrackIdentityStore:
    """Maps per-camera ByteTrack IDs to stable session identities."""

    def __init__(self) -> None:
        self._track_to_identity: Dict[Tuple[str, int], str] = {}
        self._identity_last_seen: Dict[str, float] = {}
        self._fallback_tracks: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._next_id = 1

    def _new_identity(self) -> str:
        global_id = f"Person_{self._next_id:03d}"
        self._next_id += 1
        return global_id

    def resolve(self, camera_id: str, track_id: int, now: float) -> str:
        key = (camera_id, int(track_id))
        if key not in self._track_to_identity:
            self._track_to_identity[key] = self._new_identity()
        global_id = self._track_to_identity[key]
        self._identity_last_seen[global_id] = now
        return global_id

    def resolve_by_box(self, camera_id: str, bbox: BBox, now: float) -> str:
        best_key = None
        best_score = 0.0
        cx, cy = self._center(bbox)
        height = max(1, bbox[3] - bbox[1])

        for key, data in self._fallback_tracks.items():
            if key[0] != camera_id or now - data["last_seen"] > 2.0:
                continue
            prev_bbox = data["bbox"]
            px, py = self._center(prev_bbox)
            dist = math.hypot(cx - px, cy - py)
            iou = self._iou(bbox, prev_bbox)
            score = max(iou, 1.0 - dist / max(60.0, height * 0.75))
            if score > best_score:
                best_key, best_score = key, score

        if best_key is not None and best_score >= 0.25:
            global_id = best_key[1]
        else:
            global_id = self._new_identity()
            best_key = (camera_id, global_id)

        self._fallback_tracks[best_key] = {"bbox": bbox, "last_seen": now}
        self._identity_last_seen[global_id] = now
        return global_id

    def cleanup(self, now: float, stale_after_s: float = 30.0) -> None:
        stale_ids = {gid for gid, ts in self._identity_last_seen.items() if now - ts > stale_after_s}
        if not stale_ids:
            return
        for gid in stale_ids:
            self._identity_last_seen.pop(gid, None)
        stale_keys = [key for key, gid in self._track_to_identity.items() if gid in stale_ids]
        for key in stale_keys:
            self._track_to_identity.pop(key, None)
        stale_fallback = [key for key, data in self._fallback_tracks.items() if now - data["last_seen"] > stale_after_s]
        for key in stale_fallback:
            self._fallback_tracks.pop(key, None)

    @staticmethod
    def _center(bbox: BBox) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)


class Pipeline:
    COCO_NAMES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
        34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
        38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
        42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
        47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
        52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
        57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
        61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
        66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
        70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
        74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
        78: "hair drier", 79: "toothbrush",
    }

    CATEGORY_MAP = {
        "vehicle": {1, 2, 3, 4, 5, 6, 7, 8},
        "animal": {14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        "accessory": {24, 25, 26, 27, 28},
        "sports": {29, 30, 31, 32, 33, 34, 35, 36, 37, 38},
        "food": {46, 47, 48, 49, 50, 51, 52, 53, 54, 55},
        "furniture": {13, 56, 57, 59, 60},
        "electronic": {62, 63, 64, 65, 66, 67},
        "kitchen": {39, 40, 41, 42, 43, 44, 45, 68, 69, 70, 71, 72},
        "other": {9, 10, 11, 12, 58, 61, 73, 74, 75, 76, 77, 78, 79},
    }
    VEHICLE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8}
    LUGGAGE_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}

    # Confidence thresholds are now read from config.py so they can be
    # adjusted without touching pipeline code.
    @property
    def PERSON_CONF_THRESHOLD(self) -> float:   # type: ignore[override]
        return config.person_conf_threshold

    @property
    def OBJECT_CONF_THRESHOLD(self) -> float:   # type: ignore[override]
        return config.object_conf_threshold

    @property
    def LUGGAGE_CONF_THRESHOLD(self) -> float:  # type: ignore[override]
        return config.luggage_conf_threshold

    @property
    def VEHICLE_CONF_THRESHOLD(self) -> float:
        return config.vehicle_conf_threshold

    def __init__(self, yolo_model_path: str = "", device: str = "") -> None:
        model_path = yolo_model_path or config.yolo_model_path
        infer_device = device or config.yolo_device
        print(f"[Pipeline] Loading model: {model_path} on device: {infer_device}")
        self.yolo_model = YOLO(model_path)
        self.device = infer_device
        self.identity_store = TrackIdentityStore()
        self.track_states: Dict[str, TrackState] = {}
        self.identity_records: Dict[str, Dict[str, Any]] = {}
        self._object_counter = 0
        self._temporal_buffer = TemporalBuffer(
            confirm_frames=config.temporal_confirm_frames,
            hold_frames=config.temporal_hold_frames,
            smoothing_alpha=config.bbox_smoothing_alpha,
        )
        # DB write throttle: last flush time per identity (avoids write flood)
        self._db_last_flush: Dict[str, float] = {}
        database.init_db()
        print("[Pipeline] Ready.")

    @staticmethod
    def get_object_category(cls_id: int) -> str:
        for category, ids in Pipeline.CATEGORY_MAP.items():
            if cls_id in ids:
                return category
        return "other"

    def process_frame(self, frame: np.ndarray, camera_id: str = "CAM_01") -> PipelineResult:
        now = time.time()
        now_iso = datetime.datetime.now().astimezone().isoformat()
        location = geolocation_engine.get_current_location()

        # Import production guard singletons for dynamic, scene-aware inference
        from modules.production_guard import latency_guard, scene_profiler, threshold_calibrator

        # Gate CLAHE on scene brightness (skip in well-lit conditions)
        scene = scene_profiler.get_profile(camera_id)
        inference_frame = frame_preprocessor.enhance(frame, scene_brightness=scene.avg_brightness)

        # Dynamic tracker confidence:
        # Start from calibrated camera threshold, then adjust to scene difficulty.
        dynamic_tracker_conf = (
            threshold_calibrator.get_threshold(camera_id)
            + config.tracker_conf_base_offset
        )
        if scene.is_low_light:
            dynamic_tracker_conf += config.tracker_conf_low_light_delta
        if scene.is_crowded:
            dynamic_tracker_conf += config.tracker_conf_crowded_delta
        dynamic_tracker_conf = max(
            config.tracker_conf_min,
            min(config.tracker_conf_max, dynamic_tracker_conf),
        )

        # Primary pass at config resolution (640px) with ByteTrack
        yolo = self.yolo_model.track(
            inference_frame,
            persist=True,
            verbose=False,
            imgsz=config.yolo_imgsz,
            conf=dynamic_tracker_conf,
            iou=config.tracker_nms_iou,
            device=self.device,
            tracker="bytetrack.yaml",
        )[0]

        person_candidates, object_detections = self._parse_yolo_result(yolo, frame.shape[:2], camera_id, now)

        # ── Smart Multi-Scale (gated by latency guard) ───────────────────────
        if (config.multiscale_enabled
                and latency_guard.multiscale_allowed
                and self._should_trigger_multiscale(person_candidates, frame.shape[:2])):
            extra_ms = self._multiscale_detect(inference_frame, frame.shape[:2], camera_id, now)
            if extra_ms:
                person_candidates.extend(extra_ms)

        # ── Small object re-inference (gated by latency guard) ───────────────
        if (config.small_object_reinference
                and latency_guard.reinference_allowed
                and person_candidates):
            extra_roi = self._small_object_reinference(
                inference_frame, person_candidates, frame.shape[:2], camera_id, now
            )
            if extra_roi:
                person_candidates.extend(extra_roi)

        # ── Hard Negative Filtering ──────────────────────────────────────────
        if config.hard_negative_filtering:
            person_candidates = self._hard_negative_filter(person_candidates, frame.shape[:2])

        people_raw = self._dedupe_people(person_candidates)

        # ── Temporal confirmation + hold + EMA + velocity + quality scoring ──
        people = self._temporal_buffer.update(people_raw)

        # ── Re-Detection on Track Loss (gated by latency guard) ──────────────
        if config.redetection_on_loss_enabled and latency_guard.redetection_allowed:
            recovered = self._redetect_lost_tracks(inference_frame, people_raw, frame.shape[:2], camera_id, now)
            if recovered:
                updated = self._temporal_buffer.update(recovered)
                existing_ids = {p["global_id"] for p in people}
                for p in updated:
                    if p["global_id"] not in existing_ids:
                        people.append(p)
                        existing_ids.add(p["global_id"])

        object_links = self._associate_objects_to_people(people, object_detections)

        detections: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []
        person_ids: List[str] = []
        person_positions: Dict[str, Tuple[float, float]] = {}

        for person in people:
            det, alert = self._build_person_detection(
                person,
                camera_id,
                now,
                now_iso,
                location,
                frame,
                object_links.get(person["global_id"], {}),
            )
            detections.append(det)
            person_ids.append(det["global_id"])
            x1, y1, x2, y2 = det["bbox"]
            person_positions[det["global_id"]] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if alert:
                alerts.append(alert)

        object_payloads = self._build_object_detections(object_detections, now_iso, location)
        detections.extend(object_payloads)

        luggage_events = self._update_luggage(object_detections, person_positions, now, camera_id, frame)
        alerts.extend(luggage_events)

        self._update_presence(person_ids, now, camera_id)
        self._cleanup(now)

        return PipelineResult(
            annotated_frame=frame,
            identities=list(self.identity_records.values()),
            alerts=[a for a in alerts if a],
            current_detections=detections,
        )

    def _parse_yolo_result(
        self,
        result: Any,
        frame_shape: Tuple[int, int],
        camera_id: str,
        now: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if result.boxes is None:
            return [], []

        from modules.production_guard import fp_memory, scene_profiler, threshold_calibrator

        boxes = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.int().cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        track_ids = (
            result.boxes.id.int().cpu().numpy()
            if result.boxes.id is not None
            else [None] * len(boxes)
        )

        people: List[Dict[str, Any]] = []
        objects: List[Dict[str, Any]] = []
        h, w = frame_shape
        scene = scene_profiler.get_profile(camera_id)
        dynamic_person_base = threshold_calibrator.get_threshold(camera_id)

        for box, cls_id, conf, track_id in zip(boxes, cls_ids, confs, track_ids):
            cls_id = int(cls_id)
            conf = float(conf)
            x1, y1, x2, y2 = self._clip_box(box, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            if cls_id == 0:  # person
                # Dynamic person threshold:
                # 1) per-camera adaptive calibrator baseline
                # 2) scene-aware adjustment (low-light / crowded conditions)
                # 3) false-positive hotspot confidence boost by spatial cell
                person_threshold = dynamic_person_base
                if scene.is_low_light:
                    person_threshold -= 0.03
                if scene.is_crowded:
                    person_threshold += 0.02
                person_threshold += fp_memory.get_confidence_boost(camera_id, (x1, y1, x2, y2))
                person_threshold = max(
                    config.adaptive_calib_min_conf,
                    min(config.adaptive_calib_max_conf, person_threshold),
                )

                if conf < person_threshold:
                    continue
                if not self._valid_person_box((x1, y1, x2, y2), frame_shape):
                    continue
                global_id = (
                    self.identity_store.resolve(camera_id, int(track_id), now)
                    if track_id is not None
                    else self.identity_store.resolve_by_box(camera_id, (x1, y1, x2, y2), now)
                )
                people.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "track_id": int(track_id) if track_id is not None else None,
                    "global_id": global_id,
                })
                continue

            # ── Fix 4: Per-class threshold for vehicles vs objects vs luggage ─
            if cls_id in self.VEHICLE_CLASSES:
                threshold = config.vehicle_conf_threshold
            elif cls_id in self.LUGGAGE_CLASSES:
                threshold = config.luggage_conf_threshold
            else:
                threshold = config.object_conf_threshold

            # Scene-aware object thresholding for better recall in dark scenes.
            if scene.is_low_light:
                threshold = max(
                    config.object_low_light_conf_min,
                    threshold + config.object_low_light_conf_delta,
                )

            if cls_id in self.COCO_NAMES and conf >= threshold:
                if not self._valid_object_box((x1, y1, x2, y2), frame_shape):
                    continue
                objects.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": self.COCO_NAMES.get(cls_id, f"class_{cls_id}"),
                    "track_id": int(track_id) if track_id is not None else None,
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                })

        return people, objects

    @staticmethod
    def _clip_box(box: Iterable[float], width: int, height: int) -> BBox:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        return (
            max(0, min(width - 1, x1)),
            max(0, min(height - 1, y1)),
            max(0, min(width - 1, x2)),
            max(0, min(height - 1, y2)),
        )

    @staticmethod
    def _valid_person_box(bbox: BBox, frame_shape: Tuple[int, int]) -> bool:
        h, w = frame_shape
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        # ── Fix 5: Use config-driven min dimensions ───────────────────────────
        if bw < config.person_min_width_px or bh < config.person_min_height_px:
            return False
        area_ratio = (bw * bh) / max(1.0, float(w * h))
        aspect = bw / max(1.0, float(bh))
        if area_ratio < config.person_min_area_ratio or area_ratio > config.person_max_area_ratio:
            return False
        if aspect < config.person_min_aspect or aspect > config.person_max_aspect:
            return False
        if (
            bh > h * config.person_edge_tall_height_ratio
            and bw < w * config.person_edge_tall_width_ratio
        ):
            return False
        return True

    @staticmethod
    def _valid_object_box(bbox: BBox, frame_shape: Tuple[int, int]) -> bool:
        h, w = frame_shape
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < config.object_min_width_px or bh < config.object_min_height_px:
            return False
        area_ratio = (bw * bh) / max(1.0, float(w * h))
        aspect = bw / max(1.0, float(bh))
        return (
            config.object_min_area_ratio <= area_ratio <= config.object_max_area_ratio
            and config.object_min_aspect <= aspect <= config.object_max_aspect
        )

    def _small_object_reinference(
        self,
        frame: np.ndarray,
        candidates: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        camera_id: str,
        now: float,
    ) -> List[Dict[str, Any]]:
        """
        Re-run YOLO at higher resolution on cropped regions around small persons.

        If any detected person bbox occupies less than `small_object_area_threshold`
        of the frame, we crop a padded region around it, upscale to
        `reinference_imgsz`, and run a detect-only YOLO pass. New detections
        found inside the crop are mapped back to full-frame coordinates and
        returned as extra candidates (deduplicated later by the caller).

        This catches distant/small people that the 640px main pass misses
        entirely or detects at very low confidence.
        """
        h, w = frame_shape
        frame_area = float(h * w)
        threshold = config.small_object_area_threshold

        # Identify small-person regions worth re-examining
        small_regions: List[Tuple[int, int, int, int]] = []
        for person in candidates:
            x1, y1, x2, y2 = person["bbox"]
            bbox_area = float((x2 - x1) * (y2 - y1))
            if bbox_area / frame_area < threshold:
                # Pad the crop region by 2x the person height for context
                pad_h = max(config.reinference_min_pad_px, (y2 - y1) * config.reinference_pad_scale)
                pad_w = max(config.reinference_min_pad_px, (x2 - x1) * config.reinference_pad_scale)
                rx1 = max(0, x1 - pad_w)
                ry1 = max(0, y1 - pad_h)
                rx2 = min(w, x2 + pad_w)
                ry2 = min(h, y2 + pad_h)
                # Ensure minimum crop size
                if (
                    (rx2 - rx1) >= config.reinference_min_crop_size_px
                    and (ry2 - ry1) >= config.reinference_min_crop_size_px
                ):
                    small_regions.append((rx1, ry1, rx2, ry2))

        if not small_regions:
            return []

        # Limit to avoid excessive compute
        small_regions = small_regions[:config.reinference_max_regions]

        # Merge overlapping regions to avoid redundant inference
        merged = self._merge_regions(small_regions)

        extra_people: List[Dict[str, Any]] = []

        for rx1, ry1, rx2, ry2 in merged:
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue

            # Run detection-only (no tracking) at higher resolution
            try:
                hi_res_results = self.yolo_model(
                    crop,
                    verbose=False,
                    imgsz=config.reinference_imgsz,
                    conf=config.person_conf_threshold * config.reinference_conf_scale,
                    iou=config.tracker_nms_iou,
                    device=self.device,
                    classes=[0],  # person only
                )
                if not hi_res_results or hi_res_results[0].boxes is None:
                    continue

                boxes = hi_res_results[0].boxes.xyxy.cpu().numpy()
                confs = hi_res_results[0].boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    # Map crop coordinates back to full frame
                    fx1 = int(round(box[0])) + rx1
                    fy1 = int(round(box[1])) + ry1
                    fx2 = int(round(box[2])) + rx1
                    fy2 = int(round(box[3])) + ry1

                    # Validate in full-frame context
                    full_bbox = (fx1, fy1, fx2, fy2)
                    if not self._valid_person_box(full_bbox, frame_shape):
                        continue

                    global_id = self.identity_store.resolve_by_box(
                        camera_id, full_bbox, now
                    )
                    extra_people.append({
                        "bbox": full_bbox,
                        "confidence": float(conf),
                        "track_id": None,
                        "global_id": global_id,
                    })
            except Exception as e:
                print(f"[Pipeline] Small-object reinference error: {e}")
                continue

        return extra_people

    @staticmethod
    def _merge_regions(
        regions: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping rectangular regions to avoid redundant crops."""
        if len(regions) <= 1:
            return regions

        merged: List[List[int]] = []
        for r in sorted(regions, key=lambda r: r[0]):
            if merged and r[0] <= merged[-1][2] and r[1] <= merged[-1][3]:
                # Overlapping — expand the last merged region
                merged[-1][2] = max(merged[-1][2], r[2])
                merged[-1][3] = max(merged[-1][3], r[3])
                merged[-1][0] = min(merged[-1][0], r[0])
                merged[-1][1] = min(merged[-1][1], r[1])
            else:
                merged.append(list(r))

        return [tuple(m) for m in merged]

    def _should_trigger_multiscale(
        self,
        candidates: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
    ) -> bool:
        """
        Smart multi-scale triggering: only run the expensive 960px pass when
        at least one condition is met.

        Conditions (any one triggers):
        1. Small objects: any person bbox area < multiscale_trigger_small_area
        2. Low confidence: any person confidence < multiscale_trigger_low_conf
        3. Track loss: any confirmed track is currently in hold phase
        """
        h, w = frame_shape
        frame_area = float(h * w)

        for det in candidates:
            # Condition 1: small object
            x1, y1, x2, y2 = det["bbox"]
            bbox_area = float((x2 - x1) * (y2 - y1))
            if bbox_area / frame_area < config.multiscale_trigger_small_area:
                return True

            # Condition 2: low confidence
            if det.get("confidence", 1.0) < config.multiscale_trigger_low_conf:
                return True

        # Condition 3: any confirmed track in hold phase (track loss event)
        if config.multiscale_trigger_on_track_loss:
            for gid, entry in self._temporal_buffer._entries.items():
                if entry.confirmed and entry.hold_frames < self._temporal_buffer._hold_frames:
                    return True

        # No candidates at all but tracks exist — might be total loss
        if not candidates and len(self._temporal_buffer._entries) > 0:
            return True

        return False

    def _multiscale_detect(
        self,
        frame: np.ndarray,
        frame_shape: Tuple[int, int],
        camera_id: str,
        now: float,
    ) -> List[Dict[str, Any]]:
        """
        Full-frame detection at `multiscale_imgsz` (960px) merged with the
        primary 640px pass.

        Uses the secondary confidence tier (more permissive than primary).
        """
        try:
            hi_results = self.yolo_model(
                frame,
                verbose=False,
                imgsz=config.multiscale_imgsz,
                conf=config.secondary_conf_threshold,
                iou=config.multiscale_nms_iou,
                device=self.device,
                classes=[0],   # person only
            )
            if not hi_results or hi_results[0].boxes is None:
                return []

            boxes = hi_results[0].boxes.xyxy.cpu().numpy()
            confs = hi_results[0].boxes.conf.cpu().numpy()
        except Exception as e:
            print(f"[Pipeline] Multiscale detect error: {e}")
            return []

        h, w = frame_shape
        extra: List[Dict[str, Any]] = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = self._clip_box(box, w, h)
            full_bbox = (x1, y1, x2, y2)
            if not self._valid_person_box(full_bbox, frame_shape):
                continue
            global_id = self.identity_store.resolve_by_box(camera_id, full_bbox, now)
            extra.append({
                "bbox": full_bbox,
                "confidence": float(conf),
                "track_id": None,
                "global_id": global_id,
            })
        return extra

    def _hard_negative_filter(
        self,
        candidates: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """
        Reject detections that are geometrically implausible or clearly noise.

        Rules applied (all are fast, O(1) per detection):
        1. Edge-clipped box: if the bbox is pressed hard against a frame border
           with < `hnf_edge_margin_px` pixels of clearance on ≥3 sides, it is
           almost certainly a partial detection of a background object, not a
           person walking in from the edge.  Drop it.
        2. Hyper-tall slivers: aspect ratio < 0.04 after geometry filter already
           passed person_min_aspect=0.10 — catches extreme edge artefacts.
        3. Floating detections: bbox centroid is in the top 5% of the frame AND
           bbox height < 8% of frame height.  Unlikely to be a real person.
        """
        if not config.hard_negative_filtering:
            return candidates

        h, w = frame_shape
        margin = config.hnf_edge_margin_px
        kept: List[Dict[str, Any]] = []

        for det in candidates:
            x1, y1, x2, y2 = det["bbox"]
            bw, bh = x2 - x1, y2 - y1

            # Rule 1: edge-clipped on ≥3 sides
            sides_clipped = sum([
                x1 <= margin,
                y1 <= margin,
                x2 >= w - margin,
                y2 >= h - margin,
            ])
            if sides_clipped >= config.hnf_min_clipped_sides:
                continue

            # Rule 2: hyper-tall sliver (after passing geometry filter)
            aspect = bw / max(1.0, float(bh))
            if aspect < config.hnf_min_aspect_ratio:
                continue

            # Rule 3: floating top-of-frame ghost
            cy = (y1 + y2) / 2.0
            if cy < h * config.hnf_top_band_ratio and bh < h * config.hnf_small_height_ratio:
                continue

            kept.append(det)

        return kept

    def _redetect_lost_tracks(
        self,
        frame: np.ndarray,
        seen_this_frame: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        camera_id: str,
        now: float,
    ) -> List[Dict[str, Any]]:
        """
        For confirmed tracks currently in the hold phase (not seen by YOLO
        this frame), expand their predicted bbox into an ROI and run a
        targeted high-resolution re-detection pass.

        This actively tries to rescue tracks mid-occlusion rather than just
        coasting on extrapolated positions.
        """
        seen_ids = {p["global_id"] for p in seen_this_frame}

        # Collect hold-phase confirmed tracks from TemporalBuffer
        lost_entries = [
            (gid, entry)
            for gid, entry in self._temporal_buffer._entries.items()
            if entry.confirmed
            and gid not in seen_ids
            and entry.hold_frames > 0
            and entry.smooth_bbox is not None
        ]

        if not lost_entries:
            return []

        h, w = frame_shape
        expand = config.redetection_roi_expand_factor
        recovered: List[Dict[str, Any]] = []

        for gid, entry in lost_entries:
            sx1, sy1, sx2, sy2 = [int(v) for v in entry.smooth_bbox]
            bw, bh = sx2 - sx1, sy2 - sy1

            # Expand the ROI by the configured factor, centred on predicted bbox
            pad_x = int(bw * (expand - 1.0) / 2.0)
            pad_y = int(bh * (expand - 1.0) / 2.0)
            rx1 = max(0, sx1 - pad_x)
            ry1 = max(0, sy1 - pad_y)
            rx2 = min(w, sx2 + pad_x)
            ry2 = min(h, sy2 + pad_y)

            if (rx2 - rx1) < 32 or (ry2 - ry1) < 32:
                continue

            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue

            try:
                results = self.yolo_model(
                    crop,
                    verbose=False,
                    imgsz=config.redetection_imgsz,
                    conf=config.redetection_conf_threshold_tier,
                    iou=0.50,
                    device=self.device,
                    classes=[0],
                )
                if not results or results[0].boxes is None:
                    continue

                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    # Map crop coords back to full frame
                    fx1 = max(0, int(round(box[0])) + rx1)
                    fy1 = max(0, int(round(box[1])) + ry1)
                    fx2 = min(w, int(round(box[2])) + rx1)
                    fy2 = min(h, int(round(box[3])) + ry1)
                    full_bbox = (fx1, fy1, fx2, fy2)

                    if not self._valid_person_box(full_bbox, frame_shape):
                        continue

                    # Reuse the existing global_id for this lost track
                    recovered.append({
                        "bbox": full_bbox,
                        "confidence": float(conf),
                        "track_id": None,
                        "global_id": gid,
                    })
                    break  # Best match per lost track is enough
            except Exception as e:
                print(f"[Pipeline] Re-detect error for {gid}: {e}")
                continue

        return recovered

    def _dedupe_people(self, people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best_by_track: Dict[Any, Dict[str, Any]] = {}
        for person in people:
            track_key = person["track_id"] if person["track_id"] is not None else person["global_id"]
            prev = best_by_track.get(track_key)
            if prev is None or person["confidence"] > prev["confidence"]:
                best_by_track[track_key] = person

        candidates = sorted(best_by_track.values(), key=lambda p: p["confidence"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for candidate in candidates:
            if all(
                self._iou(candidate["bbox"], prev["bbox"]) < config.dedup_iou_threshold
                and self._overlap_over_smaller(candidate["bbox"], prev["bbox"]) < config.dedup_overlap_threshold
                for prev in kept
            ):
                kept.append(candidate)
        return sorted(kept, key=lambda p: p["bbox"][0])

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    @staticmethod
    def _overlap_over_smaller(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(min(area_a, area_b))

    def _build_person_detection(
        self,
        person: Dict[str, Any],
        camera_id: str,
        now: float,
        now_iso: str,
        location: Dict[str, float],
        frame: np.ndarray,
        object_links: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        global_id = person["global_id"]
        x1, y1, x2, y2 = person["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        height = max(1.0, float(y2 - y1))

        state = self.track_states.get(global_id)
        if state is None:
            state = TrackState(global_id, now, now, (cx, cy))
            self.track_states[global_id] = state

        dt = max(1e-3, now - state.last_seen)
        dx, dy = cx - state.last_center[0], cy - state.last_center[1]
        speed_px_s = math.hypot(dx, dy) / dt
        movement_direction = self._movement_direction(dx, dy)

        state.last_seen = now
        state.last_center = (cx, cy)
        state.speed_px_s = speed_px_s
        state.movement_direction = movement_direction
        state.samples.append((cx, cy, now, height))

        behaviour_label, behaviour_score, behaviour_alert = self._classify_behavior(state)

        signals = {
            "loitering": behaviour_label == "loitering",
            "pacing": behaviour_label == "pacing",
            "running": behaviour_label == "running",
            "prolonged_stillness": behaviour_label == "loitering",
        }
        risk = risk_engine.compute_risk(global_id, signals, behaviour_score=behaviour_score, avoidance_score=0.0)

        metadata = {
            "last_seen_camera": camera_id,
            "last_seen_time": now_iso,
            "entry_time": self.identity_records.get(global_id, {}).get("metadata", {}).get("entry_time", now_iso),
            "exit_time": now_iso,
            "risk_level": risk["risk_level"],
            "risk_score": risk["risk_score"],
            "risk_factors": risk["risk_factors"],
            "movement_direction": movement_direction,
            "speed": round(speed_px_s, 1),
            "activity": behaviour_label,
            "pose_detail": behaviour_label,
            "behaviour_label": behaviour_label,
            "behaviour_score": behaviour_score,
            "face_name": None,
            "carried_objects": sorted(set(object_links.get("carried_objects", []))),
            "nearby_objects": object_links.get("nearby_objects", []),
            "luggage_status": object_links.get("luggage_status", {}),
            "dwell_time_seconds": round(now - state.first_seen, 1),
            "frequency_label": "session",
            "visit_count": 1,
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
        }

        self.identity_records[global_id] = {
            "global_id": global_id,
            "face_name": None,
            "risk_level": risk["risk_level"],
            "metadata": metadata,
            "history": [],
        }

        # DB write throttle: flush at most once every db_write_interval_s per identity
        _now_m = time.monotonic()
        if _now_m - self._db_last_flush.get(global_id, 0.0) >= config.db_write_interval_s:
            self._db_last_flush[global_id] = _now_m
            database.save_detection({
                "object_id": global_id,
                "material": "person",
                "confidence": person["confidence"],
                "size": f"{x2 - x1}x{y2 - y1}",
                "timestamp": now_iso,
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
            })
            database.save_person_log({
                "person_id": global_id,
                "timestamp": now_iso,
                "position_x": cx,
                "position_y": cy,
                "speed": speed_px_s,
                "zone": "General",
                "event_type": behaviour_label,
            })
            database.save_identity(self.identity_records[global_id])

        alert = None
        if behaviour_alert:
            alert = alert_engine.create_alert(behaviour_alert, global_id, camera_id, {
                "behaviour_score": behaviour_score,
                "speed_px_s": speed_px_s,
            }, frame)

        return {
            "global_id": global_id,
            "track_id": person["track_id"],
            "bbox": [x1, y1, x2, y2],
            "display_name": global_id,
            "is_object": False,
            "object_class": "person",
            "object_category": "person",
            "confidence": round(person["confidence"], 3),
            "timestamp": now_iso,
            **metadata,
        }, alert

    def _associate_objects_to_people(
        self,
        people: List[Dict[str, Any]],
        objects: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        links: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"carried_objects": [], "nearby_objects": [], "luggage_status": {}})
        if not people or not objects:
            return links

        for obj in objects:
            obj_cx, obj_cy = obj["center"]
            best_person = None
            best_dist = float("inf")
            for person in people:
                px1, py1, px2, py2 = person["bbox"]
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                person_h = max(1, py2 - py1)
                dist = math.hypot(obj_cx - pcx, obj_cy - pcy)
                expanded = (px1 - person_h * 0.45, py1 - person_h * 0.25, px2 + person_h * 0.45, py2 + person_h * 0.35)
                inside_expanded = expanded[0] <= obj_cx <= expanded[2] and expanded[1] <= obj_cy <= expanded[3]
                if inside_expanded and dist < best_dist:
                    best_person = person
                    best_dist = dist

            if best_person is None:
                continue

            gid = best_person["global_id"]
            obj_label = obj["class_name"]
            obj_info = {
                "class": obj_label,
                "confidence": round(obj["confidence"], 3),
                "bbox": list(obj["bbox"]),
                "distance": round(best_dist, 1),
            }
            links[gid]["nearby_objects"].append(obj_info)
            if obj["class_id"] in self.LUGGAGE_CLASSES:
                links[gid]["carried_objects"].append(obj_label)
                links[gid]["luggage_status"][f"{obj_label}_{len(links[gid]['luggage_status']) + 1}"] = {
                    "type": obj_label,
                    "status": "nearby",
                    "confidence": round(obj["confidence"], 3),
                }
        return links

    @staticmethod
    def _movement_direction(dx: float, dy: float) -> str:
        if abs(dx) < 3 and abs(dy) < 3:
            return "stationary"
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "away" if dy < 0 else "towards"

    def _classify_behavior(self, state: TrackState) -> Tuple[str, float, Optional[str]]:
        samples = list(state.samples)
        if len(samples) < 6:
            return "normal", 0.0, None

        median_height = float(np.median([max(1.0, s[3]) for s in samples[-12:]]))
        body_lengths_per_s = state.speed_px_s / max(1.0, median_height)
        if body_lengths_per_s >= 2.4:
            return "running", min(80.0, 45.0 + body_lengths_per_s * 12.0), "sudden_movement"

        recent = np.array([(s[0], s[1]) for s in samples[-45:]], dtype=np.float32)
        center = recent.mean(axis=0)
        radius = float(np.linalg.norm(recent - center, axis=1).mean())
        if radius <= 45.0:
            if state.stationary_since is None:
                state.stationary_since = samples[-1][2]
            duration = state.last_seen - state.stationary_since
            if duration >= 30.0:
                return "loitering", min(90.0, 45.0 + duration), "loitering"
            return "stationary", min(25.0, duration / 30.0 * 25.0), None

        state.stationary_since = None
        pacing_score = self._pacing_score(samples)
        if pacing_score > 0:
            return "pacing", pacing_score, None

        return "normal", 0.0, None

    @staticmethod
    def _pacing_score(samples: List[Tuple[float, float, float, float]]) -> float:
        if len(samples) < 18:
            return 0.0
        pts = np.array([(s[0], s[1]) for s in samples[-45:]], dtype=np.float32)
        deltas = np.diff(pts, axis=0)
        path = float(np.linalg.norm(deltas, axis=1).sum())
        displacement = float(np.linalg.norm(pts[-1] - pts[0]))
        if path < 120.0 or displacement < 1.0:
            return 0.0
        axis = 0 if np.std(pts[:, 0]) >= np.std(pts[:, 1]) else 1
        signs = np.sign(deltas[:, axis])
        signs = signs[np.abs(signs) > 0]
        reversals = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
        if reversals >= 3 and path / max(1.0, displacement) >= 2.0:
            return min(65.0, 30.0 + reversals * 6.0)
        return 0.0

    def _build_object_detections(
        self,
        objects: List[Dict[str, Any]],
        now_iso: str,
        location: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            obj_id = f"Obj_{obj['track_id']}" if obj.get("track_id") is not None else f"Obj_{self._object_counter}"
            self._object_counter += 1
            payload = {
                "global_id": obj_id,
                "bbox": [x1, y1, x2, y2],
                "display_name": obj["class_name"],
                "is_object": True,
                "object_class": obj["class_name"],
                "object_category": self.get_object_category(obj["class_id"]),
                "confidence": round(obj["confidence"], 3),
                "timestamp": now_iso,
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
            }
            payloads.append(payload)
            database.save_detection({
                "object_id": obj_id,
                "material": obj["class_name"],
                "confidence": obj["confidence"],
                "size": f"{x2 - x1}x{y2 - y1}",
                "timestamp": now_iso,
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
            })
        return payloads

    def _update_luggage(
        self,
        objects: List[Dict[str, Any]],
        person_positions: Dict[str, Tuple[float, float]],
        now: float,
        camera_id: str,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        object_detections = []
        for obj in objects:
            if obj["class_id"] not in self.LUGGAGE_CLASSES:
                continue
            object_detections.append({
                "type": self.LUGGAGE_CLASSES[obj["class_id"]],
                "center": obj["center"],
                "bbox": obj["bbox"],
            })

        if not object_detections:
            return []

        result = luggage_tracker.update(object_detections, person_positions, now)
        alerts = []
        for event in result.get("events", []):
            owner = event.get("owner_id") or event.get("new_holder") or "Unknown"
            alert = alert_engine.create_alert(event["type"], owner, camera_id, event, frame)
            if alert:
                alerts.append(alert)
        return alerts

    @staticmethod
    def _update_presence(person_ids: List[str], now: float, camera_id: str) -> None:
        for event in presence_tracker.update(person_ids, now):
            database.save_presence_log({
                "person_id": event.get("person_id"),
                "event_type": event.get("type"),
                "timestamp": event.get("timestamp"),
                "session_duration": event.get("session_duration", 0.0),
            })
        for event in presence_tracker.check_exits(person_ids, camera_id):
            database.save_presence_log({
                "person_id": event.get("person_id"),
                "event_type": event.get("type"),
                "timestamp": event.get("timestamp"),
                "session_duration": event.get("session_duration", 0.0),
                "camera_id": camera_id,
            })

    def _cleanup(self, now: float) -> None:
        self.identity_store.cleanup(now, stale_after_s=config.track_stale_after_s)
        stale = [gid for gid, state in self.track_states.items() if now - state.last_seen > config.track_stale_after_s]
        for gid in stale:
            self.track_states.pop(gid, None)
            self._temporal_buffer.reset(gid)
            self._db_last_flush.pop(gid, None)  # prune throttle dict
