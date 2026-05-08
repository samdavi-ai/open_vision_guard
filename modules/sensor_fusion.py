"""
SensorFusionLayer — Airport-grade multi-camera bag tracking with Re-ID.
=======================================================================

Sits between YOLO raw detections and AirportLuggageTracker.update().
Does NOT modify the state machine or alert engine.

Components
----------
SpatialAligner          : camera pixel → shared world-coordinate frame (homography)
AppearanceEmbedder      : 512-d L2-normalised embedding via existing OSNet-AIN
EmbeddingGallery        : cosine-similarity matching with TTL-based eviction
KalmanMotionPredictor   : world-space position prediction per track
SensorFusionLayer       : orchestrates all of the above
FusionConfig            : all tuneable parameters
FusionMetrics           : per-call observability, logged every 100 frames
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionConfig:
    """All tuneable parameters for the SensorFusionLayer."""
    similarity_threshold: float = 0.82      # cosine threshold for gallery match
    gallery_ttl_s: float = 300.0            # evict gallery entries after 5 min
    spatial_filter_radius: float = 150.0    # world-unit radius for Kalman pre-filter
    kalman_noise_cov: float = 1e-4          # process noise (Q diagonal)
    kalman_meas_noise: float = 1e-2         # measurement noise (R diagonal)
    embedding_dim: int = 512                # OSNet-AIN output dimension
    log_interval_frames: int = 100          # log FusionMetrics every N frames


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionMetrics:
    """Populated every fuse() call; logged every 100 frames."""
    n_new_tracks: int = 0
    n_reidentified: int = 0
    n_spatial_filtered: int = 0
    mean_reid_score: float = 0.0
    frame_id: int = 0

    def log(self) -> None:
        log.debug(
            "[FusionMetrics] frame=%d  new=%d  reidentified=%d  "
            "spatial_filtered=%d  mean_reid_score=%.3f",
            self.frame_id,
            self.n_new_tracks,
            self.n_reidentified,
            self.n_spatial_filtered,
            self.mean_reid_score,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  FusedDetection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusedDetection:
    """
    Drop-in superset of the pipeline's object detection dict.
    All original keys are preserved; these fields are added.
    """
    # original detection dict preserved here
    raw: Dict[str, Any] = field(default_factory=dict)

    # new fields
    global_track_id: str = ""               # "BAG-{uuid8}", stable across cameras
    embedding: Optional[np.ndarray] = None  # 512-d unit vector (or None if embed failed)
    reid_score: float = 0.0                 # cosine score of gallery match (0 = new)
    is_new_track: bool = True
    world_xy: Tuple[float, float] = (0.0, 0.0)
    depth_z: Optional[float] = None         # median Z from depth frame (if provided)

    def to_dict(self) -> Dict[str, Any]:
        """Merge fused fields back into the raw detection dict."""
        d = dict(self.raw)
        d["global_track_id"] = self.global_track_id
        d["reid_score"] = round(self.reid_score, 4)
        d["is_new_track"] = self.is_new_track
        d["world_xy"] = self.world_xy
        if self.depth_z is not None:
            d["depth_z"] = round(self.depth_z, 3)
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  1. SpatialAligner
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAligner:
    """
    Projects a pixel-space (cx, cy) into a shared world-coordinate frame
    using a per-camera 3×3 homography matrix loaded from a YAML file.

    Falls back to identity (pixel = world) when no homography is configured.
    """

    def __init__(self, homography_config_path: Optional[str] = None) -> None:
        self._matrices: Dict[str, np.ndarray] = {}
        if homography_config_path and os.path.exists(homography_config_path):
            self._load_yaml(homography_config_path)
        else:
            log.info("[SpatialAligner] No homography config — using identity (pixel = world)")

    def _load_yaml(self, path: str) -> None:
        try:
            import yaml  # PyYAML
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            for cam_id, cfg in data.items():
                h_list = cfg.get("H")
                if h_list:
                    H = np.array(h_list, dtype=np.float64).reshape(3, 3)
                    self._matrices[str(cam_id)] = H
                    log.info("[SpatialAligner] Loaded homography for %s", cam_id)
        except Exception as e:
            log.warning("[SpatialAligner] Failed to load YAML: %s", e)

    def align(self, cx: float, cy: float, camera_id: str) -> Tuple[float, float]:
        """Return (world_x, world_y) for a given pixel center."""
        H = self._matrices.get(camera_id)
        if H is None:
            return (cx, cy)   # identity fallback

        pts = np.array([[[cx, cy]]], dtype=np.float64)
        try:
            world = cv2.perspectiveTransform(pts, H)
            return (float(world[0][0][0]), float(world[0][0][1]))
        except Exception as e:
            log.warning("[SpatialAligner] perspectiveTransform failed for %s: %s", camera_id, e)
            return (cx, cy)


# ─────────────────────────────────────────────────────────────────────────────
#  2. AppearanceEmbedder
# ─────────────────────────────────────────────────────────────────────────────

class AppearanceEmbedder:
    """
    Generates L2-normalised embeddings from bag crops.
    Reuses the global embedding_engine (OSNet-AIN) — no second model loaded.
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        self._dim = embedding_dim
        self._engine = None   # loaded lazily to avoid import-time issues

    def _get_engine(self):
        if self._engine is None:
            try:
                from modules.embedding_engine import embedding_engine
                self._engine = embedding_engine
            except Exception as e:
                log.error("[AppearanceEmbedder] Cannot load embedding_engine: %s", e)
        return self._engine

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns a shape-(512,) float32 L2-normalised embedding, or None on failure.
        """
        if crop is None or crop.size == 0:
            return None
        engine = self._get_engine()
        if engine is None:
            return None
        try:
            emb = engine.generate_embedding(crop)   # shape (1, 512)
            flat = emb.flatten().astype(np.float32)
            norm = np.linalg.norm(flat)
            if norm < 1e-6:
                return None
            return flat / norm
        except Exception as e:
            log.warning("[AppearanceEmbedder] embed failed: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  3. EmbeddingGallery
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingGallery:
    """
    Thread-safe in-memory gallery:  global_track_id → (embedding, timestamp, world_xy)

    Eviction runs in a background thread once per second.
    """

    def __init__(self, config: FusionConfig) -> None:
        self._cfg = config
        self._store: Dict[str, Tuple[np.ndarray, float, Tuple[float, float]]] = {}
        self._lock = threading.Lock()
        self._evict_thread = threading.Thread(
            target=self._evict_loop, daemon=True, name="FusionGalleryEvict"
        )
        self._evict_thread.start()

    # ── Write ──────────────────────────────────────────────────────────────

    def upsert(
        self,
        track_id: str,
        embedding: np.ndarray,
        timestamp: float,
        world_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        with self._lock:
            self._store[track_id] = (embedding, timestamp, world_xy)

    # ── Read ───────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: np.ndarray,
        world_xy: Tuple[float, float],
        spatial_filter_radius: float,
    ) -> Tuple[Optional[str], float]:
        """
        Return (best_track_id, cosine_score) above the similarity threshold,
        after pre-filtering by spatial proximity.
        Returns (None, 0.0) if no match or gallery is empty.
        """
        best_id: Optional[str] = None
        best_score: float = 0.0
        n_filtered = 0

        with self._lock:
            entries = list(self._store.items())

        for tid, (gal_emb, _ts, gal_xy) in entries:
            # Spatial pre-filter
            dist = math.hypot(world_xy[0] - gal_xy[0], world_xy[1] - gal_xy[1])
            if dist > spatial_filter_radius:
                n_filtered += 1
                continue

            score = float(np.dot(embedding, gal_emb))  # cosine (both L2-normed)
            if score > best_score:
                best_score = score
                best_id = tid

        if best_id is not None and best_score >= self._cfg.similarity_threshold:
            return best_id, best_score
        return None, best_score   # return score even when below threshold for metrics

    def query_unconstrained(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Query without spatial filter — used as fallback."""
        best_id: Optional[str] = None
        best_score: float = 0.0

        with self._lock:
            entries = list(self._store.items())

        for tid, (gal_emb, _ts, _xy) in entries:
            score = float(np.dot(embedding, gal_emb))
            if score > best_score:
                best_score = score
                best_id = tid

        if best_score >= self._cfg.similarity_threshold:
            return best_id, best_score
        return None, 0.0

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    # ── Eviction ───────────────────────────────────────────────────────────

    def evict_stale(self, ttl_seconds: Optional[float] = None) -> int:
        ttl = ttl_seconds if ttl_seconds is not None else self._cfg.gallery_ttl_s
        cutoff = time.time() - ttl
        removed = 0
        with self._lock:
            stale = [tid for tid, (_, ts, _) in self._store.items() if ts < cutoff]
            for tid in stale:
                del self._store[tid]
                removed += 1
        if removed:
            log.debug("[EmbeddingGallery] Evicted %d stale entries", removed)
        return removed

    def _evict_loop(self) -> None:
        while True:
            time.sleep(1.0)
            try:
                self.evict_stale()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
#  4. KalmanMotionPredictor
# ─────────────────────────────────────────────────────────────────────────────

class KalmanMotionPredictor:
    """
    One cv2.KalmanFilter per active track_id.
    State vector: [x, y, vx, vy] in world coords.
    """

    def __init__(self, config: FusionConfig) -> None:
        self._cfg = config
        self._filters: Dict[str, cv2.KalmanFilter] = {}
        self._lock = threading.Lock()

    def _make_filter(self, x: float, y: float) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)   # 4 states, 2 measurements (x, y)

        # Transition matrix: constant velocity
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        q = self._cfg.kalman_noise_cov
        r = self._cfg.kalman_meas_noise
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.errorCovPre  = np.eye(4, dtype=np.float32)

        state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        kf.statePost = state.copy()   # post-correction state (used by correct())
        kf.statePre  = state.copy()   # pre-correction state  (used by predict())
        return kf


    def predict(self, track_id: str) -> Optional[Tuple[float, float]]:
        with self._lock:
            kf = self._filters.get(track_id)
        if kf is None:
            return None
        pred = kf.predict()   # shape (4, 1): [[x], [y], [vx], [vy]]
        return (float(pred[0][0]), float(pred[1][0]))

    def update(self, track_id: str, x: float, y: float) -> None:
        with self._lock:
            if track_id not in self._filters:
                self._filters[track_id] = self._make_filter(x, y)
            kf = self._filters[track_id]
        measurement = np.array([[x], [y]], dtype=np.float32)
        kf.correct(measurement)

    def remove(self, track_id: str) -> None:
        with self._lock:
            self._filters.pop(track_id, None)

    def active_count(self) -> int:
        with self._lock:
            return len(self._filters)


# ─────────────────────────────────────────────────────────────────────────────
#  5. SensorFusionLayer
# ─────────────────────────────────────────────────────────────────────────────

class SensorFusionLayer:
    """
    Orchestrates SpatialAligner, AppearanceEmbedder, EmbeddingGallery,
    and KalmanMotionPredictor to produce stable cross-frame, cross-camera
    bag identities (FusedDetection) fed into AirportLuggageTracker.

    Thread-safe: gallery and Kalman filters are guarded internally.
    The fuse() method itself is NOT re-entrant (one call per camera per frame).
    """

    def __init__(
        self,
        homography_config_path: Optional[str] = None,
        fusion_config: Optional[FusionConfig] = None,
    ) -> None:
        self._cfg = fusion_config or FusionConfig()
        self._aligner = SpatialAligner(homography_config_path)
        self._embedder = AppearanceEmbedder(self._cfg.embedding_dim)
        self._gallery = EmbeddingGallery(self._cfg)
        self._kalman = KalmanMotionPredictor(self._cfg)
        self._frame_counter = 0
        self._cumulative_metrics = FusionMetrics()

    # ── Public API ─────────────────────────────────────────────────────────

    def fuse(
        self,
        raw_detections: List[Dict[str, Any]],
        rgb_frames: Dict[str, np.ndarray],
        depth_frames: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[FusedDetection]:
        """
        Main entry point.

        Parameters
        ----------
        raw_detections
            List of object detection dicts from pipeline._parse_yolo_result().
            Expected keys: track_id, class_id, class_name, bbox, center,
                           confidence, camera_id (if present).
        rgb_frames
            Dict mapping camera_id → full BGR frame for crop extraction.
        depth_frames
            Optional dict mapping camera_id → depth map (float32, metres).
        """
        self._frame_counter += 1
        metrics = FusionMetrics(frame_id=self._frame_counter)
        reid_scores: List[float] = []
        results: List[FusedDetection] = []

        for det in raw_detections:
            camera_id = det.get("camera_id", "default")
            cx, cy = det.get("center", (0.0, 0.0))
            track_id_raw = det.get("track_id")
            x1, y1, x2, y2 = det.get("bbox", (0, 0, 0, 0))

            # ── 1. World alignment ────────────────────────────────────────
            world_xy = self._aligner.align(float(cx), float(cy), camera_id)

            # ── 2. Embedding ──────────────────────────────────────────────
            embedding: Optional[np.ndarray] = None
            frame = rgb_frames.get(camera_id)
            if frame is not None:
                crop = self._safe_crop(frame, x1, y1, x2, y2)
                embedding = self._embedder.embed(crop)

            # ── 3. Kalman prediction ──────────────────────────────────────
            local_key = f"{camera_id}_{track_id_raw}" if track_id_raw is not None else None
            predicted_xy = self._kalman.predict(local_key) if local_key else None

            # Use predicted position for spatial filter if available,
            # else fall back to current world_xy
            filter_center = predicted_xy if predicted_xy else world_xy

            # ── 4. Gallery query ──────────────────────────────────────────
            global_track_id: Optional[str] = None
            reid_score = 0.0
            is_new = True

            if embedding is not None:
                matched_id, score = self._gallery.query(
                    embedding,
                    filter_center,
                    self._cfg.spatial_filter_radius,
                )
                if matched_id is None:
                    # Fallback: try without spatial filter (cross-camera scenario)
                    matched_id, score = self._gallery.query_unconstrained(embedding)
                    if matched_id:
                        metrics.n_spatial_filtered += 1

                if matched_id:
                    global_track_id = matched_id
                    reid_score = score
                    is_new = False
                    metrics.n_reidentified += 1
                    reid_scores.append(score)

            # ── 5. Mint new ID if no match ────────────────────────────────
            if global_track_id is None:
                # Prefer stable local ByteTrack ID as seed for display,
                # but always use UUID for true cross-camera uniqueness
                short = uuid.uuid4().hex[:8].upper()
                bag_class = det.get("class_name", "bag")[:3].upper()
                global_track_id = f"BAG-{bag_class}-{short}"
                is_new = True
                metrics.n_new_tracks += 1

            # ── 6. Update gallery + Kalman ────────────────────────────────
            if embedding is not None:
                self._gallery.upsert(global_track_id, embedding, time.time(), world_xy)
            if local_key:
                self._kalman.update(local_key, world_xy[0], world_xy[1])

            # ── 7. Depth Z ────────────────────────────────────────────────
            depth_z: Optional[float] = None
            if depth_frames:
                depth_frame = depth_frames.get(camera_id)
                if depth_frame is not None:
                    depth_z = self._median_depth(depth_frame, x1, y1, x2, y2)

            # ── 8. Build FusedDetection ───────────────────────────────────
            fd = FusedDetection(
                raw=det,
                global_track_id=global_track_id,
                embedding=embedding,
                reid_score=reid_score,
                is_new_track=is_new,
                world_xy=world_xy,
                depth_z=depth_z,
            )
            results.append(fd)

        # ── Metrics ────────────────────────────────────────────────────────
        if reid_scores:
            metrics.mean_reid_score = sum(reid_scores) / len(reid_scores)
        if self._frame_counter % self._cfg.log_interval_frames == 0:
            metrics.log()

        # Sort by global_track_id for deterministic order
        results.sort(key=lambda fd: fd.global_track_id)
        return results

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _safe_crop(
        frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1c = max(0, int(x1))
        y1c = max(0, int(y1))
        x2c = min(w, int(x2))
        y2c = min(h, int(y2))
        if x2c <= x1c or y2c <= y1c:
            return None
        return frame[y1c:y2c, x1c:x2c]

    @staticmethod
    def _median_depth(
        depth_frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Optional[float]:
        h, w = depth_frame.shape[:2]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        r = max(1, min((x2 - x1), (y2 - y1)) // 4)
        dx1, dy1 = max(0, cx - r), max(0, cy - r)
        dx2, dy2 = min(w, cx + r), min(h, cy + r)
        patch = depth_frame[dy1:dy2, dx1:dx2]
        if patch.size == 0:
            return None
        val = float(np.median(patch))
        return val if val > 0 else None

    def gallery_size(self) -> int:
        return self._gallery.size()

    def kalman_active(self) -> int:
        return self._kalman.active_count()


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton — built lazily so it doesn't block import
# ─────────────────────────────────────────────────────────────────────────────

_fusion_layer: Optional[SensorFusionLayer] = None
_fusion_lock = threading.Lock()


def get_fusion_layer() -> SensorFusionLayer:
    global _fusion_layer
    if _fusion_layer is None:
        with _fusion_lock:
            if _fusion_layer is None:
                from config import config
                homography_path = os.path.join(
                    os.path.dirname(__file__), "..", "data", "homographies.yaml"
                )
                fc = FusionConfig(
                    similarity_threshold=getattr(config, "fusion_similarity_threshold", 0.82),
                    gallery_ttl_s=getattr(config, "fusion_gallery_ttl_s", 300.0),
                    spatial_filter_radius=getattr(config, "fusion_spatial_filter_radius", 150.0),
                    kalman_noise_cov=getattr(config, "fusion_kalman_noise_cov", 1e-4),
                    embedding_dim=getattr(config, "fusion_embedding_dim", 512),
                )
                _fusion_layer = SensorFusionLayer(
                    homography_config_path=homography_path,
                    fusion_config=fc,
                )
                print("[SensorFusion] Initialised (gallery TTL=%.0fs, threshold=%.2f)" %
                      (fc.gallery_ttl_s, fc.similarity_threshold))
    return _fusion_layer
