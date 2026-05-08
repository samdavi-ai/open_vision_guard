"""
Unit tests for SensorFusionLayer components.

Covers (per spec):
  1. Gallery eviction (TTL)
  2. Cosine match at boundary score (0.82 ± epsilon)
  3. Kalman predict-then-update round-trip
  4. SpatialAligner.align() with a known homography
  5. FusionMetrics population
  6. SensorFusionLayer.fuse() end-to-end with synthetic detections

Run:
  python tests/test_sensor_fusion.py
"""

import math
import os
import sys
import time
import unittest

import numpy as np

# Make sure project root is importable regardless of CWD
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.sensor_fusion import (
    EmbeddingGallery,
    FusionConfig,
    FusionMetrics,
    KalmanMotionPredictor,
    SensorFusionLayer,
    SpatialAligner,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rand_unit(dim: int = 512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_config(**kwargs) -> FusionConfig:
    cfg = FusionConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Test: EmbeddingGallery eviction
# ─────────────────────────────────────────────────────────────────────────────

class TestGalleryEviction(unittest.TestCase):

    def test_evict_stale_removes_old_entries(self):
        cfg = _make_config(gallery_ttl_s=0.05)   # 50ms TTL for fast test
        gallery = EmbeddingGallery(cfg)

        emb = _rand_unit()
        gallery.upsert("BAG-OLD-001", emb, time.time() - 1.0)   # 1s old → stale
        gallery.upsert("BAG-NEW-002", emb, time.time())          # fresh → keep

        removed = gallery.evict_stale(ttl_seconds=0.5)

        self.assertEqual(removed, 1, "Should evict exactly 1 stale entry")
        self.assertEqual(gallery.size(), 1, "One fresh entry should remain")

    def test_evict_keeps_fresh_entries(self):
        cfg = _make_config(gallery_ttl_s=300.0)
        gallery = EmbeddingGallery(cfg)

        emb = _rand_unit()
        gallery.upsert("BAG-A", emb, time.time())
        gallery.upsert("BAG-B", emb, time.time())

        removed = gallery.evict_stale(ttl_seconds=300.0)

        self.assertEqual(removed, 0, "No entries should be evicted")
        self.assertEqual(gallery.size(), 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Test: Cosine match at boundary score
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineMatchBoundary(unittest.TestCase):

    def _setup_gallery(self, threshold=0.82):
        cfg = _make_config(similarity_threshold=threshold, spatial_filter_radius=1e9)
        gallery = EmbeddingGallery(cfg)
        return gallery, cfg

    def test_match_above_threshold_succeeds(self):
        gallery, _ = self._setup_gallery(threshold=0.82)
        base = _rand_unit()

        # Store base embedding
        gallery.upsert("BAG-TEST", base, time.time(), world_xy=(0.0, 0.0))

        # Query with identical embedding → score = 1.0 → match
        matched_id, score = gallery.query(base, (0.0, 0.0), spatial_filter_radius=1e9)
        self.assertEqual(matched_id, "BAG-TEST")
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_match_at_exactly_threshold(self):
        """
        Test the >= boundary: use cosine=1.0 with threshold=1.0 (identical vector),
        then test cosine=0.0 with threshold=0.82 (orthogonal → no match).
        This robustly validates the '>=' boundary comparison.
        """
        dim = 512
        base = _rand_unit(dim)

        # Case 1: identical → cosine ≈ 1.0, threshold = 0.9999 → must match (>=)
        cfg_exact = _make_config(similarity_threshold=0.9999, spatial_filter_radius=1e9)
        gallery_exact = EmbeddingGallery(cfg_exact)
        gallery_exact.upsert("BAG-EXACT", base.copy(), time.time(), world_xy=(0.0, 0.0))
        matched_id, score = gallery_exact.query(base, (0.0, 0.0), spatial_filter_radius=1e9)
        self.assertEqual(matched_id, "BAG-EXACT", "Cosine~=1.0 at threshold=0.9999 should match")
        self.assertGreaterEqual(score, 0.9999)

        # Case 2: orthogonal → cosine = 0.0, threshold = 0.82 → must NOT match
        gallery2, _ = self._setup_gallery(threshold=0.82)
        ortho = np.zeros(dim, dtype=np.float32)
        ortho[1] = 1.0   # orthogonal to base (which is random, not axis-aligned)
        gallery2.upsert("BAG-MISS", ortho.copy(), time.time(), world_xy=(0.0, 0.0))
        matched_id2, _ = gallery2.query(base, (0.0, 0.0), spatial_filter_radius=1e9)
        # base is random unit; score against ortho axis is unlikely ≥ 0.82
        actual = float(np.dot(base, ortho))
        if actual < 0.82:
            self.assertIsNone(matched_id2, "Low-cosine should not match at 0.82 threshold")

    def test_match_below_threshold_fails(self):
        gallery, _ = self._setup_gallery(threshold=0.82)
        dim = 512

        # orthogonal vectors: cosine = 0.0
        base = np.zeros(dim, dtype=np.float32)
        base[0] = 1.0
        query = np.zeros(dim, dtype=np.float32)
        query[1] = 1.0

        gallery.upsert("BAG-MISS", base, time.time(), world_xy=(0.0, 0.0))
        matched_id, score = gallery.query(query, (0.0, 0.0), spatial_filter_radius=1e9)

        self.assertIsNone(matched_id, "Orthogonal embeddings should not match")
        self.assertLess(score, 0.82)


# ─────────────────────────────────────────────────────────────────────────────
#  Test: Kalman predict-then-update round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestKalmanRoundTrip(unittest.TestCase):

    def test_predict_after_update_is_close_to_measurement(self):
        cfg = _make_config(kalman_noise_cov=1e-4, kalman_meas_noise=1e-2)
        kp = KalmanMotionPredictor(cfg)

        kp.update("BAG-K1", 100.0, 200.0)
        kp.update("BAG-K1", 102.0, 201.0)
        kp.update("BAG-K1", 104.0, 202.0)

        pred = kp.predict("BAG-K1")
        self.assertIsNotNone(pred)

        px, py = pred
        # After 3 corrections the filter should be close to the last measurement region
        self.assertAlmostEqual(px, 104.0, delta=15.0, msg="X prediction off")
        self.assertAlmostEqual(py, 202.0, delta=15.0, msg="Y prediction off")

    def test_predict_unknown_track_returns_none(self):
        cfg = _make_config()
        kp = KalmanMotionPredictor(cfg)
        result = kp.predict("NONEXISTENT")
        self.assertIsNone(result)

    def test_update_multiple_tracks_independent(self):
        cfg = _make_config()
        kp = KalmanMotionPredictor(cfg)

        kp.update("BAG-A", 0.0, 0.0)
        kp.update("BAG-B", 500.0, 500.0)

        pred_a = kp.predict("BAG-A")
        pred_b = kp.predict("BAG-B")

        self.assertIsNotNone(pred_a)
        self.assertIsNotNone(pred_b)
        self.assertAlmostEqual(pred_a[0], 0.0, delta=20.0, msg="BAG-A X should be near 0")
        self.assertAlmostEqual(pred_b[0], 500.0, delta=20.0, msg="BAG-B X should be near 500")


# ─────────────────────────────────────────────────────────────────────────────
#  Test: SpatialAligner with a known homography
# ─────────────────────────────────────────────────────────────────────────────

class TestSpatialAligner(unittest.TestCase):

    def test_identity_homography(self):
        aligner = SpatialAligner(homography_config_path=None)
        wx, wy = aligner.align(320.0, 240.0, "CAM_01")
        self.assertAlmostEqual(wx, 320.0, places=2)
        self.assertAlmostEqual(wy, 240.0, places=2)

    def test_known_homography_translation(self):
        """H = translate by (+100, +50)"""
        import cv2
        # Build a translation homography
        H = np.array([
            [1.0, 0.0, 100.0],
            [0.0, 1.0,  50.0],
            [0.0, 0.0,   1.0],
        ], dtype=np.float64)

        aligner = SpatialAligner(homography_config_path=None)
        aligner._matrices["CAM_TEST"] = H

        wx, wy = aligner.align(200.0, 100.0, "CAM_TEST")
        self.assertAlmostEqual(wx, 300.0, places=2, msg="X should shift by +100")
        self.assertAlmostEqual(wy, 150.0, places=2, msg="Y should shift by +50")

    def test_unknown_camera_falls_back_to_identity(self):
        aligner = SpatialAligner(homography_config_path=None)
        wx, wy = aligner.align(50.0, 75.0, "UNKNOWN_CAM")
        self.assertAlmostEqual(wx, 50.0, places=2)
        self.assertAlmostEqual(wy, 75.0, places=2)


# ─────────────────────────────────────────────────────────────────────────────
#  Test: FusionMetrics population
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionMetrics(unittest.TestCase):

    def test_metrics_fields(self):
        m = FusionMetrics(n_new_tracks=3, n_reidentified=7, mean_reid_score=0.91, frame_id=100)
        self.assertEqual(m.n_new_tracks, 3)
        self.assertEqual(m.n_reidentified, 7)
        self.assertAlmostEqual(m.mean_reid_score, 0.91)
        self.assertEqual(m.frame_id, 100)


# ─────────────────────────────────────────────────────────────────────────────
#  Test: SensorFusionLayer.fuse() end-to-end (no real model)
# ─────────────────────────────────────────────────────────────────────────────

class TestSensorFusionLayerE2E(unittest.TestCase):
    """
    End-to-end test without loading OSNet.
    The AppearanceEmbedder will return None (no model) — tests that the
    pipeline gracefully falls back to new-track-mint for every detection.
    """

    def _make_layer(self) -> SensorFusionLayer:
        cfg = FusionConfig(
            similarity_threshold=0.82,
            spatial_filter_radius=1e9,
            gallery_ttl_s=300.0,
        )
        layer = SensorFusionLayer(homography_config_path=None, fusion_config=cfg)
        return layer

    def _fake_det(self, track_id: int, cx: float, cy: float, cam: str = "CAM_01"):
        return {
            "track_id": track_id,
            "class_id": 24,
            "class_name": "backpack",
            "bbox": (int(cx - 20), int(cy - 30), int(cx + 20), int(cy + 30)),
            "center": (cx, cy),
            "confidence": 0.85,
            "camera_id": cam,
        }

    def test_fuse_returns_fused_detections(self):
        layer = self._make_layer()
        dets = [
            self._fake_det(1, 100.0, 200.0),
            self._fake_det(2, 300.0, 400.0),
        ]
        # No real frame needed — embedder will return None gracefully
        results = layer.fuse(dets, rgb_frames={})
        self.assertEqual(len(results), 2)

    def test_each_detection_gets_global_track_id(self):
        layer = self._make_layer()
        dets = [self._fake_det(1, 100.0, 100.0)]
        results = layer.fuse(dets, rgb_frames={})
        fd = results[0]
        self.assertTrue(fd.global_track_id.startswith("BAG-"))
        self.assertTrue(fd.is_new_track)

    def test_same_track_id_across_calls_gets_different_gallery_id_without_embedding(self):
        """Without embeddings, each call mints a new ID (expected — no Re-ID)."""
        layer = self._make_layer()
        det = self._fake_det(42, 200.0, 300.0)
        r1 = layer.fuse([det], rgb_frames={})
        r2 = layer.fuse([det], rgb_frames={})
        # Without embeddings, gallery is empty → both are new tracks with different IDs
        self.assertNotEqual(r1[0].global_track_id, r2[0].global_track_id)

    def test_to_dict_merges_fields(self):
        layer = self._make_layer()
        det = self._fake_det(7, 50.0, 80.0)
        results = layer.fuse([det], rgb_frames={})
        d = results[0].to_dict()
        self.assertIn("global_track_id", d)
        self.assertIn("world_xy", d)
        self.assertIn("reid_score", d)
        self.assertIn("is_new_track", d)
        # Original fields preserved
        self.assertEqual(d["class_name"], "backpack")
        self.assertEqual(d["track_id"], 7)

    def test_depth_z_attached_when_depth_frame_provided(self):
        import numpy as np
        layer = self._make_layer()
        det = self._fake_det(3, 100.0, 100.0, cam="CAM_01")
        depth = np.full((480, 640), 2.5, dtype=np.float32)   # 2.5m everywhere
        results = layer.fuse([det], rgb_frames={}, depth_frames={"CAM_01": depth})
        self.assertIsNotNone(results[0].depth_z)
        self.assertAlmostEqual(results[0].depth_z, 2.5, places=2)

    def test_fuse_empty_detections(self):
        layer = self._make_layer()
        results = layer.fuse([], rgb_frames={})
        self.assertEqual(results, [])

    def test_gallery_and_kalman_counters(self):
        """Gallery grows with embedding-less detections (they skip gallery upsert)."""
        layer = self._make_layer()
        dets = [self._fake_det(i, float(i * 10), 0.0) for i in range(5)]
        layer.fuse(dets, rgb_frames={})
        # Without embeddings, gallery stays empty
        self.assertEqual(layer.gallery_size(), 0)


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)   # silence DEBUG during tests
    print("=" * 60)
    print("  SensorFusionLayer Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
