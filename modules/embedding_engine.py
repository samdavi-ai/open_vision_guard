"""
Embedding Engine — OSNet-AIN x1.0 with multi-embedding gallery.

Upgrades over the previous version:
  1. Model: osnet_ain_x1_0  (AIN = Adaptive Instance Normalisation)
     — strips lighting / colour-style bias so the same person matches
       even after removing or adding a bag/laptop.
  2. Multi-embedding gallery (up to reid_max_gallery_size per identity)
     — stores multiple appearance snapshots per person and matches
       against ALL of them (max-similarity voting). Survives
       significant appearance changes between visits.
  3. Near-duplicate suppression — only adds a new gallery embedding
     when it is sufficiently different from existing ones (>0.97 sim
     would be a duplicate, skip it).
  4. Lowered default similarity threshold to 0.72 (configured in
     config.py) — more lenient to cope with bag-on / bag-off changes
     while still discriminating between different people.
"""

import datetime
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import faiss

from config import config


class EmbeddingEngine:
    """
    OSNet-AIN-based person Re-ID engine with per-identity appearance gallery.
    Thread-safe via an internal RLock (multiple reads + one write at a time).
    """

    def __init__(self) -> None:
        # ── Device selection (M4 Mac / CUDA / CPU) ───────────────────────────
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"[EmbeddingEngine] Device: {self.device}")

        self.model = self._load_model()
        self.feature_dim = 512          # OSNet output dimension

        # Image pre-processing pipeline (standard Re-ID input size 256×128)
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # ── Multi-embedding gallery ───────────────────────────────────────────
        # global_id  →  list of L2-normalised float32 embeddings (ndim=512)
        self._gallery: Dict[str, List[np.ndarray]] = {}
        self._gallery_lock = threading.RLock()

        # ── Legacy metadata store (for backward compat with identity router) ─
        self.global_to_metadata: Dict[str, Dict[str, Any]] = {}
        self._faiss_id_counter: int = 0   # used only for new-ID minting

        # ── FAISS index — still kept for fast bulk search ─────────────────────
        self.index = faiss.IndexFlatIP(self.feature_dim)  # inner-product on L2-normed vecs = cosine
        self.id_to_global: Dict[int, str] = {}

        # Track last gallery-update time per identity to rate-limit adds
        self._last_update_ts: Dict[str, float] = {}

        print("[EmbeddingEngine] Ready (OSNet-AIN x1.0 + multi-gallery).")

    # ──────────────────────────────────────────────────────────────────────────
    #  Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        """Try several loading strategies, fall back gracefully."""
        # Strategy 1: installed torchreid pip package (pip install torchreid)
        # Import path: torchreid.reid.models.osnet_ain
        try:
            from torchreid.reid.models.osnet_ain import osnet_ain_x1_0
            model = osnet_ain_x1_0(pretrained=True)
            model.to(self.device).eval()
            print("[EmbeddingEngine] Loaded osnet_ain_x1_0 from torchreid pip package.")
            return model
        except Exception as e:
            print(f"[EmbeddingEngine] torchreid.reid.models strategy failed: {e}")

        # Strategy 2: KaiyangZhou torch.hub (clones repo into ~/.cache/torch/hub)
        try:
            import sys, os
            hub_dir = os.path.expanduser(
                "~/.cache/torch/hub/KaiyangZhou_deep-person-reid_master"
            )
            if hub_dir not in sys.path:
                sys.path.insert(0, hub_dir)
            from torchreid.models.osnet_ain import osnet_ain_x1_0  # hub repo path
            model = osnet_ain_x1_0(pretrained=True)
            model.to(self.device).eval()
            print("[EmbeddingEngine] Loaded osnet_ain_x1_0 from torch.hub cache dir.")
            return model
        except Exception as e:
            print(f"[EmbeddingEngine] torch.hub cache strategy failed: {e}")

        # Strategy 3: torch.hub.load (downloads fresh clone if needed)
        try:
            model = torch.hub.load(
                "KaiyangZhou/deep-person-reid",
                "osnet_ain_x1_0",
                pretrained=True,
                trust_repo=True,
            )
            model.to(self.device).eval()
            print("[EmbeddingEngine] Loaded osnet_ain_x1_0 via torch.hub.load.")
            return model
        except Exception as e:
            print(f"[EmbeddingEngine] torch.hub.load strategy failed: {e}")

        # Strategy 4: fall back to plain osnet_x1_0 — still better than nothing
        try:
            from torchreid.reid.models.osnet import osnet_x1_0
            model = osnet_x1_0(pretrained=True)
            model.to(self.device).eval()
            print("[EmbeddingEngine] WARNING: Using osnet_x1_0 fallback (no AIN).")
            return model
        except Exception as e:
            print(f"[EmbeddingEngine] All strategies failed: {e}")

        print("[EmbeddingEngine] CRITICAL: No Re-ID model — embeddings will be random.")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    #  Embedding generation
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_embedding(self, cv2_image: np.ndarray) -> np.ndarray:
        """
        Returns an L2-normalised float32 embedding of shape (1, 512) from a BGR crop.
        Falls back to random noise if the model is unavailable.
        """
        if self.model is None or cv2_image is None or cv2_image.size == 0:
            return np.random.rand(1, self.feature_dim).astype(np.float32)

        try:
            rgb = cv2_image[:, :, ::-1]          # BGR → RGB
            pil = Image.fromarray(rgb.astype(np.uint8))
            tensor = self.transform(pil).unsqueeze(0).to(self.device)

            features = self.model(tensor)         # (1, 512)
            norm = torch.norm(features, p=2, dim=1, keepdim=True)
            emb = (features / (norm + 1e-6)).cpu().numpy().astype(np.float32)
            return emb                            # shape: (1, 512)
        except Exception as e:
            print(f"[EmbeddingEngine] generate_embedding error: {e}")
            return np.random.rand(1, self.feature_dim).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    #  Gallery management
    # ──────────────────────────────────────────────────────────────────────────

    def _add_to_gallery(self, global_id: str, embedding: np.ndarray, now: float) -> None:
        """
        Add an embedding to this identity's gallery.

        Rules:
          - Skip if last update was <reid_min_update_interval_s ago (rate-limit).
          - Skip if the new embedding is near-identical to any existing one (sim > 0.97).
          - Evict the oldest entry (FIFO) when gallery exceeds max size.
        """
        import time
        min_interval = getattr(config, "reid_min_update_interval_s", 3.0)
        if now - self._last_update_ts.get(global_id, 0.0) < min_interval:
            return

        flat = embedding.flatten()

        with self._gallery_lock:
            gallery = self._gallery.setdefault(global_id, [])
            # Near-duplicate check
            for existing in gallery:
                sim = float(np.dot(flat, existing.flatten()))
                if sim > 0.97:
                    return                        # Already have a very similar embedding

            max_size = getattr(config, "reid_max_gallery_size", 8)
            if len(gallery) >= max_size:
                gallery.pop(0)                    # FIFO eviction
            gallery.append(flat.copy())

        self._last_update_ts[global_id] = now

    def _match_gallery(self, query: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match query embedding against every gallery entry of every known identity.
        Returns (best_global_id, best_similarity).

        Uses max-similarity voting: the per-identity score is the HIGHEST
        similarity among all of that identity's gallery embeddings.
        This is specifically designed to handle appearance changes (bag removed etc.)
        because at least ONE gallery entry should still match well.
        """
        flat = query.flatten()
        best_id: Optional[str] = None
        best_score: float = 0.0

        with self._gallery_lock:
            for gid, embeddings in self._gallery.items():
                for emb in embeddings:
                    sim = float(np.dot(flat, emb))   # cosine (both L2-normed)
                    if sim > best_score:
                        best_id, best_score = gid, sim

        return best_id, best_score

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API (backward compatible with pipeline.py)
    # ──────────────────────────────────────────────────────────────────────────

    def get_or_create_identity(self, crop_img: np.ndarray) -> str:
        """
        Match a crop against the gallery, or mint a new identity.
        This is the main Re-ID entry point called from the async AI worker.
        """
        import time as _time
        now = _time.time()

        new_embedding = self.generate_embedding(crop_img)
        threshold = getattr(config, "similarity_threshold", 0.72)

        best_id, best_score = self._match_gallery(new_embedding)

        if best_id is not None and best_score >= threshold:
            # Matched existing identity — update gallery with new appearance
            self._add_to_gallery(best_id, new_embedding, now)
            if best_id in self.global_to_metadata:
                self.global_to_metadata[best_id]["last_seen_time"] = (
                    datetime.datetime.now().astimezone().isoformat()
                )
                self.global_to_metadata[best_id]["total_appearances"] = (
                    self.global_to_metadata[best_id].get("total_appearances", 0) + 1
                )
            return best_id

        # New identity
        new_global_id = f"Person_OSN_{self._faiss_id_counter:03d}"
        self._faiss_id_counter += 1

        # Seed gallery with first embedding
        with self._gallery_lock:
            self._gallery[new_global_id] = [new_embedding.flatten().copy()]
        self._last_update_ts[new_global_id] = now

        # Also add to FAISS for bulk search (search_similar)
        self.index.add(new_embedding)
        self.id_to_global[self._faiss_id_counter - 1] = new_global_id

        now_iso = datetime.datetime.now().astimezone().isoformat()
        self.global_to_metadata[new_global_id] = {
            "global_id": new_global_id,
            "face_name": None,
            "activity": "unknown",
            "risk_level": "low",
            "clothing_color": "unknown",
            "last_seen_camera": "unknown",
            "last_seen_time": now_iso,
            "movement_direction": "stationary",
            "speed": 0.0,
            "pose_detail": "unknown",
            "entry_time": now_iso,
            "exit_time": now_iso,
            "carried_objects": [],
            "object_log": [],
            "zone_history": [],
            "latitude": None,
            "longitude": None,
            "total_appearances": 1,
        }
        return new_global_id

    def update_identity_metadata(self, global_id: str, updates: Dict[str, Any]) -> None:
        if global_id in self.global_to_metadata:
            self.global_to_metadata[global_id].update(updates)

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        if self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        D, I = self.index.search(query_embedding, k)
        return [
            self.id_to_global[int(I[0][i])]
            for i in range(k)
            if int(I[0][i]) in self.id_to_global
        ]

    def get_all_identities(self) -> List[Dict[str, Any]]:
        return [
            {"global_id": gid, "metadata": meta}
            for gid, meta in self.global_to_metadata.items()
        ]

    def get_identity(self, global_id: str) -> Optional[Dict[str, Any]]:
        if global_id in self.global_to_metadata:
            return {"global_id": global_id, "metadata": self.global_to_metadata[global_id]}
        return None


# Singleton instance
embedding_engine = EmbeddingEngine()
