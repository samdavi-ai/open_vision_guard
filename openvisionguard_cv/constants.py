"""Shared class mappings for the OpenVisionGuard detector."""

from __future__ import annotations

SURVEILLANCE_CLASSES = {
    0: "person",
    1: "backpack",
    2: "handbag",
    3: "suitcase",
}

COCO_TO_SURVEILLANCE = {
    0: 0,
    24: 1,
    26: 2,
    28: 3,
}

SURVEILLANCE_TO_COCO = {
    0: 0,
    1: 24,
    2: 26,
    3: 28,
}

BAG_CLASS_IDS = {1, 2, 3}
