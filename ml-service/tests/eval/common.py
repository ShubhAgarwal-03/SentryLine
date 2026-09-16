"""Shared utilities for the evaluation scripts — manifest loading, image
loading, and IoU math used by the detection and system-level checks.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass
class GroundTruthObject:
    label: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class ManifestEntry:
    image_path: Path
    environment: str
    expected_objects: list[GroundTruthObject]
    expected_relationship: str | None
    should_alert: bool
    notes: str = ""


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    data = json.loads(manifest_path.read_text())
    entries = []
    for raw in data.get("entries", []):
        image_path = FIXTURES_DIR / raw["image"]
        if not image_path.exists():
            raise FileNotFoundError(
                f"Manifest references '{raw['image']}' but no file exists at {image_path}. "
                f"Add the image or remove the manifest entry."
            )
        entries.append(
            ManifestEntry(
                image_path=image_path,
                environment=raw["environment"],
                expected_objects=[
                    GroundTruthObject(label=o["label"], bbox=tuple(o["bbox"]))
                    for o in raw.get("expected_objects", [])
                ],
                expected_relationship=raw.get("expected_relationship"),
                should_alert=raw["should_alert"],
                notes=raw.get("notes", ""),
            )
        )
    return entries


def load_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image at {path}")
    return image


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0