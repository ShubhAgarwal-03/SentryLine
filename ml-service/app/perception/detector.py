"""YOLO wrapper — environment-agnostic object detection, trained once.

Per PRD §3/§9: perception never changes across deployment sites. Only the
config (site_config) and the reasoning layer are environment-aware.

Two-model design
----------------
The fine-tuned hazard checkpoint (e.g. gun_fire_v1.pt) was trained with
nc=14 but only has labels for the hazard classes that were actually merged
(gun, fire, ...). Its classification head was re-initialised, so it has
effectively FORGOTTEN person/knife/chair/etc. Swapping it in as the only
model would silently remove those detections, and the relationship engine
depends on `person`.

So we run both and split responsibility using configs/hazard_classes.yaml:
  - stock COCO model      -> classes with in_stock_yolo: true
  - fine-tuned hazard model -> classes with in_stock_yolo: false
Each model's output is filtered to its own classes, so they never conflict.

If the hazard weights file is missing (e.g. fresh clone — models/*.pt is
gitignored), we log a warning and fall back to the stock model alone.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from app.schemas.inference import DetectedObject

logger = logging.getLogger(__name__)

BASE_MODEL_PATH = os.environ.get("SENTRYLINE_BASE_MODEL", "models/yolov8n.pt")
HAZARD_MODEL_PATH = os.environ.get("SENTRYLINE_HAZARD_MODEL", "models/gun_fire_v1.pt")

# Separate thresholds so the hazard model (recall ~0.46 at last validation)
# can be tuned independently of the stock model using tests/eval.
BASE_CONFIDENCE_THRESHOLD = 0.35
HAZARD_CONFIDENCE_THRESHOLD = 0.35

HAZARD_CLASSES_PATH = Path(__file__).resolve().parents[2] / "configs" / "hazard_classes.yaml"


@lru_cache(maxsize=1)
def _class_split() -> tuple[frozenset[str], frozenset[str]]:
    """(labels the stock model owns, labels the hazard model owns)."""
    data = yaml.safe_load(HAZARD_CLASSES_PATH.read_text())
    stock = frozenset(c["label"] for c in data["classes"] if c.get("in_stock_yolo"))
    hazard = frozenset(c["label"] for c in data["classes"] if not c.get("in_stock_yolo"))
    return stock, hazard


@lru_cache(maxsize=1)
def _load_base_model() -> YOLO:
    return YOLO(BASE_MODEL_PATH)


@lru_cache(maxsize=1)
def _load_hazard_model() -> YOLO | None:
    if not Path(HAZARD_MODEL_PATH).exists():
        logger.warning(
            "Hazard model %s not found — running stock COCO model only "
            "(gun/fire/etc. will not be detected).", HAZARD_MODEL_PATH,
        )
        return None
    return YOLO(HAZARD_MODEL_PATH)


def _run(model: YOLO, image: np.ndarray, conf: float, keep: frozenset[str]) -> list[DetectedObject]:
    detections: list[DetectedObject] = []
    for result in model.predict(image, verbose=False, conf=conf):
        names = result.names
        for box in result.boxes:
            label = names[int(box.cls[0])]
            if label not in keep:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                DetectedObject(label=label, confidence=float(box.conf[0]), bbox=(x1, y1, x2, y2))
            )
    return detections


def detect_objects(image: np.ndarray) -> list[DetectedObject]:
    """Run detection on a single BGR image (as loaded by cv2.imread)."""
    stock_labels, hazard_labels = _class_split()

    detections = _run(_load_base_model(), image, BASE_CONFIDENCE_THRESHOLD, stock_labels)

    hazard_model = _load_hazard_model()
    if hazard_model is not None:
        detections += _run(hazard_model, image, HAZARD_CONFIDENCE_THRESHOLD, hazard_labels)

    return detections