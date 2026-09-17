"""YOLO wrapper — environment-agnostic object detection, trained once.

Per PRD §3/§9: perception never changes across deployment sites. Only the
config (site_config) and the reasoning layer are environment-aware.
"""
from functools import lru_cache

import numpy as np
from ultralytics import YOLO

from app.schemas.inference import DetectedObject

# Pretrained YOLO checkpoint for Phase 1. Fine-tuning on a small hazard
# object set (knife, spill, ladder, etc. — PRD §14) happens once a labeled
# dataset is assembled; the wrapper interface below doesn't need to change
# when that swap happens, only the weights path.
MODEL_PATH = "models/yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.35


@lru_cache(maxsize=1)
def _load_model() -> YOLO:
    return YOLO(MODEL_PATH)


def detect_objects(image: np.ndarray) -> list[DetectedObject]:
    """Run detection on a single BGR image (as loaded by cv2.imread)."""
    model = _load_model()
    results = model.predict(image, verbose=False, conf=CONFIDENCE_THRESHOLD)

    detections: list[DetectedObject] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            label = names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detections.append(
                DetectedObject(label=label, confidence=confidence, bbox=(x1, y1, x2, y2))
            )
    return detections