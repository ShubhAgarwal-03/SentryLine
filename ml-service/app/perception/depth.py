"""Monocular relative depth estimation via Depth Anything V2 (Small).

IMPORTANT SCOPE NOTE: this produces *relative*, uncalibrated depth — "this
pixel is nearer/farther than that pixel" — not metric distance in meters.
True metric distance needs camera calibration (focal length, mount height,
etc.) which doesn't exist yet. What this DOES unlock: catching perspective
illusions in relationship_engine.py, where two objects overlap in 2D pixel
space but are actually at very different depths (e.g. a hazard object far
in the background that only *looks* close to a person in the foreground).
Real calibrated restricted-zone distance is a separate, later piece of work.
"""
from functools import lru_cache

import numpy as np
from PIL import Image
from transformers import pipeline

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


@lru_cache(maxsize=1)
def _load_pipeline():
    return pipeline(task="depth-estimation", model=MODEL_ID)


def estimate_depth(image_rgb: np.ndarray) -> np.ndarray:
    """Returns a relative depth map, same H/W as the input image.

    Convention: HIGHER value = CLOSER to the camera (matches Depth Anything's
    default output). Values are relative within a single image only — do not
    compare depth values across different images/frames.
    """
    pil_image = Image.fromarray(image_rgb)
    result = _load_pipeline()(pil_image)
    depth = np.array(result["depth"], dtype=np.float32)

    if depth.shape[:2] != image_rgb.shape[:2]:
        depth_img = Image.fromarray(depth).resize(
            (image_rgb.shape[1], image_rgb.shape[0]), Image.BILINEAR
        )
        depth = np.array(depth_img, dtype=np.float32)

    return depth