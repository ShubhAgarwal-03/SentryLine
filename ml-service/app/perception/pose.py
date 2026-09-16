"""Pose estimation — used for relationship cues like 'holding', 'reaching
toward', 'running'. Stub for Phase 1; relationship_engine.py currently
infers these from bbox proximity/overlap rather than joint keypoints.
Wire in a pose model (e.g. YOLO-pose, MediaPipe) here when finer-grained
relationships are needed.
"""
import numpy as np


def estimate_poses(image: np.ndarray) -> list[dict]:
    raise NotImplementedError("Pose estimation not wired into Phase 1 yet.")