"""MiDaS / Depth Anything wrapper — spatial understanding (PRD §4, level 2).

Stub for Phase 1: single-image reasoning in v0 doesn't strictly require
metric depth (relationship_engine.py uses 2D bbox geometry as a cheap proxy
for "near"/"reachable" in the MVP). This module is wired in once
reachability needs to account for real distance rather than pixel overlap.
"""
import numpy as np


def estimate_depth(image: np.ndarray) -> np.ndarray:
    """Returns a per-pixel relative depth map, same H/W as the input image.

    Not called yet in the Phase 1 pipeline — relationship_engine.py currently
    uses bounding-box distance only. Swap in a real MiDaS/Depth Anything
    forward pass here when reachability needs real-world distance.
    """
    raise NotImplementedError(
        "Depth estimation not wired into the Phase 1 pipeline yet — "
        "relationship_engine.py currently uses 2D bbox distance as a proxy."
    )