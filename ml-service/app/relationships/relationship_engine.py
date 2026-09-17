"""Spatial relationship logic — near / holding / reachable, from detected
object geometry. PRD §4 level 2/3.

Phase 1 baseline: pure 2D bbox pixel-distance (no depth model).
Phase 1.5 (this update): optionally depth-aware. If a relative depth map is
supplied, it's used to catch perspective illusions — two objects can be
close in pixel space but far apart in actual depth (e.g. a hazard object
in the background that only *looks* close to a foreground person). When
that happens, the relationship gets demoted a level rather than trusted at
face value.

IMPORTANT: depth here is RELATIVE, uncalibrated (see perception/depth.py).
This improves *relative* near/reachable judgments — it does NOT provide
real-world metric distance (e.g. "2 meters from the restricted zone").
That needs camera calibration and is separate, later work.
"""
import numpy as np

from app.schemas.inference import DetectedObject

NEAR_DISTANCE_PX = 150  # heuristic pixel threshold; tune once real footage is available

# Normalized depth difference (0-1 range, relative to this image's own depth
# spread) above which two pixel-close objects are considered NOT actually
# near each other — i.e. a perspective illusion. Heuristic; tune once real
# footage with known ground-truth proximity is available.
DEPTH_MISMATCH_THRESHOLD = 0.25

_DEMOTION = {"holding": "near", "near": "reachable", "reachable": None}


def _bbox_center(obj: DetectedObject) -> tuple[float, float]:
    x1, y1, x2, y2 = obj.bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _classify_by_pixel_distance(distance: float) -> str | None:
    if distance < NEAR_DISTANCE_PX * 0.4:
        return "holding"
    if distance < NEAR_DISTANCE_PX:
        return "near"
    if distance < NEAR_DISTANCE_PX * 2:
        return "reachable"
    return None


def _depth_at_bbox(depth_map: np.ndarray, bbox: tuple[float, float, float, float]) -> float | None:
    """Median depth within a bbox — more robust to noise than sampling a
    single center pixel."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, depth_map.shape[1]), min(y2, depth_map.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_map[y1:y2, x1:x2]
    return float(np.median(patch)) if patch.size else None


def infer_relationship(
    objects: list[DetectedObject],
    depth_map: np.ndarray | None = None,
) -> str | None:
    """Returns a single dominant relationship label for the scene.

    Finds the closest (hazard-object, person) pair by 2D pixel distance,
    classifies by distance band, then — if a depth map is provided — checks
    whether the pair is actually close in depth too. If they're pixel-close
    but depth-far, the relationship is demoted a level (a large depth gap
    contradicts "near"/"holding", so we don't trust the pixel-only read).
    """
    people = [o for o in objects if o.label == "person"]
    others = [o for o in objects if o.label != "person"]

    if not people or not others:
        return None

    closest_distance = None
    closest_pair: tuple[DetectedObject, DetectedObject] | None = None
    for person in people:
        for other in others:
            distance = _euclidean(_bbox_center(person), _bbox_center(other))
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                closest_pair = (person, other)

    if closest_distance is None or closest_pair is None:
        return None

    relationship = _classify_by_pixel_distance(closest_distance)

    if depth_map is not None and relationship in ("holding", "near") and depth_map.size:
        person, other = closest_pair
        person_depth = _depth_at_bbox(depth_map, person.bbox)
        other_depth = _depth_at_bbox(depth_map, other.bbox)
        depth_range = float(depth_map.max() - depth_map.min())

        if person_depth is not None and other_depth is not None and depth_range > 1e-6:
            normalized_gap = abs(person_depth - other_depth) / depth_range
            if normalized_gap > DEPTH_MISMATCH_THRESHOLD:
                relationship = _DEMOTION[relationship]

    return relationship