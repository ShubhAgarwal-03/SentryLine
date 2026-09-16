"""Spatial relationship logic — near / holding / reachable, from detected
object geometry. PRD §4 level 2/3, simplified for Phase 1 to 2D bbox math
(no depth model yet — see perception/depth.py).
"""
from app.schemas.inference import DetectedObject

NEAR_DISTANCE_PX = 150  # heuristic pixel threshold; tune once real footage is available


def _bbox_center(obj: DetectedObject) -> tuple[float, float]:
    x1, y1, x2, y2 = obj.bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def infer_relationship(objects: list[DetectedObject]) -> str | None:
    """Returns a single dominant relationship label for the scene.

    Phase 1 heuristic: find the closest (hazard-object, person) pair and
    classify by pixel distance. This is intentionally simple and
    explainable — it's swapped for pose-informed relationships (holding,
    reaching toward) once perception/pose.py is wired in.
    """
    people = [o for o in objects if o.label == "person"]
    others = [o for o in objects if o.label != "person"]

    if not people or not others:
        return None

    closest_distance = None
    for person in people:
        for other in others:
            distance = _euclidean(_bbox_center(person), _bbox_center(other))
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance

    if closest_distance is None:
        return None
    if closest_distance < NEAR_DISTANCE_PX * 0.4:
        return "holding"
    if closest_distance < NEAR_DISTANCE_PX:
        return "near"
    if closest_distance < NEAR_DISTANCE_PX * 2:
        return "reachable"
    return None