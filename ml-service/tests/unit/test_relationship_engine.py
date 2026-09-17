"""Unit tests for relationship_engine.py — pure logic, no ML dependencies.
Covers both the original pixel-only behavior (unchanged) and the new
depth-aware demotion logic.
"""
import numpy as np

from app.relationships.relationship_engine import infer_relationship
from app.schemas.inference import DetectedObject

PERSON = DetectedObject(label="person", confidence=0.9, bbox=(0, 0, 50, 200))
KNIFE_CLOSE = DetectedObject(label="knife", confidence=0.9, bbox=(55, 0, 95, 40))  # pixel-near person


def test_no_person_returns_none():
    knife = DetectedObject(label="knife", confidence=0.9, bbox=(0, 0, 40, 40))
    assert infer_relationship([knife]) is None


def test_no_hazard_objects_returns_none():
    assert infer_relationship([PERSON]) is None


def test_pixel_only_near_when_no_depth_map():
    """Backward-compatible path: no depth_map supplied, behaves exactly as
    the original Phase 1 pixel-only heuristic did."""
    relationship = infer_relationship([PERSON, KNIFE_CLOSE])
    assert relationship in ("holding", "near")


def test_depth_agreement_keeps_relationship():
    """Person and knife at the SAME depth (both foreground) — pixel-near
    AND depth-near, so the relationship should NOT be demoted."""
    depth_map = np.zeros((200, 100), dtype=np.float32)
    depth_map[:, :] = 0.9  # uniformly close/foreground everywhere

    relationship = infer_relationship([PERSON, KNIFE_CLOSE], depth_map=depth_map)
    assert relationship in ("holding", "near")


def test_depth_mismatch_demotes_relationship():
    """Person in the foreground, knife's bbox region actually in the
    background (low depth value = far) despite being pixel-close. This is
    the perspective-illusion case — should get demoted."""
    depth_map = np.zeros((200, 100), dtype=np.float32)
    depth_map[:, :] = 0.9  # default: everything close
    # Knife's bbox region (x=55-95, y=0-40) is actually far away
    depth_map[0:40, 55:95] = 0.05

    pixel_only = infer_relationship([PERSON, KNIFE_CLOSE], depth_map=None)
    depth_aware = infer_relationship([PERSON, KNIFE_CLOSE], depth_map=depth_map)

    assert pixel_only in ("holding", "near")
    assert depth_aware != pixel_only, (
        "Large depth gap despite pixel closeness should demote the "
        "relationship rather than trusting the pixel-only read."
    )


def test_reachable_is_never_promoted_by_depth():
    """Depth logic only DEMOTES holding/near — it should never touch an
    already-distant 'reachable' or None classification."""
    far_knife = DetectedObject(label="knife", confidence=0.9, bbox=(400, 0, 440, 40))
    depth_map = np.zeros((200, 300), dtype=np.float32)
    depth_map[:, :] = 0.5

    relationship = infer_relationship([PERSON, far_knife], depth_map=depth_map)
    # far_knife is > NEAR_DISTANCE_PX*2 away, so should stay None regardless of depth
    assert relationship is None


def test_empty_depth_map_falls_back_gracefully():
    """A zero-size depth map (e.g. estimation failed and returned an empty
    array) shouldn't crash — should behave like no depth_map at all."""
    empty_depth = np.zeros((0, 0), dtype=np.float32)
    relationship = infer_relationship([PERSON, KNIFE_CLOSE], depth_map=empty_depth)
    assert relationship in ("holding", "near")