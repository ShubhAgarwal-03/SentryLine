"""Unit tests for risk_scorer.py — pure logic, no ML dependencies, runs
without torch/ultralytics/open_clip installed. Locks in the formula's
behavior before anyone starts tuning weights.
"""
from app.reasoning.risk_scorer import compute_risk_score
from app.schemas.inference import DetectedObject, SiteConfig

PERSON = DetectedObject(label="person", confidence=0.9, bbox=(0, 0, 50, 200))
KNIFE_NEAR = DetectedObject(label="knife", confidence=0.9, bbox=(60, 0, 100, 40))  # ~50px from person center

HOSPITAL_CONFIG = SiteConfig(
    environment="hospital",
    restricted_roles=["visitor"],
    permitted_sharp_object_roles=["staff", "nurse"],
    restricted_zones=["operating_room"],
    notes="",
)

OFFICE_CONFIG = SiteConfig(
    environment="office",
    restricted_roles=["visitor", "employee"],
    permitted_sharp_object_roles=[],
    notes="",
)


def test_no_hazard_objects_scores_zero():
    result = compute_risk_score([PERSON], relationship=None, site_config=OFFICE_CONFIG)
    assert result.score == 0
    assert result.context_mismatch is False


def test_same_objects_score_differently_by_site():
    """The PRD's headline demo: identical detections, different verdicts,
    purely from site_config differing."""
    hospital_result = compute_risk_score(
        [PERSON, KNIFE_NEAR], relationship="near", site_config=HOSPITAL_CONFIG
    )
    office_result = compute_risk_score(
        [PERSON, KNIFE_NEAR], relationship="near", site_config=OFFICE_CONFIG
    )
    assert hospital_result.score < office_result.score, (
        "Hospital (permits sharp objects for staff) should score the same "
        "detection lower than office (permits none)."
    )


def test_permitted_sharp_objects_actually_apply_discount():
    """Regression test for the role/object vocabulary bug: previously this
    intersected object labels ('knife') against role names ('staff'),
    which could never match, so the discount silently never applied.
    """
    result = compute_risk_score(
        [PERSON, KNIFE_NEAR], relationship="near", site_config=HOSPITAL_CONFIG
    )
    assert result.environment_factor == 0.5, (
        "Hospital config lists permitted roles, so the 0.5 discount should apply."
    )


def test_no_permitted_roles_gets_no_discount():
    result = compute_risk_score(
        [PERSON, KNIFE_NEAR], relationship="near", site_config=OFFICE_CONFIG
    )
    assert result.environment_factor == 1.0


def test_relationship_ordering_holding_worse_than_reachable():
    holding = compute_risk_score([PERSON, KNIFE_NEAR], relationship="holding", site_config=OFFICE_CONFIG)
    reachable = compute_risk_score([PERSON, KNIFE_NEAR], relationship="reachable", site_config=OFFICE_CONFIG)
    assert holding.score > reachable.score


def test_context_mismatch_disables_leniency_even_at_permitted_site():
    """If CLIP is confident the scene doesn't look like the configured
    'hospital' environment, the hospital's leniency shouldn't apply —
    treat it as the strict case regardless of site_config.
    """
    matched = compute_risk_score(
        [PERSON, KNIFE_NEAR], relationship="near", site_config=HOSPITAL_CONFIG,
        observed_environment="hospital", observed_confidence=0.9,
    )
    mismatched = compute_risk_score(
        [PERSON, KNIFE_NEAR], relationship="near", site_config=HOSPITAL_CONFIG,
        observed_environment="office", observed_confidence=0.9,
    )
    assert matched.context_mismatch is False
    assert mismatched.context_mismatch is True
    assert mismatched.environment_factor == 1.0
    assert mismatched.score > matched.score


def test_low_confidence_mismatch_is_ignored():
    """CLIP disagreeing with low confidence shouldn't override the
    operator's explicit site configuration."""
    result = compute_risk_score(
        [PERSON, KNIFE_NEAR], relationship="near", site_config=HOSPITAL_CONFIG,
        observed_environment="office", observed_confidence=0.2,
    )
    assert result.context_mismatch is False
    assert result.environment_factor == 0.5


def test_score_bounded_0_to_100():
    extreme_config = SiteConfig(environment="x", permitted_sharp_object_roles=[])
    gun = DetectedObject(label="gun", confidence=0.99, bbox=(60, 0, 100, 40))
    result = compute_risk_score([PERSON, gun], relationship="holding", site_config=extreme_config)
    assert 0 <= result.score <= 100