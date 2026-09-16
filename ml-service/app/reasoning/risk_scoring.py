"""Weighted risk score calculation — PRD §12. Deliberately a simple,
transparent, hand-tuned formula for the MVP (not a learned model) so it's
fully explainable and debuggable.
"""
from dataclasses import dataclass

from app.knowledge.knowledge_graph import danger_weight
from app.schemas.inference import DetectedObject, SiteConfig

RELATIONSHIP_MULTIPLIER = {
    "holding": 1.0,
    "near": 0.7,
    "reachable": 0.5,
    None: 0.3,
}

# If CLIP is at least this confident the scene does NOT match the
# configured site environment, we stop trusting that environment's
# leniency rules — e.g. a "hospital" config's permitted-sharp-objects
# discount shouldn't apply to a scene that doesn't look like a hospital
# (wrong config, wrong camera, spoofed/misconfigured deployment).
# Below this confidence, CLIP's read is too uncertain to override the
# operator's explicit site configuration.
CONTEXT_MISMATCH_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class RiskScoreResult:
    score: int
    context_mismatch: bool  # True if CLIP's observed scene confidently disagrees with site_config.environment
    base_weight: float
    relationship_multiplier: float
    environment_factor: float


def compute_risk_score(
    objects: list[DetectedObject],
    relationship: str | None,
    site_config: SiteConfig,
    observed_environment: str | None = None,
    observed_confidence: float | None = None,
) -> RiskScoreResult:
    """Returns a RiskScoreResult with the 0-100 score plus the factors that
    produced it, so the reasoning layer can explain *why*.

    Formula (PRD §12 inputs, MVP subset — role/person-type and
    persistence/history land in Phase 3/4 once role detection and
    tracking exist):
      base = max intrinsic danger weight among non-person objects
      * relationship multiplier (holding > near > reachable)
      * environment factor (derived from site_config — the mechanism
        that makes the same objects score differently per site, PRD §9;
        disabled if CLIP's observed scene doesn't match the configured
        environment — see CONTEXT_MISMATCH_CONFIDENCE_THRESHOLD)
    """
    hazard_objects = [o for o in objects if o.label != "person"]
    if not hazard_objects:
        return RiskScoreResult(
            score=0, context_mismatch=False, base_weight=0.0,
            relationship_multiplier=0.0, environment_factor=1.0,
        )

    base_weight = max(danger_weight(o.label) for o in hazard_objects)
    rel_multiplier = RELATIONSHIP_MULTIPLIER.get(relationship, 0.3)

    # Phase 1 approximation: role detection (who is holding the object)
    # isn't implemented yet (Phase 3), so we can't check "is THIS specific
    # person permitted to carry sharp objects here." Instead we check
    # whether this site's config permits sharp objects for *any* role at
    # all — e.g. a hospital's config lists staff/nurse as permitted
    # roles, so sharp objects get a discount there; an office's config
    # lists no permitted roles, so they don't. This is intentionally
    # conservative and gets replaced once role/uniform detection lands.
    site_generally_permits_sharp_objects = len(site_config.permitted_sharp_object_roles) > 0

    context_mismatch = (
        observed_environment is not None
        and observed_confidence is not None
        and observed_environment != site_config.environment
        and observed_confidence >= CONTEXT_MISMATCH_CONFIDENCE_THRESHOLD
    )

    if context_mismatch:
        # Scene doesn't visually match the configured site — don't extend
        # that site's leniency. Treat as the strictest case.
        environment_factor = 1.0
    else:
        environment_factor = 0.5 if site_generally_permits_sharp_objects else 1.0

    score = base_weight * rel_multiplier * environment_factor * 100
    return RiskScoreResult(
        score=int(round(min(max(score, 0), 100))),
        context_mismatch=context_mismatch,
        base_weight=base_weight,
        relationship_multiplier=rel_multiplier,
        environment_factor=environment_factor,
    )