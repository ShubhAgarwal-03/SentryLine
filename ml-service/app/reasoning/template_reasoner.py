"""Phase 1-2 rule-based explanation generator — PRD §10.

Fast, deterministic, fully debuggable. Produces the same output shape the
LLM reasoner (Phase 3+) will produce, so the app layer never needs to know
which one generated a given event.
"""
from app.schemas.inference import DetectedObject, SiteConfig

RISK_LEVEL_LABELS = [
    (20, "Safe"),
    (40, "Low"),
    (60, "Medium"),
    (80, "High"),
    (101, "Critical"),
]


def risk_level_label(score: int) -> str:
    for threshold, label in RISK_LEVEL_LABELS:
        if score < threshold:
            return label
    return "Critical"


def generate_explanation(
    objects: list[DetectedObject],
    relationship: str | None,
    environment: str,
    risk_score: int,
    site_config: SiteConfig,
    context_mismatch: bool = False,
) -> tuple[str, str]:
    """Returns (reasoning, recommended_action) — the two free-text fields of
    the output template from PRD §10.
    """
    hazard_labels = [o.label for o in objects if o.label != "person"]
    has_person = any(o.label == "person" for o in objects)
    level = risk_level_label(risk_score)

    if not hazard_labels:
        return (
            "No notable hazard objects were detected in this scene.",
            "No action — log for record only.",
        )

    hazards_str = ", ".join(hazard_labels)
    rel_str = f" {relationship} a person" if relationship and has_person else ""

    reasoning = (
        f"Detected {hazards_str}{rel_str} in a {environment} environment. "
        f"{site_config.notes}"
    ).strip()

    if context_mismatch:
        reasoning += (
            f" Note: this scene doesn't visually match the configured "
            f"'{environment}' environment — this site's usual leniency for "
            f"permitted objects has been disabled as a precaution until "
            f"the mismatch is reviewed."
        )

    if level in ("Safe", "Low"):
        action = "No action — log for record only."
    elif level == "Medium":
        action = "Monitor the situation; no immediate intervention required."
    elif level == "High":
        action = "Notify on-site staff to check the situation promptly."
    else:
        action = "Immediate attention required — alert on-site staff now."

    if context_mismatch and level not in ("Safe", "Low"):
        action += " Also verify the site configuration/camera placement — the observed scene didn't match the configured environment."

    return reasoning, action