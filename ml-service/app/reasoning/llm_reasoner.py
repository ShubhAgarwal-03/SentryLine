"""Phase 3+ LLM-based reasoning layer — PRD §10.

Takes the same structured facts as template_reasoner.py and produces
richer, more natural explanations by delegating to an LLM's broad world
knowledge instead of hand-written rules. Not wired into the Phase 1
pipeline yet — routers/inference.py calls template_reasoner directly until
this is built out with a real API client and prompt.
"""
from app.schemas.inference import DetectedObject, SiteConfig


def generate_explanation_llm(
    objects: list[DetectedObject],
    relationship: str | None,
    environment: str,
    risk_score: int,
    site_config: SiteConfig,
) -> tuple[str, str]:
    raise NotImplementedError(
        "LLM reasoning is a Phase 3 feature (PRD §10/§16) — "
        "template_reasoner.generate_explanation is the Phase 1-2 implementation."
    )