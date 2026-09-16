"""POST /infer — the Phase 1-3 synchronous inference path.

Orchestrates the full pipeline: perception -> context -> relationships ->
risk scoring -> reasoning. Called by NestJS's inference.service.ts, which
already resolved and passed in the site_config from Postgres.
"""
import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from app.context.scene_classifier import classify_environment
from app.perception.detector import detect_objects
from app.relationships.relationship_engine import infer_relationship
from app.reasoning.risk_scorer import compute_risk_score
from app.reasoning.template_reasoner import generate_explanation
from app.schemas.inference import InferenceRequest, InferenceResponse

router = APIRouter()


def _decode_image(image_base64: str) -> np.ndarray:
    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image_bgr


@router.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    image_bgr = _decode_image(request.image_base64)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1. Perception — environment-agnostic, trained once (PRD §3)
    objects = detect_objects(image_bgr)

    # 2. Context — zero-shot scene classification (PRD §9 Step 1).
    # site_config.environment (chosen at deployment) remains the
    # authoritative label for the "environment" field in the response —
    # operators configure a site once, and we don't want scores flapping
    # frame-to-frame on CLIP's read alone. But CLIP's observed label now
    # actively feeds risk scoring: see risk_scorer.py's context_mismatch
    # handling, which disables a site's leniency rules if the scene
    # confidently doesn't match what's configured.
    observed_environment, observed_confidence = classify_environment(image_rgb)
    environment = request.site_config.environment

    # 3. Relationships — spatial reasoning (PRD §4 level 2/3)
    relationship = infer_relationship(objects)

    # 4. Risk scoring — environment-conditioned (PRD §12), now
    # context-checked against what the scene actually looks like
    risk_result = compute_risk_score(
        objects, relationship, request.site_config,
        observed_environment=observed_environment,
        observed_confidence=observed_confidence,
    )

    # 5. Reasoning — template-based explanation (PRD §10, Phase 1-2)
    reasoning, recommended_action = generate_explanation(
        objects, relationship, environment, risk_result.score, request.site_config,
        context_mismatch=risk_result.context_mismatch,
    )

    avg_confidence = (
        sum(o.confidence for o in objects) / len(objects) if objects else None
    )

    return InferenceResponse(
        objects=[o.label for o in objects],
        relationship=relationship,
        environment=environment,
        risk_score=risk_result.score,
        reasoning=reasoning,
        recommended_action=recommended_action,
        confidence=avg_confidence,
        observed_environment=observed_environment,
        observed_environment_confidence=observed_confidence,
        context_mismatch=risk_result.context_mismatch,
    )