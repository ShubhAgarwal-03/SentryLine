"""Pydantic request/response models for the /infer endpoint.

This is the contract the NestJS `inference.service.ts` depends on — keep the
field names in sync with `MlServiceResponse` there.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SiteConfig(BaseModel):
    """Mirrors the SiteConfig JSON shape from PRD §9 / packages/db seed data."""
    environment: str
    restricted_roles: list[str] = Field(default_factory=list)
    permitted_sharp_object_roles: list[str] = Field(default_factory=list)
    restricted_zones: list[str] = Field(default_factory=list)
    notes: str = ""


class InferenceRequest(BaseModel):
    image_base64: str
    site_config: SiteConfig


class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coords)


class InferenceResponse(BaseModel):
    objects: list[str]
    relationship: Optional[str]
    environment: str
    risk_score: int
    reasoning: str
    recommended_action: str
    confidence: Optional[float]
    observed_environment: Optional[str] = None  # CLIP's own read of the scene, for QA/logging
    observed_environment_confidence: Optional[float] = None
    context_mismatch: bool = False  # True if observed_environment confidently disagrees with the configured environment