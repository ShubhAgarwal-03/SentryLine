"""Semantic danger-weight lookup — the knowledge graph concept from the
original PEHD notes (PRD §4). Loads from configs/hazard_classes.yaml, the
single source of truth shared with the training pipeline
(training/prepare_dataset.py) — editing the yaml is the only thing needed
to add a new hazard class to both scoring and fine-tuning.
"""
from functools import lru_cache
from pathlib import Path

import yaml

HAZARD_CLASSES_PATH = Path(__file__).resolve().parents[2] / "configs" / "hazard_classes.yaml"


@lru_cache(maxsize=1)
def _load_weights() -> tuple[dict[str, float], float]:
    data = yaml.safe_load(HAZARD_CLASSES_PATH.read_text())
    weights = {entry["label"]: entry["danger_weight"] for entry in data["classes"]}
    default = data.get("default_danger_weight", 0.15)
    return weights, default


def danger_weight(label: str) -> float:
    weights, default = _load_weights()
    return weights.get(label, default)