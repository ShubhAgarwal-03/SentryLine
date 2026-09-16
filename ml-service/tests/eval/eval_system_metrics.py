"""System-level metrics — PRD §15: false alarm rate and inference latency.

False alarm rate is called out in the PRD as the metric that matters most
for user trust in a safety system, so it gets its own script rather than
being folded into detection/classification eval.
"""
import argparse
import time
from pathlib import Path

import cv2

from app.perception.detector import detect_objects
from app.relationships.relationship_engine import infer_relationship
from app.reasoning.risk_scorer import compute_risk_score
from app.schemas.inference import SiteConfig

from .common import ManifestEntry, load_manifest

# Threshold above which the system would surface an alert to a user.
# Mirrors the app-layer RiskLevel boundary (MEDIUM=40 in inference.service.ts's
# riskLevelFromScore) — kept as a local constant rather than importing across
# the Node/Python boundary; update both if the threshold changes.
ALERT_THRESHOLD = 40

# Placeholder config used only to exercise the scorer in isolation — real
# eval should carry each entry's actual site config once fixtures exist
# with per-site config references, not just an environment string.
_DEFAULT_CONFIG = SiteConfig(environment="generic")


def evaluate_system_metrics(entries: list[ManifestEntry]) -> dict:
    false_positives = 0  # flagged, but human reviewer said should_alert=False
    false_negatives = 0  # NOT flagged, but human reviewer said should_alert=True
    latencies_ms = []

    for entry in entries:
        image_bgr = cv2.imread(str(entry.image_path))

        start = time.perf_counter()
        objects = detect_objects(image_bgr)
        relationship = infer_relationship(objects)
        result = compute_risk_score(objects, relationship, _DEFAULT_CONFIG)
        score = result.score
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        predicted_alert = score >= ALERT_THRESHOLD
        if predicted_alert and not entry.should_alert:
            false_positives += 1
        elif not predicted_alert and entry.should_alert:
            false_negatives += 1

    n = len(entries)
    return {
        "num_images": n,
        "false_alarm_rate": false_positives / n if n else 0.0,
        "miss_rate": false_negatives / n if n else 0.0,
        "avg_latency_ms": sum(latencies_ms) / n if n else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(Path(__file__).parent / "manifest.json"))
    args = parser.parse_args()

    entries = load_manifest(Path(args.manifest))
    if not entries:
        print("No manifest entries found — see tests/eval/README.md.")
        return

    results = evaluate_system_metrics(entries)
    print(f"False alarm rate: {results['false_alarm_rate']:.1%}")
    print(f"Miss rate:        {results['miss_rate']:.1%}")
    print(f"Avg latency:      {results['avg_latency_ms']:.1f} ms  (single-image perception path, no network hop)")


if __name__ == "__main__":
    main()