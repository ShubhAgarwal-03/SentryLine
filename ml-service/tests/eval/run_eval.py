"""Runs every automatic eval metric from PRD §15 against a fixtures
manifest and prints a combined report. Reasoning quality is intentionally
excluded — see reasoning_rubric.md, that one is graded by a human.

Usage:
    cd apps/ml-service
    python -m tests.eval.run_eval                                          # real fixtures
    python -m tests.eval.run_eval --manifest tests/eval/synthetic_manifest.json  # harness smoke test
"""
import argparse
from pathlib import Path

from .common import load_manifest
from .eval_detection import evaluate_detection
from .eval_scene_classifier import evaluate_scene_classifier
from .eval_system_metrics import evaluate_system_metrics

DEFAULT_MANIFEST_PATH = Path(__file__).parent / "manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    entries = load_manifest(manifest_path)

    if not entries:
        print(
            "No entries in manifest.json — this is expected on a fresh checkout.\n"
            "See tests/eval/README.md: add real labeled photos to fixtures/<env>/ "
            "and describe them in manifest.json before this report means anything.\n\n"
            "In the meantime, run the synthetic smoke test to confirm the harness "
            "itself works:\n"
            "    python -m tests.eval.generate_synthetic_fixtures\n"
            "    python -m tests.eval.run_eval --manifest tests/eval/synthetic_manifest.json"
        )
        return

    print("=" * 60)
    print(f"SentryLine ML eval report — {len(entries)} labeled images")
    print("=" * 60)

    print("\n[Perception] Detection (mAP@0.5)")
    det = evaluate_detection(entries)
    print(f"  mAP@0.5: {det['mAP@0.5']:.3f}")
    for label, ap in sorted(det["ap_per_class"].items()):
        print(f"    {label:<15} AP={ap:.3f}")

    print("\n[Context] Environment classification")
    scene = evaluate_scene_classifier(entries)
    print(f"  Accuracy: {scene['accuracy']:.1%}")

    print("\n[System] False alarm rate / latency")
    sysm = evaluate_system_metrics(entries)
    print(f"  False alarm rate: {sysm['false_alarm_rate']:.1%}")
    print(f"  Miss rate:        {sysm['miss_rate']:.1%}")
    print(f"  Avg latency:      {sysm['avg_latency_ms']:.1f} ms")

    print("\n[Reasoning] Not automatic — see reasoning_rubric.md")


if __name__ == "__main__":
    main()