"""Environment classification accuracy — PRD §15 'Context layer':
"Environment classification accuracy (against a small labeled validation
set of scene photos per environment)."
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2

from app.context.scene_classifier import classify_environment

from .common import ManifestEntry, load_manifest


def evaluate_scene_classifier(entries: list[ManifestEntry]) -> dict:
    correct = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for entry in entries:
        image_bgr = cv2.imread(str(entry.image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predicted_label, confidence = classify_environment(image_rgb)

        confusion[entry.environment][predicted_label] += 1
        if predicted_label == entry.environment:
            correct += 1

    accuracy = correct / len(entries) if entries else 0.0
    return {"accuracy": accuracy, "num_images": len(entries), "confusion": confusion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(Path(__file__).parent / "manifest.json"))
    args = parser.parse_args()

    entries = load_manifest(Path(args.manifest))
    if not entries:
        print("No manifest entries found — see tests/eval/README.md.")
        return

    results = evaluate_scene_classifier(entries)
    print(f"Environment classification accuracy: {results['accuracy']:.1%} ({results['num_images']} images)")
    print("\nConfusion (true -> predicted counts):")
    for true_label, predictions in results["confusion"].items():
        for pred_label, count in predictions.items():
            marker = "correct" if pred_label == true_label else "WRONG"
            print(f"  [{marker}] {true_label} -> {pred_label}: {count}")


if __name__ == "__main__":
    main()