"""Generates a YOLO-format data.yaml from configs/hazard_classes.yaml — the
class list training uses is always derived from the same source scoring
uses, so they can't drift apart.

Expects a dataset already laid out in YOLO format:

    training/dataset/
      images/
        train/*.jpg
        val/*.jpg
      labels/
        train/*.txt   # one line per object: <class_idx> <x_center> <y_center> <w> <h>, normalized 0-1
        val/*.txt

This script does NOT create the dataset — see training/README.md for the
data collection and labeling plan. It only generates the data.yaml that
points ultralytics at whatever dataset directory you've built.

Usage:
    python prepare_dataset.py --dataset-dir ./dataset --out data.yaml
"""
import argparse
from pathlib import Path

import yaml

HAZARD_CLASSES_PATH = Path(__file__).resolve().parents[1] / "configs" / "hazard_classes.yaml"


def load_class_list() -> list[str]:
    data = yaml.safe_load(HAZARD_CLASSES_PATH.read_text())
    # "person" is included for relationship reasoning but detects fine via
    # stock YOLO already — no need to duplicate it in a fine-tuning set
    # unless you want to reinforce it. Kept in the class list either way so
    # class indices stay consistent with knowledge_graph.py's expectations.
    return [entry["label"] for entry in data["classes"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to the YOLO-format dataset root")
    parser.add_argument("--out", default="data.yaml", help="Output path for the generated data.yaml")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    classes = load_class_list()

    config = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: label for i, label in enumerate(classes)},
    }

    out_path = Path(args.out)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote {out_path} with {len(classes)} classes:")
    for i, label in enumerate(classes):
        print(f"  {i}: {label}")
    print(f"\nExpected dataset layout under {dataset_dir}:")
    print("  images/train/*.jpg, images/val/*.jpg")
    print("  labels/train/*.txt, labels/val/*.txt  (YOLO bbox format, class indices as printed above)")


if __name__ == "__main__":
    main()