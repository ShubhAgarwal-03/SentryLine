"""Randomly downsamples a class's images/labels within training/dataset/ to
a target count, to fix class imbalance between merged datasets (e.g. fire
massively outnumbering gun after merging).

Only removes images where the SPECIFIED class is the dominant/majority
content — safer approach: this operates per-split (train/val) and removes
images at random until the target count for images CONTAINING that class
is reached. Images with multiple classes are left alone (not removed) to
avoid accidentally deleting other classes' only examples — so if many
images are multi-class, true downsampling may not perfectly hit the target;
the script reports what it actually achieved.

Usage:
    python downsample_class.py --dataset-dir training/dataset --class-name fire --target 1200
"""
import argparse
import random
from pathlib import Path

import yaml

HAZARD_CLASSES_PATH = Path(__file__).resolve().parents[1] / "configs" / "hazard_classes.yaml"


def load_target_class_order() -> list[str]:
    data = yaml.safe_load(HAZARD_CLASSES_PATH.read_text())
    return [entry["label"] for entry in data["classes"]]


def get_classes_in_label(label_path: Path) -> set[int]:
    classes = set()
    for line in label_path.read_text().splitlines():
        if line.strip():
            classes.add(int(line.split()[0]))
    return classes


def downsample_split(dataset_dir: Path, split: str, target_class_idx: int, target_count: int, seed: int) -> tuple[int, int]:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    # Find every image whose label file contains ONLY the target class
    # (single-class images are safe to remove without affecting other classes)
    only_target = []
    other = []
    for label_path in labels_dir.glob("*.txt"):
        classes = get_classes_in_label(label_path)
        if classes == {target_class_idx}:
            only_target.append(label_path)
        else:
            other.append(label_path)

    current_count = len(only_target)
    if current_count <= target_count:
        return current_count, 0  # nothing to remove, already at or below target

    random.seed(seed)
    random.shuffle(only_target)
    to_remove = only_target[target_count:]

    removed = 0
    for label_path in to_remove:
        img_stem = label_path.stem
        label_path.unlink()
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = images_dir / f"{img_stem}{ext}"
            if img_path.exists():
                img_path.unlink()
                removed += 1
                break

    return target_count, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--class-name", required=True, help="Class name as it appears in hazard_classes.yaml")
    parser.add_argument("--target", type=int, required=True, help="Target count of single-class images to keep for this class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    target_classes = load_target_class_order()

    if args.class_name not in target_classes:
        print(f"ERROR: '{args.class_name}' not found in hazard_classes.yaml. Valid: {target_classes}")
        return
    target_idx = target_classes.index(args.class_name)

    print(f"Downsampling class '{args.class_name}' (index {target_idx}) to target {args.target} per split")
    for split in ("train", "val"):
        kept, removed = downsample_split(dataset_dir, split, target_idx, args.target, args.seed)
        print(f"  {split}: kept ~{kept} single-class images, removed {removed} image files")

    print("\nDone. Re-run prepare_dataset.py if you want a fresh data.yaml (class list unchanged, just noting the data changed).")


if __name__ == "__main__":
    main()