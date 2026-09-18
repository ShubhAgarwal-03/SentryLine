"""
downsample_class.py — reduce an overrepresented class to a target count,
independently per split (train/val), to fix class imbalance between
merged datasets without breaking the train/val ratio.

Usage:
    python training/downsample_class.py --class-name fire --target 900 --val-target 150
"""

import argparse
import os
import random
import yaml

DATASET_DIR = "training/dataset"
DATA_YAML = "training/data.yaml"


def load_class_id(class_name):
    with open(DATA_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        names = {v: k for k, v in names.items()}  # name -> id
    else:
        names = {n: i for i, n in enumerate(names)}
    if class_name not in names:
        raise ValueError(f"Class '{class_name}' not found in {DATA_YAML}")
    return names[class_name]


def get_single_class_images(label_dir, class_id):
    """Return label files whose ONLY class present is class_id."""
    matches = []
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(label_dir, fname)
        classes_in_file = set()
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                classes_in_file.add(int(line.split()[0]))
        if classes_in_file == {class_id}:
            matches.append(fname)
    return matches


def downsample_split(split, class_id, target, dry_run):
    label_dir = os.path.join(DATASET_DIR, "labels", split)
    image_dir = os.path.join(DATASET_DIR, "images", split)

    if not os.path.isdir(label_dir):
        print(f"[{split}] label dir missing, skipping")
        return

    single_class_files = get_single_class_images(label_dir, class_id)
    current_count = len(single_class_files)

    print(f"\n[{split}] single-class instances for target class: {current_count}")

    if current_count <= target:
        print(f"[{split}] already at or below target ({target}), nothing to remove")
        return

    n_to_remove = current_count - target
    random.shuffle(single_class_files)
    to_remove = single_class_files[:n_to_remove]

    print(f"[{split}] removing {n_to_remove} images to reach target {target}")

    for fname in to_remove:
        label_path = os.path.join(label_dir, fname)
        base = os.path.splitext(fname)[0]

        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(image_dir, base + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if dry_run:
            print(f"  would remove: {fname}" + (f" + {os.path.basename(image_path)}" if image_path else " (image not found)"))
            continue

        os.remove(label_path)
        if image_path:
            os.remove(image_path)
        else:
            print(f"  WARNING: no matching image found for {fname}")

    remaining = current_count - (0 if dry_run else n_to_remove)
    print(f"[{split}] done. remaining single-class count: {remaining}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--target", type=int, required=True, help="target count for train split")
    parser.add_argument("--val-target", type=int, default=None,
                         help="target count for val split (defaults to ~15%% of --target if omitted)")
    parser.add_argument("--dry-run", action="store_true", help="preview without deleting")
    args = parser.parse_args()

    val_target = args.val_target if args.val_target is not None else max(1, round(args.target * 0.15))

    class_id = load_class_id(args.class_name)
    print(f"Class '{args.class_name}' = id {class_id}")
    print(f"Targets — train: {args.target}, val: {val_target}" + (" (auto, 15% of train)" if args.val_target is None else ""))

    downsample_split("train", class_id, args.target, args.dry_run)
    downsample_split("val", class_id, val_target, args.dry_run)


if __name__ == "__main__":
    main()