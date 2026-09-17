"""One-off reorg: converts a Roboflow export's split-first layout
(train/images, valid/images, train/labels, valid/labels) into the
type-first layout merge_dataset.py expects (images/train, images/val,
labels/train, labels/val). Renames 'valid' -> 'val' in the process.
'test' split is intentionally left untouched/unused — not needed for
training or the merge script.

Usage:
    python reorg_roboflow.py --source Yolo-Weapon-Detection-2 --dest Yolo-Weapon-Detection-2-reorg
"""
import argparse
import shutil
from pathlib import Path

SPLIT_MAP = {"train": "train", "valid": "val"}  # test is dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    dest_dir = Path(args.dest).resolve()

    for roboflow_split, target_split in SPLIT_MAP.items():
        src_images = source_dir / roboflow_split / "images"
        src_labels = source_dir / roboflow_split / "labels"
        dst_images = dest_dir / "images" / target_split
        dst_labels = dest_dir / "labels" / target_split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        if src_images.exists():
            for f in src_images.iterdir():
                shutil.copy2(f, dst_images / f.name)
        if src_labels.exists():
            for f in src_labels.iterdir():
                shutil.copy2(f, dst_labels / f.name)

        img_count = len(list(dst_images.iterdir())) if dst_images.exists() else 0
        print(f"{roboflow_split} -> {target_split}: {img_count} images")

    print(f"\nReorganized dataset ready at: {dest_dir}")


if __name__ == "__main__":
    main()