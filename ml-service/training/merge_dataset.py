"""Merges a downloaded Roboflow (or any YOLO-format) dataset into the
canonical training/dataset/ layout, remapping its class indices to match
configs/hazard_classes.yaml's order.

Why this is needed: a dataset downloaded from Roboflow has its OWN class
list/order (e.g. their "Pistol" might be index 0), which almost never
matches your hazard_classes.yaml order. Label files reference classes by
index, not name — so merging two datasets without remapping silently
corrupts labels (a "gun" label from one dataset could become a "fire"
label after merging, with no error raised).

Usage:
    python merge_dataset.py \
        --source ./downloaded_gun_dataset \
        --source-classes gun \
        --dest ./training/dataset

--source-classes maps the SOURCE dataset's class names, in the SAME ORDER
as that dataset's own data.yaml, to your target label. If the source
dataset only contains one relevant class amid several irrelevant ones
(e.g. it has "Pistol", "Rifle", "Knife" but you only want "Pistol" mapped
to your "gun" class), list ALL of the source's classes in order, using
"SKIP" for any you don't want to import:

    --source-classes gun SKIP SKIP

This drops images/labels for skipped classes entirely (per-object, not
per-image — an image with both a wanted and skipped class keeps only the
wanted object's label line).
"""
import argparse
import shutil
from pathlib import Path

import yaml

HAZARD_CLASSES_PATH = Path(__file__).resolve().parents[1] / "configs" / "hazard_classes.yaml"


def load_target_class_order() -> list[str]:
    data = yaml.safe_load(HAZARD_CLASSES_PATH.read_text())
    return [entry["label"] for entry in data["classes"]]


def remap_label_file(src_path: Path, dst_path: Path, index_map: dict[int, int]) -> None:
    """index_map: source class index -> target class index. Source indices
    not in the map are dropped (SKIP classes)."""
    lines_out = []
    for line in src_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        src_idx = int(parts[0])
        if src_idx not in index_map:
            continue  # SKIP class — drop this object
        target_idx = index_map[src_idx]
        lines_out.append(f"{target_idx} {' '.join(parts[1:])}")

    if lines_out:  # only write if at least one object survived
        dst_path.write_text("\n".join(lines_out) + "\n")


def merge_split(source_dir: Path, dest_dir: Path, split: str, index_map: dict[int, int]) -> int:
    src_images = source_dir / "images" / split
    src_labels = source_dir / "labels" / split
    dst_images = dest_dir / "images" / split
    dst_labels = dest_dir / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    if not src_images.exists():
        return 0

    copied = 0
    for img_path in src_images.iterdir():
        if not img_path.is_file():
            continue
        label_path = src_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            continue  # no annotations for this image, skip

        # Remap into a temp buffer first so we don't copy images whose
        # labels end up empty after SKIP filtering
        tmp_label_dst = dst_labels / label_path.name
        remap_label_file(label_path, tmp_label_dst, index_map)

        if tmp_label_dst.exists():
            dst_img_path = dst_images / img_path.name
            if dst_img_path.exists():
                # Name collision across merged datasets — prefix with source dir name
                dst_img_path = dst_images / f"{source_dir.name}_{img_path.name}"
                tmp_label_dst.rename(dst_labels / f"{source_dir.name}_{label_path.name}")
            shutil.copy2(img_path, dst_img_path)
            copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to the downloaded dataset root (must have images/{train,val} and labels/{train,val})")
    parser.add_argument("--source-classes", nargs="+", required=True,
                         help="Source dataset's class names IN ORDER matching its own data.yaml, using SKIP for classes to drop")
    parser.add_argument("--dest", required=True, help="Path to training/dataset/ (created if it doesn't exist)")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    dest_dir = Path(args.dest).resolve()
    target_classes = load_target_class_order()

    # Build source_idx -> target_idx map
    index_map: dict[int, int] = {}
    for src_idx, src_label in enumerate(args.source_classes):
        if src_label == "SKIP":
            continue
        if src_label not in target_classes:
            print(f"WARNING: source class '{src_label}' not found in hazard_classes.yaml — skipping it. "
                  f"Valid target classes: {target_classes}")
            continue
        index_map[src_idx] = target_classes.index(src_label)

    if not index_map:
        print("ERROR: no valid class mappings — nothing to merge. Check --source-classes against hazard_classes.yaml.")
        return

    print(f"Mapping source indices {list(index_map.keys())} -> target indices {list(index_map.values())}")
    print(f"(target classes: {target_classes})")

    total = 0
    for split in ("train", "val"):
        count = merge_split(source_dir, dest_dir, split, index_map)
        print(f"  {split}: merged {count} images")
        total += count

    print(f"\nDone. {total} images merged into {dest_dir}")
    print("Run prepare_dataset.py next once all source datasets are merged in.")


if __name__ == "__main__":
    main()