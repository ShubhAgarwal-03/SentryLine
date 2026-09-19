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


def _to_yolo_bbox(parts: list[str]) -> str | None:
    """Return "x y w h" for a label line's coordinates. Passes plain boxes
    through; converts polygon (segmentation) lines to their bounding box.
    Returns None for malformed lines. Mixed polygon/box datasets make
    Ultralytics warn and silently drop the polygons — normalising here
    avoids that."""
    coords = parts[1:]
    if len(coords) == 4:
        return " ".join(coords)
    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = [float(v) for v in coords[0::2]]
        ys = [float(v) for v in coords[1::2]]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return f"{(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
    return None


def remap_label_lines(src_path: Path, index_map: dict[int, int]) -> list[str]:
    """index_map: source class index -> target class index. Source indices
    not in the map are dropped (SKIP classes). Returns the surviving lines
    WITHOUT writing anything, so the caller can decide the output filename
    first."""
    lines_out = []
    for line in src_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        src_idx = int(parts[0])
        if src_idx not in index_map:
            continue  # SKIP class — drop this object
        box = _to_yolo_bbox(parts)
        if box is None:
            print(f"  WARNING: malformed label line in {src_path.name}, skipped")
            continue
        lines_out.append(f"{index_map[src_idx]} {box}")
    return lines_out


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

        lines = remap_label_lines(label_path, index_map)
        if not lines:
            continue  # every object was SKIPped — don't import the image

        # Decide the final name BEFORE writing anything. Collision if EITHER
        # the image or its label already exists in dest (from an earlier
        # merged dataset); prefix with the source folder name until unique.
        # (The old version wrote the label first, which overwrote the
        # earlier dataset's label on a name clash.)
        stem, suffix = img_path.stem, img_path.suffix
        prefix = source_dir.name
        while (dst_images / f"{stem}{suffix}").exists() or (dst_labels / f"{stem}.txt").exists():
            stem = f"{prefix}_{stem}"

        shutil.copy2(img_path, dst_images / f"{stem}{suffix}")
        (dst_labels / f"{stem}.txt").write_text("\n".join(lines) + "\n")
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