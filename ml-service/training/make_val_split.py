import argparse, random, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="dataset root containing train/images and train/labels")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.source)
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    val_images = root / "valid" / "images"
    val_labels = root / "valid" / "labels"
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    stems = sorted(f.stem for f in train_labels.glob("*.txt"))
    random.seed(args.seed)
    random.shuffle(stems)

    n_val = max(1, int(len(stems) * args.val_fraction))
    val_stems = set(stems[:n_val])

    moved = 0
    for stem in val_stems:
        label_src = train_labels / f"{stem}.txt"
        img_matches = list(train_images.glob(f"{stem}.*"))
        if not img_matches:
            print(f"WARNING: no image found for {stem}, skipping")
            continue
        img_src = img_matches[0]
        shutil.move(str(label_src), str(val_labels / label_src.name))
        shutil.move(str(img_src), str(val_images / img_src.name))
        moved += 1

    print(f"Moved {moved} pairs to valid/ (target {n_val} of {len(stems)})")
    print(f"train labels remaining: {len(list(train_labels.glob('*.txt')))}")
    print(f"valid labels now: {len(list(val_labels.glob('*.txt')))}")

if __name__ == "__main__":
    main()
