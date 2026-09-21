import re, random, shutil
from pathlib import Path

SRC = Path("training/downloads/broken-glass")
random.seed(42)
VAL_FRAC = 0.15

img_dir, lbl_dir = SRC / "train" / "images", SRC / "train" / "labels"
groups = {}
for f in img_dir.iterdir():
    base = re.split(r"\.rf\.", f.stem)[0]
    groups.setdefault(base, []).append(f.stem)

keys = list(groups.keys())
random.shuffle(keys)
val_keys = set(keys[:max(1, int(len(keys) * VAL_FRAC))])

val_img, val_lbl = SRC / "valid" / "images", SRC / "valid" / "labels"
val_img.mkdir(parents=True, exist_ok=True)
val_lbl.mkdir(parents=True, exist_ok=True)

moved = 0
for k in val_keys:
    for stem in groups[k]:
        for ext in (".jpg", ".png", ".jpeg"):
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                shutil.move(str(p), str(val_img / p.name)); moved += 1
                break
        lp = lbl_dir / f"{stem}.txt"
        if lp.exists():
            shutil.move(str(lp), str(val_lbl / lp.name))

print(f"source-image groups: {len(keys)} | moved to val: {len(val_keys)} groups / {moved} images")
