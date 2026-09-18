import os
from collections import Counter
import yaml

DATASET = "training/dataset"
DATA_YAML = "training/data.yaml"

with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg["names"]
if isinstance(class_names, dict):
    class_names = [class_names[i] for i in range(len(class_names))]

for split in ["train", "val"]:
    label_dir = os.path.join(DATASET, "labels", split)
    counts = Counter()
    file_count = 0
    if not os.path.isdir(label_dir):
        print(f"[{split}] MISSING DIRECTORY: {label_dir}")
        continue
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        file_count += 1
        with open(os.path.join(label_dir, fname), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                counts[cls_id] += 1

    print(f"\n=== {split.upper()} — {file_count} label files ===")
    for i, name in enumerate(class_names):
        n = counts.get(i, 0)
        flag = "  <-- ZERO INSTANCES" if n == 0 else ""
        print(f"  [{i:2d}] {name:20s}: {n:5d}{flag}")
    print(f"  TOTAL instances: {sum(counts.values())}")