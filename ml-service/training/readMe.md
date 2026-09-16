# Hazard object fine-tuning plan

## Why this is needed

Stock YOLOv8n only knows COCO's 80 classes. Of the hazard classes SentryLine
cares about (`configs/hazard_classes.yaml`), only `knife`, `scissors`,
`bottle`, `cell phone`, `laptop`, `chair`, and `person` are COCO classes —
those work today with zero extra effort. Everything else (`gun`, `fire`,
`spill`, `ladder`, `exposed_wire`, `broken_glass`, `stove_burner_on`) will
return **nothing** from `detect_objects()` no matter how good the rest of
the pipeline is — the model has literally never seen labeled examples of
these things.

This is the actual bottleneck for Phase 1's completeness, not the reasoning
or scoring layers.

## What I can and can't do from this environment

- **Can't**: browse the web for stock photos, download datasets (Roboflow,
  Open Images, Kaggle), or run GPU training — this sandbox has no GPU and
  network access limited to package registries.
- **Can**: generate the class taxonomy, the data.yaml generator, the
  training script, and this plan — all of which are ready to run the
  moment real labeled data exists.

The actual data collection step is a manual task outside this environment.

## Priority order (by danger_weight × how achievable the data is)

| Class | Danger weight | Why this priority |
|---|---|---|
| `gun` | 0.98 | Highest weight, but also hardest to source ethical/legal training photos for. Consider a licensed dataset (e.g. a weapons-detection dataset with proper licensing) rather than manual collection. |
| `fire` | 0.95 | High weight, relatively easy to source (public domain fire/smoke imagery exists) and visually distinctive — good ROI. |
| `exposed_wire` | 0.60 | Moderate weight. Small object — will need more examples than large ones for the same recall, and close-up/varied-angle shots. |
| `stove_burner_on` | 0.55 | Moderate weight, but flagged in the taxonomy as likely needing a two-stage approach (detect stove → classify burner state) rather than one YOLO class — scope this as a design decision before collecting data, not just a labeling task. |
| `broken_glass` | 0.50 | Moderate weight. Expect false positives against reflective surfaces early on — plan for a "hard negatives" batch (photos of normal glass/reflections labeled as NOT broken_glass) alongside positives. |
| `spill` | 0.45 | Lower weight, but PRD's home-environment scenarios lean on this — worth having even at lower priority. Same reflective-surface confusion risk as broken_glass. |
| `ladder` | 0.35 | Lowest weight of the new classes. Also: `relationship_engine.py` doesn't yet have "unattended" as a relationship type, so even a perfect ladder detector won't produce good risk scores until that's built (see "Related gap" below). Deprioritize until relationship logic catches up. |

## Data collection guidance (per class)

- **Minimum viable**: ~50-100 labeled instances per class to get a first
  signal from fine-tuning; COCO-scale performance needs thousands, but
  that's not the bar for a Phase 1 proof of concept.
- **Split**: 80/10/10 train/val/test, or 80/20 train/val if the set is
  small — keep test images out of `training/dataset/` entirely and instead
  route them to `tests/eval/fixtures/` so the same images serve as the
  eval harness's ground truth (one labeling pass, two uses).
- **Diversity matters more than volume** for a small set: vary angle,
  lighting, distance, and background over adding more near-duplicate shots
  of the same object.
- **Hard negatives**: for classes prone to false positives (`broken_glass`,
  `spill`, `stove_burner_on`), include some negative examples — normal
  glass, dry floors, unlit burners — so the model learns the distinction,
  not just "this shape exists somewhere."

## Labeling

Use a bounding-box annotation tool that exports YOLO format directly —
CVAT or LabelImg both do. Output needs to match the layout
`prepare_dataset.py` expects:

\`\`\`
training/dataset/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt   # class_idx x_center y_center width height, normalized 0-1
  labels/val/*.txt
\`\`\`

Class indices must match `configs/hazard_classes.yaml`'s order —
`prepare_dataset.py` prints the exact index-to-label mapping it generated;
use that as the labeling tool's class list, don't hand-type it separately
(the two lists silently drifting apart is exactly the bug class this setup
is designed to avoid).

## Running the fine-tune once data exists

\`\`\`bash
cd apps/ml-service/training
python prepare_dataset.py --dataset-dir ./dataset --out data.yaml
python train.py --data data.yaml --epochs 100 --batch 16
\`\`\`

Transfer learning from `yolov8n.pt` (not training from scratch) — the base
model already understands general object shapes; this teaches it the new
classes on top of that, which needs far less data than starting cold.

## Verifying it actually helped

Don't just eyeball the training loss curve. Run the same fixtures through
the eval harness before and after:

\`\`\`bash
cd apps/ml-service
python -m tests.eval.eval_detection --manifest tests/eval/manifest.json
\`\`\`

Compare `mAP@0.5` and per-class AP before/after swapping
`app/perception/detector.py`'s `MODEL_PATH` to the fine-tuned weights.
If per-class AP for the new classes is near 0 after training, that's a
signal to add more/more-varied examples for that class before trusting it
in production scoring — not to ship it anyway because the demo happened to
work on one photo.

## Related gap this doesn't fix

Even a perfect `ladder` detector doesn't produce a good risk score today,
because `relationship_engine.py` only knows `holding`/`near`/`reachable` —
it has no concept of "unattended" or "duration," both of which matter more
than proximity for a ladder. That's Phase 4 territory (tracking/temporal
reasoning, PRD §16) — noting it here so it's not mistaken for a labeling
problem when it's actually a relationship-model gap.