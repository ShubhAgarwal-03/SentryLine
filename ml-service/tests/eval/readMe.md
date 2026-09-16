# Evaluation harness

Implements the metrics specified in PRD §15. This is deliberately separate
from unit tests (which check "does the code run") — this checks "is the
model's understanding actually correct," which is a different question and
needs labeled ground truth to answer.

## Important: synthetic vs. real fixtures

- `fixtures/_synthetic_smoketest/` — programmatically generated placeholder
  images (see `generate_synthetic_fixtures.py`). These exist ONLY to prove
  the eval scripts themselves work (loading, IoU matching, metric math).
  YOLO/CLIP were never trained on crude synthetic shapes, so scores against
  this set are meaningless as accuracy numbers — don't report them as if
  they mean anything about real model performance.
- `fixtures/home/`, `fixtures/hospital/`, `fixtures/office/` — EMPTY on
  purpose. This is where real, labeled photos go. Nothing in this sandbox
  can fetch real-world photos, so populating this is a manual step done
  outside this environment.

## How to add real validation data

1. Drop a handful of real photos (5-15 per environment is enough to start)
   into `fixtures/<environment>/`.
2. Add one entry per image to `manifest.json` (see format below).
3. Run `python -m tests.eval.run_eval`.

## `manifest.json` entry format

\`\`\`json
{
  "image": "hospital/photo1.jpg",
  "environment": "hospital",
  "expected_objects": [
    { "label": "person", "bbox": [120, 40, 310, 480] },
    { "label": "knife", "bbox": [340, 200, 410, 260] }
  ],
  "expected_relationship": "near",
  "should_alert": true,
  "notes": "Staff member with a scalpel near a patient — should NOT be flagged (permitted role/object per hospital config) unless proximity to a visitor-restricted zone."
}
\`\`\`

- `expected_objects` / `expected_objects[].bbox` — ground truth for the
  **detection** metric (mAP@0.5). bbox is `[x1, y1, x2, y2]` in pixels.
- `environment` — ground truth for the **scene classification** metric.
- `should_alert` — ground truth for the **false alarm rate** metric (did a
  human reviewer decide this scene genuinely warrants an alert, independent
  of what risk score the pipeline currently produces).
- `expected_relationship` is optional — omit if not applicable to the scene.

## Running

\`\`\`bash
cd apps/ml-service
python -m tests.eval.run_eval                      # everything, real fixtures
python -m tests.eval.eval_detection                 # detection mAP only
python -m tests.eval.eval_scene_classifier          # CLIP accuracy only
python -m tests.eval.eval_system_metrics            # false alarm rate only

# Harness smoke test (no real photos needed):
python -m tests.eval.generate_synthetic_fixtures
python -m tests.eval.run_eval --manifest tests/eval/synthetic_manifest.json
\`\`\`

Reasoning quality (§15 "Reasoning layer") has no script — see
`reasoning_rubric.md`. That one is graded by a human on purpose; the PRD is
explicit that automatic evaluation doesn't make sense for free-text
explanations.r