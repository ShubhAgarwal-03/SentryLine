# SentryLine

**A context-adaptive risk reasoning system for camera feeds.**

SentryLine doesn't just detect objects — it reasons about whether they're actually dangerous *given where they are*. The same detected objects (a person near a knife) should produce a different risk verdict in a hospital (staff routinely carry sharp instruments) than in an office (they shouldn't) — same perception, different risk score, purely from site configuration. No per-site retraining required.

> **Status:** Active development. ML pipeline is materialized and partially verified. Full-stack integration (`apps/api`, `apps/web`) is designed but not yet on disk. See [Project Status](#project-status) below for the honest current state.

---

## Table of Contents

- [The Core Idea](#the-core-idea)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Project Status](#project-status)
- [Getting Started](#getting-started)
- [ML Pipeline Details](#ml-pipeline-details)
- [Training Data Pipeline](#training-data-pipeline)
- [Known Issues & Gotchas](#known-issues--gotchas)
- [Roadmap](#roadmap)

---

## The Core Idea

Most computer vision safety systems detect objects and stop there. SentryLine separates three concerns that most projects collapse into one model:

- **Perception** — what's in the scene (environment-agnostic, trained once, never retrained per site)
- **Context** — what environment this is, and what the local norms are (zero-shot CLIP classification + a per-site config file, never retrained)
- **Reasoning** — does this combination of objects/relationships/environment actually constitute risk, and why (rule-based or LLM inference, producing a human-readable explanation)

**Example:** the same detections (`knife`, `person`, relationship `running_with`) produce:
- `environment: hospital` + config exception for staff → **Low risk**, logged only
- `environment: office`, no exception → **High risk**, immediate alert

Deploying to a new environment means writing a config file, not retraining a model.

Every alert follows a fixed, explainable output shape:

```
Risk Level: [Score/100] — [Safe/Low/Medium/High/Critical]
Reasoning: [Why this combination of objects/relationship/environment matters]
Potential Outcome: [What could happen if unaddressed]
Recommended Action: [What a human should do]
```

---

## Architecture

**Target stack:** Next.js frontend → NestJS backend → Python/FastAPI ML service, with a shared PostgreSQL/Prisma schema and Redis for async video processing (Phase 4+).

```
Next.js (apps/web) → NestJS (apps/api) → FastAPI (apps/ml-service)
                            ↓
                    PostgreSQL (packages/db, Prisma)
```

### ML pipeline stages (apps/ml-service)

```
Input (RGB frame + site_id)
   → Perception   (YOLOv8n object detection — environment-agnostic)
   → Context      (CLIP zero-shot scene classification + site config lookup)
   → Relationships (deterministic bbox/depth geometry: near, holding, reachable...)
   → Risk Scoring  (danger_weight × relationship_multiplier × environment_factor)
   → Reasoning     (template rules, or LLM — produces the 4-field explanation)
   → Output (JSON risk report)
```

See [`sentryline_methodology.md`](./sentryline_methodology.md) for full per-stage detail, and the PRD documents for the complete architectural rationale and phased roadmap.

---

## Repository Layout

```
sentryline/
├── apps/
│   ├── web/              # Next.js frontend — designed, mostly not materialized
│   ├── api/               # NestJS backend — designed, not materialized
│   └── ml-service/        # FastAPI ML pipeline — materialized, partially verified
│       ├── app/
│       │   ├── main.py
│       │   ├── routers/inference.py       # POST /infer
│       │   ├── perception/detector.py     # YOLOv8n wrapper
│       │   ├── perception/depth.py        # Depth Anything V2 Small
│       │   ├── context/scene_classifier.py # CLIP zero-shot
│       │   ├── relationships/relationship_engine.py  # depth-aware
│       │   ├── knowledge/knowledge_graph.py
│       │   └── reasoning/
│       │       ├── risk_scorer.py
│       │       ├── template_reasoner.py
│       │       └── llm_reasoner.py        # Phase 3 stub
│       ├── configs/
│       │   ├── hazard_classes.yaml        # 14-class taxonomy
│       │   └── sites/{home,hospital,office}.json
│       ├── training/                      # dataset prep + fine-tuning pipeline
│       ├── tests/unit/                    # 15 passing tests
│       └── tests/eval/                    # harness built, manifest.json empty
├── packages/
│   └── db/                # Prisma schema — materialized
└── docker-compose.yml      # not started
```

---

## Project Status

**Materialized and partially verified:** `apps/ml-service` (the pipeline above), `packages/db` (Prisma schema).

**Designed, not yet on disk:** `apps/web`, `apps/api`. Nothing has run end-to-end through the full target stack — no Postgres → NestJS → FastAPI request has ever completed. Only the ML service has been exercised directly.

**Hazard taxonomy (14 classes):** `knife, scissors, gun, fire, spill, ladder, exposed_wire, broken_glass, stove_burner_on, bottle, cell phone, laptop, chair, person`

- 7 classes (`knife, scissors, bottle, cell phone, laptop, chair, person`) are covered by stock COCO-pretrained YOLOv8n — no fine-tuning needed.
- 2 classes (`gun`, `fire`) have been sourced, merged, and are actively being fine-tuned (see [Training Data Pipeline](#training-data-pipeline)).
- 5 classes (`spill, ladder, exposed_wire, broken_glass, stove_burner_on`) are not yet sourced. Candidate datasets identified for `ladder` and `broken_glass`; `spill` and `stove_burner_on` likely need self-captured footage.

**Two real bugs found and fixed this session:**
1. **Vocabulary mismatch** — the risk scorer intersected object labels against permitted *roles*, two vocabularies that could never match, silently defeating the environment discount. Fixed to check site-level permission (a documented Phase 1 approximation).
2. **Dead CLIP output** — scene classification ran every request but its result was only logged, never consumed by scoring. Now a confident (≥50%) environment mismatch disables that site's leniency rules.

**No real accuracy numbers exist yet.** The eval harness (`tests/eval/`) is built and its plumbing is tested, but `manifest.json` ground truth is empty.

---

## Getting Started

### Environment (Windows, confirmed working)

```powershell
# From project root
python -m venv venv
venv\Scripts\Activate.ps1

# pip cache (optional, keeps installs off C:)
pip config set global.cache-dir D:\path\to\pip-cache

# CUDA-enabled torch (adjust index-url for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r apps/ml-service/requirements.txt
```

Verify GPU is visible:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Running the ML service

```powershell
cd apps\ml-service
uvicorn app.main:app --reload
```

### Running tests

```powershell
cd apps\ml-service
pytest tests/unit/
```

---

## ML Pipeline Details

| Stage | File | Status |
|---|---|---|
| Perception | `app/perception/detector.py` | Real YOLOv8n inference. 7/14 classes via COCO; 2/14 (`gun`, `fire`) via fine-tuning in progress |
| Depth | `app/perception/depth.py` | Real Depth Anything V2 Small implementation |
| Context | `app/context/scene_classifier.py` | Real CLIP zero-shot call, output now load-bearing |
| Relationships | `app/relationships/relationship_engine.py` | Real, depth-aware demotion logic |
| Knowledge | `app/knowledge/knowledge_graph.py` | Loads `configs/hazard_classes.yaml` — shared source of truth with training pipeline |
| Risk scoring | `app/reasoning/risk_scorer.py` | `danger_weight × relationship_multiplier × environment_factor`, clamped 0–100. 8 passing unit tests |
| Reasoning | `app/reasoning/template_reasoner.py` | Rule-based, real. `llm_reasoner.py` is a Phase 3 stub |

**Risk scale:** 0–20 Safe · 20–40 Low · 40–60 Medium · 60–80 High · 80–100 Critical

---

## Training Data Pipeline

Located in `apps/ml-service/training/`.

```
prepare_dataset.py    — generates data.yaml from hazard_classes.yaml
train.py               — transfer-learning training script (yolov8n.pt base)
reorg_roboflow.py       — converts Roboflow's train/valid/test layout to images/{train,val}, labels/{train,val}
merge_dataset.py        — remaps a downloaded dataset's class indices to the shared taxonomy, merges into training/dataset/
downsample_class.py     — reduces an overrepresented class to a target count, per split independently
count_classes.py        — audits per-class instance counts across train/val
```

### Datasets sourced so far

| Class | Source | License | Size (merged) |
|---|---|---|---|
| `gun` | "Yolo Weapon Detection" (weapon-detect-qbsiw/yolo-weapon-detection, v2) | CC BY 4.0 | 1,174 images |
| `fire` | "Fire & Smoke Detection" (mytest-blult/fire-smoke-detection-eozii, v1) | CC BY 4.0 | Downsampled from ~20,346 to 900/split-appropriate |

### Current dataset state (train/val)

- **Train:** 1,866 images — fire=900, gun=966 (balanced)
- **Val:** 343 images — fire=135, gun=208 (≈18% of train, proportional)

### Known-good training command

```powershell
python training\train.py --data training\data.yaml --epochs 100 --batch 8 --workers 2
```

**AMP must stay off** (`--amp` flag defaults to `False`) — see [Known Issues](#known-issues--gotchas).

### Candidate datasets for remaining classes

| Class | Candidate | Size |
|---|---|---|
| `ladder` | Construction Site Safety (roboflow-universe-projects) | 44k images (multi-class) |
| `broken_glass` | CLEAR BROKEN GLASS (colsecure) | 247 images |
| `exposed_wire` | "Broken Cable" datasets (wire-oezdr) | 514–581 images — semantic fit not yet confirmed |
| `spill`, `stove_burner_on` | No purpose-built dataset found | Likely needs self-captured footage |

---

## Known Issues & Gotchas

- **AMP causes NaN losses on GTX 1650.** Turing-architecture numerical instability with Automatic Mixed Precision. `--amp` defaults to off in `train.py` — do not re-enable without re-testing on this hardware.
- **`workers=8` (Ultralytics default) crashes on Windows** with `OSError: [WinError 1455] The paging file is too small`. Use `--workers 2` (or increase the Windows paging file size if you need more).
- **`.gitignore` must live in the actual repo root.** In this project's layout, that's `apps/`, not the top-level `SentryLine/` folder — check with `git status` from inside `apps/` to confirm where `.git` actually is before troubleshooting an ignore file that "isn't working."
- **CUDA OOM can occur even with a clean GPU state.** On WDDM (Windows) systems, `nvidia-smi`'s per-process memory reporting is unreliable for compositor/browser GPU usage (`N/A` doesn't mean zero). If training OOMs unexpectedly on a config that worked before, check `torch.cuda.mem_get_info()` directly for ground truth, and consider dropping `--batch` before assuming a code regression.
- **`downsample_class.py` must target train and val independently.** An early version applied one `--target` to both splits uniformly, collapsing val to the same size as train (should be ~10–20% of train). Always pass separate targets per split.
- **Roboflow's "1 epoch mAP" numbers on dataset pages are often inflated** by near-duplicate train/val images from auto-augmentation — don't take a source dataset's advertised accuracy at face value.
- **Relationship engine has no "unattended object" concept.** This means `ladder` detection won't produce meaningful risk scores even once trained — it's a relationship-engine gap (Phase 4), not a labeling problem.

---

## Roadmap

1. Finish sourcing and merging remaining hazard classes (`ladder`, `broken_glass` next; `spill`/`stove_burner_on` need self-capture)
2. Materialize `apps/api` (NestJS) and `apps/web` (Next.js) to disk
3. Run the full pipeline end-to-end against a local Postgres instance
4. Populate `tests/eval/manifest.json` with real labeled fixtures and get real accuracy numbers
5. Phase 3: LLM-based reasoning (`llm_reasoner.py`)
6. Phase 4: real-time video (tracking, Redis Streams, WebSocket push)
7. Phase 5 (stretch): industrial module, edge deployment, continual adaptation

Full phased detail lives in `PRD_SentryLine_Master.md`.

---

## License

*(Add your license here — datasets sourced so far are CC BY 4.0, which requires attribution; keep dataset credits above intact if you redistribute training data.)*
