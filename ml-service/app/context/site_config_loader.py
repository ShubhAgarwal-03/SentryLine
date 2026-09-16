"""Loads per-site config JSON — PRD §9, Step 2. Cheap, no ML involved.

In Phase 1, callers pass site_config directly in the request body (NestJS
already loaded it from Postgres via Prisma — see inference.service.ts).
This loader exists for local/offline testing of the ML service in isolation,
without needing the Node backend running.
"""
import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "sites"


def load_site_config(site_name: str) -> dict:
    path = CONFIGS_DIR / f"{site_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No config found for site '{site_name}' at {path}")
    return json.loads(path.read_text())