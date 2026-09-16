"""FastAPI app entrypoint. Exposes the perception -> context -> reasoning
pipeline as a service, decoupled from the UI framework (PRD §3.1 in v3 —
this is what replaced the Streamlit-coupled Python process from v2).
"""
from fastapi import FastAPI

from app.routers import inference

app = FastAPI(
    title="SentryLine ML Service",
    description="Context-adaptive risk reasoning pipeline (perception -> context -> reasoning)",
    version="0.1.0",
)

app.include_router(inference.router, tags=["inference"])


@app.get("/health")
def health():
    return {"status": "ok"}