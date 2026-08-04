"""
FastAPI service for the AI-Generated Text Detector.

Endpoints:
  GET  /health           - liveness check
  GET  /model-info       - which model is loaded + its held-out test metrics
  POST /analyze          - { text: str } -> document verdict + sentence breakdown
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import inference

MAX_CHARS = 8000
MODEL_DIR = Path(__file__).parent / "model"


def _ensure_models():
    """Build reference LM and classifier if pkl files are missing (e.g. first Render deploy)."""
    lm_path = MODEL_DIR / "reference_lm.pkl"
    clf_path = MODEL_DIR / "classifier.pkl"

    if not lm_path.exists():
        print("[startup] reference_lm.pkl not found — building from NLTK Brown corpus...")
        from reference_lm import build_reference_lm, save_reference_lm
        save_reference_lm(build_reference_lm())
        print("[startup] reference_lm.pkl built.")

    if not clf_path.exists():
        print("[startup] classifier.pkl not found — training classifier...")
        import train  # noqa: F401 — side-effectful import runs training
        print("[startup] classifier.pkl trained.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_models()
    inference.get_bundle()   # warm the model into memory
    yield


app = FastAPI(
    title="AI-Generated Text Detector",
    description="Statistical-feature classifier that estimates whether text was AI-generated.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    bundle = inference.get_bundle()
    return {
        "model_name":     bundle.get("model_name"),
        "test_accuracy":  bundle.get("test_accuracy"),
        "test_auc":       bundle.get("test_auc"),
        "feature_names":  bundle.get("feature_names"),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"Text exceeds {MAX_CHARS} character limit.")
    word_count = len(text.split())
    if word_count < 8:
        raise HTTPException(
            status_code=400,
            detail="Text is too short. Provide at least a couple of sentences (8+ words).",
        )
    return inference.predict_document(text)
