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

MAX_CHARS = 8_000
MIN_WORDS = 8
MODEL_DIR = Path(__file__).parent / "model"

# For production, replace "*" with your actual frontend domain:
# e.g. ["https://my-portfolio.vercel.app"]
ALLOWED_ORIGINS = ["*"]


def _ensure_models() -> None:
    """Build reference LM and classifier if pkl files are missing (cold deploy)."""
    lm_path = MODEL_DIR / "reference_lm.pkl"
    clf_path = MODEL_DIR / "classifier.pkl"

    if not lm_path.exists():
        print("[startup] reference_lm.pkl not found — building from NLTK Brown corpus…")
        from reference_lm import build_reference_lm, save_reference_lm
        save_reference_lm(build_reference_lm())
        print("[startup] reference_lm.pkl built.")

    if not clf_path.exists():
        print("[startup] classifier.pkl not found — training classifier…")
        from train import main as train_main
        train_main()
        print("[startup] classifier.pkl trained.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_models()
    inference.get_bundle()   # warm the model into memory
    yield


app = FastAPI(
    title="AI-Generated Text Detector",
    description="Statistical-feature classifier estimating whether text is AI-generated.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


class AnalyzeResponse(BaseModel):
    label: str
    ai_probability: float
    confidence: float
    features: dict
    sentences: list[dict]
    model_name: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    bundle = inference.get_bundle()
    return {
        "model_name":    bundle.get("model_name"),
        "test_accuracy": bundle.get("test_accuracy"),
        "test_auc":      bundle.get("test_auc"),
        "feature_names": bundle.get("feature_names"),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    word_count = len(text.split())
    if word_count < MIN_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too short — provide at least {MIN_WORDS} words.",
        )
    return inference.predict_document(text)
