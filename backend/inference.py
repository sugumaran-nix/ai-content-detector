"""
inference.py — Document-level and sentence-level scoring.

Loads the trained bundle (classifier + scaler + metadata) once,
then exposes predict_document() for use by both app.py (FastAPI)
and any offline scripts.
"""

import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np
from nltk.tokenize import sent_tokenize

from features import FEATURE_NAMES, extract_features, feature_vector, ensure_nltk_data

MODEL_DIR = Path(__file__).parent / "model"


# ── Typed return shapes ──────────────────────────────────────────────────────

class SentenceResult(TypedDict):
    text: str
    ai_probability: float


class DocumentResult(TypedDict):
    label: str
    ai_probability: float
    confidence: float
    features: dict
    sentences: list[SentenceResult]
    model_name: str | None


# ── Bundle loading (cached per process) ─────────────────────────────────────

_BUNDLE: dict | None = None


def get_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        path = MODEL_DIR / "classifier.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run train.py or restart the server "
                "to trigger the auto-build."
            )
        with open(path, "rb") as f:
            _BUNDLE = pickle.load(f)
    return _BUNDLE


# ── Core scoring ─────────────────────────────────────────────────────────────

def _score_text(text: str, bundle: dict) -> float:
    """Return AI probability (0–1) for a piece of text."""
    vec = np.array(feature_vector(text)).reshape(1, -1)
    vec_scaled = bundle["scaler"].transform(vec)
    return float(bundle["model"].predict_proba(vec_scaled)[0][1])


def predict_document(text: str) -> DocumentResult:
    """
    Full document analysis.

    Returns a DocumentResult with verdict, probability, confidence,
    raw feature values, and per-sentence AI probabilities.
    """
    ensure_nltk_data()
    bundle = get_bundle()

    # Document-level verdict
    feats       = extract_features(text)
    vec         = np.array([feats[name] for name in FEATURE_NAMES]).reshape(1, -1)
    vec_scaled  = bundle["scaler"].transform(vec)
    ai_prob     = float(bundle["model"].predict_proba(vec_scaled)[0][1])
    confidence  = abs(ai_prob - 0.5) * 2  # 0 (uncertain) → 1 (certain)

    if ai_prob >= 0.60:
        label = "likely_ai"
    elif ai_prob <= 0.40:
        label = "likely_human"
    else:
        label = "mixed"

    # Sentence-level breakdown
    sentences: list[SentenceResult] = []
    for sent in sent_tokenize(text):
        sent = sent.strip()
        if len(sent.split()) < 4:
            # Too short for reliable features — use document-level as fallback
            sentences.append({"text": sent, "ai_probability": ai_prob})
            continue
        try:
            s_prob = _score_text(sent, bundle)
        except Exception:
            s_prob = ai_prob
        sentences.append({"text": sent, "ai_probability": round(s_prob, 4)})

    return DocumentResult(
        label=label,
        ai_probability=round(ai_prob, 4),
        confidence=round(confidence, 4),
        features={k: feats[k] for k in FEATURE_NAMES},
        sentences=sentences,
        model_name=bundle.get("model_name"),
    )
