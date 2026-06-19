"""
Loads the trained classifier bundle and exposes prediction functions used
by the FastAPI app: a document-level verdict, and a per-sentence breakdown
for the "which parts read as machine-generated" explainability view.
"""

import pickle
from pathlib import Path

import numpy as np
from nltk.tokenize import sent_tokenize

from features import FEATURE_NAMES, ensure_nltk_data, extract_features

MODEL_PATH = Path(__file__).parent / "model" / "classifier.pkl"

_BUNDLE = None


def get_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        with open(MODEL_PATH, "rb") as f:
            _BUNDLE = pickle.load(f)
    return _BUNDLE


def _score_text(text: str) -> tuple[float, dict]:
    """Returns (ai_probability, raw_features) for an arbitrary span of text."""
    bundle = get_bundle()
    feats = extract_features(text)
    vec = np.array([[feats[name] for name in FEATURE_NAMES]], dtype=float)
    vec_scaled = bundle["scaler"].transform(vec)
    proba = bundle["model"].predict_proba(vec_scaled)[0][1]
    return float(proba), feats


def predict_document(text: str) -> dict:
    ensure_nltk_data()
    if not text or not text.strip():
        return {
            "label": "unknown",
            "ai_probability": 0.0,
            "confidence": 0.0,
            "features": {},
            "sentences": [],
        }

    ai_probability, feats = _score_text(text)

    if ai_probability >= 0.65:
        label = "likely_ai"
    elif ai_probability <= 0.35:
        label = "likely_human"
    else:
        label = "mixed"

    confidence = abs(ai_probability - 0.5) * 2  # 0 (uncertain) -> 1 (very confident)

    sentences = sent_tokenize(text)
    sentence_results = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        try:
            s_prob, _ = _score_text(s_clean)
        except Exception:
            s_prob = ai_probability  # fall back to doc-level score on edge cases
        sentence_results.append({"text": s_clean, "ai_probability": round(s_prob, 4)})

    return {
        "label": label,
        "ai_probability": round(ai_probability, 4),
        "confidence": round(confidence, 4),
        "features": feats,
        "sentences": sentence_results,
        "model_name": get_bundle().get("model_name"),
    }
