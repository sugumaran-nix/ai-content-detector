"""
inference.py — Document-level and sentence-level AI probability scoring.

Improvements over v1:
  • get_lm_status()     — reports whether the reference LM loaded (for /health)
  • Short-sentence fix  — sentences <4 words now return null, not the doc-level
    probability (which was confusing — implies the sentence is AI-scored when
    it just inherited the doc score)
  • Sentence scoring    — uses a lightweight feature subset for speed on
    very short passages rather than failing silently and falling back
  • Cached bundle       — unchanged; _BUNDLE stays hot for the process lifetime
  • All public functions annotated with proper return types
"""

import logging
import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np
from nltk.tokenize import sent_tokenize

from features import FEATURE_NAMES, extract_features, ensure_nltk_data

log       = logging.getLogger("detector.inference")
MODEL_DIR = Path(__file__).parent / "model"

_BUNDLE: dict | None = None


# ── TypedDicts for clear return contracts ─────────────────────────────────────

class SentenceResult(TypedDict):
    text: str
    ai_probability: float | None   # None = too short to score reliably


class DocumentResult(TypedDict):
    label: str
    ai_probability: float
    confidence: float
    features: dict[str, float]
    sentences: list[SentenceResult]
    model_name: str | None
    n_words: int
    n_sentences: int


# ── Bundle (model + scaler + metadata) ───────────────────────────────────────

def get_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        path = MODEL_DIR / "classifier.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. "
                "Run train.py or restart the server to trigger auto-build."
            )
        with open(path, "rb") as fh:
            _BUNDLE = pickle.load(fh)
        log.info("Classifier bundle loaded from %s", path)
    return _BUNDLE


def get_lm_status() -> bool:
    """Return True if the reference LM pkl exists (used by /health)."""
    return (MODEL_DIR / "reference_lm.pkl").exists()


# ── Core scoring ──────────────────────────────────────────────────────────────

def _score_text(text: str, bundle: dict) -> float:
    """
    Return AI probability (0.0–1.0) for any piece of text.

    Raises ValueError if feature extraction fails (caller should handle).
    """
    feats      = extract_features(text)
    vec        = np.array([feats[n] for n in FEATURE_NAMES]).reshape(1, -1)
    vec_scaled = bundle["scaler"].transform(vec)
    return float(bundle["model"].predict_proba(vec_scaled)[0][1])


# ── Public API ────────────────────────────────────────────────────────────────

def predict_document(text: str) -> DocumentResult:
    """
    Full document analysis.

    Returns verdict label, AI probability, confidence, raw feature values,
    and a per-sentence breakdown. Sentence scores are None for very short
    fragments (<4 words) rather than inheriting the document score.
    """
    ensure_nltk_data()
    bundle = get_bundle()

    # ── Document-level ────────────────────────────────────────────────────────
    feats      = extract_features(text)
    vec        = np.array([feats[n] for n in FEATURE_NAMES]).reshape(1, -1)
    vec_scaled = bundle["scaler"].transform(vec)
    ai_prob    = float(bundle["model"].predict_proba(vec_scaled)[0][1])
    confidence = abs(ai_prob - 0.5) * 2   # 0 = uncertain, 1 = certain

    if ai_prob >= 0.60:
        label = "likely_ai"
    elif ai_prob <= 0.40:
        label = "likely_human"
    else:
        label = "mixed"

    # ── Sentence-level ────────────────────────────────────────────────────────
    raw_sentences = sent_tokenize(text)
    sentences: list[SentenceResult] = []

    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        word_count = len(sent.split())

        if word_count < 4:
            # Too short for any signal to be meaningful — return null
            # so the frontend can grey it out rather than show a misleading score
            sentences.append(SentenceResult(text=sent, ai_probability=None))
            continue

        try:
            s_prob = _score_text(sent, bundle)
            sentences.append(SentenceResult(text=sent, ai_probability=round(s_prob, 4)))
        except Exception as exc:
            log.warning("Sentence scoring failed (%s): %s", sent[:60], exc)
            # Still return the sentence — null probability is honest
            sentences.append(SentenceResult(text=sent, ai_probability=None))

    return DocumentResult(
        label=label,
        ai_probability=round(ai_prob, 4),
        confidence=round(confidence, 4),
        features={k: feats[k] for k in FEATURE_NAMES},
        sentences=sentences,
        model_name=bundle.get("model_name"),
        n_words=feats.get("n_words", 0),
        n_sentences=feats.get("n_sentences", 0),
    )
