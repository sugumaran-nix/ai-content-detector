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
import pickle  # nosec B403 - loads only trusted build artifacts
from pathlib import Path
from typing import TypedDict

import numpy as np
from nltk.tokenize import sent_tokenize

from features import FEATURE_NAMES, extract_features, ensure_nltk_data, get_lm
from reference_lm import tokenize as lm_tokenize

log       = logging.getLogger("detector.inference")
MODEL_DIR = Path(__file__).parent / "model"

_BUNDLE: dict | None = None
AI_THRESHOLD = 0.70
HUMAN_THRESHOLD = 0.30
OOD_START_Z = 4.0
OOD_FULL_Z = 8.0


def calibrate_probability(raw_probability: float, features: dict, bundle: dict, unknown_ratio: float = 0.0) -> tuple[float, float]:
    """Shrink overconfident scores when feature values or vocabulary are far OOD."""
    values = np.array([features[name] for name in FEATURE_NAMES], dtype=float)
    z_scores = np.abs((values - bundle["scaler"].mean_) / bundle["scaler"].scale_)
    max_z = float(np.max(z_scores)) if len(z_scores) else 0.0
    feature_penalty = min(1.0, max(0.0, (max_z - OOD_START_Z) / OOD_FULL_Z))
    domain_penalty = 0.75 if unknown_ratio >= 0.075 and (raw_probability <= 0.15 or raw_probability >= 0.85) else 0.0
    penalty = max(feature_penalty, domain_penalty)
    calibrated = 0.5 + (raw_probability - 0.5) * (1.0 - penalty)
    return round(float(min(1.0, max(0.0, calibrated))), 4), round(max_z, 4)


def label_for_probability(ai_prob: float) -> str:
    """Map calibrated probability to the public three-band verdict contract."""
    if ai_prob >= AI_THRESHOLD:
        return "likely_ai"
    if ai_prob <= HUMAN_THRESHOLD:
        return "likely_human"
    return "mixed"


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

def _validate_bundle(bundle: object) -> dict:
    """Validate the trusted model artifact before it enters the hot cache."""
    if not isinstance(bundle, dict):
        raise RuntimeError("Classifier artifact has an invalid shape.")
    required = {"model", "scaler", "feature_names"}
    missing = required.difference(bundle)
    if missing:
        raise RuntimeError("Classifier artifact is missing required metadata.")
    if list(bundle["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("Classifier feature order does not match the runtime contract.")
    if not hasattr(bundle["model"], "predict_proba") or not hasattr(bundle["scaler"], "transform"):
        raise RuntimeError("Classifier artifact does not expose the required inference methods.")
    return bundle


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
            loaded = pickle.load(fh)  # nosec B301 - artifact is produced by the trusted build
        _BUNDLE = _validate_bundle(loaded)
        log.info("Classifier bundle loaded from %s", path)
    return _BUNDLE


def get_lm_status() -> bool:
    """Return True when the reference LM is present and loaded in memory."""
    return get_lm() is not None


def warmup() -> None:
    """Load all runtime assets so readiness means the first request is warm."""
    ensure_nltk_data()
    get_bundle()
    get_lm()


# ── Core scoring ──────────────────────────────────────────────────────────────

def _unknown_ratio(text: str) -> float:
    tokens = lm_tokenize(text)
    vocab = get_lm().unigram_counts
    return sum(1 for token in tokens if token not in vocab) / max(len(tokens), 1)


def _predict_core(text: str, bundle: dict) -> tuple[dict, dict]:
    """Return calibrated document metadata and extracted features."""
    feats = extract_features(text)
    vec = np.array([feats[n] for n in FEATURE_NAMES]).reshape(1, -1)
    vec_scaled = bundle["scaler"].transform(vec)
    raw_probability = float(bundle["model"].predict_proba(vec_scaled)[0][1])
    ai_prob, max_feature_z = calibrate_probability(
        raw_probability, feats, bundle, _unknown_ratio(text)
    )
    confidence = abs(ai_prob - 0.5) * 2
    return {
        "label": label_for_probability(ai_prob),
        "ai_probability": round(ai_prob, 4),
        "confidence": round(confidence, 4),
        "max_feature_z": round(max_feature_z, 4),
    }, feats


def _score_text(text: str, bundle: dict) -> float:
    """Return a calibrated AI probability for a sentence or short passage."""
    result, _ = _predict_core(text, bundle)
    return result["ai_probability"]


def predict_lite(text: str) -> dict:
    """Return only document-level fields for fast API consumers."""
    ensure_nltk_data()
    bundle = get_bundle()
    result, _ = _predict_core(text, bundle)
    return {
        "label": result["label"],
        "ai_probability": result["ai_probability"],
        "confidence": result["confidence"],
        "model_name": bundle.get("model_name"),
    }


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
    core, feats = _predict_core(text, bundle)
    ai_prob = core["ai_probability"]
    confidence = core["confidence"]
    label = core["label"]

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
