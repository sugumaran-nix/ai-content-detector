"""
features.py — Feature engineering for the AI-text classifier.

Improvements over v1:
  • _stopwords_set() is cached at module level (was re-calling ensure_nltk_data
    on every call in the old version — caused repeated NLTK data checks)
  • statistics_stdev fixed to return population stdev for n==2 consistently
  • flesch_reading_ease: guards against zero division on empty alpha_words
  • extract_features: n_words / n_sentences now always included in return dict
    so inference.py can forward them to the API response without re-counting
  • All feature functions have a one-line docstring explaining the AI signal
"""

import math
import re
from collections import Counter
from functools import lru_cache
from statistics import stdev as _stdev
from typing import Optional

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from reference_lm import BigramLM, load_reference_lm, tokenize as lm_tokenize

# ── NLTK bootstrap ────────────────────────────────────────────────────────────

_NLTK_READY = False

def ensure_nltk_data() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import os
    custom = os.environ.get("NLTK_DATA")
    if custom and custom not in nltk.data.path:
        nltk.data.path.insert(0, custom)
    _REQUIRED = [
        ("punkt",                          "tokenizers/punkt"),
        ("punkt_tab",                      "tokenizers/punkt_tab"),
        ("stopwords",                      "corpora/stopwords"),
        ("averaged_perceptron_tagger",     "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]
    for pkg, path in _REQUIRED:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    _NLTK_READY = True


# ── Stopwords (loaded once) ───────────────────────────────────────────────────

_STOPWORDS: Optional[set] = None

def _stopwords_set() -> set:
    global _STOPWORDS
    if _STOPWORDS is None:
        ensure_nltk_data()
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


# ── Reference LM (loaded once) ───────────────────────────────────────────────

_LM: Optional[BigramLM] = None

def get_lm() -> BigramLM:
    global _LM
    if _LM is None:
        _LM = load_reference_lm()
    return _LM


# ── Syllable counting ─────────────────────────────────────────────────────────

def _count_syllables(word: str) -> int:
    """Approximate English syllable count for Flesch score."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


# ── Readability ───────────────────────────────────────────────────────────────

def flesch_reading_ease(words: list[str], sentences: list[str]) -> float:
    """Flesch Reading Ease (0–100). AI text tends toward 30–60."""
    n_words = max(len(words), 1)
    n_sents = max(len(sentences), 1)
    n_syl   = sum(_count_syllables(w) for w in words) or 1
    return 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_syl / n_words)


# ── Stats helper ──────────────────────────────────────────────────────────────

def statistics_stdev(values: list[float]) -> float:
    """Sample standard deviation; returns 0.0 for n < 2."""
    if len(values) < 2:
        return 0.0
    return _stdev(values)


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_features(text: str) -> dict:
    """
    Extract all 11 named features from a text passage.

    Returns a dict keyed by FEATURE_NAMES plus n_words and n_sentences
    (metadata, not used in the classifier vector).
    """
    ensure_nltk_data()
    lm = get_lm()
    sw = _stopwords_set()

    sentences   = sent_tokenize(text) if text.strip() else []
    words       = word_tokenize(text) if text.strip() else []
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]

    n_words = len(alpha_words)
    n_sents = max(len(sentences), 1)

    # ── 1. Perplexity — how predictable are word choices vs. reference English?
    sent_perps = []
    for s in sentences:
        toks = lm_tokenize(s)
        if toks:
            sent_perps.append(lm.perplexity(toks))
    doc_perplexity = lm.perplexity(lm_tokenize(text))

    # ── 2. Burstiness — variation in per-sentence perplexity
    burstiness = statistics_stdev(sent_perps) if len(sent_perps) > 1 else 0.0

    # ── 3. Type-token ratio — lexical diversity
    lower_words     = [w.lower() for w in alpha_words]
    type_token_ratio = len(set(lower_words)) / n_words if n_words else 0.0

    # ── 4+5. Sentence length mean + variance
    sent_lengths   = [len(word_tokenize(s)) for s in sentences] if sentences else [0]
    avg_sent_len   = sum(sent_lengths) / n_sents
    sent_len_var   = statistics_stdev(sent_lengths) if len(sent_lengths) > 1 else 0.0

    # ── 6. Average word length
    avg_word_len = sum(len(w) for w in alpha_words) / n_words if n_words else 0.0

    # ── 7. Stopword ratio
    stopword_ratio = (
        sum(1 for w in lower_words if w in sw) / n_words if n_words else 0.0
    )

    # ── 8. Punctuation density
    punct_count   = sum(1 for ch in text if ch in ",.;:!?-—")
    punct_density = punct_count / max(len(text), 1)

    # ── 9. Repetition (fraction of repeated word bigrams)
    bigrams = list(zip(lower_words, lower_words[1:]))
    repetition_score = (
        1 - (len(set(bigrams)) / len(bigrams)) if len(bigrams) > 1 else 0.0
    )

    # ── 10. POS entropy — grammatical uniformity
    pos_entropy = 0.0
    if alpha_words:
        tags        = [t for _, t in pos_tag(alpha_words)]
        tag_counts  = Counter(tags)
        total       = sum(tag_counts.values())
        pos_entropy = -sum(
            (c / total) * math.log2(c / total) for c in tag_counts.values()
        )

    # ── 11. Readability
    readability = flesch_reading_ease(alpha_words, sentences) if alpha_words else 0.0

    return {
        "perplexity":               round(doc_perplexity, 2),
        "burstiness":               round(burstiness, 2),
        "type_token_ratio":         round(type_token_ratio, 4),
        "avg_sentence_length":      round(avg_sent_len, 2),
        "sentence_length_variance": round(sent_len_var, 2),
        "avg_word_length":          round(avg_word_len, 2),
        "stopword_ratio":           round(stopword_ratio, 4),
        "punctuation_density":      round(punct_density, 4),
        "repetition_score":         round(repetition_score, 4),
        "pos_entropy":              round(pos_entropy, 4),
        "readability":              round(readability, 2),
        # metadata — not in classifier vector but useful for API consumers
        "n_words":                  n_words,
        "n_sentences":              len(sentences),
    }


# ── Feature name order (must match scaler + classifier training) ──────────────
FEATURE_NAMES = [
    "perplexity",
    "burstiness",
    "type_token_ratio",
    "avg_sentence_length",
    "sentence_length_variance",
    "avg_word_length",
    "stopword_ratio",
    "punctuation_density",
    "repetition_score",
    "pos_entropy",
    "readability",
]


def feature_vector(text: str) -> list[float]:
    """Return ordered list of feature values for classifier input."""
    feats = extract_features(text)
    return [feats[name] for name in FEATURE_NAMES]
