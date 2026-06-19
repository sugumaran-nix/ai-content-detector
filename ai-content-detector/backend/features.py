"""
Feature engineering for the AI-text classifier.

Each function below returns a single named feature so the full set can be
inspected (and surfaced to the frontend) as a transparent "diagnostic
readout" rather than a black-box score — this mirrors the explainability
approach used in the fake-job-posting-ml project.
"""

import math
import re
from collections import Counter

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from reference_lm import BigramLM, load_reference_lm, tokenize as lm_tokenize

_STOPWORDS = None


def ensure_nltk_data():
    for pkg, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def _stopwords_set():
    global _STOPWORDS
    if _STOPWORDS is None:
        ensure_nltk_data()
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


# Cached reference LM (loaded once per process)
_LM: BigramLM | None = None


def get_lm() -> BigramLM:
    global _LM
    if _LM is None:
        _LM = load_reference_lm()
    return _LM


def _count_syllables(word: str) -> int:
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(words: list[str], sentences: list[str]) -> float:
    n_words = max(len(words), 1)
    n_sents = max(len(sentences), 1)
    n_syllables = sum(_count_syllables(w) for w in words) or 1
    return 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_syllables / n_words)


def extract_features(text: str) -> dict:
    """Document-level (or single-sentence-level) feature extraction."""
    ensure_nltk_data()
    lm = get_lm()
    sw = _stopwords_set()

    sentences = sent_tokenize(text) if text.strip() else []
    words = word_tokenize(text) if text.strip() else []
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]

    n_words = len(alpha_words)
    n_sents = max(len(sentences), 1)

    # --- Perplexity & burstiness ---
    sent_perplexities = []
    for s in sentences:
        toks = lm_tokenize(s)
        if toks:
            sent_perplexities.append(lm.perplexity(toks))
    doc_perplexity = lm.perplexity(lm_tokenize(text))
    burstiness = (
        statistics_stdev(sent_perplexities) if len(sent_perplexities) > 1 else 0.0
    )

    # --- Lexical diversity ---
    lower_words = [w.lower() for w in alpha_words]
    type_token_ratio = len(set(lower_words)) / n_words if n_words else 0.0

    # --- Sentence length stats ---
    sent_lengths = [len(word_tokenize(s)) for s in sentences] if sentences else [0]
    avg_sent_len = sum(sent_lengths) / n_sents
    sent_len_variance = statistics_stdev(sent_lengths) if len(sent_lengths) > 1 else 0.0

    # --- Word-level stats ---
    avg_word_len = (
        sum(len(w) for w in alpha_words) / n_words if n_words else 0.0
    )
    stopword_ratio = (
        sum(1 for w in lower_words if w in sw) / n_words if n_words else 0.0
    )

    # --- Punctuation density ---
    punct_count = sum(1 for ch in text if ch in ",.;:!?-—")
    punct_density = punct_count / max(len(text), 1)

    # --- Repetition: ratio of repeated word bigrams ---
    bigrams = list(zip(lower_words, lower_words[1:]))
    repetition_score = (
        1 - (len(set(bigrams)) / len(bigrams)) if len(bigrams) > 1 else 0.0
    )

    # --- POS tag entropy (distributional uniformity of grammar) ---
    pos_entropy = 0.0
    if alpha_words:
        tags = [t for _, t in pos_tag(alpha_words)]
        tag_counts = Counter(tags)
        total = sum(tag_counts.values())
        pos_entropy = -sum(
            (c / total) * math.log2(c / total) for c in tag_counts.values()
        )

    # --- Readability ---
    readability = flesch_reading_ease(alpha_words, sentences) if alpha_words else 0.0

    return {
        "perplexity": round(doc_perplexity, 2),
        "burstiness": round(burstiness, 2),
        "type_token_ratio": round(type_token_ratio, 4),
        "avg_sentence_length": round(avg_sent_len, 2),
        "sentence_length_variance": round(sent_len_variance, 2),
        "avg_word_length": round(avg_word_len, 2),
        "stopword_ratio": round(stopword_ratio, 4),
        "punctuation_density": round(punct_density, 4),
        "repetition_score": round(repetition_score, 4),
        "pos_entropy": round(pos_entropy, 4),
        "readability": round(readability, 2),
        "n_words": n_words,
        "n_sentences": len(sentences),
    }


FEATURE_NAMES = [
    "perplexity", "burstiness", "type_token_ratio", "avg_sentence_length",
    "sentence_length_variance", "avg_word_length", "stopword_ratio",
    "punctuation_density", "repetition_score", "pos_entropy", "readability",
]


def feature_vector(text: str) -> list[float]:
    feats = extract_features(text)
    return [feats[name] for name in FEATURE_NAMES]


def statistics_stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)
