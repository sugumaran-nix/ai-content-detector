"""
reference_lm.py — Bigram language model trained on HC3 human answers.

Replaces the original Brown corpus baseline with real modern human text
from the HC3 dataset (Reddit / StackExchange style answers).  This fixes
the perplexity signal: modern AI text should score differently from
modern human text, rather than both looking alien to a 1960s corpus.

Usage:
    python reference_lm.py
Outputs:
    model/reference_lm.pkl
"""

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "reference_lm.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)


# ── Tokenizer (shared with features.py and export_lm_json.py) ────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


# ── Bigram LM ─────────────────────────────────────────────────────────────────

class BigramLM:
    def __init__(self):
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: dict[str, Counter] = defaultdict(Counter)
        self.vocab_size: int = 0

    def fit(self, token_lists: list[list[str]]) -> "BigramLM":
        for tokens in token_lists:
            self.unigram_counts.update(tokens)
            for a, b in zip(tokens, tokens[1:]):
                self.bigram_counts[a][b] += 1
        self.vocab_size = len(self.unigram_counts)
        return self

    def perplexity(self, tokens: list[str]) -> float:
        if len(tokens) < 2:
            return float("inf")
        total_uni = sum(self.unigram_counts.values())
        log_prob = 0.0
        for a, b in zip(tokens, tokens[1:]):
            uni = self.unigram_counts.get(a, 0)
            bi  = self.bigram_counts.get(a, {}).get(b, 0)
            # Add-1 (Laplace) smoothing
            p = (bi + 1) / (uni + self.vocab_size + 1) if uni else 1 / (total_uni + 1)
            log_prob += math.log2(p)
        n = len(tokens) - 1
        return 2 ** (-log_prob / n)


def load_reference_lm() -> BigramLM:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Build from HC3 human answers ──────────────────────────────────────────────

def build_reference_lm() -> BigramLM:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("Downloading HC3 dataset (human answers)…")
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")

    token_lists = []
    for row in ds:
        for ans in row["human_answers"]:
            if not ans:
                continue
            toks = tokenize(ans)
            if len(toks) >= 20:
                token_lists.append(toks)

    print(f"Training bigram LM on {len(token_lists)} human passages…")
    lm = BigramLM().fit(token_lists)
    print(f"Vocab size: {lm.vocab_size}")
    return lm


if __name__ == "__main__":
    lm = build_reference_lm()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(lm, f)
    print(f"Saved → {MODEL_PATH}")
