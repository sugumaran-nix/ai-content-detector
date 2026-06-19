"""
A small bigram language model (Laplace/add-one smoothing) trained on a
held-out slice of NLTK's Brown corpus, used to compute a perplexity proxy
for arbitrary input text.

Why a bigram LM instead of a neural model: this keeps the whole pipeline
CPU-only and deployable on free-tier hosting (no torch/transformers runtime),
consistent with the rest of the classical-NLP feature set. It's a proxy for
"how predictable is this text under typical English word-pair statistics" —
not a true measure of model-internal perplexity the way GPT-2 perplexity is,
but it captures the same underlying signal (AI text tends to be more
locally predictable / lower perplexity than human text, which has more
idiosyncratic word choice).

Train/inference leakage note: the LM is built from a *different* 70% slice
of Brown fileids than the 30% slice used to sample human-class paragraphs
in data/build_dataset.py, so the model isn't scoring verbatim text it was
built from.
"""

import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize

MODEL_PATH = Path(__file__).parent / "model" / "reference_lm.pkl"
SPLIT_SEED = 99
LM_FRACTION = 0.7  # fraction of Brown fileids used to build the LM


def ensure_nltk_data():
    for pkg, path in [
        ("brown", "corpora/brown"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def brown_fileid_split(seed: int = SPLIT_SEED, lm_fraction: float = LM_FRACTION):
    from nltk.corpus import brown

    fileids = list(brown.fileids())
    rng = random.Random(seed)
    rng.shuffle(fileids)
    cut = int(len(fileids) * lm_fraction)
    return fileids[:cut], fileids[cut:]  # (lm_fileids, pool_fileids)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in word_tokenize(text) if any(c.isalnum() for c in t)]


class BigramLM:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()

    def fit(self, token_lists: list[list[str]]):
        for tokens in token_lists:
            padded = ["<s>"] + tokens + ["</s>"]
            self.vocab.update(padded)
            for w in padded:
                self.unigram_counts[w] += 1
            for i in range(len(padded) - 1):
                self.bigram_counts[padded[i]][padded[i + 1]] += 1
        self.vocab_size = len(self.vocab)

    def _bigram_logprob(self, prev: str, cur: str) -> float:
        import math
        prev_count = self.unigram_counts.get(prev, 0)
        pair_count = self.bigram_counts.get(prev, {}).get(cur, 0)
        prob = (pair_count + 1) / (prev_count + self.vocab_size)
        return math.log(prob)

    def perplexity(self, tokens: list[str]) -> float:
        import math
        if not tokens:
            return 0.0
        padded = ["<s>"] + tokens + ["</s>"]
        total_logprob = 0.0
        n = len(padded) - 1
        for i in range(n):
            total_logprob += self._bigram_logprob(padded[i], padded[i + 1])
        avg_neg_logprob = -total_logprob / n
        return math.exp(min(avg_neg_logprob, 20))  # cap to avoid overflow on pathological input


def build_reference_lm() -> BigramLM:
    ensure_nltk_data()
    from nltk.corpus import brown

    lm_fileids, _pool_fileids = brown_fileid_split()
    token_lists = []
    for fid in lm_fileids:
        for sent in brown.sents(fid):
            token_lists.append([w.lower() for w in sent if any(c.isalnum() for c in w)])

    lm = BigramLM()
    lm.fit(token_lists)
    return lm


def save_reference_lm(lm: BigramLM, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(lm, f)


def load_reference_lm(path: Path = MODEL_PATH) -> BigramLM:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Building reference bigram LM from held-out Brown corpus split...")
    lm = build_reference_lm()
    save_reference_lm(lm)
    print(f"Vocab size: {lm.vocab_size}. Saved to {MODEL_PATH}")

    # quick sanity check
    test_human = tokenize("The committee deliberated for several hours before reaching a decision.")
    test_ai = tokenize("It's important to note that this topic offers a number of clear benefits.")
    print("Perplexity (human-like):", lm.perplexity(test_human))
    print("Perplexity (ai-like):", lm.perplexity(test_ai))
