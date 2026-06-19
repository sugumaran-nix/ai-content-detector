"""
Assembles the starter training dataset: data.csv with columns [text, label]
(label 1 = AI-style, 0 = human-written).

Human side: real, public-domain-licensed paragraphs from NLTK's Brown corpus,
sampled across all 15 genres for stylistic diversity (news, fiction,
editorial, science, humor, etc.) and detokenized into natural-looking prose.

AI side: synthetic paragraphs from ai_samples.py (see that file's docstring
for the honest limitation this implies).
"""

import csv
import random
import sys
from pathlib import Path

import nltk
from nltk.corpus import brown
from nltk.tokenize.treebank import TreebankWordDetokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference_lm import brown_fileid_split  # noqa: E402
from ai_samples import generate_ai_paragraphs  # noqa: E402

OUT_PATH = Path(__file__).parent / "data.csv"
SENTENCES_PER_CHUNK = (4, 7)  # inclusive range, randomly chosen per chunk
N_PER_CLASS = 450


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


def build_human_paragraphs(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    det = TreebankWordDetokenizer()
    _lm_fileids, pool_fileids = brown_fileid_split()
    categories_seen = {}
    for fid in pool_fileids:
        cat = brown.categories(fid)[0]
        categories_seen.setdefault(cat, []).append(fid)

    cat_cycle = list(categories_seen.keys())
    paragraphs = []

    while len(paragraphs) < n:
        rng.shuffle(cat_cycle)
        for cat in cat_cycle:
            if len(paragraphs) >= n:
                break
            fid = rng.choice(categories_seen[cat])
            sents = brown.sents(fid)
            if len(sents) < 6:
                continue
            chunk_len = rng.randint(*SENTENCES_PER_CHUNK)
            start = rng.randint(0, max(0, len(sents) - chunk_len - 1))
            chunk = sents[start:start + chunk_len]
            text = " ".join(det.detokenize(s) for s in chunk)
            if len(text.split()) >= 25:  # skip too-short chunks
                paragraphs.append(text)

    return paragraphs[:n]


def main():
    ensure_nltk_data()

    human = build_human_paragraphs(N_PER_CLASS)
    ai = generate_ai_paragraphs(N_PER_CLASS)

    rows = [(t, 0) for t in human] + [(t, 1) for t in ai]
    random.Random(123).shuffle(rows)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows ({len(human)} human / {len(ai)} ai) to {OUT_PATH}")


if __name__ == "__main__":
    main()
