"""
Export the trained bigram language model from reference_lm.pkl → ../lm.json

Run this once after training (or whenever you retrain reference_lm.py):
    cd backend
    python export_lm_json.py

The output lm.json is loaded by index.html at page load to enable the
perplexity and burstiness signals in the browser. Without it, those two
signals are silently disabled and inference is less accurate.

lm.json format expected by index.html:
  {
    "u": { word: count, ... },          -- unigram counts
    "b": { prev: { cur: count }, ... }, -- bigram counts
    "v": vocab_size                     -- int
  }
"""

import json
import pickle  # nosec B403 - reads only the trusted repository build artifact
import sys
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "reference_lm.pkl"
OUT_PATH = Path(__file__).parent.parent / "lm.json"


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        print("Run `python reference_lm.py` first to train and save the LM.")
        sys.exit(1)

    print(f"Loading {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        lm = pickle.load(f)  # nosec B301 - trusted artifact conversion, never user-supplied

    print(f"Vocab size : {lm.vocab_size}")
    print(f"Unigrams   : {len(lm.unigram_counts)}")
    print(f"Bigram keys: {len(lm.bigram_counts)}")

    payload = {
        "u": dict(lm.unigram_counts),
        "b": {prev: dict(nexts) for prev, nexts in lm.bigram_counts.items()},
        "v": lm.vocab_size,
    }

    print(f"Writing {OUT_PATH} ...")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"Done. lm.json = {size_mb:.1f} MB")
    print(f"Commit lm.json to the repo root before deploying index.html.")


if __name__ == "__main__":
    main()
