"""
train.py — Retrain the AI-text classifier on HC3 data.

HC3 (Hello-SimpleAI/HC3) contains real ChatGPT responses paired with
real human answers from Reddit / StackExchange. This replaces the original
Brown corpus + synthetic AI paragraph approach, which caused the classifier
to mislabel modern AI text as human.

Outputs:
    model/classifier.pkl
"""

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, feature_vector

MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)


def build_dataset(n_per_class: int = 2000) -> tuple[list[str], list[int]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    import random
    random.seed(42)

    print("Downloading HC3…")
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")

    human_texts, ai_texts = [], []

    for row in ds:
        for ans in row["human_answers"]:
            if ans and len(ans.split()) >= 40:
                human_texts.append(ans.strip())
        for ans in row["chatgpt_answers"]:
            if ans and len(ans.split()) >= 40:
                ai_texts.append(ans.strip())

    n = min(n_per_class, len(human_texts), len(ai_texts))
    human_texts = random.sample(human_texts, n)
    ai_texts    = random.sample(ai_texts,    n)

    texts  = human_texts + ai_texts
    labels = [0] * n + [1] * n
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    print(f"Dataset: {n} human + {n} AI = {2 * n} total")
    return list(texts), list(labels)


def extract_all_features(texts: list[str]) -> np.ndarray:
    print(f"Extracting features for {len(texts)} documents…")
    rows = []
    for i, text in enumerate(texts, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(texts)}")
        rows.append(feature_vector(text))
    return np.array(rows, dtype=float)


def main():
    t0 = time.time()

    texts, labels = build_dataset()
    X = extract_all_features(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train)
    Xte_sc  = scaler.transform(X_test)

    print("Training LinearSVC with Platt calibration…")
    clf = CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=5)
    clf.fit(Xtr_sc, y_train)

    y_pred  = clf.predict(Xte_sc)
    y_proba = clf.predict_proba(Xte_sc)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test accuracy: {acc:.4f}   AUC: {auc:.4f}")

    bundle = {
        "model":          clf,
        "scaler":         scaler,
        "feature_names":  FEATURE_NAMES,
        "model_name":     "LinearSVC (Platt)",
        "test_accuracy":  round(acc, 4),
        "test_auc":       round(auc, 4),
    }

    out = MODEL_DIR / "classifier.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved → {out}  ({out.stat().st_size // 1024} KB)")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
