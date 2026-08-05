"""
train.py — Build and evaluate the AI-text classifier.

Pipeline:
  1. Build dataset (AI samples + Brown corpus human samples)
  2. Extract 11 features per document
  3. Compare LR, RF, LinearSVC via 5-fold stratified CV
  4. Refit winner on full train set, evaluate on held-out test set
  5. Serialize (scaler + model + metadata) to model/classifier.pkl

Usage:
  python train.py
"""

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, feature_vector

MODEL_DIR = Path(__file__).parent / "model"
DATA_DIR  = Path(__file__).parent / "data"

CANDIDATES = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "LinearSVC":          CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=5),
}


def build_dataset():
    """Return (texts, labels) where label=1 → AI, label=0 → Human."""
    print("[data] Generating AI samples…")
    from data.ai_samples import generate_ai_paragraphs
    ai_texts = generate_ai_paragraphs(450, seed=42)

    print("[data] Loading human samples from Brown corpus…")
    import nltk
    from nltk.corpus import brown

    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)

    # Use 30% of Brown corpus for training, 70% was used to build the LM
    # to prevent verbatim memorisation bleeding into perplexity signal.
    sents = brown.sents()
    n = len(sents)
    split = int(n * 0.70)
    human_pool = sents[split:]

    human_texts = []
    buf = []
    for sent in human_pool:
        buf.extend(sent)
        if len(buf) >= 60:
            human_texts.append(" ".join(buf))
            buf = []
            if len(human_texts) >= 450:
                break

    if len(human_texts) < 450:
        raise RuntimeError("Not enough Brown corpus sentences for 450 human samples.")

    texts  = ai_texts + human_texts
    labels = [1] * len(ai_texts) + [0] * len(human_texts)
    print(f"[data] Dataset: {len(ai_texts)} AI, {len(human_texts)} human")
    return texts, labels


def extract_all_features(texts: list[str]) -> np.ndarray:
    print(f"[features] Extracting features for {len(texts)} documents…")
    rows = []
    for i, text in enumerate(texts, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(texts)}")
        rows.append(feature_vector(text))
    return np.array(rows, dtype=float)


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)

    texts, labels = build_dataset()
    X = extract_all_features(texts)
    y = np.array(labels)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardise
    scaler   = StandardScaler()
    Xtr_sc   = scaler.fit_transform(X_train)
    Xte_sc   = scaler.transform(X_test)

    # 5-fold CV to choose best model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n[train] 5-fold cross-validation:")
    best_name, best_auc, best_clf = None, -1, None
    for name, clf in CANDIDATES.items():
        aucs = cross_val_score(clf, Xtr_sc, y_train, cv=cv, scoring="roc_auc")
        print(f"  {name:<22} AUC = {aucs.mean():.4f} ± {aucs.std():.4f}")
        if aucs.mean() > best_auc:
            best_auc  = aucs.mean()
            best_name = name
            best_clf  = clf

    print(f"\n[train] Winner: {best_name}  (CV AUC = {best_auc:.4f})")

    # Refit on full training set, evaluate on held-out test
    best_clf.fit(Xtr_sc, y_train)
    y_pred  = best_clf.predict(Xte_sc)
    y_proba = best_clf.predict_proba(Xte_sc)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    print(f"[eval] Test accuracy = {acc:.4f}  AUC = {auc:.4f}")

    # Persist
    bundle = {
        "model":         best_clf,
        "scaler":        scaler,
        "feature_names": FEATURE_NAMES,
        "model_name":    best_name,
        "test_accuracy": round(acc, 4),
        "test_auc":      round(auc, 4),
    }
    out = MODEL_DIR / "classifier.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[save] Saved to {out}  ({out.stat().st_size // 1024} KB)")
    print(f"[done] Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
