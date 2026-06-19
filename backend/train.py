"""
Trains the AI-vs-human text classifier on the handcrafted feature set.

Compares Logistic Regression, Random Forest, and Linear SVM (consistent with
the multi-model comparison approach used in the fake-job-posting-ml project),
picks the best performer on a held-out test split, and saves the fitted
scaler + model + feature names for use by the FastAPI inference service.
"""

import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_NAMES, extract_features  # noqa: E402

DATA_PATH = Path(__file__).parent / "data" / "data.csv"
MODEL_DIR = Path(__file__).parent / "model"


def load_dataset():
    texts, labels = [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def build_feature_matrix(texts: list[str]) -> np.ndarray:
    rows = []
    t0 = time.time()
    for i, text in enumerate(texts):
        feats = extract_features(text)
        rows.append([feats[name] for name in FEATURE_NAMES])
        if (i + 1) % 100 == 0:
            print(f"  extracted features for {i + 1}/{len(texts)} ({time.time() - t0:.1f}s)")
    return np.array(rows, dtype=float)


def main():
    print("Loading dataset...")
    texts, labels = load_dataset()
    y = np.array(labels)
    print(f"{len(texts)} examples ({sum(y)} AI / {len(y) - sum(y)} human)")

    print("Extracting features (this calls the bigram LM + POS tagger per example)...")
    X = build_feature_matrix(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
        ),
        "LinearSVC": LinearSVC(class_weight="balanced", max_iter=5000),
    }

    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        # LinearSVC has no predict_proba; use decision_function for AUC instead.
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_test_s)[:, 1]
        else:
            scores = clf.decision_function(X_test_s)
        auc = roc_auc_score(y_test, scores)
        results[name] = (clf, acc, auc)
        print(f"\n=== {name} ===  accuracy={acc:.3f}  auc={auc:.3f}")
        print(classification_report(y_test, preds, target_names=["human", "ai"]))

    best_name = max(results, key=lambda k: results[k][2])
    best_clf, best_acc, best_auc = results[best_name]
    print(f"\nBest model: {best_name} (accuracy={best_acc:.3f}, auc={best_auc:.3f})")

    # Wrap LinearSVC so the inference code can always call a uniform
    # predict_proba-like interface regardless of which model won.
    needs_calibration = not hasattr(best_clf, "predict_proba")
    if needs_calibration:
        from sklearn.calibration import CalibratedClassifierCV
        print("Calibrating LinearSVC for probability outputs...")
        best_clf = CalibratedClassifierCV(best_clf, cv=5)
        best_clf.fit(X_train_s, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "classifier.pkl", "wb") as f:
        pickle.dump(
            {
                "model": best_clf,
                "scaler": scaler,
                "feature_names": FEATURE_NAMES,
                "model_name": best_name,
                "test_accuracy": best_acc,
                "test_auc": best_auc,
            },
            f,
        )
    print(f"Saved classifier bundle to {MODEL_DIR / 'classifier.pkl'}")


if __name__ == "__main__":
    main()
