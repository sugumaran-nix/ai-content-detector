"""
export_weights.py — Export classifier weights to model.json for browser inference.

Run after train.py:
    python export_weights.py

Outputs:
    ../model.json

Format consumed by index.html predict():
{
  "featureNames": [...],
  "scalerMean":   [...],
  "scalerStd":    [...],
  "calibrated": [
    {"coef": [...], "intercept": 0.0, "cal_a": 0.0, "cal_b": 0.0},
    ...  (one entry per CV fold)
  ]
}
"""

import json
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "classifier.pkl"
OUT_PATH   = Path(__file__).parent.parent / "model.json"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found — run train.py first.")

    print(f"Loading {MODEL_PATH}…")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    clf     = bundle["model"]       # CalibratedClassifierCV
    scaler  = bundle["scaler"]      # StandardScaler
    names   = bundle["feature_names"]

    calibrated = []
    for cc in clf.calibrated_classifiers_:
        svc = cc.estimator                  # fitted LinearSVC
        cal = cc.calibrators[0]             # _SigmoidCalibration (binary: one per class)
        calibrated.append({
            "coef":      svc.coef_[0].tolist(),
            "intercept": float(svc.intercept_[0]),
            "cal_a":     float(cal.a_),
            "cal_b":     float(cal.b_),
        })

    payload = {
        "featureNames": names,
        "scalerMean":   scaler.mean_.tolist(),
        "scalerStd":    scaler.scale_.tolist(),
        "calibrated":   calibrated,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Done → {OUT_PATH}  ({size_kb:.1f} KB)")
    print(f"Folds exported: {len(calibrated)}")


if __name__ == "__main__":
    main()
