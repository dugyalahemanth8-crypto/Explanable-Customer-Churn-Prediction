"""
check_model.py — Diagnostic script to verify model health
Run from project root: python src/check_model.py
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_PATH, DATASET_PATH, MODEL_COLUMNS_PATH, SCALER_PATH

print("=" * 70)
print("MODEL DIAGNOSTIC CHECK")
print("=" * 70)

# ── 1. Load model ──────────────────────────────────────────────────────────────
print("\n1. Loading model …")
if not MODEL_PATH.exists():
    print(f"   ❌ Model not found: {MODEL_PATH}")
    print("   Run `python src/train.py` first.")
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print(f"   Model type     : {type(model).__name__}")
print(f"   Features needed: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'unknown'}")

if hasattr(model, "coef_"):
    coef = model.coef_[0]
    print(f"\n2. Coefficients")
    print(f"   Shape          : {coef.shape}")
    print(f"   All zeros?     : {np.all(coef == 0)}")
    print(f"   Max |coef|     : {np.abs(coef).max():.4f}")

if hasattr(model, "intercept_"):
    print(f"   Intercept      : {model.intercept_[0]:.4f}")

# ── 2. Load column names ───────────────────────────────────────────────────────
print("\n3. Column schema …")
if MODEL_COLUMNS_PATH.exists():
    with open(MODEL_COLUMNS_PATH, "rb") as f:
        cols = pickle.load(f)
    print(f"   ✅ {len(cols)} columns loaded from model_columns.pkl")
else:
    print("   ⚠️  model_columns.pkl not found — will derive from dataset")

# ── 3. Load scaler ─────────────────────────────────────────────────────────────
print("\n4. Scaler …")
if SCALER_PATH.exists():
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"   ✅ StandardScaler loaded")
else:
    print("   ⚠️  scaler.pkl not found — predictions will use unscaled data")
    scaler = None

# ── 4. End-to-end prediction test ─────────────────────────────────────────────
print("\n5. End-to-end prediction test on first 10 training rows …")
df = pd.read_csv(DATASET_PATH)
df.drop(columns=["customerID"], inplace=True, errors="ignore")
y_true = df["Churn"].map({"Yes": 1, "No": 0})
df.drop(columns=["Churn"], inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df_enc = pd.get_dummies(df, drop_first=True)

sample = df_enc.head(10)
X = scaler.transform(sample) if scaler is not None else sample.values

try:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    actual = y_true.head(10).values

    print(f"\n   {'#':<4} {'Actual':<8} {'Predicted':<10} {'P(Churn)':<10} {'Correct'}")
    print("   " + "-" * 40)
    for i in range(10):
        tick = "✔" if preds[i] == actual[i] else "✗"
        print(f"   {i+1:<4} {'Churn' if actual[i] else 'Stay':<8} "
              f"{'Churn' if preds[i] else 'Stay':<10} "
              f"{probs[i]:.3f}{'':4} {tick}")

    acc = (preds == actual).mean()
    print(f"\n   Accuracy on these 10: {acc:.0%}")

    if np.all(preds == preds[0]):
        print("\n   ❌ WARNING: Model predicts the same class for all customers!")
    else:
        print("\n   ✅ Model produces varied predictions — looks healthy.")

except Exception as exc:
    print(f"   ❌ Prediction error: {exc}")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)