"""
view_predictions.py — Batch prediction viewer
Run from project root: python src/view_predictions.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predict import batch_predict
from src.config  import DATASET_PATH

print("=" * 70)
print("CUSTOMER CHURN PREDICTIONS — BATCH VIEW")
print("=" * 70)

# Load full dataset (without target)
df = pd.read_csv(DATASET_PATH)
customer_ids = df["customerID"].values if "customerID" in df.columns else None
df.drop(columns=["customerID", "Churn"], inplace=True, errors="ignore")

# Make predictions
print(f"\nRunning predictions on {len(df):,} customers …")
results = batch_predict(df)

if customer_ids is not None:
    results.insert(0, "customerID", customer_ids)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("RISK DISTRIBUTION")
print(f"{'='*70}")
risk_counts = results["Risk_Level"].value_counts()
for level in ["High", "Medium", "Low"]:
    count = risk_counts.get(level, 0)
    pct   = count / len(results) * 100
    print(f"  {level:8} risk: {count:5,} customers  ({pct:.1f}%)")

# ── First 20 ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FIRST 20 CUSTOMERS")
print(f"{'='*70}")
display_cols = ["customerID"] if "customerID" in results.columns else []
display_cols += ["Predicted_Churn", "Churn_Probability", "Risk_Level",
                 "tenure", "Contract", "MonthlyCharges"]
print(results[display_cols].head(20).to_string(index=False))

# ── High risk ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("HIGH-RISK CUSTOMERS (Churn Probability > 70%)")
print(f"{'='*70}")
high_risk = results[results["Churn_Probability"] > 0.70]
print(high_risk[display_cols].head(20).to_string(index=False))
print(f"\nTotal high-risk: {len(high_risk):,}")

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = DATASET_PATH.parent / "customer_predictions.csv"
results.to_csv(out_path, index=False)
print(f"\n✅  Full predictions saved → {out_path}")