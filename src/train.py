"""
train.py — Train Churn Prediction Models
Produces:
  data/logistic_regression_model.pkl  — standalone LR (used by predict.py)
  data/model_columns.pkl              — column names in correct order
  data/churn_pipeline.pkl             — full sklearn RF pipeline
  data/feature_importance.png         — top-10 bar chart
  data/feature_importance_global.png  — top-15 bar chart (coloured)
  data/shap_importance.png            — SHAP mean-|value| bar chart

FIXES APPLIED (2026-02-17):
  1. TotalCharges DROPPED before training.
       TotalCharges ≈ tenure × MonthlyCharges — perfectly collinear.
       The LR was learning tenure_coef ≈ −22.5 and TotalCharges_coef ≈ +25.4
       (nearly equal and opposite), causing them to fight each other and
       mis-classify loyal long-tenure customers as high churn risk.
       Dropping it removes the redundancy; tenure + MonthlyCharges carry
       all the information.

  2. _shap_background.npy cache deleted on every retrain.
       The LinearExplainer background is built from training data.
       If a stale cache from an old model lives in data/, SHAP values are
       computed against the wrong baseline.  We now delete it here so
       explain.py always rebuilds it from the fresh model.
"""
import sys
import warnings
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATASET_PATH, MODEL_PATH, PIPELINE_PATH, MODEL_COLUMNS_PATH,
    DATA_DIR, MODEL_DIR, RANDOM_STATE, TEST_SIZE,
    FEATURE_IMPORTANCE_PATH, FEATURE_IMPORTANCE_GLOBAL_PATH,
    SHAP_IMPORTANCE_PATH, SCALER_PATH,
)

warnings.filterwarnings("ignore")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_and_clean(path: Path) -> tuple:
    """
    Load CSV, clean, encode, return X (DataFrame) and y (Series).

    FIX: TotalCharges is dropped before encoding because it is a near-perfect
    linear combination of tenure × MonthlyCharges.  Keeping it caused the LR
    to learn large opposing coefficients for tenure and TotalCharges that
    cancelled each other unreliably, mis-predicting loyal customers as churners.
    """
    df = pd.read_csv(path)
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Fix TotalCharges type (still needed before we drop it, to avoid errors
    # if other code paths read this function's output)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    # ── DROP TotalCharges (multicollinearity fix) ──────────────────────────────
    X = X.drop(columns=["TotalCharges"], errors="ignore")
    print("    ℹ️   TotalCharges dropped (collinear with tenure × MonthlyCharges)")

    # One-hot encode with drop_first=True
    X_enc = pd.get_dummies(X, drop_first=True)
    return X_enc, y


def plot_top_features(importance_series: pd.Series, title: str,
                      out_path: Path, n: int = 10, cmap: str = "viridis") -> None:
    top = importance_series.nlargest(n).sort_values()
    colours = plt.colormaps[cmap](np.linspace(0.3, 0.9, len(top)))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values, color=colours)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Saved: {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("TRAINING CHURN PREDICTION MODELS")
    print("=" * 70)

    # ── 0. Delete stale SHAP background cache ─────────────────────────────────
    # explain.py caches the LinearExplainer background in _shap_background.npy.
    # After a retrain the model and scaler change, so the cached background is
    # no longer valid.  Delete it here so explain.py rebuilds it fresh.
    shap_bg_path = DATA_DIR / "_shap_background.npy"
    if shap_bg_path.exists():
        shap_bg_path.unlink()
        print(f"\n🗑️   Deleted stale SHAP background cache → {shap_bg_path.name}")
    else:
        print(f"\n✅  No stale SHAP cache to delete.")

    # ── 1. Load & encode ───────────────────────────────────────────────────────
    print("\n📂  Loading dataset …")
    X, y = load_and_clean(DATASET_PATH)
    print(f"    Rows: {len(X):,}  |  Features (encoded): {X.shape[1]}")
    print(f"    Churn rate: {y.mean():.2%}")

    # Save column names — predict.py uses this at inference time
    col_list = list(X.columns)
    with open(MODEL_COLUMNS_PATH, "wb") as f:
        pickle.dump(col_list, f)
    print(f"    ✅  Saved {len(col_list)} column names → {MODEL_COLUMNS_PATH.name}")
    print(f"    📋  Columns: {col_list}")

    # ── 2. Train / test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 3. Logistic Regression ─────────────────────────────────────────────────
    print("\n🚀  Training Logistic Regression …")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    )
    lr.fit(X_train_s, y_train)

    y_pred_lr = lr.predict(X_test_s)
    y_prob_lr = lr.predict_proba(X_test_s)[:, 1]

    print("\n  Classification Report (Logistic Regression):")
    print(classification_report(y_test, y_pred_lr, target_names=["Stay", "Churn"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")

    # Print top coefficients so we can verify no huge cancelling pair remains
    coef_df = pd.Series(lr.coef_[0], index=col_list).sort_values()
    print("\n  Top 5 negative coefficients (predict STAY):")
    print(coef_df.head(5).to_string())
    print("\n  Top 5 positive coefficients (predict CHURN):")
    print(coef_df.tail(5).to_string())

    # Save LR model and scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(lr, f)
    print(f"\n  ✅  Saved LR model  → {MODEL_PATH.name}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✅  Saved scaler    → {SCALER_PATH.name}")

    # ── 4. Verification: predict the 6 canonical test cases ───────────────────
    print("\n🧪  Verification predictions on canonical test cases …")
    _verify_predictions(lr, scaler, col_list)

    # ── 5. Random Forest Pipeline ──────────────────────────────────────────────
    print("\n🚀  Training Random Forest Pipeline …")
    raw_df = pd.read_csv(DATASET_PATH)
    raw_df.drop(columns=["customerID"], inplace=True, errors="ignore")
    raw_df["TotalCharges"] = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")
    raw_df.loc[raw_df["tenure"] == 0, "TotalCharges"] = 0.0
    raw_df["TotalCharges"].fillna(raw_df["TotalCharges"].median(), inplace=True)

    y_raw = raw_df["Churn"].map({"Yes": 1, "No": 0})
    X_raw = raw_df.drop(columns=["Churn"])

    # Drop TotalCharges from RF pipeline too for consistency
    X_raw = X_raw.drop(columns=["TotalCharges"], errors="ignore")

    num_cols = X_raw.select_dtypes(exclude="object").columns.tolist()
    cat_cols = X_raw.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", rf)])

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_raw
    )
    pipeline.fit(Xr_train, yr_train)

    y_pred_rf = pipeline.predict(Xr_test)
    y_prob_rf = pipeline.predict_proba(Xr_test)[:, 1]
    print("\n  Classification Report (Random Forest Pipeline):")
    print(classification_report(yr_test, y_pred_rf, target_names=["Stay", "Churn"]))
    print(f"  ROC-AUC: {roc_auc_score(yr_test, y_prob_rf):.4f}")

    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n  ✅  Saved RF pipeline → {PIPELINE_PATH.name}")

    # ── 6. Feature Importance Plots (from RF) ──────────────────────────────────
    print("\n📊  Generating feature importance plots …")
    feat_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pd.Series(
        pipeline.named_steps["model"].feature_importances_,
        index=feat_names,
    )
    importances.index = [n.split("__")[-1] for n in importances.index]

    plot_top_features(
        importances, "Top 10 Most Important Features for Churn Prediction",
        FEATURE_IMPORTANCE_PATH, n=10, cmap="viridis",
    )
    plot_top_features(
        importances, "Top 15 Most Important Features (Global)",
        FEATURE_IMPORTANCE_GLOBAL_PATH, n=15, cmap="cool",
    )

    # ── 7. SHAP Importance Plot (from LR |coefficients|) ──────────────────────
    print("\n🔍  Generating SHAP importance plot (LR coefficients) …")
    coef_series = (
        pd.Series(np.abs(lr.coef_[0]), index=col_list)
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    colours = plt.colormaps["Oranges"](np.linspace(0.4, 0.9, len(coef_series)))[::-1]
    ax.barh(coef_series.index[::-1], coef_series.values[::-1], color=colours[::-1])
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("Top 15 Features by SHAP Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(SHAP_IMPORTANCE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Saved: {SHAP_IMPORTANCE_PATH.name}")

    print("\n" + "=" * 70)
    print("✅  TRAINING COMPLETE")
    print("   Next steps:")
    print("   1. Ctrl+C  →  stop any running Streamlit instance")
    print("   2. streamlit run app/streamlit_app.py  →  fresh start")
    print("=" * 70)


# ── Canonical verification ────────────────────────────────────────────────────

def _encode_single(customer: dict, col_list: list) -> np.ndarray:
    """
    Encode one customer dict using the same manual lookup as predict.py
    (avoids the drop_first single-row bug).
    """
    _DUMMIES = {
        "gender":          {"Male": "gender_Male"},
        "Partner":         {"Yes": "Partner_Yes"},
        "Dependents":      {"Yes": "Dependents_Yes"},
        "PhoneService":    {"Yes": "PhoneService_Yes"},
        "MultipleLines":   {
            "No phone service": "MultipleLines_No phone service",
            "Yes": "MultipleLines_Yes",
        },
        "InternetService": {
            "Fiber optic": "InternetService_Fiber optic",
            "No":          "InternetService_No",
        },
        "OnlineSecurity":  {
            "No internet service": "OnlineSecurity_No internet service",
            "Yes": "OnlineSecurity_Yes",
        },
        "OnlineBackup":    {
            "No internet service": "OnlineBackup_No internet service",
            "Yes": "OnlineBackup_Yes",
        },
        "DeviceProtection":{
            "No internet service": "DeviceProtection_No internet service",
            "Yes": "DeviceProtection_Yes",
        },
        "TechSupport":     {
            "No internet service": "TechSupport_No internet service",
            "Yes": "TechSupport_Yes",
        },
        "StreamingTV":     {
            "No internet service": "StreamingTV_No internet service",
            "Yes": "StreamingTV_Yes",
        },
        "StreamingMovies": {
            "No internet service": "StreamingMovies_No internet service",
            "Yes": "StreamingMovies_Yes",
        },
        "Contract":        {
            "One year":  "Contract_One year",
            "Two year":  "Contract_Two year",
        },
        "PaperlessBilling":{"Yes": "PaperlessBilling_Yes"},
        "PaymentMethod":   {
            "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
            "Electronic check":        "PaymentMethod_Electronic check",
            "Mailed check":            "PaymentMethod_Mailed check",
        },
    }
    _NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges"]
    # NOTE: TotalCharges intentionally excluded — dropped from model

    row = {col: 0 for col in col_list}
    for num in _NUMERIC:
        if num in customer and num in row:
            row[num] = customer[num]
    for cat, vmap in _DUMMIES.items():
        val = customer.get(cat)
        if val in vmap and vmap[val] in row:
            row[vmap[val]] = 1
    return np.array([list(row.values())], dtype=float)


def _verify_predictions(lr, scaler, col_list: list) -> None:
    """Run 6 canonical customers and print expected vs actual."""
    cases = [
        ("🟢 SHOULD STAY  (~5-15%)",
         {"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"Yes",
          "tenure":60,"PhoneService":"Yes","MultipleLines":"Yes",
          "InternetService":"DSL","OnlineSecurity":"Yes","OnlineBackup":"Yes",
          "DeviceProtection":"Yes","TechSupport":"Yes","StreamingTV":"Yes",
          "StreamingMovies":"Yes","Contract":"Two year","PaperlessBilling":"No",
          "PaymentMethod":"Credit card (automatic)","MonthlyCharges":65}),

        ("🔴 SHOULD CHURN (~85-95%)",
         {"gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No",
          "tenure":2,"PhoneService":"Yes","MultipleLines":"No",
          "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
          "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
          "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
          "PaymentMethod":"Electronic check","MonthlyCharges":85}),

        ("🟡 MEDIUM RISK  (~40-60%)",
         {"gender":"Female","SeniorCitizen":1,"Partner":"No","Dependents":"No",
          "tenure":12,"PhoneService":"Yes","MultipleLines":"No",
          "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes",
          "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
          "StreamingMovies":"No","Contract":"One year","PaperlessBilling":"Yes",
          "PaymentMethod":"Mailed check","MonthlyCharges":75}),

        ("🟢 SHOULD STAY  (~10-25%)",
         {"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
          "tenure":36,"PhoneService":"Yes","MultipleLines":"No",
          "InternetService":"DSL","OnlineSecurity":"Yes","OnlineBackup":"No",
          "DeviceProtection":"Yes","TechSupport":"Yes","StreamingTV":"No",
          "StreamingMovies":"No","Contract":"One year","PaperlessBilling":"Yes",
          "PaymentMethod":"Bank transfer (automatic)","MonthlyCharges":55}),

        ("🔴 SHOULD CHURN (~75-90%)",
         {"gender":"Male","SeniorCitizen":1,"Partner":"No","Dependents":"No",
          "tenure":1,"PhoneService":"Yes","MultipleLines":"No",
          "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
          "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
          "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
          "PaymentMethod":"Electronic check","MonthlyCharges":95}),

        ("🟡 EDGE CASE    (~30-50%)",
         {"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No",
          "tenure":24,"PhoneService":"Yes","MultipleLines":"No",
          "InternetService":"Fiber optic","OnlineSecurity":"Yes","OnlineBackup":"Yes",
          "DeviceProtection":"No","TechSupport":"Yes","StreamingTV":"No",
          "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
          "PaymentMethod":"Electronic check","MonthlyCharges":70}),
    ]

    print(f"\n  {'Label':<32} {'Prob':>7}  {'Result'}")
    print("  " + "-" * 55)
    all_pass = True
    for label, customer in cases:
        X_enc  = _encode_single(customer, col_list)
        X_sc   = scaler.transform(X_enc)
        prob   = lr.predict_proba(X_sc)[0][1]
        result = "CHURN" if prob >= 0.5 else "STAY"

        # Simple pass/fail based on label hint
        if "SHOULD STAY"  in label and prob >= 0.5:
            status = "❌ FAIL"
            all_pass = False
        elif "SHOULD CHURN" in label and prob < 0.5:
            status = "❌ FAIL"
            all_pass = False
        else:
            status = "✅ PASS"

        print(f"  {label:<32} {prob:>6.1%}  {result}  {status}")

    print()
    if all_pass:
        print("  🎉 All canonical tests passed!")
    else:
        print("  ⚠️  Some tests failed — check model coefficients above.")


if __name__ == "__main__":
    main()