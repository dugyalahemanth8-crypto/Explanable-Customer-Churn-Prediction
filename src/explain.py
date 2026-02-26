"""
explain.py â€” SHAP Explainability for Churn Prediction
Works with the Logistic Regression model using shap.LinearExplainer.
Also supports the Random Forest pipeline via shap.TreeExplainer.

FIX (2026-02-17):
  _get_background_data() now drops TotalCharges before encoding,
  matching the change made in train.py.  Previously TotalCharges was
  still being read from the CSV and encoded, causing a column-count
  mismatch between the background array and model_columns (29 features).
  The reindex() call masked the error but produced an incorrect
  background baseline, shifting all SHAP expected_values.
"""
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import shap

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_PATH, PIPELINE_PATH, MODEL_COLUMNS_PATH, DATASET_PATH,
    SCALER_PATH, DATA_DIR, SHAP_IMPORTANCE_PATH
)
from src.predict import build_feature_vector, _load_artefacts

warnings.filterwarnings("ignore")


# â”€â”€ Explainer factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_lr_explainer(model, X_background: np.ndarray):
    """
    Build a SHAP LinearExplainer for a Logistic Regression model.
    X_background should be the scaled training data (numpy array).
    """
    return shap.LinearExplainer(model, X_background)


def get_rf_explainer(pipeline):
    """
    Build a SHAP TreeExplainer for the Random Forest inside the pipeline.
    Only use this when the pipeline model is a tree-based model.
    """
    rf_model = pipeline.named_steps["model"]
    return shap.TreeExplainer(rf_model)


# â”€â”€ Individual (local) explanation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def explain_single_prediction(
    customer_data: dict,
    top_n: int = 10,
    save_plot: bool = False,
    customer_label: str = "Customer"
) -> pd.DataFrame:
    """
    Compute SHAP values for a single customer and return a ranked DataFrame.

    Parameters
    ----------
    customer_data : dict
        Raw customer feature dictionary (same format used by predict_churn).
    top_n : int
        Number of top features to include.
    save_plot : bool
        If True, save a bar chart PNG to DATA_DIR.
    customer_label : str
        Label used in the chart title.

    Returns
    -------
    pd.DataFrame with columns: Feature, SHAP_Value, Abs_Impact, Impact
    """
    model, scaler, model_columns = _load_artefacts()

    # Build encoded feature vector
    df_enc = build_feature_vector(customer_data)

    # Scale (same as training)
    if scaler is not None:
        X_scaled = scaler.transform(df_enc)
    else:
        X_scaled = df_enc.values

    # Background dataset for LinearExplainer
    X_background = _get_background_data(scaler, model_columns)

    # Create LinearExplainer and compute SHAP values
    explainer  = get_lr_explainer(model, X_background)
    shap_vals  = explainer.shap_values(X_scaled)

    # shap_vals shape: (1, n_features) for LR binary classification
    if isinstance(shap_vals, list):
        # Older SHAP versions return [class0_vals, class1_vals]
        values = np.array(shap_vals[1], dtype=float).flatten()
    else:
        values = np.array(shap_vals, dtype=float).flatten()

    explanation_df = pd.DataFrame({
        "Feature":    model_columns,
        "SHAP_Value": values.astype(float),
    })
    explanation_df["Abs_Impact"] = explanation_df["SHAP_Value"].abs()
    explanation_df["Impact"] = explanation_df["SHAP_Value"].apply(
        lambda v: "â†‘ Increases Churn Risk" if v > 0 else "â†“ Decreases Churn Risk"
    )
    explanation_df = (
        explanation_df
        .sort_values("Abs_Impact", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    if save_plot:
        _save_explanation_plot(explanation_df, customer_label)

    return explanation_df


def _get_background_data(scaler, model_columns: list) -> np.ndarray:
    """
    Return a background dataset for LinearExplainer.
    Tries to load from cache; rebuilds from training CSV if cache missing.

    FIX: TotalCharges is now dropped before encoding so the background
    array has exactly 29 columns matching model_columns (same as train.py).
    Previously it was included, causing a silent mismatch that shifted
    all SHAP expected_values away from the true model baseline.
    """
    cache_path = DATA_DIR / "_shap_background.npy"
    if cache_path.exists():
        return np.load(cache_path, allow_pickle=True)

    # Build from training data â€” only done once, then cached
    df = pd.read_csv(DATASET_PATH)
    df.drop(columns=["customerID", "Churn"], inplace=True, errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # â”€â”€ DROP TotalCharges â€” matches train.py (multicollinearity fix) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df = df.drop(columns=["TotalCharges"], errors="ignore")

    df_enc = pd.get_dummies(df, drop_first=True).reindex(
        columns=model_columns, fill_value=0
    )

    if scaler is not None:
        X_bg = scaler.transform(df_enc)
    else:
        X_bg = df_enc.values

    np.save(cache_path, X_bg)
    print(f"  âœ…  Built & cached SHAP background: {X_bg.shape} â†’ {cache_path.name}")
    return X_bg


def _save_explanation_plot(explanation_df: pd.DataFrame, label: str) -> Path:
    """Save a horizontal bar chart of SHAP values."""
    df      = explanation_df.copy()
    colours = ["#e53e3e" if v > 0 else "#38a169" for v in df["SHAP_Value"]]

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.55)))
    ax.barh(df["Feature"][::-1], df["SHAP_Value"][::-1], color=colours[::-1])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value  (Impact on Churn Probability)", fontsize=12)
    ax.set_title(
        f"Top Feature Contributions â€” {label}\n"
        f"Red = increases churn risk  |  Green = decreases churn risk",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = DATA_DIR / f"shap_explanation_{label.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  âœ…  Saved SHAP plot â†’ {out.name}")
    return out


# â”€â”€ Matplotlib figure (for Streamlit st.pyplot) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_explanation_figure(explanation_df: pd.DataFrame, title: str = "") -> plt.Figure:
    """
    Return a matplotlib Figure showing the SHAP explanation.
    Call st.pyplot(fig) in Streamlit to display it.
    """
    df      = explanation_df.copy().head(10)
    colours = ["#e53e3e" if v > 0 else "#38a169" for v in df["SHAP_Value"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.6)))
    ax.barh(
        df["Feature"][::-1],
        df["SHAP_Value"][::-1],
        color=colours[::-1],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value  (Impact on Churn Probability)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# â”€â”€ Global explanations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_global_explanations(n_samples: int = 300) -> pd.DataFrame:
    """
    Compute mean |SHAP| across n_samples customers.
    Returns importance DataFrame and saves shap_importance.png.
    """
    model, scaler, model_columns = _load_artefacts()
    X_bg = _get_background_data(scaler, model_columns)

    rng      = np.random.default_rng(42)
    idx      = rng.choice(len(X_bg), size=min(n_samples, len(X_bg)), replace=False)
    X_sample = X_bg[idx]

    explainer = get_lr_explainer(model, X_bg)
    shap_vals = explainer.shap_values(X_sample)

    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1])
    else:
        sv = np.array(shap_vals)

    mean_abs     = np.abs(sv).mean(axis=0)
    importance_df = (
        pd.DataFrame({"Feature": model_columns, "Mean_Abs_SHAP": mean_abs})
        .sort_values("Mean_Abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )

    # Save SHAP importance plot
    top     = importance_df.head(15)
    colours = plt.colormaps["Oranges"](np.linspace(0.4, 0.9, len(top)))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Feature"][::-1], top["Mean_Abs_SHAP"][::-1], color=colours[::-1])
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("Top 15 Features by SHAP Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(SHAP_IMPORTANCE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  âœ…  Saved global SHAP importance â†’ {SHAP_IMPORTANCE_PATH.name}")

    return importance_df


# â”€â”€ CLI entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    print("=" * 70)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    print("\nðŸ“Š  Generating global SHAP explanations â€¦")
    importance_df = generate_global_explanations(n_samples=300)
    print("\nTop 10 features by mean |SHAP|:")
    print(importance_df.head(10).to_string(index=False))

    # Example â€” high-risk customer
    # NOTE: TotalCharges is omitted â€” it is no longer used by the model
    high_risk = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
    }

    print("\nðŸ“Š  Individual explanation â€” high-risk customer:")
    exp = explain_single_prediction(
        high_risk, top_n=10, save_plot=True, customer_label="High Risk Customer"
    )
    print(exp[["Feature", "SHAP_Value", "Impact"]].to_string(index=False))
    print("\nâœ…  Done.")