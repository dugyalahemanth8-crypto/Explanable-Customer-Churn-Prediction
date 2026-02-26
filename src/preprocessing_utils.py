"""
preprocessing_utils.py — Shared Preprocessing Utilities
Provides clean_dataframe(), encode_features(), validate_customer_data(),
and other helpers used across the project.
"""
import pandas as pd
import numpy as np


# ── Dataset-level helpers ──────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco Churn CSV.
    Steps:
      - Drop customerID
      - Fix TotalCharges (object → float)
      - Encode Churn → 0/1
    """
    df = df.copy()

    # Remove customerID if present
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # TotalCharges is stored as a string in the raw CSV
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all categorical columns using drop_first=True.
    This produces the same 30-column schema the LR model was trained on.
    """
    return pd.get_dummies(X, drop_first=True)


# ── Single-customer helpers ────────────────────────────────────────────────────

def validate_customer_data(customer_data: dict) -> tuple:
    """
    Validate raw customer feature dict.

    Returns
    -------
    (is_valid: bool, errors: list[str])
    """
    errors = []
    required = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "InternetService", "Contract",
        "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    for field in required:
        if field not in customer_data:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    try:
        t = customer_data["tenure"]
        mc = customer_data["MonthlyCharges"]
        tc = customer_data["TotalCharges"]

        if not (0 <= t <= 100):
            errors.append("tenure must be 0–100")
        if not (0 <= mc <= 500):
            errors.append("MonthlyCharges must be 0–500")
        if tc < 0:
            errors.append("TotalCharges cannot be negative")

        if customer_data["gender"] not in ("Male", "Female"):
            errors.append("gender must be 'Male' or 'Female'")
        if customer_data["SeniorCitizen"] not in (0, 1):
            errors.append("SeniorCitizen must be 0 or 1")
        if customer_data["Partner"] not in ("Yes", "No"):
            errors.append("Partner must be 'Yes' or 'No'")
        if customer_data["Dependents"] not in ("Yes", "No"):
            errors.append("Dependents must be 'Yes' or 'No'")
        if customer_data["PhoneService"] not in ("Yes", "No"):
            errors.append("PhoneService must be 'Yes' or 'No'")
        if customer_data["InternetService"] not in ("DSL", "Fiber optic", "No"):
            errors.append("InternetService must be 'DSL', 'Fiber optic', or 'No'")
        if customer_data["Contract"] not in ("Month-to-month", "One year", "Two year"):
            errors.append("Contract must be one of the three standard values")
        if customer_data["PaperlessBilling"] not in ("Yes", "No"):
            errors.append("PaperlessBilling must be 'Yes' or 'No'")

        valid_pm = [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ]
        if customer_data["PaymentMethod"] not in valid_pm:
            errors.append(f"PaymentMethod must be one of: {', '.join(valid_pm)}")

    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"Data validation error: {exc}")

    return len(errors) == 0, errors


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing TotalCharges using tenure × MonthlyCharges."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    mask = df["TotalCharges"].isna() & (df["tenure"] > 0)
    df.loc[mask, "TotalCharges"] = (
        df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
    )
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features (optional, not used by the main model)."""
    df = df.copy()

    df["AvgMonthlyCharges"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 100],
        labels=["0–12 months", "12–24 months", "24–48 months", "48+ months"],
    )

    svc_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    present = [c for c in svc_cols if c in df.columns]
    if present:
        df["TotalServices"] = sum((df[c] == "Yes").astype(int) for c in present)

    return df


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0, "TotalCharges": 1020.0,
    }
    ok, errs = validate_customer_data(sample)
    print("Validation:", "✅ PASSED" if ok else "❌ FAILED")
    for e in errs:
        print(" -", e)