"""
predict.py â€” Churn Prediction
Loads the Logistic Regression model and the saved column schema.
No CSV is read at prediction time â€” column names come from model_columns.pkl.

FIX 1 (2026-02-17) â€” drop_first single-row encoding bug:
  pd.get_dummies(single_row, drop_first=True) produces an EMPTY DataFrame
  when the single row contains only reference-category values, because
  drop_first removes the sole category present, leaving no column.
  After reindex(fill_value=0) every dummy column was always 0, making
  every customer look identical (only tenure/charges/SeniorCitizen varied).
  Fix: replaced get_dummies with _manual_encode(), which explicitly maps
  each categorical value to its known dummy column(s) without drop_first.

FIX 2 (2026-02-17) â€” TotalCharges multicollinearity:
  TotalCharges â‰ˆ tenure Ã— MonthlyCharges is perfectly collinear.
  The LR learned tenure_coef â‰ˆ âˆ’22.5 and TotalCharges_coef â‰ˆ +25.4,
  nearly equal and opposite, causing loyal long-tenure customers to be
  mis-classified as high churn risk.
  Fix: TotalCharges is no longer passed to the model (dropped in train.py).
  _NUMERIC_COLS and build_feature_vector() updated accordingly.
"""
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_PATH, MODEL_COLUMNS_PATH, DATASET_PATH, SCALER_PATH


# â”€â”€ Module-level cache (loaded once per process) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_model         = None
_scaler        = None
_model_columns = None


def _load_artefacts():
    """Load model, scaler, and column list once, then cache."""
    global _model, _scaler, _model_columns

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "Run `python src/train.py` first to generate the model files."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)

    if _scaler is None and SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)

    if _model_columns is None:
        if not MODEL_COLUMNS_PATH.exists():
            _model_columns = _derive_columns_from_dataset()
        else:
            with open(MODEL_COLUMNS_PATH, "rb") as f:
                _model_columns = pickle.load(f)

    return _model, _scaler, _model_columns


def _derive_columns_from_dataset() -> list:
    """
    Fallback: read dataset, apply same encoding as training, return column list.
    Only called once if model_columns.pkl is missing.
    """
    df = pd.read_csv(DATASET_PATH)
    df.drop(columns=["customerID", "Churn"], inplace=True, errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return list(pd.get_dummies(df, drop_first=True).columns)


# â”€â”€ Manual Encoding (THE FIX) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Reference categories (alphabetically first value per column, dropped by
# drop_first=True during training). Any other value maps to a 1 in its dummy.
_CATEGORICAL_DUMMIES = {
    # column_name: {value -> dummy_column_name}  (reference value not listed = 0)
    "gender": {
        "Male": "gender_Male",
    },
    "Partner": {
        "Yes": "Partner_Yes",
    },
    "Dependents": {
        "Yes": "Dependents_Yes",
    },
    "PhoneService": {
        "Yes": "PhoneService_Yes",
    },
    "MultipleLines": {
        "No phone service": "MultipleLines_No phone service",
        "Yes":              "MultipleLines_Yes",
    },
    "InternetService": {
        "Fiber optic": "InternetService_Fiber optic",
        "No":          "InternetService_No",
    },
    "OnlineSecurity": {
        "No internet service": "OnlineSecurity_No internet service",
        "Yes":                 "OnlineSecurity_Yes",
    },
    "OnlineBackup": {
        "No internet service": "OnlineBackup_No internet service",
        "Yes":                 "OnlineBackup_Yes",
    },
    "DeviceProtection": {
        "No internet service": "DeviceProtection_No internet service",
        "Yes":                 "DeviceProtection_Yes",
    },
    "TechSupport": {
        "No internet service": "TechSupport_No internet service",
        "Yes":                 "TechSupport_Yes",
    },
    "StreamingTV": {
        "No internet service": "StreamingTV_No internet service",
        "Yes":                 "StreamingTV_Yes",
    },
    "StreamingMovies": {
        "No internet service": "StreamingMovies_No internet service",
        "Yes":                 "StreamingMovies_Yes",
    },
    "Contract": {
        "One year":  "Contract_One year",
        "Two year":  "Contract_Two year",
    },
    "PaperlessBilling": {
        "Yes": "PaperlessBilling_Yes",
    },
    "PaymentMethod": {
        "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
        "Electronic check":        "PaymentMethod_Electronic check",
        "Mailed check":            "PaymentMethod_Mailed check",
    },
}

# Numeric columns passed through as-is.
# NOTE: TotalCharges intentionally excluded â€” it was dropped from the model
# in train.py to fix the multicollinearity issue with tenure Ã— MonthlyCharges.
_NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges"]


def _manual_encode(customer_data: dict, model_columns: list) -> pd.DataFrame:
    """
    Encode a single customer dict into the exact dummy schema the model expects.

    This avoids pd.get_dummies(drop_first=True) on a single row, which silently
    drops every category leaving all dummies = 0.

    Instead we:
      1. Start with a zero-filled row for all model_columns.
      2. Set numeric columns directly.
      3. For each categorical column look up the value in _CATEGORICAL_DUMMIES
         and set the matching dummy column(s) to 1.
    """
    row = {col: 0 for col in model_columns}

    # â”€â”€ Numeric features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for num_col in _NUMERIC_COLS:
        if num_col in customer_data and num_col in row:
            row[num_col] = customer_data[num_col]

    # â”€â”€ Categorical â†’ dummy mapping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for cat_col, value_map in _CATEGORICAL_DUMMIES.items():
        value = customer_data.get(cat_col)
        if value in value_map:
            dummy_col = value_map[value]
            if dummy_col in row:
                row[dummy_col] = 1
        # reference category value â†’ dummy stays 0 (already set above)

    return pd.DataFrame([row], columns=model_columns)


# â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_feature_vector(customer_data: dict) -> pd.DataFrame:
    """
    Convert raw customer dict â†’ correctly encoded, correctly ordered DataFrame
    matching the columns the model was trained on.

    Uses _manual_encode() to avoid the drop_first single-row encoding bug.
    TotalCharges is intentionally ignored â€” it was removed from the model
    to fix the multicollinearity problem with tenure Ã— MonthlyCharges.
    """
    _, _, model_columns = _load_artefacts()

    # Work on a copy so we don't mutate caller's dict
    data = dict(customer_data)

    return _manual_encode(data, model_columns)


def predict_churn(customer_data: dict, debug: bool = False):
    """
    Predict churn for a single customer.

    Returns
    -------
    prediction  : int   â€” 0 (Stay) or 1 (Churn)
    probability : float â€” churn probability (0â€“1)
    model       : fitted LogisticRegression object
    """
    model, scaler, _ = _load_artefacts()

    df = build_feature_vector(customer_data)

    if debug:
        non_zero = df.loc[0, df.iloc[0] != 0]
        print(f"\nðŸ” Non-zero features ({len(non_zero)}):")
        print(non_zero)

    if scaler is not None:
        X = scaler.transform(df)
    else:
        X = df.values

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    if debug:
        print(f"\n  P(Stay)  = {prob[0]:.4f}")
        print(f"  P(Churn) = {prob[1]:.4f}")

    return pred, float(prob[1]), model


def batch_predict(df_customers: pd.DataFrame) -> pd.DataFrame:
    """
    Predict churn for a DataFrame of customers.
    Returns original DataFrame with Predicted_Churn, Churn_Probability,
    and Risk_Level columns appended.
    """
    model, scaler, model_columns = _load_artefacts()

    df = df_customers.copy()

    # Drop TotalCharges â€” removed from model (multicollinearity fix)
    df = df.drop(columns=["TotalCharges"], errors="ignore")

    # For batch predictions use get_dummies â€” multiple rows means all categories
    # are present so drop_first works correctly here.
    df_enc = pd.get_dummies(df, drop_first=True).reindex(
        columns=model_columns, fill_value=0
    )

    X = scaler.transform(df_enc) if scaler is not None else df_enc.values

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    result = df_customers.copy()
    result["Predicted_Churn"]   = ["Yes" if p == 1 else "No" for p in preds]
    result["Churn_Probability"] = np.round(probs, 4)
    result["Risk_Level"] = pd.cut(
        probs, bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"]
    )
    return result


# â”€â”€ Quick test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    test_cases = {
        "LOW RISK  (60mo, Two year, DSL, all services)": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "Yes", "tenure": 60,
            "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes", "DeviceProtection": "Yes",
            "TechSupport": "Yes", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 65.0,
            # TotalCharges intentionally omitted â€” no longer used by model
        },
        "HIGH RISK (2mo, Month-to-month, Fiber, no services)": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
            "Dependents": "No", "tenure": 2,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.0,
        },
        "MEDIUM RISK (12mo, One year, Fiber, senior)": {
            "gender": "Female", "SeniorCitizen": 1, "Partner": "No",
            "Dependents": "No", "tenure": 12,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "Yes", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "No", "Contract": "One year",
            "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
            "MonthlyCharges": 75.0,
        },
        "VERY HIGH RISK (1mo, senior, no services)": {
            "gender": "Male", "SeniorCitizen": 1, "Partner": "No",
            "Dependents": "No", "tenure": 1,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
        },
    }

    print("=" * 70)
    print("PREDICTION TEST â€” verify both fixes")
    print("=" * 70)
    for label, customer in test_cases.items():
        pred, prob, _ = predict_churn(customer, debug=False)
        result = "CHURN" if pred == 1 else "STAY"
        print(f"\n  {label}")
        print(f"  â†’ {result}  ({prob:.1%} churn probability)")

    # Debug the LOW RISK case to show non-zero features
    print("\n" + "=" * 70)
    print("DEBUG: non-zero features for LOW RISK customer")
    print("=" * 70)
    predict_churn(list(test_cases.values())[0], debug=True)