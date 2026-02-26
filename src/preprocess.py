"""
preprocess.py â€” Standalone Data Preprocessing Pipeline
Cleans, encodes, scales and saves processed data for external use.
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

# âœ… FIX: was importing bare `config`; must import from src.config
from src.config import (
    DATASET_PATH as RAW_DATA_PATH,
    PROCESSED_DIR,
    X_TRAIN_PATH, X_TEST_PATH,
    Y_TRAIN_PATH, Y_TEST_PATH,
    SCALER_PATH, FEATURE_NAMES_PATH,
    TEST_SIZE, RANDOM_STATE
)
from src.preprocessing_utils import clean_dataframe, encode_features


def preprocess_pipeline():
    """Complete preprocessing pipeline â€” saves arrays and artefacts."""

    print("=" * 60)
    print("CHURN PREDICTION â€” DATA PREPROCESSING")
    print("=" * 60)

    # â”€â”€ 1. Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n1. Loading data from {RAW_DATA_PATH} â€¦")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   âœ” Loaded {len(df):,} customers")

    # â”€â”€ 2. Clean â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n2. Cleaning data â€¦")
    df = clean_dataframe(df)
    print(f"   âœ” Churn rate: {df['Churn'].mean():.1%}")

    # â”€â”€ 3. Split features / target â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n3. Separating features and target â€¦")
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    print(f"   âœ” {X.shape[1]} raw features  |  target: Churn (0/1)")

    # â”€â”€ 4. Encode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n4. Encoding categorical features â€¦")
    X_encoded = encode_features(X)
    feature_names = X_encoded.columns.tolist()
    print(f"   âœ” {len(feature_names)} encoded features")

    # â”€â”€ 5. Train / test split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n5. Splitting train/test sets â€¦")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   âœ” Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # â”€â”€ 6. Scale â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n6. Scaling features â€¦")
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                              columns=X_train.columns, index=X_train.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),
                              columns=X_test.columns,  index=X_test.index)
    print("   âœ” StandardScaler fitted on training data")

    # â”€â”€ 7. Save arrays â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n7. Saving processed data â€¦")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(X_TRAIN_PATH, X_train_s.values)
    np.save(X_TEST_PATH,  X_test_s.values)
    np.save(Y_TRAIN_PATH, y_train.values)
    np.save(Y_TEST_PATH,  y_test.values)
    print(f"   âœ” Arrays saved to {PROCESSED_DIR}")

    # â”€â”€ 8. Save artefacts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"   âœ” Scaler saved â†’ {SCALER_PATH.name}")

    with open(FEATURE_NAMES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    print(f"   âœ” Feature names saved â†’ {FEATURE_NAMES_PATH.name}")

    # â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 60)
    print("âœ…  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n  Training samples : {X_train_s.shape[0]:,}")
    print(f"  Test samples     : {X_test_s.shape[0]:,}")
    print(f"  Features         : {X_train_s.shape[1]}")
    print(f"  Churn rate train : {y_train.mean():.1%}")
    print(f"  Churn rate test  : {y_test.mean():.1%}")

    return X_train_s, X_test_s, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    try:
        preprocess_pipeline()
    except Exception as e:
        print(f"\nâŒ  ERROR: {e}")
        raise