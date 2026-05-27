"""
Preprocessing pipeline for NF-UQ-NIDS-v3.

Uses sklearn ColumnTransformer with three processing paths:
  - Numeric features:       median imputation + StandardScaler
  - Protocol features:      constant-0 imputation (0 = protocol not applicable) + StandardScaler
  - Categorical features:   constant-0 imputation + OrdinalEncoder

All fitting happens on the training set only; transform is applied to val/test.
L7_PROTO non-integer cleaning is performed before the split.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

from config import (
    UNSW_PATH, ATTACK_COL, LABEL_COL, DROP_COLUMNS,
    NUMERIC_FEATURES, PROTOCOL_FEATURES, CATEGORICAL_FEATURES,
    TEST_SIZE, VAL_SIZE, SEED, RANDOM_STATE,
)


def load_and_clean_data(filepath):
    """Load CSV, drop identifier columns, replace inf with NaN."""
    df = pd.read_csv(filepath, low_memory=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def clean_l7_proto(df):
    """
    Clean L7_PROTO non-integer values (v3-specific issue).
    1. fillna(0) — NaN means no application-layer protocol
    2. round() -> astype(int) — clean non-integer values like 7.126
    """
    if "L7_PROTO" not in df.columns:
        return df

    df = df.copy()
    col = df["L7_PROTO"]

    # Track cleaning stats
    n_nan_before = col.isnull().sum()
    nonint_mask = col.notna() & (col % 1 != 0)
    n_nonint = nonint_mask.sum()

    col = col.fillna(0).round().astype(int)
    df["L7_PROTO"] = col

    if n_nan_before > 0 or n_nonint > 0:
        print(f"L7_PROTO cleaned: {n_nan_before} NaN filled with 0, "
              f"{n_nonint} non-integer values rounded")

    return df


def split_data(df, stratify_col=ATTACK_COL):
    """
    Stratified 70/10/20 split: train / validation / test.
    Val is split from train (10% of total = 1/7 of train+val).
    """
    y = df[stratify_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    # val_size_relative = VAL_SIZE / (1 - TEST_SIZE) — but simpler: 1/7 of train_val
    val_frac_of_trainval = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_frac_of_trainval,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    print(f"Split: train={len(X_train):,} ({len(X_train)/len(df)*100:.0f}%), "
          f"val={len(X_val):,} ({len(X_val)/len(df)*100:.0f}%), "
          f"test={len(X_test):,} ({len(X_test)/len(df)*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_cicids_chunk(df):
    """Apply CICIDS-specific cleaning: replace inf, clean L7_PROTO, drop identifier columns."""
    DROP_COLS = [
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
        "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = clean_l7_proto(df)
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def build_preprocessor():
    """Build an unfitted ColumnTransformer with three processing paths."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    protocol_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                    unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("protocol", protocol_pipeline, PROTOCOL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")

    return preprocessor


def fit_preprocessor(preprocessor, X_train):
    """Fit the ColumnTransformer on training data only."""
    preprocessor.fit(X_train)
    return preprocessor


def preprocess_pipeline(filepath):
    """
    End-to-end preprocessing pipeline.

    1. load_and_clean_data
    2. clean_l7_proto
    3. split_data
    4. build_preprocessor
    5. fit on train, transform train/val/test

    Returns a dict with processed data and metadata.
    """
    print(f"Loading {filepath} ...")
    df = load_and_clean_data(filepath)
    print(f"Shape after dropping ID columns: {df.shape}")

    df = clean_l7_proto(df)

    # Extract target and features
    y = df[ATTACK_COL]
    X = df.drop(columns=[ATTACK_COL, LABEL_COL])

    # Identify all feature columns present
    feature_cols = [c for c in X.columns
                    if c in NUMERIC_FEATURES + PROTOCOL_FEATURES + CATEGORICAL_FEATURES]
    print(f"Feature columns used: {len(feature_cols)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X.assign(**{ATTACK_COL: y}), stratify_col=ATTACK_COL
    )

    # Separate X from the appended y
    y_train = X_train.pop(ATTACK_COL)
    y_val = X_val.pop(ATTACK_COL)
    y_test = X_test.pop(ATTACK_COL)

    preprocessor = build_preprocessor()
    preprocessor = fit_preprocessor(preprocessor, X_train)

    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    print(f"Transformed shapes: train={X_train_t.shape}, val={X_val_t.shape}, test={X_test_t.shape}")

    class_names = sorted(y.unique())

    return {
        "X_train": X_train_t,
        "X_val": X_val_t,
        "X_test": X_test_t,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "class_names": class_names,
    }
