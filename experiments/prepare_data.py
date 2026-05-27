"""
Phase 2b: End-to-end preprocessing pipeline execution.

Loads UNSW, cleans L7_PROTO, splits 70/10/20, fits ColumnTransformer,
transforms all splits, fits LabelEncoder, saves all artifacts.

Generates:
  artifacts/preprocessor.joblib
  artifacts/label_encoder.joblib
  artifacts/class_mapping.json
  artifacts/training_config.json
"""

import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

from config import (
    UNSW_PATH, ATTACK_COL, LABEL_COL, DROP_COLUMNS,
    NUMERIC_FEATURES, PROTOCOL_FEATURES, CATEGORICAL_FEATURES,
    SEED, RANDOM_STATE,
)
from preprocessing.preprocess import (
    load_and_clean_data, clean_l7_proto, split_data,
    build_preprocessor, fit_preprocessor,
)
from preprocessing.balance import get_class_weights, compute_sample_weights


def verify_split_distribution(y_train, y_val, y_test, df_y):
    """Verify class proportions are similar across splits."""
    print("\n=== Split Distribution Verification ===")
    total = len(y_train) + len(y_val) + len(y_test)
    for label in sorted(df_y.unique()):
        tr_pct = (y_train == label).sum() / len(y_train) * 100
        vl_pct = (y_val == label).sum() / len(y_val) * 100
        ts_pct = (y_test == label).sum() / len(y_test) * 100
        orig_pct = (df_y == label).sum() / len(df_y) * 100
        print(f"  {label:20s}: train={tr_pct:.2f}%  val={vl_pct:.2f}%  "
              f"test={ts_pct:.2f}%  original={orig_pct:.2f}%")
    print()


def main():
    print("=" * 60)
    print("Phase 2b: Preprocessing Pipeline")
    print("=" * 60)

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)

    # ---- Step 1: Load and clean ----
    t0 = time.time()
    print(f"\nLoading {UNSW_PATH} ...")
    df = load_and_clean_data(UNSW_PATH)
    print(f"Shape after dropping ID columns: {df.shape}")

    df = clean_l7_proto(df)

    # ---- Step 2: Split ----
    y_full = df[ATTACK_COL]
    X_full = df.drop(columns=[ATTACK_COL, LABEL_COL])

    # Put y back temporarily for split_data
    df_for_split = X_full.copy()
    df_for_split[ATTACK_COL] = y_full

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_for_split)

    y_train = X_train.pop(ATTACK_COL)
    y_val = X_val.pop(ATTACK_COL)
    y_test = X_test.pop(ATTACK_COL)

    verify_split_distribution(y_train, y_val, y_test, y_full)

    # ---- Step 3: Fit preprocessor on train only ----
    print("Fitting ColumnTransformer on training data ...")
    preprocessor = build_preprocessor()
    preprocessor = fit_preprocessor(preprocessor, X_train)

    # ---- Step 4: Transform all splits ----
    print("Transforming train / val / test ...")
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # Count output features
    n_num = len(NUMERIC_FEATURES)
    n_proto = len(PROTOCOL_FEATURES)
    n_cat = len(CATEGORICAL_FEATURES)
    expected_features = n_num + n_proto + n_cat

    print(f"\nOutput feature matrix shape:")
    print(f"  X_train: {X_train_t.shape}  (expected {expected_features} features)")
    print(f"  X_val:   {X_val_t.shape}")
    print(f"  X_test:  {X_test_t.shape}")

    if X_train_t.shape[1] != expected_features:
        print(f"  WARNING: Expected {expected_features} features, got {X_train_t.shape[1]}")

    # ---- Step 5: Fit LabelEncoder ----
    print("\nFitting LabelEncoder ...")
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    # ---- Step 6: Compute class weights ----
    class_weights = get_class_weights(y_train)
    sample_weights = compute_sample_weights(y_train)

    print(f"\nClass weights:")
    for cls, w in sorted(class_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {cls}: {w:.4f}")

    # ---- Step 7: Save artifacts ----
    print("\nSaving artifacts ...")
    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
    print("  artifacts/preprocessor.joblib")

    joblib.dump(label_encoder, "artifacts/label_encoder.joblib")
    print("  artifacts/label_encoder.joblib")

    class_mapping = {int(i): str(cls) for i, cls in enumerate(label_encoder.classes_)}
    with open("artifacts/class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print("  artifacts/class_mapping.json")

    training_config = {
        "seed": SEED,
        "random_state": RANDOM_STATE,
        "train_samples": len(X_train_t),
        "val_samples": len(X_val_t),
        "test_samples": len(X_test_t),
        "n_features": X_train_t.shape[1],
        "numeric_features": n_num,
        "protocol_features": n_proto,
        "categorical_features": n_cat,
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
    }
    with open("artifacts/training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    print("  artifacts/training_config.json")

    # ---- Step 8: Summary ----
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete. ({elapsed:.1f}s)")
    print(f"  Train: {len(X_train_t):,} samples x {X_train_t.shape[1]} features")
    print(f"  Val:   {len(X_val_t):,} samples")
    print(f"  Test:  {len(X_test_t):,} samples")
    print(f"  Classes: {len(label_encoder.classes_)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
