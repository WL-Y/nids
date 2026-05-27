"""
Phase 2b (pre): Data validation and feature column metadata generation.

1. Extracts and saves the final modeling feature list to data/metadata/feature_columns.json.
2. Checks feature compatibility between UNSW and CICIDS2018 datasets.
3. Saves common features to data/metadata/common_features_unsw_cicids.json.
4. Validates the Attack and Label columns on the full UNSW dataset.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import (
    UNSW_PATH, CICIDS_PATH,
    ATTACK_COL, LABEL_COL, DROP_COLUMNS,
    NUMERIC_FEATURES, PROTOCOL_FEATURES, CATEGORICAL_FEATURES,
    SEED,
)


def extract_feature_columns(df_columns):
    """Extract modeling feature columns by excluding target, label, and drop columns."""
    exclude = set(DROP_COLUMNS + [ATTACK_COL, LABEL_COL])
    feature_cols = [c for c in df_columns if c not in exclude]
    return feature_cols


def check_feature_compatibility(unsw_cols, cicids_cols):
    """Compare feature columns between UNSW and CICIDS2018."""
    unsw_features = extract_feature_columns(unsw_cols)
    cicids_features = extract_feature_columns(cicids_cols)

    only_in_unsw = sorted(set(unsw_features) - set(cicids_features))
    only_in_cicids = sorted(set(cicids_features) - set(unsw_features))
    common = sorted(set(unsw_features).intersection(set(cicids_features)))

    print("\n=== Feature Compatibility: UNSW vs CICIDS2018 ===")
    print(f"UNSW feature columns:     {len(unsw_features)}")
    print(f"CICIDS2018 feature columns: {len(cicids_features)}")
    print(f"Common feature columns:   {len(common)}")
    print(f"Only in UNSW:             {only_in_unsw if only_in_unsw else '(none)'}")
    print(f"Only in CICIDS2018:       {only_in_cicids if only_in_cicids else '(none)'}")

    if only_in_unsw or only_in_cicids:
        print("\nWARNING: Feature sets differ. Stress B will use common_features subset.")
    else:
        print("\nFeature sets are identical. Stress B can directly reuse UNSW preprocessor.")

    return {
        "unsw_feature_count": len(unsw_features),
        "cicids_feature_count": len(cicids_features),
        "common_feature_count": len(common),
        "only_in_unsw": only_in_unsw,
        "only_in_cicids": only_in_cicids,
        "common_features": common,
        "identical": len(only_in_unsw) == 0 and len(only_in_cicids) == 0,
    }


def check_dtype_compatibility(df_unsw_head, df_cicids_head, common_features):
    """Compare dtypes of common features between the two datasets."""
    mismatches = []
    for col in common_features:
        dt_unsw = df_unsw_head[col].dtype
        dt_cicids = df_cicids_head[col].dtype
        if dt_unsw != dt_cicids:
            mismatches.append((col, str(dt_unsw), str(dt_cicids)))

    print("\n=== Dtype Compatibility (common features) ===")
    if mismatches:
        print(f"Dtype mismatches: {len(mismatches)}")
        for col, dt_u, dt_c in mismatches:
            print(f"  {col}: UNSW={dt_u}, CICIDS2018={dt_c}")
    else:
        print("All common features have matching dtypes.")
    return mismatches


def validate_target_columns(df):
    """Validate the Attack and Label columns on the full UNSW dataset."""
    print("\n=== Target Column Validation ===")

    # Attack column
    attack_classes = df[ATTACK_COL].value_counts()
    print(f"\n{ATTACK_COL} column:")
    print(f"  Unique classes: {len(attack_classes)}")
    print(f"  Dtype: {df[ATTACK_COL].dtype}")
    print(f"  Nulls: {df[ATTACK_COL].isnull().sum()}")
    print(f"  Class distribution:")
    for cls, cnt in attack_classes.items():
        print(f"    {cls}: {cnt:,} ({cnt / len(df) * 100:.2f}%)")

    # Label column
    print(f"\n{LABEL_COL} column:")
    print(f"  Dtype: {df[LABEL_COL].dtype}")
    print(f"  Unique values: {sorted(df[LABEL_COL].dropna().unique())}")
    print(f"  Nulls: {df[LABEL_COL].isnull().sum()}")
    label_counts = df[LABEL_COL].value_counts().sort_index()
    for val, cnt in label_counts.items():
        print(f"    {val}: {cnt:,} ({cnt / len(df) * 100:.2f}%)")

    # Verify Label=0 corresponds to Benign, Label=1 to non-Benign
    benign_label_vals = set(df[df[ATTACK_COL] == "Benign"][LABEL_COL].unique())
    attack_label_vals = set(df[df[ATTACK_COL] != "Benign"][LABEL_COL].unique())
    print(f"\n  Label values for Benign rows:  {sorted(benign_label_vals)}")
    print(f"  Label values for Attack rows:  {sorted(attack_label_vals)}")

    if benign_label_vals == {0} and attack_label_vals == {1}:
        print("  Label <-> Attack consistency: OK (0=Benign, 1=Attack)")
    else:
        print("  WARNING: Label <-> Attack mapping is inconsistent!")

    # Map between Attack string labels and their binary Label values
    print("\n  Attack class -> Label mapping:")
    for cls in sorted(attack_classes.index):
        lbl = df[df[ATTACK_COL] == cls][LABEL_COL].iloc[0]
        print(f"    {cls}: Label={lbl}")

    return {
        "num_classes": len(attack_classes),
        "classes": sorted(attack_classes.index.tolist()),
        "class_distribution": {k: int(v) for k, v in attack_classes.items()},
        "total_samples": len(df),
        "label_values": sorted(df[LABEL_COL].dropna().unique().tolist()),
    }


def build_meta_lists(common_features):
    """Build common-feature sub-lists for numeric, protocol, and categorical pipelines."""
    common_numeric = [c for c in NUMERIC_FEATURES if c in common_features]
    common_protocol = [c for c in PROTOCOL_FEATURES if c in common_features]
    common_categorical = [c for c in CATEGORICAL_FEATURES if c in common_features]
    return common_numeric, common_protocol, common_categorical


def main():
    print("=" * 60)
    print("Data Validation & Feature Metadata Generation")
    print("=" * 60)

    os.makedirs("data/metadata", exist_ok=True)

    # --- Step 1: Load UNSW full data for target validation ---
    print(f"\nLoading {UNSW_PATH} ...")
    df_unsw = pd.read_csv(UNSW_PATH, low_memory=False)
    print(f"Shape: {df_unsw.shape}")

    # --- Step 2: Extract and save feature columns ---
    feature_cols = extract_feature_columns(df_unsw.columns)
    print(f"\nTotal columns in CSV: {len(df_unsw.columns)}")
    print(f"Modeling feature columns: {len(feature_cols)}")
    print(f"  Numeric (regular): {len([c for c in NUMERIC_FEATURES if c in feature_cols])}")
    print(f"  Numeric (protocol): {len([c for c in PROTOCOL_FEATURES if c in feature_cols])}")
    print(f"  Categorical:        {len([c for c in CATEGORICAL_FEATURES if c in feature_cols])}")

    feature_meta = {
        "description": "Final modeling feature columns for NF-UQ-NIDS-v3 (after removing labels, IPs, timestamps)",
        "total_features": len(feature_cols),
        "features": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "protocol_features": PROTOCOL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "dropped_columns": DROP_COLUMNS,
        "target_column": ATTACK_COL,
        "label_column": LABEL_COL,
    }

    with open("data/metadata/feature_columns.json", "w") as f:
        json.dump(feature_meta, f, indent=2)
    print("Saved: data/metadata/feature_columns.json")

    # --- Step 3: Feature compatibility with CICIDS2018 ---
    print(f"\nLoading header of {CICIDS_PATH} ...")
    try:
        df_cicids_head = pd.read_csv(CICIDS_PATH, nrows=5, low_memory=False)
        print(f"CICIDS2018 columns: {len(df_cicids_head.columns)}")

        compat = check_feature_compatibility(df_unsw.columns, df_cicids_head.columns)

        # Dtype check
        dtype_mismatches = check_dtype_compatibility(
            df_unsw.head(5), df_cicids_head, compat["common_features"]
        )

        # Build common-feature sub-lists
        common_num, common_proto, common_cat = build_meta_lists(compat["common_features"])

        common_meta = {
            "description": "Common feature columns between UNSW and CICIDS2018 for Stress B",
            "identical_feature_sets": compat["identical"],
            "common_features": compat["common_features"],
            "unsw_only": compat["only_in_unsw"],
            "cicids_only": compat["only_in_cicids"],
            "common_numeric_features": common_num,
            "common_protocol_features": common_proto,
            "common_categorical_features": common_cat,
            "dtype_mismatches": [
                {"feature": col, "unsw_dtype": du, "cicids_dtype": dc}
                for col, du, dc in dtype_mismatches
            ],
        }

        with open("data/metadata/common_features_unsw_cicids.json", "w") as f:
            json.dump(common_meta, f, indent=2)
        print("Saved: data/metadata/common_features_unsw_cicids.json")

    except FileNotFoundError:
        print(f"WARNING: {CICIDS_PATH} not found. Skipping compatibility check.")

    # --- Step 4: Validate target columns ---
    target_info = validate_target_columns(df_unsw)

    # Save class mapping
    class_mapping = {i: cls for i, cls in enumerate(target_info["classes"])}
    with open("data/metadata/class_mapping.json", "w") as f:
        json.dump({
            "description": "Integer index to Attack class name mapping",
            "mapping": class_mapping,
            "num_classes": target_info["num_classes"],
        }, f, indent=2)
    print("Saved: data/metadata/class_mapping.json")

    # --- Step 5: Summary ---
    print("\n" + "=" * 60)
    print("Validation complete.")
    print(f"  Feature columns:    {len(feature_cols)} -> data/metadata/feature_columns.json")
    print(f"  Target classes:     {target_info['num_classes']} -> data/metadata/class_mapping.json")
    print(f"  Total samples:      {target_info['total_samples']:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
