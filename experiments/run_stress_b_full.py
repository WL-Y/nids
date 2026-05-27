"""
Full Stress B evaluation on complete CICIDS v3 dataset using chunked reading.

Evaluates the trained Random Forest model (artifacts/best_model.joblib) on the
full NF-CSE-CIC-IDS2018-v3 target domain without retraining. Uses streaming
chunked evaluation to keep memory usage manageable while scoring every row.

Usage:
  python experiments/run_stress_b_full.py
  python experiments/run_stress_b_full.py --chunk-size 100000
"""

import os
import sys
import gc
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

from config import CICIDS_PATH, ATTACK_COL
from preprocessing.preprocess import prepare_cicids_chunk


def get_expected_columns(preprocessor, chunk):
    """Recover expected training feature columns from the preprocessor."""
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    return [c for c in chunk.columns if c not in [ATTACK_COL, "Label"]]


def update_binary_counts(y_true_bin, y_pred_bin, counts):
    counts["tp"] += int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    counts["tn"] += int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    counts["fp"] += int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    counts["fn"] += int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    counts["n"] += int(len(y_true_bin))


def compute_binary_metrics(counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    n = counts["n"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    return {
        "n_samples": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full Stress B evaluation on CICIDS v3 with chunked reading")
    parser.add_argument("--chunk-size", type=int, default=200000,
                        help="Rows per chunk (default: 200000)")
    parser.add_argument("--in-dist-f1", type=float, default=None,
                        help="UNSW in-distribution Macro-F1 for delta_f1. "
                             "Defaults to RandomForest macro_f1 in results/baseline_results.csv.")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    chunk_size = args.chunk_size

    print("Loading trained artifacts...")
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    model = joblib.load("artifacts/best_model.joblib")

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
    expected_cols = None

    print("Starting FULL Stress B evaluation on CICIDS")
    print(f"Path: {CICIDS_PATH}")
    print(f"Chunk size: {chunk_size:,}")

    reader = pd.read_csv(CICIDS_PATH, chunksize=chunk_size, low_memory=False)

    for chunk_id, chunk in enumerate(reader, start=1):
        print(f"\nProcessing chunk {chunk_id}: {len(chunk):,} rows")

        chunk = prepare_cicids_chunk(chunk)

        if expected_cols is None:
            expected_cols = get_expected_columns(preprocessor, chunk)
            print(f"Expected feature columns: {len(expected_cols)}")

        y_true = chunk[ATTACK_COL]
        y_true_bin = (y_true != "Benign").astype(int).values

        drop_target_cols = [ATTACK_COL]
        if "Label" in chunk.columns:
            drop_target_cols.append("Label")

        X_raw = chunk.drop(columns=drop_target_cols)
        X_raw = X_raw.reindex(columns=expected_cols, fill_value=0)
        X_t = preprocessor.transform(X_raw)

        y_pred = model.predict(X_t)
        y_pred_bin = (y_pred != "Benign").astype(int)

        update_binary_counts(y_true_bin, y_pred_bin, counts)

        partial = compute_binary_metrics(counts)
        print(
            f"Processed={partial['n_samples']:,} | "
            f"F1={partial['f1']:.4f} | "
            f"FPR={partial['fpr']:.4f} | "
            f"FNR={partial['fnr']:.4f}"
        )

        del chunk, X_raw, X_t, y_true, y_true_bin, y_pred, y_pred_bin
        gc.collect()

    metrics = compute_binary_metrics(counts)
    in_dist_f1 = args.in_dist_f1
    if in_dist_f1 is None and os.path.exists("results/baseline_results.csv"):
        baseline = pd.read_csv("results/baseline_results.csv")
        row = baseline[baseline["model"] == "RandomForest"]
        if len(row) > 0:
            in_dist_f1 = float(row["macro_f1"].iloc[0])
    if in_dist_f1 is not None:
        metrics["in_dist_f1"] = in_dist_f1
        metrics["delta_f1"] = in_dist_f1 - metrics["f1"]

    print("\n" + "=" * 50)
    print("FULL Stress B final results:")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    pd.DataFrame([metrics]).to_csv("results/stress_b_full_results.csv", index=False)
    print("\nSaved: results/stress_b_full_results.csv")


if __name__ == "__main__":
    main()
