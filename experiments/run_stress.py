"""
Phase 4: Stress Tests.

Usage:
  python experiments/run_stress.py --stress A    # Stress A only
  python experiments/run_stress.py --stress B    # Stress B only
  python experiments/run_stress.py --stress C    # Stress C only
  python experiments/run_stress.py --stress all  # all three
"""

import os
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from config import (
    UNSW_PATH, CICIDS_PATH, ATTACK_COL, HELD_OUT_CLASSES_SETS,
    NUMERIC_FEATURES, PROTOCOL_FEATURES, CATEGORICAL_FEATURES, SEED,
    STRESS_B_CHUNK_SIZE,
)
from preprocessing.preprocess import (
    load_and_clean_data, clean_l7_proto, split_data,
    build_preprocessor, fit_preprocessor,
)
from robustness.stress_tests import run_stress_a, run_stress_b_full_chunked, run_stress_c
from evaluation.plots import (
    plot_confidence_distribution, plot_confusion_matrix,
    plot_degradation_curve,
)
from evaluation.metrics import macro_f1_score


def prepare_data():
    """Load, clean, split, and preprocess UNSW data."""
    print("Loading and preparing UNSW data ...")
    df = load_and_clean_data(UNSW_PATH)
    df = clean_l7_proto(df)

    y_full = df[ATTACK_COL]
    X_full = df.drop(columns=[ATTACK_COL, "Label"])

    df_for_split = X_full.copy()
    df_for_split[ATTACK_COL] = y_full
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_for_split)

    # Keep raw copies for Stress B/C
    X_train_raw = X_train.copy()
    X_test_raw = X_test.copy()

    y_train = X_train.pop(ATTACK_COL)
    y_val = X_val.pop(ATTACK_COL)
    y_test = X_test.pop(ATTACK_COL)

    preprocessor = build_preprocessor()
    preprocessor = fit_preprocessor(preprocessor, X_train)

    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    class_names = sorted(y_train.unique())
    le = joblib.load("artifacts/label_encoder.joblib")

    return {
        "X_train": X_train_t, "X_val": X_val_t, "X_test": X_test_t,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_raw": X_train, "X_val_raw": X_val, "X_test_raw": X_test_raw,
        "preprocessor": preprocessor, "class_names": class_names,
        "label_encoder": le,
    }


def train_full_model(data):
    """Train a RandomForest on full UNSW training set using known best params."""
    print("\nTraining RF on full UNSW training set ...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=2,
        min_samples_split=5, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    model.fit(data["X_train"], data["y_train"])

    # Evaluate on UNSW test (in-distribution reference)
    y_pred = model.predict(data["X_test"])
    in_dist_f1 = macro_f1_score(data["y_test"], y_pred)
    print(f"In-distribution (UNSW test) Macro-F1: {in_dist_f1:.4f}")

    return model, in_dist_f1


def run_stress_a_experiments(data, model_name, use_gpu):
    """Run Stress A for both held-out groups with strict preprocessing."""
    print("\n" + "=" * 60)
    print("Phase 4a: Stress Test A — Held-Out Attack Classes")
    print("=" * 60)

    all_results = []

    for i, held_out in enumerate(HELD_OUT_CLASSES_SETS):
        group_name = f"set{i+1}_{'_'.join(held_out)}"
        print(f"\n{'='*50}")
        print(f"Group {i+1}: {held_out}")
        print(f"{'='*50}")

        # Strict preprocessing: fit preprocessor only on known-class data
        known_classes = sorted(set(data["class_names"]) - set(held_out))
        train_mask = data["y_train"].isin(known_classes)
        val_mask = data["y_val"].isin(known_classes)

        X_train_raw_k = data["X_train_raw"].loc[train_mask]
        y_train_k = data["y_train"][train_mask]
        X_val_raw_k = data["X_val_raw"].loc[val_mask]
        y_val_k = data["y_val"][val_mask]

        preprocessor_k = build_preprocessor()
        preprocessor_k.fit(X_train_raw_k)
        X_train_k = preprocessor_k.transform(X_train_raw_k)
        X_val_k = preprocessor_k.transform(X_val_raw_k)
        X_test_k = preprocessor_k.transform(data["X_test_raw"])

        print(f"Strict preprocessing: fitted on {len(X_train_k):,} known-class samples")

        result = run_stress_a(
            X_train=X_train_k, y_train=y_train_k,
            X_val=X_val_k, y_val=y_val_k,
            X_test=X_test_k, y_test=data["y_test"],
            held_out_classes=held_out,
            model_name=model_name,
            class_names=data["class_names"],
            label_encoder=data["label_encoder"],
            use_gpu=use_gpu,
        )
        all_results.append(result)

        os.makedirs("reports/figures", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        conf_df = result["confidence_df"].copy()
        conf_df["status"] = "unknown"
        known_mask = ~conf_df["is_unknown"]
        conf_df.loc[known_mask & conf_df["is_correct"], "status"] = "known_correct"
        conf_df.loc[known_mask & ~conf_df["is_correct"], "status"] = "known_wrong"

        plot_confidence_distribution(conf_df,
                                     f"reports/figures/stress_a_confidence_{group_name}.png")
        # Known-class confusion matrix
        plot_confusion_matrix(result["confusion_matrix_known"],
                              result["known_classes"],
                              f"reports/figures/stress_a_confusion_{group_name}.png")
        # Full 10-class confusion matrix (known + unknown)
        plot_confusion_matrix(result["confusion_matrix_full"],
                              data["class_names"],
                              f"reports/figures/stress_a_full_confusion_{group_name}.png")
        result["confusion_matrix_full"].to_csv(
            f"results/stress_a_full_confusion_{group_name}.csv", index=True)
        print(f"  Saved: reports/figures/stress_a_full_confusion_{group_name}.png")
        print(f"  Saved: results/stress_a_full_confusion_{group_name}.csv")

        result["per_class_report"].to_csv(
            f"results/stress_a_report_{group_name}.csv", index=False)

        # Save unknown-to-known mapping
        mapping_rows = []
        for cls, info in result["unknown_mapping"].items():
            for rank, (pred_cls, count) in enumerate(info["top3_predictions"]):
                mapping_rows.append({
                    "held_out_class": cls,
                    "held_out_count": info["count"],
                    "rank": rank + 1,
                    "predicted_as": pred_cls,
                    "count": count,
                })
        pd.DataFrame(mapping_rows).to_csv(
            f"results/stress_a_mapping_{group_name}.csv", index=False)
        print(f"  Saved: results/stress_a_mapping_{group_name}.csv")

    summary_rows = []
    for result, held_out in zip(all_results, HELD_OUT_CLASSES_SETS):
        summary_rows.append({
            "held_out_classes": ", ".join(held_out),
            "known_macro_f1": round(result["known_macro_f1"], 4),
            "unknown_mean_conf": round(result["unknown_mean_conf"], 4),
            "known_correct_mean_conf": round(result["known_correct_mean_conf"], 4),
            "known_wrong_mean_conf": round(result["known_wrong_mean_conf"], 4),
            "auroc_confidence_unknown": round(result["auroc_confidence_unknown"], 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("results/stress_a_results.csv", index=False)
    print(f"\nSaved: results/stress_a_results.csv")
    print(summary.to_string(index=False))

    return all_results


def run_stress_b_experiment(data, model, in_dist_f1):
    """Run Stress B: UNSW -> full CICIDS2018 cross-dataset with chunked reading."""
    print("\n" + "=" * 60)
    print("Phase 4b: Stress Test B — Cross-Dataset Generalization")
    print("=" * 60)

    result = run_stress_b_full_chunked(
        model, data["preprocessor"], CICIDS_PATH, chunk_size=STRESS_B_CHUNK_SIZE
    )

    delta_f1 = in_dist_f1 - result["f1"]
    print(f"\n  Delta F1 (in-dist - target): {delta_f1:.4f}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{**result, "in_dist_f1": in_dist_f1, "delta_f1": delta_f1}]).to_csv(
        "results/stress_b_full_results.csv", index=False
    )
    print("\nSaved: results/stress_b_full_results.csv")

    return result


def run_stress_c_experiment(data, model):
    """Run Stress C: Feature degradation."""
    print("\n" + "=" * 60)
    print("Phase 4b: Stress Test C — Feature Degradation")
    print("=" * 60)

    # Extract numeric feature importances
    n_numeric = len(NUMERIC_FEATURES)
    feature_importances = model.feature_importances_[:n_numeric]
    numeric_names = list(NUMERIC_FEATURES)

    print("Top 5 most important numeric features:")
    ranked = sorted(zip(numeric_names, feature_importances), key=lambda x: x[1], reverse=True)
    for name, imp in ranked[:5]:
        print(f"  {name}: {imp:.4f}")

    results = run_stress_c(
        X_test_raw=data["X_test_raw"],
        y_test=data["y_test"],
        preprocessor=data["preprocessor"],
        model=model,
        class_names=data["class_names"],
        feature_importances=feature_importances,
        numeric_feature_names=numeric_names,
    )

    # Save results
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # Build summary table
    rows = []
    for sigma, f1 in zip(results["noise"]["levels"], results["noise"]["macro_f1"]):
        rows.append({"type": "noise", "level": sigma, "macro_f1": round(f1, 4)})
    for p, f1 in zip(results["masking"]["levels"], results["masking"]["macro_f1"]):
        rows.append({"type": "masking", "level": p, "macro_f1": round(f1, 4)})
    for k, f1 in zip(results["dropout_top"]["levels"], results["dropout_top"]["macro_f1"]):
        rows.append({"type": "dropout_top", "level": k, "macro_f1": round(f1, 4)})
    for k, f1 in zip(results["dropout_bottom"]["levels"], results["dropout_bottom"]["macro_f1"]):
        rows.append({"type": "dropout_bottom", "level": k, "macro_f1": round(f1, 4)})

    summary = pd.DataFrame(rows)
    summary.to_csv("results/stress_c_results.csv", index=False)
    print(f"\nSaved: results/stress_c_results.csv")
    print(summary.to_string(index=False))

    # Degradation curve
    deg_dict = {
        "Noise": (results["noise"]["levels"], results["noise"]["macro_f1"]),
        "Masking": (results["masking"]["levels"], results["masking"]["macro_f1"]),
        "Dropout (top)": (results["dropout_top"]["levels"], results["dropout_top"]["macro_f1"]),
        "Dropout (bottom)": (results["dropout_bottom"]["levels"], results["dropout_bottom"]["macro_f1"]),
    }
    plot_degradation_curve(deg_dict, "reports/figures/stress_c_degradation_curve.png")
    print("Saved: reports/figures/stress_c_degradation_curve.png")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", default="all",
                        help="Which stress test to run (A, B, C, or 'all')")
    parser.add_argument("--model", default="RandomForest",
                        help="Model to use for Stress A (default: RandomForest)")
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    use_gpu = args.use_gpu and not args.no_gpu
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    data = prepare_data()

    if args.stress in ("A", "all"):
        run_stress_a_experiments(data, args.model, use_gpu)

    if args.stress in ("B", "C", "all"):
        # Train one model for both B and C
        model, in_dist_f1 = train_full_model(data)

        if args.stress in ("B", "all"):
            run_stress_b_experiment(data, model, in_dist_f1)

        if args.stress in ("C", "all"):
            run_stress_c_experiment(data, model)

    print(f"\n{'=' * 60}")
    print("Phase 4 complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
