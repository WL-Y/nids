"""
Phase 3: Baseline Classifier Experiments.

Trains and evaluates 4 models:
  1. Majority Classifier (floor baseline)
  2. Logistic Regression
  3. Random Forest
  4. XGBoost

Saves results to results/baseline_results.csv and confusion matrices to
reports/figures/. For statistical significance, use experiments/run_significance.py.
"""

import os
import sys
import json
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

from config import (
    UNSW_PATH, ATTACK_COL, SIGNIFICANCE_SEEDS, SEED,
)
from preprocessing.preprocess import (
    load_and_clean_data, clean_l7_proto, split_data,
    build_preprocessor, fit_preprocessor,
)
from models.baseline import (
    MajorityClassifier, train_logistic_regression, train_random_forest,
)
from models.chosen_model import train_xgboost
from evaluation.metrics import (
    classification_report_full, confusion_matrix_df, macro_f1_score,
    weighted_f1_score, run_statistics,
)
from evaluation.plots import plot_confusion_matrix


def prepare_data():
    """Load, clean, split, and preprocess data. Returns processed dict."""
    print("Loading and preparing data ...")
    df = load_and_clean_data(UNSW_PATH)
    df = clean_l7_proto(df)

    y_full = df[ATTACK_COL]
    X_full = df.drop(columns=[ATTACK_COL, "Label"])

    df_for_split = X_full.copy()
    df_for_split[ATTACK_COL] = y_full
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_for_split)

    y_train = X_train.pop(ATTACK_COL)
    y_val = X_val.pop(ATTACK_COL)
    y_test = X_test.pop(ATTACK_COL)

    preprocessor = build_preprocessor()
    preprocessor = fit_preprocessor(preprocessor, X_train)

    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    class_names = sorted(y_train.unique())

    # Load LabelEncoder for XGBoost integer encoding
    le = joblib.load("artifacts/label_encoder.joblib")

    return {
        "X_train": X_train_t, "X_val": X_val_t, "X_test": X_test_t,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "preprocessor": preprocessor, "class_names": class_names,
        "label_encoder": le,
    }


def evaluate_model(model, X_test, y_test, class_names, model_name, le):
    """Evaluate a trained model and return metrics dict."""
    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - t0

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    # Per-class report
    report = classification_report_full(y_test, y_pred, classes=class_names)

    # Confusion matrix
    cm = confusion_matrix_df(y_test, y_pred, class_names)

    # Key metrics
    macro_f1 = macro_f1_score(y_test, y_pred)
    weighted_f1 = weighted_f1_score(y_test, y_pred)

    print(f"\n  {model_name} Results:")
    print(f"  Macro-F1:    {macro_f1:.4f}")
    print(f"  Weighted-F1: {weighted_f1:.4f}")
    print(f"  Inference:   {infer_time:.2f}s ({infer_time / len(y_test) * 1000:.2f}ms / 1k samples)")

    return {
        "model": model_name,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "inference_time_s": infer_time,
        "per_class_report": report,
        "confusion_matrix": cm,
        "y_prob": y_prob,
    }


def train_single_model(model_name, data, use_gpu):
    """Train a single model by name. Returns (model, result_dict)."""
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    class_names = data["class_names"]
    le = data["label_encoder"]

    t0 = time.time()

    if model_name == "Majority":
        model = MajorityClassifier()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"\n  Majority training time: {train_time:.1f}s")

    elif model_name == "LogisticRegression":
        model, best_params, cv_df = train_logistic_regression(
            X_train, y_train, X_val, y_val)
        train_time = time.time() - t0

    elif model_name == "RandomForest":
        model, best_params, cv_df = train_random_forest(
            X_train, y_train, X_val, y_val)
        train_time = time.time() - t0

    elif model_name == "XGBoost":
        y_train_enc = le.transform(y_train)
        y_val_enc = le.transform(y_val)
        model, best_params, cv_df = train_xgboost(
            X_train, y_train_enc, X_val, y_val_enc, use_gpu=use_gpu)
        train_time = time.time() - t0
        # Wrap XGBoost to output string labels directly
        model = XGBWrapper(model, le)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    result = evaluate_model(model, X_test, y_test, class_names, model_name, le)
    result["train_time_s"] = train_time
    return model, result


class XGBWrapper:
    """Wrap XGBoost so predict/predict_proba return string labels."""

    def __init__(self, xgb_model, label_encoder):
        self.model = xgb_model
        self.le = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X):
        enc = self.model.predict(X)
        return self.le.inverse_transform(enc)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


_XGBWrapper = XGBWrapper  # backward-compat: old pickles reference _XGBWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gpu", action="store_true", default=True,
                        help="Use GPU for XGBoost (default: True)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU for XGBoost")
    parser.add_argument("--models", nargs="+",
                        default=["Majority", "LogisticRegression", "RandomForest", "XGBoost"],
                        help="Models to train (default: all)")
    args = parser.parse_args()

    use_gpu = args.use_gpu and not args.no_gpu

    os.makedirs("results", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    print("=" * 60)
    print("Phase 3: Baseline Classifier Experiments")
    print("=" * 60)

    data = prepare_data()

    results = []
    models_dict = {}

    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")

        model, result = train_single_model(model_name, data, use_gpu)
        results.append(result)
        models_dict[model_name] = model

        # Save confusion matrix
        cm = result["confusion_matrix"]
        cm_path = f"reports/figures/confusion_matrix_{model_name}.png"
        plot_confusion_matrix(cm, data["class_names"], cm_path)
        print(f"  Saved: {cm_path}")

        # Save per-class report
        report = result["per_class_report"]
        report.to_csv(f"results/per_class_report_{model_name}.csv", index=False)
        print(f"  Saved: results/per_class_report_{model_name}.csv")

    # ---- Save summary ----
    summary_rows = []
    for r in results:
        row = {
            "model": r["model"],
            "macro_f1": round(r["macro_f1"], 4),
            "weighted_f1": round(r["weighted_f1"], 4),
            "train_time_s": round(r.get("train_time_s", 0), 1),
            "inference_time_s": round(r.get("inference_time_s", 0), 2),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("results/baseline_results.csv", index=False)
    print(f"\nSaved: results/baseline_results.csv")
    print(summary.to_string(index=False))

    # Save best model (highest Macro-F1)
    best_result = max(results, key=lambda r: r["macro_f1"])
    best_model_name = best_result["model"]
    print(f"\nBest model: {best_model_name} (Macro-F1={best_result['macro_f1']:.4f})")

    # Save best model to artifacts/
    import joblib
    from pathlib import Path
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    best_model = models_dict[best_model_name]
    joblib.dump(best_model, "artifacts/best_model.joblib")
    print("Saved: artifacts/best_model.joblib")

    # Update training_config with best model name
    config_path = "artifacts/training_config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            tc = json.load(f)
        tc["best_model_name"] = best_model_name
        with open(config_path, "w") as f:
            json.dump(tc, f, indent=2)
        print("Updated: artifacts/training_config.json")

    print(f"\n{'=' * 60}")
    print("Phase 3 complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
