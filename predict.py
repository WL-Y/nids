"""
CLI prototype for network flow intrusion detection.

Usage:
  python predict.py --input sample_flow.csv
  python predict.py --input sample.csv --strategy confidence_threshold --tau 0.85
  python predict.py --input sample.csv --strategy ensemble
  python predict.py --input sample.csv --strategy ensemble+threshold --tau 0.85

Strategy options:
  - none:                    Direct prediction output
  - confidence_threshold:    Confidence threshold rejection (requires --tau)
  - ensemble:                Ensemble majority vote + disagreement score
  - ensemble+threshold:      Combined: confidence threshold + ensemble disagreement
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

from config import DROP_COLUMNS, ATTACK_COL, LABEL_COL


def load_model_and_artifacts():
    """Load all required artifacts. Never fits on inference data."""
    print("Loading artifacts ...")

    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    print("  preprocessor.joblib")

    model = joblib.load("artifacts/best_model.joblib")
    print("  best_model.joblib")

    label_encoder = joblib.load("artifacts/label_encoder.joblib")
    print("  label_encoder.joblib")

    # Load strategy config (optional)
    strategy_config = None
    if os.path.exists("artifacts/strategy_config.json"):
        with open("artifacts/strategy_config.json") as f:
            strategy_config = json.load(f)
        print("  strategy_config.json")

    return preprocessor, model, label_encoder, strategy_config


def load_ensemble_models():
    """Load ensemble models for ensemble-based strategies."""
    import experiments.run_baseline  # noqa: needed by joblib to unpickle _XGBWrapper/XGBWrapper
    ensemble_dir = "artifacts/ensemble_models"
    if not os.path.exists(ensemble_dir):
        print("ERROR: Ensemble models not found. Run Phase 5 first.")
        sys.exit(1)

    models = []
    for fname in sorted(os.listdir(ensemble_dir)):
        if fname.endswith(".joblib"):
            m = joblib.load(os.path.join(ensemble_dir, fname))
            name = fname.replace(".joblib", "")
            models.append((name, m))
            print(f"  ensemble/{fname}")
    return models


def preprocess_input(df):
    """
    Preprocess raw input for inference.
    Applies same cleaning as training but NEVER fits.
    """
    df = df.copy()

    # 1. Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2. Clean L7_PROTO if present
    if "L7_PROTO" in df.columns:
        col = df["L7_PROTO"]
        col = col.fillna(0).round().astype(int)
        df["L7_PROTO"] = col

    # 3. Drop identifier columns if present
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 4. Remove label columns if present (inference data should not have labels)
    for col in [ATTACK_COL, LABEL_COL]:
        if col in df.columns:
            df = df.drop(columns=col)

    return df


def decision_for_prediction(prediction, suffix=""):
    """Return operator-facing decision text for a class prediction."""
    if prediction == "Benign":
        return f"NO ALERT{suffix}"
    return f"ALERT{suffix}"


def predict_none(model, X, class_names):
    """Direct prediction without any strategy."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    max_probs = np.max(y_prob, axis=1)
    pred_indices = np.argmax(y_prob, axis=1)

    results = []
    for i in range(len(X)):
        results.append({
            "flow_index": i,
            "prediction": y_pred[i],
            "confidence": max_probs[i],
            "disagreement": np.nan,
            "decision": decision_for_prediction(y_pred[i]),
        })
    return results


def predict_confidence_threshold(model, X, class_names, tau):
    """Strategy 1: Confidence threshold rejection."""
    y_prob = model.predict_proba(X)
    max_probs = np.max(y_prob, axis=1)
    pred_indices = np.argmax(y_prob, axis=1)

    results = []
    for i in range(len(X)):
        if max_probs[i] >= tau:
            pred = class_names[pred_indices[i]] if class_names else str(pred_indices[i])
            decision = decision_for_prediction(pred, " (above threshold)")
        else:
            pred = "UNKNOWN"
            decision = "FLAG FOR REVIEW (below threshold)"

        results.append({
            "flow_index": i,
            "prediction": pred,
            "confidence": max_probs[i],
            "disagreement": np.nan,
            "decision": decision,
        })
    return results


def predict_ensemble(ensemble_models, X, class_names):
    """Strategy 2: Ensemble majority vote with disagreement."""
    from robustness.strategies import ensemble_disagreement, ensemble_predict_proba

    disagreement, majority = ensemble_disagreement(ensemble_models, X)
    avg_proba = ensemble_predict_proba(ensemble_models, X)
    max_probs = np.max(avg_proba, axis=1)

    results = []
    for i in range(len(X)):
        if disagreement[i] >= 0.4:
            decision = "UNCERTAIN (high disagreement)"
        else:
            decision = decision_for_prediction(majority[i])

        results.append({
            "flow_index": i,
            "prediction": majority[i],
            "confidence": max_probs[i],
            "disagreement": disagreement[i],
            "decision": decision,
        })
    return results


def predict_ensemble_threshold(ensemble_models, X, class_names, tau):
    """Combined: Ensemble + confidence threshold."""
    from robustness.strategies import ensemble_disagreement, ensemble_predict_proba

    disagreement, majority = ensemble_disagreement(ensemble_models, X)
    avg_proba = ensemble_predict_proba(ensemble_models, X)
    max_probs = np.max(avg_proba, axis=1)

    results = []
    for i in range(len(X)):
        pred = majority[i]

        if max_probs[i] < tau:
            pred = "UNKNOWN"
            decision = "FLAG FOR REVIEW (below threshold)"
        elif disagreement[i] >= 0.4:
            decision = "UNCERTAIN (high disagreement)"
        else:
            decision = decision_for_prediction(pred, " (high confidence + consensus)")

        results.append({
            "flow_index": i,
            "prediction": pred,
            "confidence": max_probs[i],
            "disagreement": disagreement[i],
            "decision": decision,
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="NetFlow Intrusion Detection — CLI Prototype")
    parser.add_argument("--input", required=True,
                        help="Path to input CSV file")
    parser.add_argument("--strategy", default="none",
                        choices=["none", "confidence_threshold", "ensemble",
                                 "ensemble+threshold"],
                        help="Prediction strategy (default: none)")
    parser.add_argument("--tau", type=float, default=0.85,
                        help="Confidence threshold for rejection strategies (default: 0.85)")
    parser.add_argument("--output", default=None,
                        help="Save results to CSV (optional)")
    args = parser.parse_args()

    # ---- Load ----
    preprocessor, model, label_encoder, strategy_config = load_model_and_artifacts()
    class_names = label_encoder.classes_.tolist()

    # Load ensemble models if needed
    ensemble_models = None
    if args.strategy in ("ensemble", "ensemble+threshold"):
        ensemble_models = load_ensemble_models()

    # Use saved tau if strategy uses threshold and not explicitly overridden
    if args.strategy in ("confidence_threshold", "ensemble+threshold"):
        if args.tau == 0.85 and strategy_config and "tau" in strategy_config:
            args.tau = strategy_config["tau"]
            print(f"Using saved tau from strategy_config: {args.tau}")

    # ---- Read & preprocess input ----
    print(f"\nReading input: {args.input}")
    df_raw = pd.read_csv(args.input, low_memory=False)
    print(f"Input shape: {df_raw.shape}")

    df = preprocess_input(df_raw)
    X = preprocessor.transform(df)  # transform ONLY, never fit
    print(f"Preprocessed shape: {X.shape}")

    # ---- Predict ----
    print(f"\nApplying strategy: {args.strategy}")
    if args.strategy == "none":
        results = predict_none(model, X, class_names)
    elif args.strategy == "confidence_threshold":
        results = predict_confidence_threshold(model, X, class_names, args.tau)
    elif args.strategy == "ensemble":
        results = predict_ensemble(ensemble_models, X, class_names)
    elif args.strategy == "ensemble+threshold":
        results = predict_ensemble_threshold(ensemble_models, X, class_names, args.tau)

    # ---- Output ----
    print(f"\n{'='*50}")
    for r in results[:20]:  # Print first 20
        print(f"Flow #{r['flow_index']}")
        print(f"  Predicted class: {r['prediction']}")
        print(f"  Confidence:      {r['confidence']:.4f}")
        if not np.isnan(r['disagreement']):
            print(f"  Disagreement:    {r['disagreement']:.4f}")
        print(f"  Decision:        {r['decision']}")
        print()

    if len(results) > 20:
        print(f"... ({len(results) - 20} more flows)")

    if args.output:
        df_out = pd.DataFrame(results)
        df_out.to_csv(args.output, index=False)
        print(f"\nSaved all results to: {args.output}")

    # Summary stats
    df_r = pd.DataFrame(results)
    print(f"\nSummary ({len(results)} flows):")
    print(f"  Decision distribution:")
    for dec, cnt in df_r["decision"].value_counts().items():
        print(f"    {dec}: {cnt} ({cnt/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
