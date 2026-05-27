"""
Stress test implementations.

Stress A: Held-out attack classes (open-set conditions).
Stress B: Cross-dataset generalization (distribution shift).
Stress C: Feature degradation (noise, masking, dropout).
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from config import (
    SEED, NUMERIC_FEATURES, GAUSSIAN_NOISE_STDS,
    MASKING_RATES, FEATURE_DROPOUT_COUNTS,
)
from models.baseline import MajorityClassifier, train_logistic_regression, train_random_forest
from models.chosen_model import train_xgboost
from evaluation.metrics import (
    classification_report_full, confusion_matrix_df, macro_f1_score,
    confidence_analysis, per_class_confidence_stats,
)


def _train_model(model_name, X_train, y_train, X_val, y_val, use_gpu, label_encoder):
    """Train a model by name using known best params (no grid search). Returns fitted model."""
    if model_name == "Majority":
        return MajorityClassifier().fit(X_train, y_train)

    elif model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=10, max_iter=3000, penalty="l2", solver="lbfgs",
                                   class_weight="balanced", random_state=SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        return model

    elif model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_leaf=2,
            min_samples_split=5, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        from models.chosen_model import get_xgb_device
        device_params = get_xgb_device(use_gpu)
        y_train_enc = label_encoder.transform(y_train)
        y_val_enc = label_encoder.transform(y_val)
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=1.0, colsample_bytree=0.8,
            objective="multi:softprob", eval_metric="mlogloss",
            early_stopping_rounds=20, random_state=SEED, n_jobs=-1,
            **device_params,
        )
        model.fit(X_train, y_train_enc, eval_set=[(X_val, y_val_enc)], verbose=False)
        from experiments.run_baseline import XGBWrapper
        return XGBWrapper(model, label_encoder)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_stress_a(X_train, y_train, X_val, y_val, X_test, y_test,
                 held_out_classes, model_name, class_names, label_encoder,
                 use_gpu=True):
    """
    Stress Test A: Evaluate model with held-out attack classes.

    1. Remove held-out classes from train and val
    2. Train model on known classes only
    3. Evaluate on full test set (known + unknown)

    Returns dict with per-class results, unknown mapping, confidence stats.
    """
    known_classes = sorted(set(class_names) - set(held_out_classes))
    print(f"\nHeld-out classes: {held_out_classes}")
    print(f"Known classes ({len(known_classes)}): {known_classes}")

    # Filter train and val to known classes only
    train_mask = y_train.isin(known_classes)
    val_mask = y_val.isin(known_classes)

    X_train_k = X_train[train_mask]
    y_train_k = y_train[train_mask]
    X_val_k = X_val[val_mask]
    y_val_k = y_val[val_mask]

    print(f"Train (known only): {len(X_train_k):,} samples")
    print(f"Val   (known only): {len(X_val_k):,} samples")

    # Fit a new label encoder on known classes only (for XGBoost)
    from sklearn.preprocessing import LabelEncoder
    le_known = LabelEncoder()
    le_known.fit(known_classes)

    # Train model
    model = _train_model(model_name, X_train_k, y_train_k, X_val_k, y_val_k,
                         use_gpu, le_known)

    # Predict on full test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # ---- Per-class metrics ----
    report = classification_report_full(y_test, y_pred, classes=class_names)

    # Separate known vs unknown
    is_unknown = ~y_test.isin(known_classes)
    known_mask_test = ~is_unknown

    # Known-class Macro-F1
    known_f1 = macro_f1_score(y_test[known_mask_test], y_pred[known_mask_test])

    # ---- Unknown -> Known mapping ----
    unknown_mapping = {}
    for cls in held_out_classes:
        cls_mask = y_test == cls
        if cls_mask.sum() == 0:
            continue
        preds = y_pred[cls_mask]
        # Top-3 predicted known classes
        top3 = pd.Series(preds).value_counts().head(3)
        unknown_mapping[cls] = {
            "count": int(cls_mask.sum()),
            "top3_predictions": [(str(c), int(n)) for c, n in top3.items()],
        }

    # ---- Confidence analysis ----
    # Use model's actual class order (known_classes only), not full class_names
    actual_classes = list(model.classes_) if hasattr(model, "classes_") else known_classes
    conf_df = confidence_analysis(y_prob, y_test, actual_classes)
    conf_df["is_unknown"] = is_unknown.values

    # Confidence distribution by category
    unknown_conf = conf_df[conf_df["is_unknown"]]["confidence"]
    known_correct_conf = conf_df[~conf_df["is_unknown"] & conf_df["is_correct"]]["confidence"]
    known_wrong_conf = conf_df[~conf_df["is_unknown"] & ~conf_df["is_correct"]]["confidence"]

    # AUROC of confidence as unknown detector
    # Score: 1 - confidence (high score = more likely unknown)
    try:
        auroc_unknown = roc_auc_score(is_unknown.astype(int), 1 - conf_df["confidence"])
    except ValueError:
        auroc_unknown = np.nan

    # Per-class confidence stats (on known classes in test)
    known_conf_stats = per_class_confidence_stats(
        y_prob[known_mask_test], y_test[known_mask_test], known_classes)

    # ---- Confusion matrix for known classes only ----
    cm_known = confusion_matrix_df(
        y_test[known_mask_test], y_pred[known_mask_test], known_classes)

    # ---- Full 10-class confusion matrix (known + unknown) ----
    cm_full = confusion_matrix_df(y_test, y_pred, class_names)

    results = {
        "held_out_classes": held_out_classes,
        "known_classes": known_classes,
        "per_class_report": report,
        "known_macro_f1": known_f1,
        "unknown_mapping": unknown_mapping,
        "unknown_mean_conf": unknown_conf.mean() if len(unknown_conf) > 0 else np.nan,
        "unknown_std_conf": unknown_conf.std() if len(unknown_conf) > 0 else np.nan,
        "known_correct_mean_conf": known_correct_conf.mean(),
        "known_wrong_mean_conf": known_wrong_conf.mean(),
        "auroc_confidence_unknown": auroc_unknown,
        "confusion_matrix_known": cm_known,
        "confusion_matrix_full": cm_full,
        "confidence_df": conf_df,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    # Print summary
    print(f"\n  Stress A Results ({held_out_classes}):")
    print(f"  Known-class Macro-F1: {known_f1:.4f}")
    print(f"  Unknown mean confidence: {unknown_conf.mean():.4f} (known correct: {known_correct_conf.mean():.4f}, known wrong: {known_wrong_conf.mean():.4f})")
    print(f"  AUROC (confidence as unknown detector): {auroc_unknown:.4f}")
    for cls, info in unknown_mapping.items():
        top_str = ", ".join([f"{c}({n})" for c, n in info["top3_predictions"]])
        print(f"  {cls} ({info['count']} samples) -> top-3: {top_str}")

    return results


def run_stress_b_full_chunked(model, preprocessor, cicids_path, chunk_size=200_000):
    """
    Stress Test B: Cross-dataset generalization (UNSW -> full CICIDS2018).

    Streams the full target dataset in chunks. Each chunk is cleaned,
    transformed with the UNSW-fitted preprocessor, predicted, and folded into
    binary TP/TN/FP/FN counts. The target preprocessor is never fitted.

    Returns dict with binary metrics and count totals.
    """
    import gc
    from preprocessing.preprocess import prepare_cicids_chunk
    from config import ATTACK_COL, LABEL_COL

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n_samples": 0}
    expected_cols = None

    print(f"\nStreaming full CICIDS2018 from {cicids_path} ...")
    print(f"Chunk size: {chunk_size:,}")
    reader = pd.read_csv(cicids_path, chunksize=chunk_size, low_memory=False)

    for chunk_id, chunk in enumerate(reader, start=1):
        chunk = prepare_cicids_chunk(chunk)
        if expected_cols is None:
            if hasattr(preprocessor, "feature_names_in_"):
                expected_cols = list(preprocessor.feature_names_in_)
            else:
                expected_cols = [c for c in chunk.columns if c not in (ATTACK_COL, LABEL_COL)]
            print(f"Expected feature columns: {len(expected_cols)}")

        y_true_bin = (chunk[ATTACK_COL] != "Benign").astype(int).values
        drop_cols = [ATTACK_COL]
        if LABEL_COL in chunk.columns:
            drop_cols.append(LABEL_COL)
        X_raw = chunk.drop(columns=drop_cols).reindex(columns=expected_cols, fill_value=0)
        X_t = preprocessor.transform(X_raw)
        y_pred = model.predict(X_t)
        y_pred_bin = (pd.Series(y_pred) != "Benign").astype(int).values

        counts["tp"] += int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
        counts["tn"] += int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
        counts["fp"] += int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
        counts["fn"] += int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
        counts["n_samples"] += int(len(y_true_bin))

        if chunk_id == 1 or chunk_id % 10 == 0:
            metrics = _binary_metrics_from_counts(counts)
            print(
                f"  chunk={chunk_id} processed={metrics['n_samples']:,} "
                f"F1={metrics['f1']:.4f} FPR={metrics['fpr']:.4f} FNR={metrics['fnr']:.4f}"
            )

        del chunk, X_raw, X_t, y_true_bin, y_pred, y_pred_bin
        gc.collect()

    return _binary_metrics_from_counts(counts)


def _binary_metrics_from_counts(counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    n = counts["n_samples"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / n if n else 0.0

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


def run_stress_c(X_test_raw, y_test, preprocessor, model, class_names,
                 feature_importances=None, numeric_feature_names=None, seed=SEED):
    """
    Stress Test C: Feature degradation.

    Applies 3 types of corruption at raw feature level, then transforms
    through the fitted preprocessor. Uses a fixed RNG for reproducibility.

    Returns a dict of degradation results.
    """
    rng = np.random.default_rng(seed)

    if numeric_feature_names is None:
        numeric_feature_names = [c for c in NUMERIC_FEATURES if c in X_test_raw.columns]

    results = {
        "noise": {"levels": [], "macro_f1": []},
        "masking": {"levels": [], "macro_f1": []},
        "dropout_top": {"levels": [], "macro_f1": []},
        "dropout_bottom": {"levels": [], "macro_f1": []},
    }

    baseline_f1 = macro_f1_score(y_test, model.predict(
        preprocessor.transform(X_test_raw)))
    print(f"\nStress C baseline (no corruption): Macro-F1 = {baseline_f1:.4f}")

    # ---- Gaussian Noise ----
    print("\n--- Gaussian Noise ---")
    for sigma in GAUSSIAN_NOISE_STDS:
        X_noisy = X_test_raw.copy()
        for col in numeric_feature_names:
            noise = rng.normal(0, sigma * X_noisy[col].std(), size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise
        X_t = preprocessor.transform(X_noisy)
        f1 = macro_f1_score(y_test, model.predict(X_t))
        results["noise"]["levels"].append(f"sigma={sigma}")
        results["noise"]["macro_f1"].append(f1)
        print(f"  sigma={sigma}: Macro-F1 = {f1:.4f}")

    # ---- Random Masking ----
    print("\n--- Random Masking ---")
    for p in MASKING_RATES:
        f1s = []
        for _ in range(3):  # average over 3 random masks
            X_masked = X_test_raw.copy()
            for col in numeric_feature_names:
                mask = rng.random(len(X_masked)) < p
                X_masked.loc[mask, col] = 0
            X_t = preprocessor.transform(X_masked)
            f1s.append(macro_f1_score(y_test, model.predict(X_t)))
        avg_f1 = np.mean(f1s)
        results["masking"]["levels"].append(f"p={p}")
        results["masking"]["macro_f1"].append(avg_f1)
        print(f"  p={p}: Macro-F1 = {avg_f1:.4f} (avg of 3)")

    # ---- Feature Dropout ----
    if feature_importances is not None and len(feature_importances) == len(numeric_feature_names):
        print("\n--- Feature Dropout ---")
        # Rank numeric features by importance
        ranked = sorted(zip(numeric_feature_names, feature_importances),
                        key=lambda x: x[1])
        ranked_names = [n for n, _ in ranked]

        for k in FEATURE_DROPOUT_COUNTS:
            # Drop top-k (most important)
            topk = ranked_names[-k:]
            X_dropped = X_test_raw.copy()
            for col in topk:
                X_dropped[col] = 0
            X_t = preprocessor.transform(X_dropped)
            f1_top = macro_f1_score(y_test, model.predict(X_t))
            results["dropout_top"]["levels"].append(f"k={k}")
            results["dropout_top"]["macro_f1"].append(f1_top)

            # Drop bottom-k (least important)
            bottomk = ranked_names[:k]
            X_dropped2 = X_test_raw.copy()
            for col in bottomk:
                X_dropped2[col] = 0
            X_t2 = preprocessor.transform(X_dropped2)
            f1_bottom = macro_f1_score(y_test, model.predict(X_t2))
            results["dropout_bottom"]["levels"].append(f"k={k}")
            results["dropout_bottom"]["macro_f1"].append(f1_bottom)

            print(f"  Drop top-{k} ({', '.join(topk[:3])}{'...' if k > 3 else ''}): "
                  f"Macro-F1 = {f1_top:.4f}")
            print(f"  Drop bottom-{k} ({', '.join(bottomk[:3])}{'...' if k > 3 else ''}): "
                  f"Macro-F1 = {f1_bottom:.4f}")
    else:
        print("\n--- Feature Dropout: SKIPPED (no feature importances) ---")

    return results
