"""
Phase 6 (part 1): Ablation Studies.

Ablation A: Strategy 1 threshold sensitivity — sweep tau under Stress A.
Ablation B: Strategy 2 ensemble size — vary M, measure AUROC + time.
Ablation C: Class imbalance methods — none vs class_weight vs SMOTE minority recall.
"""

import os
import sys
import gc
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score

from config import (
    UNSW_PATH, ATTACK_COL, HELD_OUT_CLASSES_SETS, SEED, NUMERIC_FEATURES,
)
from preprocessing.preprocess import (
    load_and_clean_data, clean_l7_proto, split_data,
    build_preprocessor, fit_preprocessor,
)
from preprocessing.balance import apply_smote
from robustness.strategies import (
    predict_with_rejection, ensemble_disagreement,
)
from evaluation.metrics import disagreement_auroc, macro_f1_score
from models.chosen_model import get_xgb_device
from experiments.run_baseline import XGBWrapper


def prepare_data():
    """Load and preprocess data."""
    print("Preparing data ...")
    df = load_and_clean_data(UNSW_PATH)
    df = clean_l7_proto(df)

    y = df[ATTACK_COL]
    X = df.drop(columns=[ATTACK_COL, "Label"])
    X["Attack"] = y
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X)
    y_tr = X_tr.pop("Attack"); y_v = X_v.pop("Attack"); y_te = X_te.pop("Attack")

    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    X_train_t = preprocessor.transform(X_tr)
    X_val_t = preprocessor.transform(X_v)
    X_test_t = preprocessor.transform(X_te)

    class_names = sorted(y_tr.unique())
    return {
        "X_train": X_train_t, "X_val": X_val_t, "X_test": X_test_t,
        "X_train_raw": X_tr, "X_val_raw": X_v, "X_test_raw": X_te,
        "y_train": y_tr, "y_val": y_v, "y_test": y_te,
        "class_names": class_names,
    }


# ==================== Ablation A ====================

def run_ablation_a(model, data):
    """
    Strategy 1 threshold sensitivity (STRICT open-set).
    For each held-out group: fit known-only preprocessor, train known-only model,
    then sweep tau. tau=0 means no rejection (baseline on known-only model).
    """
    from sklearn.ensemble import RandomForestClassifier

    print(f"\n{'='*50}")
    print("Ablation A: Threshold Sensitivity (strict open-set)")
    print(f"{'='*50}")

    tau_values = [0, 0.5, 0.7, 0.85, 0.9, 0.95]
    all_rows = []

    for held_out in HELD_OUT_CLASSES_SETS:
        known_classes = sorted(set(data["class_names"]) - set(held_out))
        train_mask = data["y_train"].isin(known_classes)

        # Strict preprocessing: fit on known-only raw data
        X_raw_k = data["X_train_raw"].loc[train_mask]
        y_train_k = data["y_train"][train_mask]
        prep_k = build_preprocessor()
        prep_k.fit(X_raw_k)
        X_test_k = prep_k.transform(data["X_test_raw"])

        # Train known-only model
        model_k = RandomForestClassifier(
            n_estimators=50, max_depth=15, min_samples_leaf=2,
            min_samples_split=5, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        )
        model_k.fit(prep_k.transform(X_raw_k), y_train_k)
        print(f"  Known-only model trained for held-out: {held_out}")

        is_unknown = ~data["y_test"].isin(known_classes)
        known_mask = ~is_unknown.values

        for tau in tau_values:
            y_pred, rejected, _ = predict_with_rejection(model_k, X_test_k, tau)
            coverage = (~rejected).mean()
            unknown_rej = rejected[is_unknown.values].mean()
            known_false_rej = rejected[known_mask].mean()
            accepted = ~rejected & known_mask
            acc = np.mean(y_pred[accepted] == data["y_test"].values[accepted]) if accepted.sum() > 0 else np.nan

            all_rows.append({
                "held_out": ", ".join(held_out),
                "tau": tau,
                "coverage": round(coverage, 4),
                "accepted_accuracy": round(acc, 4) if not np.isnan(acc) else np.nan,
                "unknown_rejection_rate": round(unknown_rej, 4),
                "known_false_rejection_rate": round(known_false_rej, 4),
            })
        gc.collect()

    df = pd.DataFrame(all_rows)
    df.to_csv("results/ablation_a_threshold_sensitivity.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("Saved: results/ablation_a_threshold_sensitivity.csv")
    return df


# ==================== Ablation B ====================

def _train_ensemble_for_size(X_train, y_train, M, use_gpu):
    """Train ensemble of size M. Returns (models, training_time)."""
    from sklearn.preprocessing import LabelEncoder
    from experiments.run_baseline import XGBWrapper

    # Use subset for efficiency (50%)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    _, idx = next(splitter.split(X_train, y_train))
    X_sub = X_train[idx] 
    y_sub = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]

    le = LabelEncoder()
    y_sub_enc = le.fit_transform(y_sub)
    device_params = get_xgb_device(use_gpu)

    models = []
    t0 = time.time()

    if M == 1:
        rf = RandomForestClassifier(n_estimators=50, max_depth=15, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1)
        rf.fit(X_sub, y_sub)
        models.append(("RF_1", rf))

    elif M == 3:
        for seed in [42, 123]:
            rf = RandomForestClassifier(n_estimators=50, max_depth=15, class_weight="balanced",
                                        random_state=seed, n_jobs=-1)
            rf.fit(X_sub, y_sub)
            models.append((f"RF_{seed}", rf))
        xgb = XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1,
                            objective="multi:softprob", eval_metric="mlogloss",
                            random_state=SEED, **device_params)
        xgb.fit(X_sub, y_sub_enc)
        models.append(("XGB", XGBWrapper(xgb, le)))

    elif M == 5:
        for seed in [42, 123, 456]:
            rf = RandomForestClassifier(n_estimators=50, max_depth=15, class_weight="balanced",
                                        random_state=seed, n_jobs=-1)
            rf.fit(X_sub, y_sub)
            models.append((f"RF_{seed}", rf))
        xgb = XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1,
                            objective="multi:softprob", eval_metric="mlogloss",
                            random_state=SEED, **device_params)
        xgb.fit(X_sub, y_sub_enc)
        models.append(("XGB", XGBWrapper(xgb, le)))
        lr = LogisticRegression(C=10, max_iter=2000, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
        lr.fit(X_sub, y_sub)
        models.append(("LR", lr))

    elif M == 10:
        # Same per-model capacity as M=3,5 (fixed per-model, varying only M)
        for seed in [42, 123, 456, 789, 111, 222, 333, 444]:
            rf = RandomForestClassifier(n_estimators=50, max_depth=15, class_weight="balanced",
                                        random_state=seed, n_jobs=-1)
            rf.fit(X_sub, y_sub)
            models.append((f"RF_{seed}", rf))
        xgb = XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1,
                            objective="multi:softprob", eval_metric="mlogloss",
                            random_state=SEED, **device_params)
        xgb.fit(X_sub, y_sub_enc)
        models.append(("XGB", XGBWrapper(xgb, le)))
        lr = LogisticRegression(C=10, max_iter=2000, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
        lr.fit(X_sub, y_sub)
        models.append(("LR", lr))

    train_time = time.time() - t0
    return models, train_time


def run_ablation_b(data, use_gpu):
    """
    Strategy 2 ensemble size ablation.
    Vary M in [1, 3, 5, 10], measure AUROC + time.
    M=1 has no disagreement (AUROC undefined, reported as 0.5).
    """
    print(f"\n{'='*50}")
    print("Ablation B: Ensemble Size")
    print(f"{'='*50}")

    rows = []
    for M in [1, 3, 5, 10]:
        print(f"\n  Training M={M} ensemble ...")
        models, train_time = _train_ensemble_for_size(
            data["X_train"], data["y_train"], M, use_gpu)

        t0 = time.time()
        if M == 1:
            # Single model: disagreement is always 0, AUROC undefined
            auroc = 0.5  # random baseline
            disagreement = np.zeros(len(data["y_test"]))
        else:
            disagreement, majority = ensemble_disagreement(models, data["X_test"])
            is_wrong = majority != data["y_test"].values
            auroc = disagreement_auroc(disagreement, is_wrong)
        infer_time = time.time() - t0

        rows.append({
            "M": M,
            "train_time_s": round(train_time, 1),
            "inference_time_s": round(infer_time, 3),
            "disagreement_auroc": round(auroc, 4),
        })
        print(f"    M={M}: AUROC={auroc:.4f}, train={train_time:.1f}s, infer={infer_time:.3f}s")
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv("results/ablation_b_ensemble_size.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("Saved: results/ablation_b_ensemble_size.csv")
    return df


# ==================== Ablation C ====================

def run_ablation_c():
    """
    Class imbalance methods comparison.
    Train RF with: none (raw), class_weight='balanced', SMOTE.
    Compare minority class recall: Worms, Analysis, Shellcode.
    """
    print(f"\n{'='*50}")
    print("Ablation C: Class Imbalance Methods")
    print(f"{'='*50}")

    # Use 20% subset to keep SMOTE tractable
    print("Loading raw data ...")
    df = load_and_clean_data(UNSW_PATH)
    df = clean_l7_proto(df)
    y = df[ATTACK_COL]
    X = df.drop(columns=[ATTACK_COL, "Label"])

    X["Attack"] = y
    X_tr, _, X_te, y_tr, _, y_te = split_data(X)
    y_tr = X_tr.pop("Attack"); y_te = X_te.pop("Attack")

    # Subset for SMOTE efficiency
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    _, idx = next(splitter.split(X_tr, y_tr))
    X_sub = X_tr.iloc[idx]
    y_sub = y_tr.iloc[idx]
    print(f"Training subset: {len(X_sub):,} samples")

    # Preprocess
    prep = build_preprocessor()
    prep.fit(X_sub)
    X_sub_t = prep.transform(X_sub)
    X_te_t = prep.transform(X_te)

    minority_classes = ["Worms", "Analysis", "Shellcode"]
    rows = []

    # Method 1: No balancing
    print("\n  Method 1: No balancing")
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=SEED, n_jobs=-1)
    rf.fit(X_sub_t, y_sub)
    y_pred = rf.predict(X_te_t)
    rec = recall_score(y_te, y_pred, labels=minority_classes, average=None, zero_division=0)
    rows.append({"method": "none", "Worms_recall": rec[0], "Analysis_recall": rec[1],
                 "Shellcode_recall": rec[2], "macro_f1": macro_f1_score(y_te, y_pred)})

    # Method 2: class_weight='balanced'
    print("  Method 2: class_weight='balanced'")
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
    rf.fit(X_sub_t, y_sub)
    y_pred = rf.predict(X_te_t)
    rec = recall_score(y_te, y_pred, labels=minority_classes, average=None, zero_division=0)
    rows.append({"method": "class_weight", "Worms_recall": rec[0], "Analysis_recall": rec[1],
                 "Shellcode_recall": rec[2], "macro_f1": macro_f1_score(y_te, y_pred)})

    # Method 3: SMOTE
    print("  Method 3: SMOTE")
    try:
        X_smote, y_smote = apply_smote(X_sub_t, y_sub)
        rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=SEED, n_jobs=-1)
        rf.fit(X_smote, y_smote)
        y_pred = rf.predict(X_te_t)
        rec = recall_score(y_te, y_pred, labels=minority_classes, average=None, zero_division=0)
        rows.append({"method": "SMOTE", "Worms_recall": rec[0], "Analysis_recall": rec[1],
                     "Shellcode_recall": rec[2], "macro_f1": macro_f1_score(y_te, y_pred)})
    except Exception as e:
        print(f"    SMOTE failed: {e}")
        rows.append({"method": "SMOTE", "Worms_recall": np.nan, "Analysis_recall": np.nan,
                     "Shellcode_recall": np.nan, "macro_f1": np.nan})

    df = pd.DataFrame(rows)
    df.to_csv("results/ablation_c_imbalance_methods.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("Saved: results/ablation_c_imbalance_methods.csv")
    return df


# ==================== Ablation D: Per-Class Tau ====================

def run_ablation_d(model, data):
    """
    Per-class tau vs global tau comparison (STRICT open-set).
    For each held-out group: fit known-only preprocessor, train known-only model,
    compute per-class tau on known-only val, evaluate both strategies.
    """
    from sklearn.ensemble import RandomForestClassifier
    from robustness.strategies import predict_with_rejection

    print(f"\n{'='*50}")
    print("Ablation D: Global vs Per-Class Tau (strict open-set)")
    print(f"{'='*50}")

    global_tau = 0.85
    rows = []

    for held_out in HELD_OUT_CLASSES_SETS:
        known_classes = sorted(set(data["class_names"]) - set(held_out))
        train_mask = data["y_train"].isin(known_classes)
        val_mask = data["y_val"].isin(known_classes)

        # Strict preprocessing: fit on known-only
        X_raw_k = data["X_train_raw"].loc[train_mask]
        y_train_k = data["y_train"][train_mask]
        X_val_raw_k = data["X_val_raw"].loc[val_mask]
        y_val_k = data["y_val"][val_mask]

        prep_k = build_preprocessor()
        prep_k.fit(X_raw_k)
        X_train_k_t = prep_k.transform(X_raw_k)
        X_val_k_t = prep_k.transform(X_val_raw_k)
        X_test_k_t = prep_k.transform(data["X_test_raw"])

        # Train known-only model
        model_k = RandomForestClassifier(
            n_estimators=50, max_depth=15, min_samples_leaf=2,
            min_samples_split=5, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        )
        model_k.fit(X_train_k_t, y_train_k)
        print(f"  Known-only model trained for held-out: {held_out}")

        # Per-class tau from known-only validation
        y_val_prob_k = model_k.predict_proba(X_val_k_t)
        y_val_pred_k = model_k.predict(X_val_k_t)
        max_val_probs_k = np.max(y_val_prob_k, axis=1)
        per_class_tau = {}
        for cls in known_classes:
            cls_mask = y_val_pred_k == cls
            if cls_mask.sum() < 10:
                per_class_tau[cls] = 0.9
                continue
            per_class_tau[cls] = np.percentile(max_val_probs_k[cls_mask], 15)

        is_unknown = ~data["y_test"].isin(known_classes)
        known_mask = ~is_unknown.values

        # --- Global tau ---
        y_pred_g, rej_g, _ = predict_with_rejection(model_k, X_test_k_t, global_tau)
        coverage_g = (~rej_g).mean()
        unknown_rej_g = rej_g[is_unknown.values].mean() if is_unknown.sum() > 0 else np.nan
        known_false_rej_g = rej_g[known_mask].mean()

        # --- Per-class tau ---
        y_prob_test = model_k.predict_proba(X_test_k_t)
        max_test_probs = np.max(y_prob_test, axis=1)
        y_pred_test = model_k.predict(X_test_k_t)
        rej_pc = np.zeros(len(X_test_k_t), dtype=bool)
        for i in range(len(X_test_k_t)):
            pred_cls = y_pred_test[i]
            rej_pc[i] = max_test_probs[i] < per_class_tau.get(pred_cls, 0.9)
        coverage_pc = (~rej_pc).mean()
        unknown_rej_pc = rej_pc[is_unknown.values].mean() if is_unknown.sum() > 0 else np.nan
        known_false_rej_pc = rej_pc[known_mask].mean()

        held_out_str = ", ".join(held_out)
        rows.append({"method": "global_tau", "held_out": held_out_str, "tau": global_tau,
                     "coverage": round(coverage_g, 4),
                     "unknown_rejection_rate": round(unknown_rej_g, 4),
                     "known_false_rejection_rate": round(known_false_rej_g, 4)})
        rows.append({"method": "per_class_tau", "held_out": held_out_str, "tau": "per-class",
                     "coverage": round(coverage_pc, 4),
                     "unknown_rejection_rate": round(unknown_rej_pc, 4),
                     "known_false_rejection_rate": round(known_false_rej_pc, 4)})
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv("results/per_class_tau_ablation.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("Saved: results/per_class_tau_ablation.csv")
    return df


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("Phase 6 (part 1): Ablation Studies")
    print("=" * 60)

    data = prepare_data()
    model = joblib.load("artifacts/best_model.joblib")

    run_ablation_a(model, data)
    run_ablation_b(data, use_gpu=True)
    run_ablation_c()
    run_ablation_d(model, data)

    print(f"\n{'=' * 60}")
    print("Ablation studies complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
