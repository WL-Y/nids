"""
Robustness improvement strategies.

Strategy 1: Confidence threshold rejection.
Strategy 2: Heterogeneous ensemble disagreement detection.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import SEED
from models.chosen_model import get_xgb_device
from evaluation.metrics import coverage_accuracy_curve


# ==================== Strategy 1: Confidence Threshold ====================

def predict_with_rejection(model, X, tau):
    """
    Predict with confidence threshold rejection.

    Rule: if max(p) >= tau -> predicted class, else -> "REJECT"

    Returns (predicted_labels, rejected_mask, max_probabilities).
    """
    y_prob = model.predict_proba(X)
    max_probs = np.max(y_prob, axis=1)
    pred_indices = np.argmax(y_prob, axis=1)

    rejected = max_probs < tau

    # Get class names if available
    if hasattr(model, "classes_"):
        y_pred = np.array([model.classes_[i] for i in pred_indices], dtype=object)
    else:
        y_pred = pred_indices.astype(str)

    y_pred[rejected] = "REJECT"
    return y_pred, rejected, max_probs


def select_tau_on_validation(y_val, y_val_prob, min_coverage=0.85,
                              tau_range=None, classes=None):
    """
    Select best confidence threshold tau on validation set.

    Sweeps tau from 0.50 to 0.99, selects tau that maximizes accepted accuracy
    under the constraint coverage >= min_coverage.

    Returns (best_tau, tau_results_df).
    """
    tau_results = coverage_accuracy_curve(y_val_prob, y_val, tau_range, classes)

    valid = tau_results[tau_results["coverage"] >= min_coverage]
    if len(valid) == 0:
        # Relax constraint: pick tau with max coverage that still has reasonable accuracy
        valid = tau_results[tau_results["coverage"] >= 0.5]

    best_row = valid.loc[valid["accuracy"].idxmax()]
    best_tau = best_row["tau"]

    print(f"\n  Tau selection (min_coverage={min_coverage}):")
    print(f"  Best tau: {best_tau}")
    print(f"  Coverage: {best_row['coverage']:.4f}")
    print(f"  Accepted accuracy: {best_row['accuracy']:.4f}")

    return best_tau, tau_results


# ==================== Strategy 2: Ensemble Disagreement ====================

def train_heterogeneous_ensemble(X_train, y_train, use_gpu=True):
    """
    Train M=5 heterogeneous ensemble:
      3 x RandomForest (different seeds, reduced depth/size for memory)
      1 x XGBoost (GPU)
      1 x LogisticRegression

    Uses 50% stratified subset for faster training and lower memory.
    All models output the same string label format for disagreement computation.

    Returns list of (name, model) tuples.
    """
    import gc
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder
    from experiments.run_baseline import XGBWrapper

    print("\nTraining heterogeneous ensemble (M=5) ...")

    # Use 50% stratified subset for ensemble training (memory constraint)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    _, idx_subset = next(splitter.split(X_train, y_train))
    X_sub = X_train[idx_subset]
    y_sub = y_train.iloc[idx_subset] if hasattr(y_train, "iloc") else y_train[idx_subset]
    print(f"  Training on 50% subset: {len(X_sub):,} samples")

    models = []

    # LabelEncoder for XGBoost
    le = LabelEncoder()
    y_sub_enc = le.fit_transform(y_sub)

    # 3x Random Forest (plan: n_estimators=200, max_depth=20)
    for seed in [42, 123, 456]:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            min_samples_split=5, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        rf.fit(X_sub, y_sub)
        models.append((f"RF_{seed}", rf))
        print(f"  RF (seed={seed}): trained")
        gc.collect()

    # 1x XGBoost (plan: n_estimators=200, max_depth=10)
    device_params = get_xgb_device(use_gpu)
    xgb = XGBClassifier(
        n_estimators=200, max_depth=10, learning_rate=0.1,
        subsample=1.0, colsample_bytree=0.8,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=SEED,
        **device_params,
    )
    xgb.fit(X_sub, y_sub_enc)
    wrapped_xgb = XGBWrapper(xgb, le)
    models.append(("XGB", wrapped_xgb))
    print(f"  XGBoost: trained")
    gc.collect()

    # 1x LogisticRegression (plan: max_iter=1000, but uses 3000 to converge)
    lr = LogisticRegression(
        C=10, max_iter=3000, penalty="l2", solver="lbfgs",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    lr.fit(X_sub, y_sub)
    models.append(("LR", lr))
    print(f"  LR: trained")
    gc.collect()

    return models


def ensemble_disagreement(models, X):
    """
    Compute disagreement score for ensemble.

    disagreement(x) = 1 - (1/M) * sum_i I[yi_hat(x) == y_majority(x)]

    Uses pandas mode for robust string majority voting.

    Returns (disagreement_scores, majority_vote_predictions).
    """
    all_preds = []
    for name, m in models:
        all_preds.append(m.predict(X))
    all_preds = np.array(all_preds, dtype=object)  # (M, N)

    # Vectorized majority vote via pandas mode (handles ties by picking first)
    majority = pd.DataFrame(all_preds).mode(axis=0).iloc[0].values.astype(object)

    agreement = (all_preds == majority).mean(axis=0)
    disagreement = 1.0 - agreement

    return disagreement, majority


def ensemble_predict_proba(models, X):
    """
    Average predicted probabilities across ensemble members.

    Handles class alignment: different models may have different classes_
    orderings or subsets. Builds a unified class list from all models,
    maps each model's probabilities to the unified list, filling 0 for
    missing classes.
    """
    # Determine unified class ordering across all models
    all_classes = []
    for name, m in models:
        if hasattr(m, "classes_"):
            for cls in m.classes_:
                if cls not in all_classes:
                    all_classes.append(cls)
    if not all_classes:
        # Fallback: all models have same number of classes, assume aligned
        probas = [m.predict_proba(X) for _, m in models if hasattr(m, "predict_proba")]
        return np.mean(probas, axis=0) if probas else None

    # Align each model's probabilities to the unified class list
    aligned = np.zeros((len(X), len(all_classes)))
    n_models = 0
    for name, m in models:
        if not hasattr(m, "predict_proba") or not hasattr(m, "classes_"):
            continue
        proba = m.predict_proba(X)
        for j, cls in enumerate(m.classes_):
            k = all_classes.index(cls)
            aligned[:, k] += proba[:, j]
        n_models += 1

    if n_models == 0:
        return None
    return aligned / n_models
