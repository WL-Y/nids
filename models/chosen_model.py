"""
XGBoost classifier — the primary model for this project.

Supports GPU acceleration with automatic CPU fallback.
Uses integer-encoded labels (required by XGBoost multi:softprob).
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from config import SEED, XGB_PARAMS


def get_xgb_device(use_gpu=True):
    """Return XGBoost device params. Tests CUDA with a tiny fit, falls back to CPU."""
    if use_gpu:
        try:
            import numpy as np
            X_test = np.random.randn(10, 4)
            y_test = np.random.randint(0, 3, 10)
            m = XGBClassifier(tree_method="hist", device="cuda")
            m.fit(X_test, y_test)
            print("GPU (CUDA) is available for XGBoost.")
            return {"tree_method": "hist", "device": "cuda"}
        except Exception as e:
            print(f"GPU check failed ({e}), falling back to CPU.")
    print("Using CPU for XGBoost.")
    return {"tree_method": "hist"}


def _tuning_subset(X, y, frac=0.1):
    """Stratified subset for fast hyperparameter search."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=frac,
                                       random_state=SEED)
    _, idx_subset = next(splitter.split(X, y))
    return X[idx_subset], y[idx_subset]


def train_xgboost(X_train, y_train, X_val, y_val, param_grid=None, use_gpu=True):
    """
    XGBClassifier with early_stopping_rounds=20, eval_metric='mlogloss'.

    GridSearchCV on 10% tuning subset -> best params -> full training.
    GPU acceleration if available, falls back to CPU automatically.

    y_train/y_val must be integer-encoded (use LabelEncoder).
    """
    if param_grid is None:
        param_grid = XGB_PARAMS

    print(f"\n{'='*50}")
    print("XGBoost")
    print(f"{'='*50}")

    # Determine device
    device_params = get_xgb_device(use_gpu)
    device_str = device_params.get("device", "cpu")
    print(f"Using device: {device_str}")

    # Tuning on 10% subset
    X_tune, y_tune = _tuning_subset(X_train, y_train, frac=0.1)
    print(f"Tuning subset: {len(X_tune):,} samples")

    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
        **device_params,
    )

    grid = GridSearchCV(xgb, param_grid, cv=3, scoring="f1_macro",
                        n_jobs=1, verbose=1)  # n_jobs=1: XGBoost uses internal threading

    t0 = time.time()
    grid.fit(X_tune, y_tune)
    tuning_time = time.time() - t0
    print(f"Tuning time: {tuning_time:.1f}s")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV Macro-F1: {grid.best_score_:.4f}")

    cv_df = pd.DataFrame(grid.cv_results_)

    # Full training with best params + early stopping on validation set
    best_params = grid.best_params_.copy()

    best = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=20,
        **best_params,
        **device_params,
    )

    t0 = time.time()
    best.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - t0
    print(f"Full training time: {train_time:.1f}s")
    print(f"Best iteration: {best.best_iteration}")
    print(f"Best eval mlogloss: {best.best_score:.4f}")

    return best, best_params, cv_df
