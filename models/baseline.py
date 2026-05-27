"""
Baseline classifiers: Majority, Logistic Regression, Random Forest.

Each training function:
  1. Stratified 10% tuning subset -> GridSearchCV (3-fold)
  2. Best params -> full training set
  3. Returns (fitted_model, best_params, cv_results_df)
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from config import SEED, LR_PARAMS, RF_PARAMS


class MajorityClassifier:
    """Always predicts the most frequent class from training data."""

    def __init__(self):
        self.majority_class = None
        self.classes_ = None

    def fit(self, X, y):
        counts = pd.Series(y).value_counts()
        self.majority_class = counts.index[0]
        self.classes_ = sorted(y.unique())
        self.majority_index = self.classes_.index(self.majority_class)
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class)

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, self.majority_index] = 1.0
        return proba


def _tuning_subset(X, y, frac=0.1):
    """Stratified subset for fast hyperparameter search."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=frac,
                                       random_state=SEED)
    _, idx_subset = next(splitter.split(X, y))
    return X[idx_subset], y.iloc[idx_subset] if hasattr(y, "iloc") else y[idx_subset]


def train_logistic_regression(X_train, y_train, X_val, y_val, param_grid=None):
    """
    LogisticRegression(max_iter=2000, class_weight='balanced').
    GridSearchCV on 10% tuning subset -> best params -> full training.
    """
    if param_grid is None:
        param_grid = LR_PARAMS

    print(f"\n{'='*50}")
    print("Logistic Regression")
    print(f"{'='*50}")

    # Tuning on 10% subset
    X_tune, y_tune = _tuning_subset(X_train, y_train, frac=0.1)
    print(f"Tuning subset: {len(X_tune):,} samples")

    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=SEED, n_jobs=-1)

    grid = GridSearchCV(lr, param_grid, cv=3, scoring="f1_macro",
                        n_jobs=-1, verbose=1)
    t0 = time.time()
    grid.fit(X_tune, y_tune)
    tuning_time = time.time() - t0
    print(f"Tuning time: {tuning_time:.1f}s")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV Macro-F1: {grid.best_score_:.4f}")

    # Full training with best params
    best_params = grid.best_params_.copy()
    best = LogisticRegression(**best_params,
                              class_weight="balanced", random_state=SEED,
                              n_jobs=-1)
    t0 = time.time()
    best.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Full training time: {train_time:.1f}s")

    cv_df = pd.DataFrame(grid.cv_results_)
    return best, best_params, cv_df


def train_random_forest(X_train, y_train, X_val, y_val, param_grid=None):
    """
    RandomForestClassifier(n_jobs=-1, random_state=SEED).
    GridSearchCV on 10% tuning subset -> best params -> full training.
    """
    if param_grid is None:
        param_grid = RF_PARAMS

    print(f"\n{'='*50}")
    print("Random Forest")
    print(f"{'='*50}")

    X_tune, y_tune = _tuning_subset(X_train, y_train, frac=0.1)
    print(f"Tuning subset: {len(X_tune):,} samples")

    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)

    grid = GridSearchCV(rf, param_grid, cv=3, scoring="f1_macro",
                        n_jobs=-1, verbose=1)
    t0 = time.time()
    grid.fit(X_tune, y_tune)
    tuning_time = time.time() - t0
    print(f"Tuning time: {tuning_time:.1f}s")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV Macro-F1: {grid.best_score_:.4f}")

    # Full training
    best_params = grid.best_params_.copy()
    best = RandomForestClassifier(**best_params, random_state=SEED,
                                   n_jobs=-1)
    t0 = time.time()
    best.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Full training time: {train_time:.1f}s")

    cv_df = pd.DataFrame(grid.cv_results_)
    return best, best_params, cv_df
