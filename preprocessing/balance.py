"""
Class balancing utilities.

Default strategy: class_weight / sample_weight (safe, no synthetic data).
SMOTE is reserved for Phase 6 ablation experiments only.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from config import SEED


def get_class_weights(y_train):
    """
    Compute class_weight dict for sklearn models (LR, RF).

    Returns dict mapping class label -> weight.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


def compute_sample_weights(y_train):
    """
    Compute per-sample weights for XGBoost sample_weight parameter.

    Returns np.ndarray of same length as y_train.
    """
    classes = np.unique(y_train)
    n_samples = len(y_train)
    n_classes = len(classes)

    counts = np.array([(y_train == c).sum() for c in classes])
    class_weight = n_samples / (n_classes * counts)

    weight_map = dict(zip(classes, class_weight))
    sample_weights = np.array([weight_map[y] for y in y_train])
    return sample_weights


def apply_smote(X_train, y_train, random_state=SEED):
    """
    Apply SMOTE oversampling — for Phase 6 ablation only.

    Handles the Worms class (158 samples) edge case by using
    k_neighbors=1 when necessary.
    """
    from collections import Counter
    from imblearn.over_sampling import SMOTE

    counts = Counter(y_train)
    min_count = min(counts.values())

    # k_neighbors must be <= min_count - 1. For Worms (158), use default=5.
    # If a class has fewer than 6 samples, reduce k_neighbors.
    k = min(5, min_count - 1) if min_count > 1 else 1

    print(f"SMOTE: k_neighbors={k} (min class count={min_count})")
    print(f"Before SMOTE: {dict(counts)}")

    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    after_counts = Counter(y_res)
    print(f"After SMOTE:  {dict(after_counts)}")
    return X_res, y_res
