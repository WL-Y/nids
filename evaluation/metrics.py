"""
Evaluation metrics for multi-class network intrusion detection.

Primary metric: Macro-F1 (class-imbalanced setting).
All functions operate on (y_true, y_pred, y_prob) triples where applicable.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from scipy.stats import ks_2samp, chi2


def classification_report_full(y_true, y_pred, classes=None, digits=4):
    """Per-class precision/recall/f1 + macro/weighted averages as DataFrame."""
    if classes is None:
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    report = classification_report(y_true, y_pred, labels=classes,
                                   output_dict=True, zero_division=0)

    rows = []
    for cls in classes:
        d = report[str(cls)]
        rows.append({
            "class": cls,
            "precision": round(d["precision"], digits),
            "recall": round(d["recall"], digits),
            "f1": round(d["f1-score"], digits),
            "support": int(d["support"]),
        })
    df = pd.DataFrame(rows)

    # Append macro and weighted averages
    for avg in ["macro avg", "weighted avg"]:
        d = report[avg]
        df.loc[len(df)] = [
            avg, round(d["precision"], digits), round(d["recall"], digits),
            round(d["f1-score"], digits), int(d["support"]),
        ]
    return df


def confusion_matrix_df(y_true, y_pred, classes):
    """Labeled confusion matrix as DataFrame."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return pd.DataFrame(cm, index=classes, columns=classes)


def macro_f1_score(y_true, y_pred):
    """Macro-averaged F1 score (equal weight per class)."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def weighted_f1_score(y_true, y_pred):
    """Weighted-averaged F1 score (by class support)."""
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def confidence_analysis(y_prob, y_true, classes):
    """
    Per-sample confidence breakdown.

    Returns DataFrame with columns:
      true_class, predicted_class, confidence (max prob), is_correct
    """
    max_indices = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)

    df = pd.DataFrame({
        "true_class": y_true,
        "predicted_class": [classes[i] for i in max_indices],
        "confidence": confidences,
        "is_correct": y_true == np.array([classes[i] for i in max_indices]),
    })
    return df


def coverage_accuracy_curve(y_prob, y_true, tau_range=None, classes=None):
    """
    Sweep confidence threshold tau.

    Returns DataFrame: tau, coverage (fraction accepted), accuracy (on accepted).
    """
    if tau_range is None:
        tau_range = np.arange(0.5, 1.0, 0.01)

    max_probs = np.max(y_prob, axis=1)
    pred_indices = np.argmax(y_prob, axis=1)

    # Map indices back to class labels if classes provided
    if classes is not None:
        y_pred = np.array([classes[i] for i in pred_indices])
    else:
        y_pred = pred_indices

    rows = []
    for tau in tau_range:
        accepted = max_probs >= tau
        coverage = accepted.mean()
        if accepted.sum() > 0:
            acc = accuracy_score(y_true[accepted], y_pred[accepted])
        else:
            acc = np.nan
        rows.append({"tau": round(tau, 2), "coverage": coverage, "accuracy": acc})
    return pd.DataFrame(rows)


def per_class_confidence_stats(y_prob, y_true, classes):
    """
    Per-class confidence statistics, split by correct vs incorrect prediction.

    Returns DataFrame: class, mean_conf_correct, std_conf_correct,
                       mean_conf_incorrect, std_conf_incorrect, count
    """
    conf_df = confidence_analysis(y_prob, y_true, classes)
    rows = []
    for cls in classes:
        mask = conf_df["true_class"] == cls
        subset = conf_df[mask]
        correct = subset[subset["is_correct"]]
        incorrect = subset[~subset["is_correct"]]
        rows.append({
            "class": cls,
            "count": len(subset),
            "accuracy": correct["confidence"].count() / max(len(subset), 1),
            "mean_conf_correct": correct["confidence"].mean() if len(correct) > 0 else np.nan,
            "std_conf_correct": correct["confidence"].std() if len(correct) > 0 else np.nan,
            "mean_conf_incorrect": incorrect["confidence"].mean() if len(incorrect) > 0 else np.nan,
            "std_conf_incorrect": incorrect["confidence"].std() if len(incorrect) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def degradation_curve_table(degradation_levels, metric_values):
    """Build a degradation table: degradation level x metric value."""
    return pd.DataFrame({
        "degradation_level": degradation_levels,
        "macro_f1": metric_values,
    })


def run_statistics(result_list):
    """
    Compute mean +/- std from list of metric values.

    Returns formatted string: "mean +/- std"
    """
    arr = np.array(result_list)
    return f"{arr.mean():.4f} +/- {arr.std():.4f}"


def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test for paired nominal data.
    Compares whether two classifiers differ significantly.

    Returns (chi2_statistic, p_value).
    """
    both_correct = (pred_a == y_true) & (pred_b == y_true)
    both_wrong = (pred_a != y_true) & (pred_b != y_true)
    a_better = (pred_a == y_true) & (pred_b != y_true)
    b_better = (pred_a != y_true) & (pred_b == y_true)

    b = a_better.sum()
    c = b_better.sum()

    if b + c == 0:
        return 0.0, 1.0

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return stat, p


def kl_divergence(p, q, epsilon=1e-10):
    """KL divergence between two discrete probability distributions."""
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def ks_test_two_samples(sample1, sample2):
    """
    Two-sample KS test: are sample1 and sample2 from the same distribution?

    Returns (ks_statistic, p_value).
    """
    s1 = np.asarray(sample1, dtype=float)
    s2 = np.asarray(sample2, dtype=float)
    s1 = s1[np.isfinite(s1)]
    s2 = s2[np.isfinite(s2)]
    return ks_2samp(s1, s2)


def calculate_ece(y_prob, y_true, n_bins=10, classes=None):
    """
    Expected Calibration Error (ECE).

    Partitions predictions into n_bins by confidence, computes the
    weighted absolute difference between accuracy and average confidence per bin.

    If classes is provided, maps argmax indices to class labels for comparison
    with string y_true. If y_true is already integer-encoded, pass classes=None.

    [Future Work] Not required for the main experiments.
    """
    max_probs = np.max(y_prob, axis=1)
    pred_indices = np.argmax(y_prob, axis=1)

    if classes is not None:
        y_pred = np.array([classes[i] for i in pred_indices])
    else:
        y_pred = pred_indices
    correct = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (max_probs > bin_edges[i]) & (max_probs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = max_probs[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)

    return ece


def disagreement_auroc(disagreement_scores, y_true_is_wrong):
    """
    AUROC of disagreement scores as a wrong-prediction detector.
    Higher AUROC = disagreement is better at flagging incorrect predictions.
    """
    return roc_auc_score(y_true_is_wrong, disagreement_scores)
