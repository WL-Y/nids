"""
Visualization functions for multi-class NIDS evaluation.

All plots use matplotlib + seaborn with Agg backend for non-interactive use.
Default DPI: 150, tight bounding box on save.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


# Global style
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
})


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_class_distribution(y, class_names, save_path):
    """Bar chart of class distribution with log-scale y-axis."""
    counts = pd.Series(y).value_counts()
    counts = counts.reindex(class_names, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue" if c == "Benign" else "coral" for c in class_names]
    bars = ax.bar(class_names, counts.values, color=colors, alpha=0.85)

    ax.set_yscale("log")
    ax.set_ylabel("Sample Count (log scale)")
    ax.set_xlabel("Attack Class")
    ax.set_title("Class Distribution")
    ax.tick_params(axis="x", rotation=45)

    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{cnt:,}", ha="center", fontsize=6)

    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(cm, classes, save_path, normalize=False):
    """Confusion matrix heatmap. Pass cm as a DataFrame or 2D array."""
    if isinstance(cm, pd.DataFrame):
        values = cm.values
        labels = list(cm.columns)
    else:
        values = cm
        labels = classes

    if normalize:
        values = values.astype(float) / values.sum(axis=1, keepdims=True)
        values = np.nan_to_num(values, 0)
        fmt = ".2f"
        vmax = 1.0
    else:
        fmt = "d"
        vmax = None

    figsize = max(8, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))
    sns.heatmap(values, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                square=True, linewidths=0.3, vmin=0, vmax=vmax, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)


def plot_feature_importance(importances, feature_names, save_path, top_k=20):
    """Top-K feature importance bar chart."""
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(0.3 + 0.7 * np.linspace(0, 1, len(imp_df)))[::-1]
    ax.barh(range(len(imp_df)), imp_df["importance"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Feature Importances")
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)


def plot_confidence_distribution(conf_df, save_path):
    """
    Box plot of confidence grouped by true class, split by correct/wrong.
    conf_df must have columns: true_class, is_correct, confidence.
    May also have a pre-set "status" column (e.g. "unknown"/"known_correct"/"known_wrong").
    """
    plot_df = conf_df.copy()
    if "status" not in plot_df.columns:
        plot_df["status"] = plot_df["is_correct"].map({True: "Correct", False: "Wrong"})

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = {"Correct": "steelblue", "Wrong": "coral",
               "known_correct": "steelblue", "known_wrong": "coral", "unknown": "darkorange"}
    sns.boxplot(data=plot_df, x="true_class", y="confidence", hue="status",
                palette=palette,
                fliersize=1, linewidth=0.5, ax=ax)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("True Class")
    ax.set_ylabel("Confidence (max probability)")
    ax.set_title("Confidence Distribution by Class (Correct vs Wrong)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)


def plot_coverage_accuracy_curve(tau_results, save_path):
    """
    Coverage vs accuracy curve across tau thresholds.
    tau_results: DataFrame with columns tau, coverage, accuracy.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "steelblue"
    ax1.plot(tau_results["tau"], tau_results["coverage"], marker="o",
             color=color1, markersize=3, label="Coverage")
    ax1.set_xlabel("Threshold tau")
    ax1.set_ylabel("Coverage", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.plot(tau_results["tau"], tau_results["accuracy"], marker="s",
             color=color2, markersize=3, label="Accuracy (accepted)")
    ax2.set_ylabel("Accuracy on Accepted", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Coverage-Accuracy Trade-off (Strategy 1)")
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)


def plot_degradation_curve(degradation_dict, save_path):
    """
    Degradation curve: multiple lines (noise / masking / dropout) x level vs macro-F1.
    degradation_dict = {"Noise": ([levels], [f1s]), "Mask": ([levels], [f1s]), ...}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Noise": "steelblue", "Masking": "coral", "Dropout (top)": "darkorange",
              "Dropout (bottom)": "green"}
    markers = {"Noise": "o", "Masking": "s", "Dropout (top)": "^", "Dropout (bottom)": "v"}

    for label, (levels, f1_vals) in degradation_dict.items():
        ax.plot(levels, f1_vals, marker=markers.get(label, "o"),
                color=colors.get(label, None), label=label, alpha=0.85)

    ax.set_xlabel("Degradation Level")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Model Performance Under Feature Degradation (Stress C)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)


def plot_disagreement_histogram(disagreement, is_wrong, save_path):
    """
    Overlaid histogram of ensemble disagreement scores:
    correct predictions vs wrong predictions.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(disagreement[~is_wrong], bins=30, alpha=0.6, color="steelblue",
            density=True, label="Correct")
    ax.hist(disagreement[is_wrong], bins=30, alpha=0.6, color="coral",
            density=True, label="Wrong")
    ax.set_xlabel("Disagreement Score")
    ax.set_ylabel("Density")
    ax.set_title("Ensemble Disagreement: Correct vs Wrong Predictions")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)


def plot_feature_distribution_comparison(source_values, target_values,
                                          feature_name, save_path):
    """
    Source vs target domain distribution comparison for a single feature (Stress B).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(source_values.dropna(), bins=50, alpha=0.5, color="steelblue",
            density=True, label="UNSW (source)")
    ax.hist(target_values.dropna(), bins=50, alpha=0.5, color="coral",
            density=True, label="CICIDS2018 (target)")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.set_title(f"Feature Distribution Shift: {feature_name}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)


def plot_disagreement_roc_curve(disagreement_scores, y_true_is_wrong, auroc_value, save_path):
    """ROC curve of disagreement score as a wrong-prediction detector."""
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(y_true_is_wrong, disagreement_scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"ROC (AUROC = {auroc_value:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Ensemble Disagreement as Wrong-Prediction Detector")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)


def plot_reliability_diagram(y_prob, y_true, n_bins=10, save_path=None):
    """
    Reliability diagram for probability calibration assessment.
    [Future Work] Not required for main experiments.
    """
    max_probs = np.max(y_prob, axis=1)
    y_pred = np.argmax(y_prob, axis=1)
    correct = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (max_probs > bin_edges[i]) & (max_probs <= bin_edges[i + 1])
        if mask.sum() > 0:
            accs[i] = correct[mask].mean()
            confs[i] = max_probs[mask].mean()
        else:
            accs[i] = np.nan
            confs[i] = bin_centers[i]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    ax.plot(confs, accs, "o-", color="steelblue", label="Model")
    ax.set_xlabel("Mean Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax
