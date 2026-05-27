"""
Phase 2a: Exploratory Data Analysis (EDA).

Loads the main training dataset, analyzes data quality, class distribution,
feature statistics, correlations, and per-class feature patterns. Saves all
figures to reports/figures/ and writes findings to reports/notes/.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    UNSW_PATH, ATTACK_COL, LABEL_COL, DROP_COLUMNS,
    NUMERIC_FEATURES, PROTOCOL_FEATURES, CATEGORICAL_FEATURES, SEED,
)


def set_plot_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
    })


def load_data():
    """Load the main UNSW dataset."""
    print(f"Loading {UNSW_PATH} ...")
    df = pd.read_csv(UNSW_PATH, low_memory=False)
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Dtype counts:\n{df.dtypes.value_counts()}")
    return df


def analyze_missing_values(df):
    """Analyze and report missing values."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"count": missing, "pct": missing_pct})
    report = report[report["count"] > 0].sort_values("count", ascending=False)

    print(f"\n=== Missing Values ({len(report)} columns with missing data) ===")
    print(report.to_string())

    # Decision rules summary
    high_missing = report[report["pct"] > 50]
    mid_missing = report[(report["pct"] > 5) & (report["pct"] <= 50)]
    low_missing = report[report["pct"] <= 5]

    decisions = {}
    for col in high_missing.index:
        decisions[col] = "DROP or flag as N/A (missing > 50%)"
    for col in mid_missing.index:
        if col in PROTOCOL_FEATURES:
            decisions[col] = "Fill with 0 (protocol not applicable)"
        else:
            decisions[col] = "Fill with median (traffic statistic)"
    for col in low_missing.index:
        decisions[col] = "Fill with median"

    return report, decisions


def check_l7_proto(df):
    """Check L7_PROTO for non-integer values (v3-specific issue)."""
    print("\n=== L7_PROTO Quality Check ===")
    l7 = df["L7_PROTO"].dropna()
    decimal_mask = l7 % 1 != 0
    n_nonint = decimal_mask.sum()
    print(f"Non-integer values: {n_nonint} / {len(l7)} ({n_nonint / len(l7) * 100:.2f}%)")
    unique_vals = sorted(l7[decimal_mask].unique())
    print(f"Unique non-integer values (first 30): {unique_vals[:30]}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: histogram of all L7_PROTO values
    axes[0].hist(l7.dropna(), bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].axvline(l7.median(), color="red", linestyle="--", label=f"Median: {l7.median():.1f}")
    axes[0].set_xlabel("L7_PROTO Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("L7_PROTO Distribution (All Values)")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Right: bar chart of non-integer counts
    nonint_counts = l7[decimal_mask].value_counts().sort_index().head(30)
    axes[1].bar(range(len(nonint_counts)), nonint_counts.values, color="darkorange", alpha=0.8)
    axes[1].set_xticks(range(0, len(nonint_counts), 5))
    axes[1].set_xticklabels([f"{v:.3f}" for v in nonint_counts.index[::5]], rotation=45, ha="right")
    axes[1].set_xlabel("L7_PROTO Value (non-integer)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Non-Integer L7_PROTO Values (Top 30, n={n_nonint})")

    plt.tight_layout()
    fig.savefig("reports/figures/l7_proto_quality.png")
    plt.close(fig)
    print("Saved: reports/figures/l7_proto_quality.png")

    return n_nonint, unique_vals


def analyze_class_distribution(df):
    """Analyze and plot class distribution."""
    print("\n=== Class Distribution ===")
    class_counts = df[ATTACK_COL].value_counts()
    class_pct = df[ATTACK_COL].value_counts(normalize=True) * 100

    dist_df = pd.DataFrame({"count": class_counts, "pct": class_pct})
    print(dist_df.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue" if c == "Benign" else "coral" for c in class_counts.index]
    bars = ax.bar(class_counts.index, class_counts.values, color=colors, alpha=0.85)

    ax.set_yscale("log")
    ax.set_ylabel("Sample Count (log scale)")
    ax.set_xlabel("Attack Class")
    ax.set_title("Class Distribution — NF-UNSW-NB15-v3")
    ax.tick_params(axis="x", rotation=45)

    # Add count labels
    for bar, count in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{count:,}", ha="center", fontsize=7)

    plt.tight_layout()
    fig.savefig("reports/figures/class_distribution.png")
    plt.close(fig)
    print("Saved: reports/figures/class_distribution.png")

    return class_counts.to_dict()


def analyze_feature_statistics(df):
    """Compute descriptive statistics for numeric features."""
    print("\n=== Feature Statistics ===")
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    desc = df[numeric_cols].describe().T
    desc["skewness"] = df[numeric_cols].skew()
    desc["range"] = desc["max"] - desc["min"]
    print(desc[["mean", "std", "min", "max", "skewness"]].to_string())
    return desc


def analyze_correlations(df):
    """Correlation analysis of numeric features."""
    print("\n=== Correlation Analysis ===")
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[numeric_cols].corr()

    # Identify highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.85:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    print(f"Highly correlated pairs (|r| > 0.85): {len(high_corr_pairs)}")
    for c1, c2, r in high_corr_pairs:
        print(f"  {c1} <-> {c2}: r = {r:.4f}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.1,
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.6, "label": "Pearson r"},
                ax=ax)
    ax.set_title("Feature Correlation Matrix — NF-UNSW-NB15-v3", fontsize=12)
    ax.tick_params(labelsize=5)

    plt.tight_layout()
    fig.savefig("reports/figures/correlation_heatmap.png")
    plt.close(fig)
    print("Saved: reports/figures/correlation_heatmap.png")

    return high_corr_pairs


def analyze_feature_distribution_by_class(df):
    """Plot key feature distributions per attack class."""
    print("\n=== Feature Distribution by Class ===")
    key_features = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS", "PROTOCOL"]
    available = [f for f in key_features if f in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, feat in enumerate(available):
        # Limit data for plotting performance: sample 5000 per class
        dfs = []
        for cls in df[ATTACK_COL].unique():
            cls_data = df[df[ATTACK_COL] == cls][feat].dropna()
            n_sample = min(5000, len(cls_data))
            dfs.append(pd.DataFrame({
                "value": cls_data.sample(n_sample, random_state=SEED),
                "class": cls,
            }))
        plot_df = pd.concat(dfs, ignore_index=True)

        sns.boxplot(data=plot_df, x="class", y="value", ax=axes[i],
                    palette="Set2", fliersize=1, linewidth=0.5)
        axes[i].set_yscale("log")
        axes[i].set_xlabel("Attack Class")
        axes[i].set_ylabel(feat)
        axes[i].set_title(f"{feat} Distribution by Class")
        axes[i].tick_params(axis="x", rotation=45, labelsize=7)

    for j in range(len(available), 4):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig("reports/figures/feature_distribution_by_class.png")
    plt.close(fig)
    print("Saved: reports/figures/feature_distribution_by_class.png")


def analyze_protocol_by_class(df):
    """Crosstab: protocol usage by attack class."""
    print("\n=== Protocol Distribution by Class ===")
    if "PROTOCOL" not in df.columns:
        print("PROTOCOL column not found, skipping.")
        return

    proto_map = {1: "ICMP", 6: "TCP", 17: "UDP"}
    proto_names = df["PROTOCOL"].map(proto_map).fillna(df["PROTOCOL"].astype(str))

    crosstab = pd.crosstab(df[ATTACK_COL], proto_names, normalize="index") * 100
    print(crosstab.round(1).to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    crosstab.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", alpha=0.85)
    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel("Attack Class")
    ax.set_title("Protocol Distribution by Attack Class")
    ax.legend(title="Protocol", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig("reports/figures/protocol_by_class.png")
    plt.close(fig)
    print("Saved: reports/figures/protocol_by_class.png")

    return crosstab


def save_notes(df, missing_report, decisions, l7_nonint, class_dist,
               high_corr_pairs, feature_desc):
    """Write EDA findings to reports/notes/data_and_eda_notes.md."""
    lines = []
    lines.append("# Data and EDA Notes\n")
    lines.append(f"## Dataset Version\n")
    lines.append(f"NF-UQ-NIDS-v3 collection.\n")
    lines.append(f"Main dataset: NF-UNSW-NB15-v3\n")
    lines.append(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    lines.append(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

    lines.append("\n## Target Columns\n")
    lines.append(f"- `{ATTACK_COL}`: multi-class target ({df[ATTACK_COL].nunique()} classes)\n")
    lines.append(f"- `{LABEL_COL}`: binary label (0=Benign, 1=Attack)\n")

    lines.append("\n## Class Distribution\n")
    lines.append("| Class | Count | Percentage |\n")
    lines.append("|-------|-------|------------|\n")
    for cls, cnt in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
        pct = cnt / len(df) * 100
        lines.append(f"| {cls} | {cnt:,} | {pct:.2f}% |\n")

    lines.append("\n## Missing Values\n")
    if len(missing_report) == 0:
        lines.append("No missing values found.\n")
    else:
        lines.append("| Column | Missing Count | Missing % | Decision |\n")
        lines.append("|--------|---------------|-----------|----------|\n")
        for col in missing_report.index:
            cnt = missing_report.loc[col, "count"]
            pct = missing_report.loc[col, "pct"]
            dec = decisions.get(col, "Fill with median")
            lines.append(f"| {col} | {int(cnt):,} | {pct:.2f}% | {dec} |\n")

    lines.append("\n## L7_PROTO Quality\n")
    lines.append(f"Non-integer values found: {l7_nonint:,} (must be rounded before encoding)\n")

    lines.append("\n## Feature Statistics\n")
    lines.append(f"| Feature | Mean | Std | Min | Max | Skewness |\n")
    lines.append(f"|---------|------|-----|-----|-----|----------|\n")
    for feat in feature_desc.index:
        row = feature_desc.loc[feat]
        lines.append(f"| {feat} | {row['mean']:.3f} | {row['std']:.3f} | "
                     f"{row['min']:.3f} | {row['max']:.3f} | {row['skewness']:.3f} |\n")

    lines.append("\n## High Correlation Pairs (|r| > 0.85)\n")
    if high_corr_pairs:
        for c1, c2, r in high_corr_pairs:
            lines.append(f"- {c1} <-> {c2}: r = {r:.4f}\n")
    else:
        lines.append("None found.\n")

    lines.append("\n## Implications for Modeling\n")
    lines.append("- Accuracy is insufficient — Macro-F1 and per-class recall are primary metrics.\n")
    lines.append("- Minority classes (Worms, Analysis, Shellcode) require special attention.\n")
    lines.append("- Use class_weight / sample_weight, not SMOTE as default.\n")
    lines.append("- High correlation pairs may be candidates for Stress C feature dropout analysis.\n")

    out_path = "reports/notes/data_and_eda_notes.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nSaved: {out_path}")


def main():
    print("=" * 60)
    print("Phase 2a: Exploratory Data Analysis")
    print("=" * 60)

    set_plot_style()
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/notes", exist_ok=True)

    df = load_data()
    missing_report, decisions = analyze_missing_values(df)
    l7_nonint, _ = check_l7_proto(df)
    class_dist = analyze_class_distribution(df)
    feature_desc = analyze_feature_statistics(df)
    high_corr_pairs = analyze_correlations(df)
    analyze_feature_distribution_by_class(df)
    analyze_protocol_by_class(df)

    save_notes(df, missing_report, decisions, l7_nonint, class_dist,
               high_corr_pairs, feature_desc)

    print("\n" + "=" * 60)
    print("EDA complete.")
    print("Figures: reports/figures/")
    print("Notes:   reports/notes/data_and_eda_notes.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
