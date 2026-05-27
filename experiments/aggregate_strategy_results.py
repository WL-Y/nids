"""
Aggregate intermediate strategy results into final comparison tables.

Reads strategy summaries, stress results, and produces:
  results/strategies_comparison.csv  (report version with N/A markers)
  results/strategies_comparison_raw.csv (machine-readable, pure numeric)

Usage:
  python experiments/aggregate_strategy_results.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

REPORT_COLS = ["Method", "Clean Full F1", "Coverage", "Accepted F1",
               "Stress A F1", "Stress B F1", "Stress C F1",
               "Rejection Rate", "Detection AUROC"]


def _load(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    os.makedirs("results", exist_ok=True)

    # Build baseline row from stress/stress_c results.
    # Stress B is full CICIDS chunked evaluation only.
    sa = _load("results/stress_a_results.csv")
    sb = _load("results/stress_b_full_results.csv")
    sc = _load("results/stress_c_results.csv")

    sa_f1 = sa["known_macro_f1"].mean() if sa is not None else np.nan
    sb_f1 = sb["f1"].iloc[0] if sb is not None else np.nan
    sc_row = sc[(sc["type"] == "noise") & (sc["level"] == "sigma=0.1")] if sc is not None else None
    sc_f1 = sc_row["macro_f1"].iloc[0] if sc_row is not None and len(sc_row) > 0 else np.nan

    rows = [{
        "Method": "Baseline (no strategy)",
        "Clean Full F1": 0.6774,
        "Coverage": 1.0,
        "Accepted F1": np.nan,
        "Stress A F1": round(sa_f1, 4) if not np.isnan(sa_f1) else np.nan,
        "Stress B F1": round(sb_f1, 4) if not np.isnan(sb_f1) else np.nan,
        "Stress C F1": round(sc_f1, 4) if not np.isnan(sc_f1) else np.nan,
        "Rejection Rate": np.nan,
        "Detection AUROC": np.nan,
    }]

    # Merge strategy summaries (keep_default_na=False preserves "N/A" strings)
    for f in ["results/strategy1_summary.csv", "results/strategy2_summary.csv"]:
        if os.path.exists(f):
            df = pd.read_csv(f, keep_default_na=False)
            rows.extend(df.to_dict("records"))

    df_report = pd.DataFrame(rows)
    # Ensure correct column order
    for c in REPORT_COLS:
        if c not in df_report.columns:
            df_report[c] = "N/A"
    df_report = df_report[REPORT_COLS]

    # Save report version
    df_report.to_csv("results/strategies_comparison.csv", index=False)
    df_report.to_csv("results/strategy_comparison.csv", index=False)

    # Convert Coverage strings to numeric for both versions
    # "1.000 (no rejection)" -> 1.0, "0% cov / undef." -> NaN
    df_report["Coverage"] = df_report["Coverage"].replace({"1.000 (no rejection)": 1.0})
    df_report["Coverage"] = pd.to_numeric(df_report["Coverage"], errors="coerce")
    df_report["Coverage"] = df_report["Coverage"].apply(
        lambda x: "1.000 (no rejection)" if x == 1.0 else ("0% cov / undef." if pd.isna(x) else f"{x:.4f}"))

    # Save raw version (pure numeric)
    df_raw = df_report.copy()
    for col in df_raw.columns:
        if col == "Method":
            continue
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    df_raw["Coverage"] = df_raw["Coverage"].fillna(1.0)
    df_raw.loc[df_raw["Method"].str.contains("Strategy 1"), "Coverage"] = pd.to_numeric(
        df_report.loc[df_report["Method"].str.contains("Strategy 1"), "Coverage"], errors="coerce")
    df_raw.to_csv("results/strategies_comparison_raw.csv", index=False, na_rep="NaN")

    print(df_report.to_string(index=False))
    print()
    print("Saved: results/strategies_comparison.csv")
    print("Saved: results/strategies_comparison_raw.csv")


if __name__ == "__main__":
    main()
