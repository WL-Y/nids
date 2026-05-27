"""Consistency check: schema, S2 no NaN, S1 N/A, README numbers, file inventory."""

import os, re
import pandas as pd
import numpy as np

errors = []

df = pd.read_csv("results/strategies_comparison.csv")
df_dup = pd.read_csv("results/strategy_comparison.csv")
assert df.equals(df_dup), "FAIL: mismatch"
print("PASS: strategies == strategy")

cols = list(df.columns)
assert "Method" in cols and "Clean Full F1" in cols and "Detection AUROC" in cols
print(f"PASS: schema {len(cols)} cols")

# Summary rows
for f in ["strategy1_summary.csv", "strategy2_summary.csv"]:
    d = pd.read_csv(f"results/{f}")
    assert len(d) == 1, f"FAIL: {f} rows={len(d)}"
print("PASS: summaries 1 row each")

# No NaN in S2
s2 = df[df["Method"].str.contains("Strategy 2")].iloc[0]
for c in ["Stress A F1", "Stress B F1", "Stress C F1"]:
    assert not np.isnan(float(s2[c])), f"FAIL: S2 {c}=NaN"
print("PASS: S2 no NaN")

# S1 Stress B/C contain "undef" or "cov"
with open("results/strategies_comparison.csv") as f:
    raw = f.read()
assert "0% cov / undef." in raw, "FAIL: S1 Stress B/C missing coverage note"
print("PASS: S1 Stress B/C = 0% cov / undef. (verified in raw CSV)")

# Raw version all numeric
df_raw = pd.read_csv("results/strategies_comparison_raw.csv")
for c in df_raw.columns:
    if c == "Method": continue
    assert df_raw[c].dtype.kind in "fi", f"FAIL: raw {c} dtype={df_raw[c].dtype}"
print("PASS: raw CSV all numeric")

# README numbers
with open("README.md") as f:
    rm = f.read()
for pat, expected in [("Baseline.*?RF.*?Macro-F1.*?(\\d+\\.\\d+)", 0.6774),
                       ("Stress C.*?\\| (\\d+\\.\\d+)", 0.2688)]:
    m = re.search(pat, rm)
    if not m or abs(float(m.group(1)) - expected) > 0.01:
        errors.append(f"README mismatch: {pat}")
if not errors:
    print("PASS: README consistent")

# Files
required = ["results/strategies_comparison.csv", "results/strategies_comparison_raw.csv",
            "results/strategy1_summary.csv", "results/strategy2_summary.csv",
            "results/mcnemar_results.csv", "results/statistical_significance.csv",
            "reports/notes/table_footnotes.txt"]
for p in required:
    if not os.path.exists(p): errors.append(f"MISSING: {p}")
if not errors:
    print(f"PASS: all {len(required)} files")
    print("\nALL CHECKS PASSED.")
else:
    print(f"\n{len(errors)} ERROR(S):")
    for e in errors: print(f"  {e}")
