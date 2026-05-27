"""Verify items 10, 11, 12 are complete."""
import pandas as pd, os

# === 10. Stress C ===
sc = pd.read_csv("results/stress_c_results.csv")
print("=== 10. Stress C ===")
print(sc.to_string(index=False))
types = sc["type"].unique()
print(f"Degradation types: {len(types)} ({list(types)})")
assert "noise" in types and "masking" in types and "dropout_top" in types and "dropout_bottom" in types
print("PASS: All 3 degradation types present")

degradation_fig = "reports/figures/stress_c_degradation_curve.png"
assert os.path.exists(degradation_fig)
print(f"PASS: degradation curve exists")

# === 11. Baseline ===
bl = pd.read_csv("results/baseline_results.csv")
print(f"\n=== 11. Baseline ===")
print(bl.to_string(index=False))
required_models = ["Majority", "LogisticRegression", "RandomForest", "XGBoost"]
for m in required_models:
    report_ok = os.path.exists(f"results/per_class_report_{m}.csv")
    cm_ok = os.path.exists(f"reports/figures/confusion_matrix_{m}.png")
    print(f"  {m}: per_class={report_ok}, confusion_matrix={cm_ok}")
    assert report_ok and cm_ok, f"MISSING for {m}"

# Check required columns in baseline_results
for col in ["macro_f1", "weighted_f1", "train_time_s", "inference_time_s"]:
    assert col in bl.columns, f"MISSING column {col}"
print(f"PASS: All 4 models with per-class reports, confusion matrices, timing")

# === 12. Significance ===
ss = pd.read_csv("results/statistical_significance.csv")
print(f"\n=== 12. Significance ===")
print(ss.to_string(index=False))
assert len(ss) >= 2, "Need at least Clean + Stress A"
assert "mean" in ss.columns and "std" in ss.columns
assert "n_runs" in ss.columns and int(ss["n_runs"].iloc[0]) == 5
print("PASS: 5-run mean/std reported")

# McNemar
mc = pd.read_csv("results/mcnemar_results.csv")
assert len(mc) >= 2
print(f"PASS: McNemar {len(mc)} rows (S1 + S2)")

print("\n=== ALL ITEMS 10/11/12 VERIFIED ===")
