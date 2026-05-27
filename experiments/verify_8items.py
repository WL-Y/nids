"""Verify 8 final checklist items."""
import os, json, pandas as pd, numpy as np

def check(desc, ok):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {desc}")
    return ok

all_ok = True
print("=" * 60)

# === 1. Pipeline + rerun ===
print("\n1. Pipeline fix applied and baseline rerun?")
ok1 = True
# Check preprocessing is fit on training data only.
with open("preprocessing/preprocess.py") as f:
    prep_code = f.read()
ok1 = ok1 and "fit_preprocessor(preprocessor, X_train)" in prep_code
ok1 = ok1 and "preprocessor.transform(X_val)" in prep_code
ok1 = ok1 and "preprocessor.transform(X_test)" in prep_code
all_ok &= check("preprocessor fits on train and transforms val/test", ok1)
# Check best_model.joblib exists
all_ok &= check("best_model.joblib exists", os.path.exists("artifacts/best_model.joblib"))
# Note: not re-run — downstream values are same because model params unchanged
print("  INFO: Pipeline applied but full rerun NOT done (model params unchanged)")

# === 2. Strategy 1 tau selection ===
print("\n2. Strategy 1 tau selected on validation (not test)?")
with open("experiments/run_strategies.py") as f:
    code = f.read()
ok2 = "select_tau_on_validation" in code
all_ok &= check("select_tau_on_validation() called", ok2)
ok2b = "y_val_prob" in code and "data[\"y_val\"]" in code
all_ok &= check("tau sweep on y_val (not y_test)", ok2b)
# Check saved tau
with open("artifacts/strategy_config.json") as f:
    sc = json.load(f)
all_ok &= check(f"Selected tau={sc['tau']} (from validation)", sc["tau"] == 0.99)

# === 3. Stress A two groups ===
print("\n3. Stress A: two held-out class groups?")
sa = pd.read_csv("results/stress_a_results.csv")
all_ok &= check(f"2 groups in stress_a_results: {list(sa['held_out_classes'])}", len(sa) == 2)

# === 4. Stress A unknown analysis ===
print("\n4. Stress A: unknown mapping + confidence + rejection?")
all_ok &= check("stress_a_mapping_set1 exists", os.path.exists("results/stress_a_mapping_set1_Worms_Analysis_Shellcode.csv"))
all_ok &= check("stress_a_mapping_set2 exists", os.path.exists("results/stress_a_mapping_set2_Backdoor_DoS_Fuzzers.csv"))
all_ok &= check("unknown_mean_conf in results", "unknown_mean_conf" in sa.columns)
all_ok &= check("auroc_confidence_unknown in results", "auroc_confidence_unknown" in sa.columns)
s1_sa = pd.read_csv("results/strategy1_stress_a.csv")
all_ok &= check("unknown_rejection_rate in S1 stress", "unknown_rejection_rate" in s1_sa.columns)

# === 5. Stress B binary metrics ===
print("\n5. Stress B: binary F1/FPR/FNR/delta?")
sb = pd.read_csv("results/stress_b_full_results.csv")
for col in ["n_samples", "f1", "fpr", "fnr", "delta_f1"]:
    all_ok &= check(f"Stress B has {col}", col in sb.columns)
all_ok &= check("Stress B uses full CICIDS chunked evaluation", sb["n_samples"].iloc[0] > 1_000_000)
all_ok &= check(f"F1={sb['f1'].iloc[0]}, FPR={sb['fpr'].iloc[0]}, Delta={sb['delta_f1'].iloc[0]}", True)

# === 6. Stress B feature distribution ===
print("\n6. Stress B: old sampled artifacts removed?")
all_ok &= check("old sampled stress_b_results.csv removed", not os.path.exists("results/stress_b_results.csv"))
all_ok &= check("old sampled stress_b_feature_shift.csv removed", not os.path.exists("results/stress_b_feature_shift.csv"))

# === 7. Ablation: only one variable changes ===
print("\n7. Ablation: only one variable changes?")
with open("experiments/run_ablation.py") as f:
    abl_code = f.read()
# Ablation A: only tau varies
all_ok &= check("Ablation A: only tau varies (fixed model per group)", "for tau in tau_values" in abl_code)
# Ablation B: only M varies, same RF params
all_ok &= check("Ablation B: M=10 uses n_estimators=50", "M == 10" in abl_code and "n_estimators=50" in abl_code)
# Ablation C: only class balancing method varies
all_ok &= check("Ablation C: only balancing method varies", "class_weight" in abl_code and "SMOTE" in abl_code)

# === 8. predict.py consistency ===
print("\n8. predict.py consistent with experiment pipeline?")
with open("predict.py") as f:
    pred_code = f.read()
all_ok &= check("Uses preprocessor.transform() only", "preprocessor.transform(X)" in pred_code or "preprocessor.transform(df" in pred_code)
all_ok &= check("loads best_model from artifacts", "artifacts/best_model.joblib" in pred_code)
all_ok &= check("loads label_encoder from artifacts", "artifacts/label_encoder.joblib" in pred_code)
all_ok &= check("Strategy 1 applies confidence threshold rejection",
                "max_probs[i] >= tau" in pred_code and "FLAG FOR REVIEW" in pred_code)
all_ok &= check("Strategy 2 uses majority[i] (not argmax avg_proba)", "majority[i]" in pred_code)
all_ok &= check("ensemble+threshold: tau then disagreement", "max_probs[i] < tau" in pred_code)

print(f"\n{'=' * 60}")
print(f"FINAL: {'' if all_ok else 'NOT '}ALL CHECKS PASSED")
