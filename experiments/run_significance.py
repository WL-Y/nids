"""
Seed sensitivity check with proper seed propagation (reduced RF).

NOT a final-model significance test. Uses a reduced RF (n_estimators=50,
max_depth=15) for computational feasibility. Reports seed-to-seed consistency
of the experimental pipeline rather than final model significance.

Loops through SIGNIFICANCE_SEEDS, trains reduced models with each seed,
evaluates on Clean + Stress A (strict open-set), and generates
statistical_significance.csv.

Each seed controls: data split, preprocessing, model initialization.
"""

import os
import sys
import json
import gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import (
    UNSW_PATH, ATTACK_COL, HELD_OUT_CLASSES_SETS,
    SIGNIFICANCE_SEEDS, SEED,
)
from preprocessing.preprocess import (
    load_and_clean_data, clean_l7_proto,
    build_preprocessor,
)
from evaluation.metrics import (
    macro_f1_score, classification_report_full, run_statistics,
)
from evaluation.plots import plot_confusion_matrix


def run_one_seed(seed):
    """Run a complete train/eval cycle with a given seed, returning metrics.

    Fix P1-1: seed controls data split AND model initialization.
    Fix P1-5: Stress A uses strict preprocessing (preprocessor fit on known-only).
    """
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

    print(f"\n--- Seed={seed} ---")

    # 1. Load + clean data
    df = load_and_clean_data(UNSW_PATH)
    df = clean_l7_proto(df)
    y = df[ATTACK_COL]
    X = df.drop(columns=[ATTACK_COL, "Label"])

    # 2. Split with THIS seed (Bug 1 fix: pass seed into split)
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed,
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_frac, stratify=y_train_val,
        random_state=seed,
    )
    y_tr = y_train
    X_tr = X_train
    y_te = y_test
    X_te = X_test
    class_names_full = sorted(y_tr.unique())

    # Use subset for speed
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    _, idx = next(splitter.split(X_tr, y_tr))
    X_tr_sub = X_tr.iloc[idx]
    y_tr_sub = y_tr.iloc[idx]

    # 3. CLEAN evaluation: preprocessor fit on all-class subset
    prep_clean = build_preprocessor()
    prep_clean.fit(X_tr_sub)
    X_train_t = prep_clean.transform(X_tr_sub)
    X_test_t_clean = prep_clean.transform(X_te)

    # 4. Train model with this seed
    model = RandomForestClassifier(
        n_estimators=50, max_depth=15, min_samples_leaf=2,
        min_samples_split=5, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    model.fit(X_train_t, y_tr_sub)

    # 5. Evaluate: Clean
    y_pred = model.predict(X_test_t_clean)
    clean_f1 = macro_f1_score(y_te, y_pred)

    # 6. Evaluate: Stress A — STRICT preprocessing (Bug 2 fix)
    sa_f1s = []
    for held_out in HELD_OUT_CLASSES_SETS:
        known_classes = sorted(set(class_names_full) - set(held_out))

        # Filter RAW data to known-only BEFORE fitting preprocessor
        train_mask = y_tr_sub.isin(known_classes)
        X_raw_k = X_tr_sub.loc[train_mask]
        y_tr_k = y_tr_sub[train_mask]

        # ASSERT: no held-out classes in training
        leaked = set(held_out) & set(y_tr_k.unique())
        assert len(leaked) == 0, f"DATA LEAK: held-out {leaked} in known-only train!"

        # Fit FRESH preprocessor on known-only raw data
        prep_k = build_preprocessor()
        prep_k.fit(X_raw_k)
        X_tr_k = prep_k.transform(X_raw_k)
        X_te_k = prep_k.transform(X_te)

        model_k = RandomForestClassifier(
            n_estimators=50, max_depth=15, min_samples_leaf=2,
            min_samples_split=5, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        model_k.fit(X_tr_k, y_tr_k)
        y_pred_k = model_k.predict(X_te_k)
        known_mask = y_te.isin(known_classes)
        sa_f1s.append(macro_f1_score(y_te[known_mask], y_pred_k[known_mask]))
        gc.collect()

    stress_a_f1 = np.mean(sa_f1s)
    gc.collect()

    return {"seed": seed, "clean_macro_f1": clean_f1, "stress_a_macro_f1": stress_a_f1}


def main():
    os.makedirs("results", exist_ok=True)
    print("=" * 60)
    print("Statistical Significance Runs")
    print(f"Seeds: {SIGNIFICANCE_SEEDS}")
    print("=" * 60)

    all_results = []
    for seed in SIGNIFICANCE_SEEDS:
        res = run_one_seed(seed)
        all_results.append(res)
        print(f"  Seed={seed}: Clean F1={res['clean_macro_f1']:.4f}, "
              f"Stress A F1={res['stress_a_macro_f1']:.4f}")

    # Aggregate
    clean_f1s = [r["clean_macro_f1"] for r in all_results]
    sa_f1s = [r["stress_a_macro_f1"] for r in all_results]

    print(f"\nResults (mean ± std):")
    print(f"  Clean Macro-F1:    {run_statistics(clean_f1s)}")
    print(f"  Stress A Macro-F1:  {run_statistics(sa_f1s)}")

    # Save
    df = pd.DataFrame([{
        "model": "RandomForest",
        "condition": "Clean",
        "metric": "macro_f1",
        "mean": round(np.mean(clean_f1s), 4),
        "std": round(np.std(clean_f1s), 4),
        "n_runs": len(SIGNIFICANCE_SEEDS),
        "seeds": ";".join(str(s) for s in SIGNIFICANCE_SEEDS),
    }, {
        "model": "RandomForest",
        "condition": "Stress_A",
        "metric": "macro_f1",
        "mean": round(np.mean(sa_f1s), 4),
        "std": round(np.std(sa_f1s), 4),
        "n_runs": len(SIGNIFICANCE_SEEDS),
        "seeds": ";".join(str(s) for s in SIGNIFICANCE_SEEDS),
    }])
    df.to_csv("results/statistical_significance.csv", index=False)
    print("\nSaved: results/statistical_significance.csv")
    print(df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("Significance test complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
