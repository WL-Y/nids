"""
Phase 5: Robustness Strategy Experiments (per implementation_plan.md).

1. Load Phase 3 best model (artifacts/best_model.joblib)
2. Strategy 1: confidence threshold — sweep tau, select on val, evaluate
3. Strategy 2: heterogeneous ensemble — train, disagreement AUROC, evaluate
4. Compare vs baseline on Clean + Stress A conditions
5. Save strategy config + results

Usage:
  python experiments/run_strategies.py              # full run
  python experiments/run_strategies.py --strategy 1  # Strategy 1 only
  python experiments/run_strategies.py --strategy 2  # Strategy 2 only
"""

import os
import sys
import json
import gc
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

from config import (
    UNSW_PATH, CICIDS_PATH, ATTACK_COL, LABEL_COL, HELD_OUT_CLASSES_SETS,
    TAU_RANGE, ENSEMBLE_SIZE, SEED, NUMERIC_FEATURES,
    STRESS_B_CHUNK_SIZE,
)
from preprocessing.preprocess import (
    load_and_clean_data, clean_l7_proto, split_data,
    build_preprocessor, fit_preprocessor, prepare_cicids_chunk,
)
from robustness.strategies import (
    predict_with_rejection, select_tau_on_validation,
    train_heterogeneous_ensemble, ensemble_disagreement,
)
from robustness.streaming_ensemble import (
    train_predict_clean_eval_sets,
    majority_and_disagreement_from_prediction_files,
    full_stress_b_single_pass_ensemble,
)
from evaluation.metrics import (
    macro_f1_score, disagreement_auroc, mcnemar_test, run_statistics,
)
from evaluation.plots import (
    plot_coverage_accuracy_curve, plot_disagreement_histogram,
    plot_disagreement_roc_curve,
)


def prepare_data():
    """Load, clean, split, preprocess. Returns dict with processed data."""
    print("Preparing data ...")
    df = load_and_clean_data(UNSW_PATH)
    df = clean_l7_proto(df)

    y_full = df[ATTACK_COL]
    X_full = df.drop(columns=[ATTACK_COL, "Label"])

    df_for_split = X_full.copy()
    df_for_split[ATTACK_COL] = y_full
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_for_split)

    y_train = X_train.pop(ATTACK_COL)
    y_val = X_val.pop(ATTACK_COL)
    y_test = X_test.pop(ATTACK_COL)

    # Load pre-trained preprocessor
    preprocessor = joblib.load("artifacts/preprocessor.joblib")

    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    class_names = sorted(y_train.unique())

    return {
        "X_train": X_train_t, "X_val": X_val_t, "X_test": X_test_t,
        "X_train_raw": X_train, "X_val_raw": X_val, "X_test_raw": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "preprocessor": preprocessor, "class_names": class_names,
    }


def _stratified_limit(X, y, max_samples=100000, seed=SEED):
    """Limit training size with stratified sampling to reduce memory."""
    if len(y) <= max_samples:
        return X, y
    from sklearn.model_selection import StratifiedShuffleSplit
    frac = max_samples / len(y)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
    idx, _ = next(splitter.split(X, y))
    return X[idx], y.iloc[idx] if hasattr(y, "iloc") else y[idx]


def train_lightweight_ensemble(X_train, y_train, seed=SEED):
    """Lightweight ensemble for memory-constrained Stress A. Uses LR + one small RF."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    models = []
    lr = LogisticRegression(max_iter=500, class_weight="balanced", solver="saga",
                            n_jobs=1, random_state=seed)
    lr.fit(X_train, y_train)
    models.append(("LR_light", lr))
    rf = RandomForestClassifier(n_estimators=30, max_depth=12, min_samples_leaf=3,
                                min_samples_split=8, max_features="sqrt",
                                class_weight="balanced", random_state=seed, n_jobs=1)
    rf.fit(X_train, y_train)
    models.append(("RF_light", rf))
    return models


def _train_known_only_model(data, held_out):
    """Train RF on known-class data only (strict preprocessing per P1-5)."""
    from sklearn.ensemble import RandomForestClassifier
    known_classes = sorted(set(data["class_names"]) - set(held_out))
    train_mask = data["y_train"].isin(known_classes)
    val_mask = data["y_val"].isin(known_classes)

    # ASSERT: held-out classes must NOT appear in training data
    held_in_train = set(data["y_train"][train_mask].unique())
    leaked = set(held_out) & held_in_train
    assert len(leaked) == 0, f"DATA LEAKAGE: held-out classes {leaked} found in training set!"

    X_train_raw_k = data["X_train_raw"].loc[train_mask]
    y_train_k = data["y_train"][train_mask]
    X_val_raw_k = data["X_val_raw"].loc[val_mask]
    y_val_k = data["y_val"][val_mask]

    # ASSERT: preprocessor fit ONLY on known-class training data
    prep_k = build_preprocessor()
    prep_k.fit(X_train_raw_k)
    X_train_k = prep_k.transform(X_train_raw_k)
    X_val_k = prep_k.transform(X_val_raw_k)
    X_test_k = prep_k.transform(data["X_test_raw"])

    model_k = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=2,
        min_samples_split=5, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    model_k.fit(X_train_k, y_train_k)
    return model_k, X_test_k, known_classes, y_train_k, X_train_k, y_val_k


def evaluate_strategy_stress_a(data, held_out, tau=None, use_gpu=True,
                                lightweight=False, max_s2_samples=100000):
    """
    Evaluate Strategy 1 or 2 on Stress A with known-only training.
    Returns dict with known_macro_f1, unknown_rejection_rate (S1), or disagreement_auroc (S2).
    """
    model_k, X_test_k, known_classes, y_train_k, X_train_k, y_val_k = \
        _train_known_only_model(data, held_out)

    is_unknown = ~data["y_test"].isin(known_classes)
    known_mask = ~is_unknown.values
    result = {"held_out": ", ".join(held_out)}

    if tau is not None:
        # Strategy 1
        y_pred, rejected, _ = predict_with_rejection(model_k, X_test_k, tau)
        accepted = ~rejected & known_mask
        result["known_macro_f1"] = macro_f1_score(
            data["y_test"][accepted], y_pred[accepted]) if accepted.sum() > 0 else np.nan
        result["unknown_rejection_rate"] = rejected[is_unknown.values].mean()
        result["known_false_rejection_rate"] = rejected[known_mask].mean()
        result["coverage"] = (~rejected).mean()
    else:
        # Strategy 2
        if lightweight:
            X_train_s, y_train_s = _stratified_limit(
                X_train_k, y_train_k, max_samples=max_s2_samples, seed=SEED)
            print(f"    Lightweight S2 training samples: {X_train_s.shape[0]:,}",
                  flush=True)
            ensemble = train_lightweight_ensemble(X_train_s, y_train_s)
        else:
            ensemble = train_heterogeneous_ensemble(X_train_k, y_train_k, use_gpu=use_gpu)

        disagreement, majority = ensemble_disagreement(ensemble, X_test_k)
        is_wrong = majority != data["y_test"].values
        result["known_macro_f1"] = macro_f1_score(
            data["y_test"][known_mask], majority[known_mask])
        # AUROC(disagreement, wrong vs correct) — wrong-prediction detector
        result["disagreement_auroc"] = disagreement_auroc(disagreement, is_wrong)
        # AUROC(disagreement, unknown vs known) — unknown-class detector
        result["unknown_detection_auroc"] = disagreement_auroc(disagreement, ~known_mask)
        result["unknown_mean_disagreement"] = disagreement[~known_mask].mean() if (~known_mask).sum() > 0 else np.nan
        result["known_mean_disagreement"] = disagreement[known_mask].mean()

        del ensemble
        del X_train_k
        del X_test_k
        gc.collect()

    return result


def evaluate_strategy_stress_b(model, preprocessor, tau=None, models=None):
    """
    Evaluate strategy on Stress B (full CICIDS cross-dataset).
    Returns binary F1 + coverage (S1) or majority F1 (S2).
    """
    import gc

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "accepted": 0, "n": 0}
    expected_cols = None

    print("  Streaming full CICIDS for strategy Stress B evaluation ...")
    reader = pd.read_csv(CICIDS_PATH, chunksize=STRESS_B_CHUNK_SIZE, low_memory=False)

    for chunk_id, chunk in enumerate(reader, start=1):
        chunk = prepare_cicids_chunk(chunk)
        if expected_cols is None:
            if hasattr(preprocessor, "feature_names_in_"):
                expected_cols = list(preprocessor.feature_names_in_)
            else:
                expected_cols = [c for c in chunk.columns if c not in (ATTACK_COL, LABEL_COL)]

        y_true_bin = (chunk[ATTACK_COL] != "Benign").astype(int).values
        drop_cols = [ATTACK_COL]
        if LABEL_COL in chunk.columns:
            drop_cols.append(LABEL_COL)
        X_c = chunk.drop(columns=drop_cols).reindex(columns=expected_cols, fill_value=0)
        # STRESS B RULE: transform() only; never fit on CICIDS.
        X_c_t = preprocessor.transform(X_c)

        if tau is not None:
            y_pred, rejected, _ = predict_with_rejection(model, X_c_t, tau)
            y_pred_bin = np.where(y_pred == "REJECT", 0, (y_pred != "Benign").astype(int))
            accepted = ~rejected
        elif models is not None:
            _, majority = ensemble_disagreement(models, X_c_t)
            y_pred_bin = (majority != "Benign").astype(int)
            accepted = np.ones(len(y_pred_bin), dtype=bool)
        else:
            raise ValueError("Pass either tau for Strategy 1 or models for Strategy 2.")

        y_eval = y_true_bin[accepted]
        p_eval = y_pred_bin[accepted]
        counts["tp"] += int(((y_eval == 1) & (p_eval == 1)).sum())
        counts["tn"] += int(((y_eval == 0) & (p_eval == 0)).sum())
        counts["fp"] += int(((y_eval == 0) & (p_eval == 1)).sum())
        counts["fn"] += int(((y_eval == 1) & (p_eval == 0)).sum())
        counts["accepted"] += int(accepted.sum())
        counts["n"] += int(len(y_true_bin))

        if chunk_id == 1 or chunk_id % 10 == 0:
            partial = _strategy_stress_b_metrics(counts)
            print(
                f"    chunk={chunk_id} processed={partial['stress_b_n_samples']:,} "
                f"coverage={partial['stress_b_coverage']:.4f} "
                f"f1={partial['stress_b_f1']:.4f}"
            )

        del chunk, X_c, X_c_t, y_true_bin, y_pred_bin, accepted
        gc.collect()

    return _strategy_stress_b_metrics(counts)


def _strategy_stress_b_metrics(counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    accepted = counts["accepted"]
    n = counts["n"]
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (
        not np.isnan(precision) and not np.isnan(recall) and precision + recall
    ) else np.nan
    return {
        "stress_b_n_samples": n,
        "stress_b_accepted": accepted,
        "stress_b_coverage": accepted / n if n else np.nan,
        "stress_b_precision": precision,
        "stress_b_recall": recall,
        "stress_b_f1": f1,
        "stress_b_tp": tp,
        "stress_b_tn": tn,
        "stress_b_fp": fp,
        "stress_b_fn": fn,
    }


def prepare_stress_c_eval_set(preprocessor, data):
    """
    Prepare Stress C noisy test set for Strategy 2.
    Uses Gaussian noise sigma=0.1 for strategy comparison.
    """
    rng = np.random.default_rng(SEED)
    X_noisy = data["X_test_raw"].copy()
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X_noisy.columns]
    for col in numeric_cols:
        col_std = X_noisy[col].std()
        noise = rng.normal(0, 0.1 * col_std, size=len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise
    X_noisy_t = preprocessor.transform(X_noisy)
    del X_noisy
    gc.collect()
    return X_noisy_t


def evaluate_strategy_stress_c(model, preprocessor, data, tau=None, models=None):
    """
    Evaluate strategy on Stress C (Gaussian noise sigma=0.1).
    Returns Macro-F1 on accepted (S1) or majority vote (S2).
    """
    rng = np.random.default_rng(SEED)
    X_noisy = data["X_test_raw"].copy()
    # Only add noise to NUMERIC_FEATURES (consistent with main Stress C in stress_tests.py)
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X_noisy.columns]
    for col in numeric_cols:
        if col in X_noisy.columns:
            noise = rng.normal(0, 0.1 * X_noisy[col].std(), size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise
    X_t = preprocessor.transform(X_noisy)

    if tau is not None:
        y_pred, rejected, _ = predict_with_rejection(model, X_t, tau)
        accepted = ~rejected
        return macro_f1_score(data["y_test"][accepted], y_pred[accepted]) if accepted.sum() > 0 else np.nan
    elif models is not None:
        disagreement, majority = ensemble_disagreement(models, X_t)
        return macro_f1_score(data["y_test"], majority)
    return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=int, choices=[1, 2], default=0,
                        help="Run specific strategy only (1 or 2); default: both")
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--lightweight-ensemble", action="store_true",
                        help="Use lightweight ensemble for memory-constrained Stress A runs")
    parser.add_argument("--stress-a-group", type=int, choices=[1, 2], default=None,
                        help="Run only one Stress A held-out group for Strategy 2")
    parser.add_argument("--max-s2-samples", type=int, default=100000,
                        help="Max training samples for Strategy 2 lightweight Stress A")
    parser.add_argument("--stress-b-only", action="store_true",
                        help="Only run Strategy 2 full Stress B (skip clean + Stress C)")
    parser.add_argument("--ensemble-train-frac", type=float, default=0.5,
                        help="Training fraction for Strategy 2 ensemble members")
    parser.add_argument("--ensemble-n-jobs", type=int, default=4,
                        help="Parallel jobs per ensemble member")
    parser.add_argument("--stress-b-chunk-size", type=int, default=500000,
                        help="Chunk size for full CICIDS Strategy 2 Stress B")
    parser.add_argument("--pred-batch-size", type=int, default=200000,
                        help="Prediction batch size for clean and Stress C evaluation")
    args = parser.parse_args()

    use_gpu = args.use_gpu and not args.no_gpu
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    run_s1 = args.strategy in (0, 1)
    run_s2 = args.strategy in (0, 2)

    print("=" * 60)
    print("Phase 5: Robustness Strategy Experiments")
    print("=" * 60)

    data = prepare_data()

    # ---- Load Phase 3 best model ----
    print("\nLoading Phase 3 best model from artifacts/best_model.joblib ...")
    model = joblib.load("artifacts/best_model.joblib")

    # Baseline predictions
    y_pred_base = model.predict(data["X_test"])
    y_prob_base = model.predict_proba(data["X_test"])
    baseline_f1 = macro_f1_score(data["y_test"], y_pred_base)
    print(f"Baseline Macro-F1 (clean): {baseline_f1:.4f}")

    # ================================================================
    # Strategy 1
    # ================================================================
    # Default values (filled if strategies run)
    sb_s1, sc_s1 = {"stress_b_f1": np.nan}, np.nan
    sb_s2, sc_s2 = {"stress_b_f1": np.nan}, np.nan
    best_tau, rejected_s1 = 0.85, np.zeros(len(data["y_test"]), dtype=bool)
    majority_f1, auroc_d = baseline_f1, 0.5
    ensemble_models = None

    if run_s1:
        print(f"\n{'='*50}")
        print("Strategy 1: Confidence Threshold Rejection")
        print(f"{'='*50}")

        # Sweep tau on validation
        y_val_prob = model.predict_proba(data["X_val"])
        best_tau, tau_results = select_tau_on_validation(
            data["y_val"], y_val_prob, min_coverage=0.85,
            classes=data["class_names"])

        # Save tau configuration
        strategy_config = {"strategy": "confidence_threshold", "tau": float(best_tau)}
        with open("artifacts/strategy_config.json", "w") as f:
            json.dump(strategy_config, f, indent=2)
        print("Saved: artifacts/strategy_config.json")

        # Plot coverage-accuracy curve
        plot_coverage_accuracy_curve(tau_results,
                                     "reports/figures/coverage_accuracy_curve.png")
        print("Saved: reports/figures/coverage_accuracy_curve.png")

        # Evaluate on clean test set
        y_pred_s1, rejected_s1, _ = predict_with_rejection(model, data["X_test"], best_tau)
        accepted = ~rejected_s1
        s1_clean_acc = accuracy_score(data["y_test"][accepted], y_pred_s1[accepted])
        print(f"\nStrategy 1 — Clean test (tau={best_tau}):")
        print(f"  Coverage: {accepted.mean():.4f}")
        print(f"  Accepted accuracy: {s1_clean_acc:.4f}")
        print(f"  Rejection rate: {rejected_s1.mean():.4f}")

        # Stress A evaluation (known-only training per P1-5)
        s1_stress = []
        for held_out in HELD_OUT_CLASSES_SETS:
            res = evaluate_strategy_stress_a(data, held_out, tau=best_tau)
            s1_stress.append(res)
            print(f"  Stress A [{res['held_out']}]: "
                  f"known_macro_f1={res['known_macro_f1']:.4f}, "
                  f"unknown_rej={res['unknown_rejection_rate']:.4f}")

        pd.DataFrame(s1_stress).to_csv("results/strategy1_stress_a.csv", index=False)

        # Stress B evaluation (single tau)
        sb_s1 = evaluate_strategy_stress_b(model, data["preprocessor"], tau=best_tau)
        print(f"  Stress B: F1={sb_s1.get('stress_b_f1', np.nan):.4f}, "
              f"coverage={sb_s1.get('stress_b_coverage', np.nan):.4f}")

        # Stress B tau sweep: show coverage-accuracy trade-off on cross-domain data
        print("\n  Stress B tau sweep ...")
        sb_rows = []
        for tau_sweep in [0.5, 0.7, 0.85, 0.9, 0.95, 0.99]:
            res = evaluate_strategy_stress_b(model, data["preprocessor"], tau=tau_sweep)
            sb_rows.append({
                "tau": tau_sweep,
                "coverage": round(res.get("stress_b_coverage", np.nan), 4),
                "rejection_rate": round(1 - res.get("stress_b_coverage", 0), 4),
                "accepted_precision": round(res.get("stress_b_precision", np.nan), 4),
                "accepted_recall": round(res.get("stress_b_recall", np.nan), 4),
                "accepted_f1": round(res.get("stress_b_f1", np.nan), 4),
            })
            print(f"    tau={tau_sweep}: coverage={sb_rows[-1]['coverage']:.4f}, "
                  f"prec={sb_rows[-1]['accepted_precision']:.4f}, "
                  f"rec={sb_rows[-1]['accepted_recall']:.4f}, "
                  f"f1={sb_rows[-1]['accepted_f1']:.4f}")
        pd.DataFrame(sb_rows).to_csv("results/stress_b_tau_sweep.csv", index=False)
        print("  Saved: results/stress_b_tau_sweep.csv")

        # Stress C evaluation
        sc_s1 = evaluate_strategy_stress_c(model, data["preprocessor"], data, tau=best_tau)
        print(f"  Stress C (noise=0.1): accepted Macro-F1={sc_s1:.4f}" if not np.isnan(sc_s1) else "  Stress C: skipped")

        # McNemar test: Strategy 1 (on accepted) vs Baseline (on same samples)
        accepted_mask = ~rejected_s1
        if accepted_mask.sum() > 0:
            chi2_s1, pval_s1 = mcnemar_test(
                data["y_test"].values[accepted_mask],
                y_pred_base[accepted_mask],
                y_pred_s1[accepted_mask])
            print(f"\n  McNemar test (baseline vs Strategy 1 on accepted): "
                  f"chi2={chi2_s1:.2f}, p={pval_s1:.4f}")
        else:
            chi2_s1, pval_s1 = np.nan, np.nan

    # ================================================================
    # Strategy 2
    # ================================================================
    if run_s2:
        print(f"\n{'='*50}")
        print("Strategy 2: Streaming Ensemble Disagreement Detection")
        print(f"{'='*50}")

        # ---- Fast path: full Stress B only (single-pass ensemble) ----
        if args.strategy == 2 and args.stress_b_only:
            print("\nRunning Strategy 2 full Stress B only (single-pass ensemble)")

            stress_b_metrics = full_stress_b_single_pass_ensemble(
                X_train=data["X_train"],
                y_train=data["y_train"],
                preprocessor=data["preprocessor"],
                use_gpu=use_gpu,
                train_frac=args.ensemble_train_frac,
                n_jobs=args.ensemble_n_jobs,
                chunk_size=args.stress_b_chunk_size,
            )

            s2_sb_df = pd.DataFrame([{
                "strategy": "ensemble_disagreement_single_pass",
                "ensemble_size": 5,
                "stress_b_n_samples": stress_b_metrics["n_samples"],
                "stress_b_accuracy": stress_b_metrics["accuracy"],
                "stress_b_precision": stress_b_metrics["precision"],
                "stress_b_recall": stress_b_metrics["recall"],
                "stress_b_f1": stress_b_metrics["f1"],
                "stress_b_fpr": stress_b_metrics["fpr"],
                "stress_b_fnr": stress_b_metrics["fnr"],
                "stress_b_tp": stress_b_metrics["tp"],
                "stress_b_tn": stress_b_metrics["tn"],
                "stress_b_fp": stress_b_metrics["fp"],
                "stress_b_fn": stress_b_metrics["fn"],
                "full_stress_b": True,
                "ensemble_train_frac": args.ensemble_train_frac,
                "ensemble_n_jobs": args.ensemble_n_jobs,
                "chunk_size": args.stress_b_chunk_size,
            }])
            s2_sb_df.to_csv("results/strategy2_stress_b_full_results.csv", index=False)
            print("\nSaved: results/strategy2_stress_b_full_results.csv")
            return

        # ---- Streaming ensemble path ----
        X_stress_c_t = prepare_stress_c_eval_set(data["preprocessor"], data)

        eval_sets = {
            "clean": data["X_test"],
            "stress_c": X_stress_c_t,
        }

        pred_paths = train_predict_clean_eval_sets(
            X_train=data["X_train"],
            y_train=data["y_train"],
            eval_sets=eval_sets,
            out_dir="artifacts/ensemble_predictions/strategy2_clean_stressc",
            use_gpu=use_gpu,
            train_frac=args.ensemble_train_frac,
            n_jobs=args.ensemble_n_jobs,
            batch_size=args.pred_batch_size,
        )

        majority_clean, disagreement_clean = \
            majority_and_disagreement_from_prediction_files(pred_paths["clean"])

        is_wrong = majority_clean != data["y_test"].values

        auroc_d = disagreement_auroc(disagreement_clean, is_wrong)
        majority_acc = accuracy_score(data["y_test"], majority_clean)
        majority_f1 = macro_f1_score(data["y_test"], majority_clean)

        print(f"\nStrategy 2 Clean:")
        print(f"  Majority vote accuracy:    {majority_acc:.4f}")
        print(f"  Majority vote Macro-F1:    {majority_f1:.4f}")
        print(f"  Disagreement AUROC:        {auroc_d:.4f}")

        print(f"\n  Disagreement bucket analysis:")
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(bins) - 1):
            mask = (disagreement_clean >= bins[i]) & (disagreement_clean < bins[i + 1])
            if mask.sum() > 0:
                acc = accuracy_score(data["y_test"].values[mask], majority_clean[mask])
                print(f"    [{bins[i]}, {bins[i+1]}): "
                      f"count={mask.sum():,}, accuracy={acc:.4f}")

        plot_disagreement_histogram(disagreement_clean, is_wrong,
                                    "reports/figures/disagreement_histogram.png")
        print("Saved: reports/figures/disagreement_histogram.png")
        plot_disagreement_roc_curve(disagreement_clean, is_wrong, auroc_d,
                                    "reports/figures/disagreement_auroc.png")
        print("Saved: reports/figures/disagreement_auroc.png")

        majority = majority_clean
        disagreement = disagreement_clean

        majority_c, disagreement_c = \
            majority_and_disagreement_from_prediction_files(pred_paths["stress_c"])
        sc_s2 = macro_f1_score(data["y_test"], majority_c)
        print(f"\nStrategy 2 Stress C:")
        print(f"  Macro-F1: {sc_s2:.4f}")

        stress_b_metrics = full_stress_b_single_pass_ensemble(
            X_train=data["X_train"],
            y_train=data["y_train"],
            preprocessor=data["preprocessor"],
            use_gpu=use_gpu,
            train_frac=args.ensemble_train_frac,
            n_jobs=args.ensemble_n_jobs,
            chunk_size=args.stress_b_chunk_size,
        )
        sb_s2 = {"stress_b_f1": stress_b_metrics["f1"]}
        stress_b_f1 = stress_b_metrics["f1"]

        # Stress A evaluation (shared between both paths)
        s2_stress = []

        if args.stress_a_group is not None:
            held_out_sets = [HELD_OUT_CLASSES_SETS[args.stress_a_group - 1]]
            out_path = f"results/strategy2_stress_a_group{args.stress_a_group}.csv"
        else:
            held_out_sets = HELD_OUT_CLASSES_SETS
            out_path = "results/strategy2_stress_a.csv"

        for held_out in held_out_sets:
            print(f"  Running Strategy 2 Stress A for held-out: {held_out}", flush=True)
            res = evaluate_strategy_stress_a(
                data, held_out, tau=None, use_gpu=use_gpu,
                lightweight=args.lightweight_ensemble,
                max_s2_samples=args.max_s2_samples,
            )
            s2_stress.append(res)
            print(f"  Stress A [{res['held_out']}]: "
                  f"known_macro_f1={res['known_macro_f1']:.4f}, "
                  f"disagreement_auroc={res['disagreement_auroc']:.4f}",
                  flush=True)
            pd.DataFrame(s2_stress).to_csv(out_path, index=False)
            print(f"  Saved partial results: {out_path}", flush=True)

        pd.DataFrame(s2_stress).to_csv(out_path, index=False)
        print(f"Saved: {out_path}", flush=True)

        # McNemar's test: ensemble majority vote vs baseline
        chi2, pval = mcnemar_test(data["y_test"].values, y_pred_base, majority)
        print(f"\n  McNemar test (baseline vs ensemble): chi2={chi2:.2f}, p={pval:.4f}")

    # ================================================================
    # Summary: Build complete strategies_comparison.csv
    # ================================================================

    # Collect stress metrics from disk (never from this run's variables)
    baseline_macro_f1 = baseline_f1

    def _read_csv_val(path, col, row=0, default=np.nan):
        if os.path.exists(path):
            df = pd.read_csv(path)
            if col in df.columns and len(df) > row:
                return df[col].iloc[row]
        return default

    sa_path = "results/stress_a_results.csv"
    sa_f1 = np.nan
    if os.path.exists(sa_path):
        sa = pd.read_csv(sa_path)
        if "known_macro_f1" in sa.columns:
            sa_f1 = sa["known_macro_f1"].mean()

    sb_f1 = _read_csv_val("results/stress_b_full_results.csv", "f1")

    sc_f1 = np.nan
    sc_path = "results/stress_c_results.csv"
    if os.path.exists(sc_path):
        sc = pd.read_csv(sc_path)
        noise_row = sc[(sc["type"] == "noise") & (sc["level"] == "sigma=0.1")]
        if len(noise_row) > 0:
            sc_f1 = noise_row["macro_f1"].iloc[0]

    print(f"\n{'='*50}")
    print("Building final strategies_comparison.csv")
    print(f"{'='*50}")

    # Save individual strategy summaries (never overwritten by single-strategy runs)
    # NOTE: Strategy 1 "AUROC" = confidence-as-unknown-detector AUROC (from Stress A results).
    #       Strategy 2 "AUROC" = disagreement-as-wrong-prediction AUROC.
    if run_s1:
        s1_sa = pd.read_csv("results/strategy1_stress_a.csv") if os.path.exists("results/strategy1_stress_a.csv") else None
        s1_unk_rej = s1_sa["unknown_rejection_rate"].mean() if s1_sa is not None and "unknown_rejection_rate" in s1_sa.columns else np.nan
        s1_sa_f1 = s1_sa["known_macro_f1"].mean() if s1_sa is not None and "known_macro_f1" in s1_sa.columns else np.nan
        # Use actual confidence AUROC from Stress A results, not rejection rate
        sa_results = pd.read_csv("results/stress_a_results.csv") if os.path.exists("results/stress_a_results.csv") else None
        s1_auroc = sa_results["auroc_confidence_unknown"].mean() if sa_results is not None and "auroc_confidence_unknown" in sa_results.columns else np.nan
        # S1 Clean Full F1: Macro-F1 treating REJECT as an error class
        s1_full_pred = y_pred_s1.copy()
        s1_full_pred[rejected_s1] = "__REJECT__"
        s1_full_f1 = macro_f1_score(data["y_test"], s1_full_pred)
        # S1 Accepted F1: Macro-F1 on accepted samples only
        s1_accepted_f1 = macro_f1_score(data["y_test"][~rejected_s1], y_pred_s1[~rejected_s1]) if (~rejected_s1).sum() > 0 else np.nan

        s1_df = pd.DataFrame([{
            "Method": f"Strategy 1 (tau={best_tau})",
            "Clean Full F1": round(s1_full_f1, 4),
            "Coverage": round((~rejected_s1).mean(), 4),
            "Accepted F1": round(s1_accepted_f1, 4),
            "Stress A F1": round(s1_sa_f1, 4) if not np.isnan(s1_sa_f1) else np.nan,
            "Stress B F1": "0% cov / undef." if np.isnan(sb_s1.get("stress_b_f1", np.nan)) else round(sb_s1["stress_b_f1"], 4),
            "Stress C F1": "0% cov / undef." if np.isnan(sc_s1) else round(sc_s1, 4),
            "Rejection Rate": round(rejected_s1.mean(), 4),
            "Detection AUROC": round(s1_auroc, 4) if not np.isnan(s1_auroc) else np.nan,
        }])
        s1_df.to_csv("results/strategy1_summary.csv", index=False)
        print("Saved: results/strategy1_summary.csv")

    if run_s2:
        s2_sa = pd.read_csv("results/strategy2_stress_a.csv") if os.path.exists("results/strategy2_stress_a.csv") else None
        s2_auroc = s2_sa["disagreement_auroc"].mean() if s2_sa is not None and "disagreement_auroc" in s2_sa.columns else np.nan
        s2_sa_f1 = s2_sa["known_macro_f1"].mean() if s2_sa is not None and "known_macro_f1" in s2_sa.columns else np.nan
        s2_df = pd.DataFrame([{
            "Method": "Strategy 2 (M=5 ensemble)",
            "Clean Full F1": round(majority_f1, 4),
            "Coverage": "1.000 (no rejection)",
            "Accepted F1": "N/A",
            "Stress A F1": round(s2_sa_f1, 4) if not np.isnan(s2_sa_f1) else np.nan,
            "Stress B F1": round(sb_s2.get("stress_b_f1", np.nan), 4),
            "Stress C F1": round(sc_s2, 4) if not np.isnan(sc_s2) else np.nan,
            "Rejection Rate": "N/A",
            "Detection AUROC": round(auroc_d, 4),
            "full_stress_b": True,
            "ensemble_train_frac": args.ensemble_train_frac if run_s2 else np.nan,
            "ensemble_n_jobs": args.ensemble_n_jobs if run_s2 else np.nan,
        }])
        s2_df.to_csv("results/strategy2_summary.csv", index=False)
        print("Saved: results/strategy2_summary.csv")

    # Build final strategies_comparison.csv by merging all available summaries
    print(f"\n{'='*50}")
    print("Building final strategies_comparison.csv")
    print(f"{'='*50}")

    all_rows = [{
        "Method": "Baseline (no strategy)",
        "Clean Full F1": round(baseline_macro_f1, 4),
        "Coverage": "1.000 (no rejection)",
        "Accepted F1": "N/A",
        "Stress A F1": round(sa_f1, 4) if not np.isnan(sa_f1) else np.nan,
        "Stress B F1": round(sb_f1, 4) if not np.isnan(sb_f1) else np.nan,
        "Stress C F1": round(sc_f1, 4) if not np.isnan(sc_f1) else np.nan,
        "Rejection Rate": "N/A",
        "Detection AUROC": "N/A",
    }]

    # Merge from intermediate summary files
    for summary_file in ["results/strategy1_summary.csv", "results/strategy2_summary.csv"]:
        if os.path.exists(summary_file):
            df_s = pd.read_csv(summary_file, keep_default_na=False, na_values=[])
            all_rows.extend(df_s.to_dict("records"))

    summary = pd.DataFrame(all_rows)
    summary.to_csv("results/strategies_comparison.csv", index=False)
    summary.to_csv("results/strategy_comparison.csv", index=False)
    print(summary.to_string(index=False))
    print("\nSaved: results/strategies_comparison.csv")

    # Save McNemar results
    mcnemar_rows = []
    if run_s1 and accepted_mask.sum() > 0:
        mcnemar_rows.append({
            "comparison": "Strategy 1 vs Baseline",
            "condition": "Clean (accepted)",
            "statistic": round(chi2_s1, 2) if not np.isnan(chi2_s1) else np.nan,
            "p_value": round(pval_s1, 4) if not np.isnan(pval_s1) else np.nan,
            "significant": "yes" if not np.isnan(pval_s1) and pval_s1 < 0.05 else "no",
            "alpha": 0.05,
        })
    if run_s2:
        mcnemar_rows.append({
            "comparison": "Strategy 2 vs Baseline",
            "condition": "Clean",
            "statistic": round(chi2, 2),
            "p_value": round(pval, 4),
            "significant": "yes" if pval < 0.05 else "no",
            "alpha": 0.05,
        })
    if mcnemar_rows:
        # Save per-strategy intermediate files (never overwritten by single runs)
        run_label = "s1" if run_s1 else "s2"
        pd.DataFrame(mcnemar_rows).to_csv(f"results/mcnemar_{run_label}.csv", index=False)

    # Merge all McNemar intermediate files into final mcnemar_results.csv
    all_mcnemar = []
    for f in ["results/mcnemar_s1.csv", "results/mcnemar_s2.csv"]:
        if os.path.exists(f):
            all_mcnemar.append(pd.read_csv(f))
    if all_mcnemar:
        pd.concat(all_mcnemar, ignore_index=True).to_csv(
            "results/mcnemar_results.csv", index=False)
        print("Saved: results/mcnemar_results.csv")

    print(f"\n{'=' * 60}")
    print("Phase 5 complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
