"""
Streaming heterogeneous ensemble for Strategy 2.

Trains one ensemble member at a time, saves predictions to disk, and releases
the model from memory. This keeps the full M=5 ensemble design while avoiding
OOM from holding all models simultaneously.

Ensemble members:
  3 x RandomForest (seeds 42, 123, 456)
  1 x XGBoost
  1 x LogisticRegression

Provides:
  - Streaming evaluation on clean test set and Stress C
  - Chunked full-CICIDS Stress B evaluation
"""

import os
import gc
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from config import SEED
from preprocessing.preprocess import prepare_cicids_chunk


def build_member_specs(use_gpu=True, n_jobs=-1):
    """Return list of (name, factory_fn) tuples for all ensemble members."""
    specs = [
        ("RF_42", lambda: RandomForestClassifier(
            n_estimators=100, max_depth=16, min_samples_leaf=3,
            min_samples_split=8, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=n_jobs,
        )),
        ("RF_123", lambda: RandomForestClassifier(
            n_estimators=100, max_depth=16, min_samples_leaf=3,
            min_samples_split=8, max_features="sqrt",
            class_weight="balanced", random_state=123, n_jobs=n_jobs,
        )),
        ("RF_456", lambda: RandomForestClassifier(
            n_estimators=100, max_depth=16, min_samples_leaf=3,
            min_samples_split=8, max_features="sqrt",
            class_weight="balanced", random_state=456, n_jobs=n_jobs,
        )),
        ("XGB", lambda: _make_xgb(use_gpu=use_gpu)),
        ("LR", lambda: LogisticRegression(
            C=10, max_iter=1000, penalty="l2", solver="saga",
            class_weight="balanced", random_state=SEED, n_jobs=n_jobs,
        )),
    ]
    return specs


def _make_xgb(use_gpu=True):
    from xgboost import XGBClassifier
    from models.chosen_model import get_xgb_device
    from experiments.run_baseline import XGBWrapper
    device_params = get_xgb_device(use_gpu)
    xgb = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss", random_state=SEED,
        **device_params,
    )
    le = LabelEncoder()
    return xgb, le, XGBWrapper


def fit_one_member(name, X_train, y_train, train_frac=0.5, n_jobs=None,
                   use_gpu=True):
    """
    Train a single ensemble member on a stratified subset.

    Args:
        name: Member identifier matching build_member_specs keys.
        X_train: Full training features (numpy array from preprocessor.transform).
        y_train: Full training labels (pandas Series of strings).
        train_frac: Fraction of training data to use.
        n_jobs: Override n_jobs for this member.
        use_gpu: Whether XGBoost should attempt GPU.

    Returns:
        (name, fitted_model)
    """
    specs = dict(build_member_specs(use_gpu=use_gpu, n_jobs=n_jobs if n_jobs else -1))

    if name not in specs:
        raise ValueError(f"Unknown member: {name}")

    # Stratified subset
    if train_frac < 1.0:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=1 - train_frac, random_state=SEED)
        _, idx_subset = next(splitter.split(X_train, y_train))
        X_sub = X_train[idx_subset]
        y_sub = y_train.iloc[idx_subset] if hasattr(y_train, "iloc") else y_train[idx_subset]
    else:
        X_sub = X_train
        y_sub = y_train

    print(f"    Training {name} on {len(X_sub):,} samples", flush=True)

    if name == "XGB":
        # XGBoost requires integer labels — factory returns (model, encoder, wrapper_cls)
        xgb, le, wrapper_cls = specs[name]()
        y_enc = le.fit_transform(y_sub)
        xgb.fit(X_sub, y_enc)
        model = wrapper_cls(xgb, le)
    else:
        model = specs[name]()
        if n_jobs is not None and hasattr(model, "n_jobs"):
            model.n_jobs = n_jobs
        model.fit(X_sub, y_sub)

    print(f"    {name}: trained", flush=True)
    return name, model


def predict_in_batches(model, X, batch_size=200000, desc="predict"):
    """
    Predict in batches to control memory. Returns concatenated predictions.
    """
    n = len(X)
    preds = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        preds.append(model.predict(X[start:end]))
        if (start // batch_size) % 10 == 0:
            print(f"    {desc}: {end:,} / {n:,}", flush=True)

    return np.concatenate(preds)


def majority_and_disagreement_from_matrix(pred_matrix):
    """
    Compute majority vote and disagreement from (M, N) prediction matrix.

    Args:
        pred_matrix: numpy array of shape (M, N) with string predictions.

    Returns:
        (majority_votes, disagreement_scores)
    """
    # Vectorized majority vote via pandas mode (handles ties by picking first)
    majority = pd.DataFrame(pred_matrix).mode(axis=0).iloc[0].values.astype(object)

    agreement = (pred_matrix == majority).mean(axis=0)
    disagreement = 1.0 - agreement

    return majority, disagreement


def majority_and_disagreement_from_prediction_files(pred_dir):
    """
    Load per-member prediction files from disk and compute majority/disagreement.

    Each file in pred_dir should be a .npy file named <name>_preds.npy.

    Returns:
        (majority_votes, disagreement_scores)
    """
    all_preds = []
    for fname in sorted(os.listdir(pred_dir)):
        if fname.endswith("_preds.npy"):
            all_preds.append(np.load(os.path.join(pred_dir, fname)))

    if not all_preds:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")

    pred_matrix = np.array(all_preds, dtype=object)
    return majority_and_disagreement_from_matrix(pred_matrix)


def train_predict_clean_eval_sets(X_train, y_train, eval_sets, out_dir,
                                   use_gpu=True, train_frac=0.5,
                                   n_jobs=2, batch_size=200000):
    """
    Train each ensemble member, predict on eval_sets, save predictions to disk.
    Releases each model from memory after use.

    Args:
        X_train, y_train: training data
        eval_sets: dict of {name: X_matrix} to predict on
        out_dir: directory to save predictions
        use_gpu: whether XGBoost should attempt GPU
        train_frac: training subset fraction
        n_jobs: n_jobs for tree models
        batch_size: prediction batch size

    Returns:
        dict of {eval_name: pred_dir}
    """
    os.makedirs(out_dir, exist_ok=True)
    member_names = [name for name, _ in build_member_specs(use_gpu=use_gpu)]

    # One subdirectory per eval set
    eval_dirs = {}
    for eval_name in eval_sets:
        edir = os.path.join(out_dir, eval_name)
        os.makedirs(edir, exist_ok=True)
        eval_dirs[eval_name] = edir

    for member_name in member_names:
        print(f"\n  --- Ensemble member: {member_name} ---", flush=True)
        name, model = fit_one_member(
            member_name, X_train, y_train,
            train_frac=train_frac, n_jobs=n_jobs, use_gpu=use_gpu,
        )

        for eval_name, X_eval in eval_sets.items():
            print(f"    Predicting on {eval_name} ({len(X_eval):,} samples)", flush=True)
            preds = predict_in_batches(model, X_eval, batch_size=batch_size,
                                       desc=f"{member_name}/{eval_name}")
            np.save(
                os.path.join(eval_dirs[eval_name], f"{member_name}_preds.npy"),
                preds,
            )
            print(f"    Saved predictions for {member_name}/{eval_name}", flush=True)

        del model, preds
        gc.collect()

    return eval_dirs


def full_stress_b_streaming_ensemble(X_train, y_train, preprocessor,
                                      out_dir, use_gpu=True,
                                      train_frac=0.5, n_jobs=2,
                                      chunk_size=100000):
    """
    Full CICIDS Stress B evaluation using streaming ensemble.

    For each ensemble member:
      1. Train on UNSW subset
      2. Read CICIDS in chunks, transform, predict, save chunk predictions
      3. Delete model

    After all members are done, compute majority vote chunk by chunk.

    Args:
        X_train, y_train: UNSW training data
        preprocessor: UNSW-fitted preprocessor (transform only, never fit on CICIDS)
        out_dir: directory for prediction chunks
        use_gpu: passed through
        train_frac: training subset fraction
        n_jobs: n_jobs for tree models
        chunk_size: CICIDS rows per chunk

    Returns:
        dict with binary metrics over full CICIDS dataset
    """
    from config import CICIDS_PATH, ATTACK_COL

    os.makedirs(out_dir, exist_ok=True)
    member_names = [name for name, _ in build_member_specs(use_gpu=use_gpu)]
    M = len(member_names)

    # ---- Phase 1: per-member training + chunked prediction ----
    n_chunks = None

    for member_idx, member_name in enumerate(member_names):
        print(f"\n{'='*50}")
        print(f"Ensemble member {member_idx+1}/{M}: {member_name}")
        print(f"{'='*50}", flush=True)

        name, model = fit_one_member(
            member_name, X_train, y_train,
            train_frac=train_frac, n_jobs=n_jobs, use_gpu=use_gpu,
        )

        member_dir = os.path.join(out_dir, member_name)
        os.makedirs(member_dir, exist_ok=True)

        print(f"  Streaming CICIDS chunk predictions ...", flush=True)
        reader = pd.read_csv(CICIDS_PATH, chunksize=chunk_size, low_memory=False)

        for chunk_id, chunk in enumerate(reader, start=1):
            chunk = prepare_cicids_chunk(chunk)

            # Save ground truth only once (first member)
            if member_idx == 0:
                y_true_bin = (chunk[ATTACK_COL] != "Benign").astype(int).values
                y_true_path = os.path.join(out_dir, f"y_true_chunk_{chunk_id:04d}.npy")
                np.save(y_true_path, y_true_bin)

            # Feature extraction
            drop_target_cols = [ATTACK_COL]
            if "Label" in chunk.columns:
                drop_target_cols.append("Label")
            X_raw = chunk.drop(columns=drop_target_cols)

            # Align with preprocessor expected columns
            if hasattr(preprocessor, "feature_names_in_"):
                expected_cols = list(preprocessor.feature_names_in_)
            else:
                expected_cols = [c for c in X_raw.columns
                                 if c not in drop_target_cols]
            X_raw = X_raw.reindex(columns=expected_cols, fill_value=0)

            X_t = preprocessor.transform(X_raw)
            preds = model.predict(X_t)

            chunk_pred_path = os.path.join(
                member_dir, f"preds_chunk_{chunk_id:04d}.npy")
            np.save(chunk_pred_path, preds)

            del chunk, X_raw, X_t, preds
            gc.collect()

            if chunk_id % 10 == 0:
                print(f"    {member_name}: chunk {chunk_id} done", flush=True)

        if n_chunks is None:
            n_chunks = chunk_id

        del model
        gc.collect()
        print(f"  {member_name}: all {n_chunks} chunks done", flush=True)

    if n_chunks is None:
        print("  WARNING: CICIDS file is empty, returning zero metrics.")
        metrics = _compute_binary_metrics(
            {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0})
        return metrics

    # ---- Phase 2: compute majority vote + metrics chunk by chunk ----
    print(f"\n{'='*50}")
    print("Computing majority vote from saved chunk predictions")
    print(f"{'='*50}", flush=True)

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}

    for chunk_id in range(1, n_chunks + 1):
        # Load predictions from all members for this chunk
        chunk_preds = []
        for member_name in member_names:
            pred_path = os.path.join(out_dir, member_name,
                                     f"preds_chunk_{chunk_id:04d}.npy")
            chunk_preds.append(np.load(pred_path, allow_pickle=True))

        pred_matrix = np.array(chunk_preds, dtype=object)
        majority, _ = majority_and_disagreement_from_matrix(pred_matrix)

        y_true_bin = np.load(os.path.join(out_dir, f"y_true_chunk_{chunk_id:04d}.npy"))
        y_pred_bin = (majority != "Benign").astype(int)

        counts["tp"] += int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
        counts["tn"] += int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
        counts["fp"] += int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
        counts["fn"] += int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
        counts["n"] += int(len(y_true_bin))

        if chunk_id % 10 == 0:
            partial = _compute_binary_metrics(counts)
            print(f"    Majority chunk {chunk_id}/{n_chunks}: "
                  f"n={partial['n_samples']:,}, F1={partial['f1']:.4f}", flush=True)

    metrics = _compute_binary_metrics(counts)

    print(f"\n  Full Stress B Streaming Ensemble:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    pd.DataFrame([metrics]).to_csv(
        os.path.join(out_dir, "stress_b_ensemble_metrics.csv"), index=False)
    print(f"\n  Saved: {os.path.join(out_dir, 'stress_b_ensemble_metrics.csv')}")

    return metrics


# ---------------------------------------------------------------------------
# Helper functions for single-pass ensemble
# ---------------------------------------------------------------------------

def make_ensemble_training_subset(X_train, y_train, train_frac=0.5, seed=SEED):
    """Stratified subset of UNSW training data for ensemble member training."""
    if train_frac >= 1.0:
        return X_train, y_train
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - train_frac, random_state=seed)
    _, idx_subset = next(splitter.split(X_train, y_train))
    X_sub = X_train[idx_subset]
    y_sub = y_train.iloc[idx_subset] if hasattr(y_train, "iloc") else y_train[idx_subset]
    print(f"  Ensemble training subset: {len(X_sub):,} samples ({train_frac:.0%})")
    return X_sub, y_sub


def get_expected_columns(preprocessor, chunk):
    """Recover expected training feature columns from the preprocessor."""
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    return [c for c in chunk.columns if c not in ("Attack", "Label")]


def update_binary_counts(y_true_bin, y_pred_bin, counts):
    """Accumulate TP/TN/FP/FN counts from a chunk."""
    counts["tp"] += int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    counts["tn"] += int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    counts["fp"] += int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    counts["fn"] += int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    counts["n"] += int(len(y_true_bin))


# ---------------------------------------------------------------------------
# Single-pass full CICIDS ensemble (64GB RAM optimal path)
# ---------------------------------------------------------------------------

def full_stress_b_single_pass_ensemble(
    X_train,
    y_train,
    preprocessor,
    use_gpu=True,
    train_frac=0.5,
    n_jobs=4,
    chunk_size=500_000,
):
    """
    Full CICIDS Strategy 2 evaluation — single-pass ensemble.

    Trains all five ensemble members once, keeps them in memory, and reads
    the full CICIDS target dataset only once. Each chunk is predicted by all
    models, majority vote is computed immediately, and binary detection counts
    are accumulated. No intermediate prediction files are saved.

    Designed for machines with sufficient RAM (e.g. 64 GB workstation).
    """
    from config import CICIDS_PATH, ATTACK_COL

    print("\n" + "=" * 60)
    print("Strategy 2: Single-Pass Full CICIDS Ensemble")
    print("=" * 60)
    print(f"  train_frac={train_frac}")
    print(f"  n_jobs={n_jobs}")
    print(f"  chunk_size={chunk_size:,}")

    # 1. Training subset
    X_sub, y_sub = make_ensemble_training_subset(
        X_train, y_train, train_frac=train_frac)

    # 2. Train all members, keep in memory
    specs = build_member_specs(use_gpu=use_gpu, n_jobs=n_jobs)
    models = []

    for name, factory in specs:
        print(f"\n  Training {name} ...", flush=True)

        fitted_name, fitted_model = fit_one_member(
            name,
            X_sub,
            y_sub,
            train_frac=1.0,  # subset already applied above
            n_jobs=n_jobs,
            use_gpu=use_gpu,
        )

        models.append((fitted_name, fitted_model))

    del X_sub, y_sub
    gc.collect()

    # 3. Single pass over full CICIDS
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
    expected_cols = None

    print(f"\n  Streaming full CICIDS (chunk_size={chunk_size:,}) ...", flush=True)
    reader = pd.read_csv(CICIDS_PATH, chunksize=chunk_size, low_memory=False)

    for chunk_id, chunk in enumerate(reader, start=1):
        chunk = prepare_cicids_chunk(chunk)

        if expected_cols is None:
            expected_cols = get_expected_columns(preprocessor, chunk)
            print(f"  Expected feature columns: {len(expected_cols)}")

        y_true_bin = (chunk[ATTACK_COL] != "Benign").astype(int).values

        drop_target_cols = [ATTACK_COL]
        if "Label" in chunk.columns:
            drop_target_cols.append("Label")

        X_raw = chunk.drop(columns=drop_target_cols)
        X_raw = X_raw.reindex(columns=expected_cols, fill_value=0)
        X_t = preprocessor.transform(X_raw)

        # Predict with all ensemble members and convert to binary
        member_bin_preds = []

        for name, m in models:
            print(f"  Predicting with {name}...", flush=True)
            pred = m.predict(X_t)
            pred_bin = (pred != "Benign").astype(np.int8)
            member_bin_preds.append(pred_bin)

        vote_matrix = np.vstack(member_bin_preds)

        # Binary majority vote for Stress B
        attack_votes = vote_matrix.sum(axis=0)
        y_pred_bin = (attack_votes >= 3).astype(np.int8)

        update_binary_counts(y_true_bin, y_pred_bin, counts)

        partial = _compute_binary_metrics(counts)
        print(
            f"  Chunk {chunk_id}: n={partial['n_samples']:,} | "
            f"F1={partial['f1']:.4f} | "
            f"P={partial['precision']:.4f} | "
            f"R={partial['recall']:.4f} | "
            f"FPR={partial['fpr']:.4f} | "
            f"FNR={partial['fnr']:.4f}",
            flush=True,
        )

        del chunk
        del X_raw
        del X_t
        del y_true_bin
        del member_bin_preds
        del vote_matrix
        del attack_votes
        del y_pred_bin
        gc.collect()

    # 4. Final metrics
    metrics = _compute_binary_metrics(counts)

    print(f"\n  {'='*50}")
    print(f"  Strategy 2 Full Stress B Final Results:")
    print(f"  {'='*50}")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    del models
    gc.collect()

    return metrics


def _compute_binary_metrics(counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    n = counts["n"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    return {
        "n_samples": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
