# Robust Multi-Class Network Intrusion Detection

Multi-class network intrusion detection on the NF-UQ-NIDS-v3 dataset with robustness evaluation.  
Training set: **NF-UNSW-NB15-v3** (2.37M flows, 10 classes).  
Target domain: **NF-CSE-CIC-IDS2018-v3** (20.1M flows).

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Place dataset CSV files under `data/`:

- `data/NF-UNSW-NB15-v3.csv`
- `data/NF-CSE-CIC-IDS2018-v3.csv`

Download: [https://staff.itee.uq.edu.au/marius/NIDS_datasets/](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)

## Reproduce All Experiments

Run scripts in order:

```bash
# Phase 2a: EDA
python experiments/run_eda.py

# Phase 2b: Data validation + preprocessing
python experiments/validate_data.py
python experiments/prepare_data.py

# Phase 3: Baseline classifiers
python experiments/run_baseline.py --no-gpu

# Phase 3 (supplementary): Seed sensitivity check (reduced RF, 5 seeds)
python experiments/run_significance.py

# Phase 4: Stress tests (A / B / C)
python experiments/run_stress.py --stress all

# Phase 4 (supplementary): Full Stress B baseline on complete CICIDS (20.1M flows)
python experiments/run_stress_b_full.py

# Phase 5: Robustness strategies (see Reproducibility Notes below)
python experiments/run_strategies.py

# Phase 5 (post-processing): Aggregate strategy results into final comparison tables
python experiments/aggregate_strategy_results.py

# Phase 6: Ablation + failure analysis
python experiments/run_ablation.py
python experiments/run_failure_analysis.py
```

With GPU (XGBoost only):

```bash
python experiments/run_baseline.py --use-gpu
python experiments/run_strategies.py --use-gpu
```

## Reproducibility Notes

**Fully reproducible (single-command end-to-end):**
- `run_eda.py`, `validate_data.py`, `prepare_data.py`
- `run_baseline.py`
- `run_stress.py` (all stress tests)
- `run_ablation.py` (all ablation studies)
- `run_significance.py` (seed sensitivity check)
- `run_failure_analysis.py`
- `predict.py`

**Segmented / memory-constrained execution:**
- `run_strategies.py --strategy 1` — runs independently, no memory issues
- `run_strategies.py --strategy 2 --lightweight-ensemble --stress-a-group 1 --max-s2-samples 100000` — lightweight Stress A group 1 only
- `run_strategies.py --strategy 2 --lightweight-ensemble --stress-a-group 2 --max-s2-samples 100000` — lightweight Stress A group 2 only

**Important**: `--strategy 2` still trains a full clean ensemble (3xRF+XGB+LR)
before reaching lightweight Stress A, which requires >8 GB RAM even with the
lightweight flag. The lightweight flag only applies to the Stress A portion.
On memory-constrained machines (<16 GB), Strategy 2 clean ensemble training may
OOM before reaching the Stress A lightweight stage.

Full Strategy 2 reproduction requires sufficient memory. The reported results
are reproducible with the commands above. Lightweight diagnostic commands are
provided for memory-constrained machines; the final reported Strategy 2 table
uses the full ensemble outputs saved in `results/`.

## Full Stress B Evaluation

Stress Test B is evaluated on the full NF-CSE-CIC-IDS2018-v3 target domain
(20,115,529 flows). The dataset is processed in chunks to avoid loading the
entire dataset into memory at once. Each chunk is cleaned, transformed using the
UNSW-fitted preprocessor (transform only — never fit on CICIDS), and predicted.
Binary TP/TN/FP/FN counts are accumulated across all chunks to compute full-target
metrics.

Command:

```bash
python experiments/run_stress_b_full.py
```

Output:

```
results/stress_b_full_results.csv
```

`run_stress.py --stress B` also uses this full chunked path. The earlier 10%
sampled Stress B path has been removed to avoid mixing sampled and full-target
results.

## Strategy 2 Full Stress B Evaluation

Strategy 2 uses a five-member heterogeneous ensemble (3x RandomForest + 1x XGBoost
+ 1x LogisticRegression). On a 64GB workstation, all five ensemble members are kept
in memory after training, while the full CICIDS target domain is streamed once in
chunks. For each chunk, all five models predict, binary majority vote is computed
(>= 3 votes for attack), and TP/TN/FP/FN counts are accumulated. No intermediate
prediction files are saved.

Command:

```bash
python experiments/run_strategies.py \
  --strategy 2 \
  --stress-b-only \
  --ensemble-train-frac 0.5 \
  --ensemble-n-jobs 4 \
  --stress-b-chunk-size 1000000
```

Output:

```
results/strategy2_stress_b_full_results.csv
```

## CLI Prototype

```bash
# Direct prediction
python predict.py --input data/sample_test.csv

# Confidence threshold rejection
python predict.py --input data/sample_test.csv --strategy confidence_threshold --tau 0.85

# Ensemble majority vote with disagreement detection
python predict.py --input data/sample_test.csv --strategy ensemble

# Combined strategy
python predict.py --input data/sample_test.csv --strategy ensemble+threshold --tau 0.85

# Save results to CSV
python predict.py --input data/sample_test.csv --strategy confidence_threshold --output results.csv
```

## Project Structure

```
project/
├── config.py                  # Global configuration
├── predict.py                 # CLI prototype
├── requirements.txt
├── README.md
├── implementation_plan.md     # Full implementation plan
│
├── data/
│   ├── NF-UNSW-NB15-v3.csv
│   ├── NF-CSE-CIC-IDS2018-v3.csv
│   ├── NF-BoT-IoT-v3.csv
│   ├── NF-ToN-IoT-v3.csv
│   ├── processed/
│   └── metadata/
│       ├── feature_columns.json
│       ├── common_features_unsw_cicids.json
│       └── class_mapping.json
│
├── preprocessing/
│   ├── __init__.py
│   ├── preprocess.py          # ColumnTransformer pipeline
│   └── balance.py             # Class weight / SMOTE utilities
│
├── models/
│   ├── __init__.py
│   ├── baseline.py            # Majority, LR, RF
│   └── chosen_model.py        # XGBoost with GPU support
│
├── robustness/
│   ├── __init__.py
│   ├── stress_tests.py           # Stress A / B / C
│   ├── strategies.py             # Confidence threshold + ensemble
│   └── streaming_ensemble.py     # Single-pass full CICIDS ensemble
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # All evaluation metrics
│   └── plots.py               # All visualization functions
│
├── experiments/
│   ├── run_eda.py
│   ├── validate_data.py
│   ├── prepare_data.py
│   ├── run_baseline.py
│   ├── run_significance.py
│   ├── run_stress.py
│   ├── run_stress_b_full.py      # Full CICIDS Stress B evaluation
│   ├── run_strategies.py
│   ├── run_ablation.py
│   └── run_failure_analysis.py
│
├── results/                   # All output tables (CSV)
├── reports/
│   ├── figures/               # All figures (PNG)
│   └── notes/                 # Stage-level notes (MD)
└── artifacts/                 # Trained models and configs
    ├── preprocessor.joblib
    ├── best_model.joblib
    ├── label_encoder.joblib
    ├── class_mapping.json
    ├── strategy_config.json
    ├── training_config.json
    └── ensemble_models/
```

## Key Results

| Phase | Key Metric | Value |
|-------|-----------|-------|
| Baseline (RF) | Macro-F1 | 0.6774 |
| Stress A | AUROC (unknown detection) | 0.983–0.987 |
| Stress B | Binary F1 / FPR / FNR (full CICIDS, 20.1M flows) | 0.2593 / 0.5607 / 0.2887 |
| Stress C | F1 under 10% noise (sigma=0.1) | 0.2688 |
| Strategy 1 (τ=0.99) | Full F1 (reject=error) / Accepted F1 | 0.1157 / 0.8667 |
| Strategy 1 (τ=0.99) | Coverage / Unknown rejection | 94.6% / 97-100% |
| Strategy 2 (M=5) | Disagreement AUROC | 0.8529 |
| Strategy 2 (M=5) | Stress B F1 (full CICIDS) / Stress C F1 | 0.2588 / 0.2526 |

Note: Strategy 1 metrics are selective (accepted-only) unless labeled "Full F1".
Stress B results are computed on the full NF-CSE-CIC-IDS2018-v3 target domain
using chunked evaluation (see Full Stress B Evaluation section below). Strategy 2
Stress B uses binary majority voting (>= 3 of 5) over a five-member heterogeneous
ensemble kept in memory while the target dataset is streamed once in chunks.

## Reference

Dataset: Sarhan, M., Layeghy, S., & Portmann, M. (2024). NF-UQ-NIDS-v3: A Network Flow-Based Benchmark Dataset for Intrusion Detection.
