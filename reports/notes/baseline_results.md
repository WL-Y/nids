# Baseline Results Notes

## Models Compared

Four baseline models were evaluated on NF-UNSW-NB15-v3:

1. **Majority Classifier** — always predicts "Benign" (floor baseline)
2. **Logistic Regression** — C=10, max_iter=3000, class_weight='balanced'
3. **Random Forest** — n_estimators=100, max_depth=None, class_weight='balanced'
4. **XGBoost** — n_estimators=200, max_depth=6, learning_rate=0.1

## Key Findings

| Model | Macro-F1 | Weighted-F1 | Training Time |
|-------|----------|-------------|---------------|
| Majority | 0.0972 | 0.9198 | 0.1s |
| Logistic Regression | 0.4211 | 0.9744 | 21.6min |
| **Random Forest** | **0.6774** | 0.9863 | 2.0min |
| XGBoost | 0.6398 | 0.9857 | 5.6min |

- **Best model**: Random Forest (Macro-F1 = 0.6774)
- Macro-F1 is the primary metric because the dataset is highly imbalanced (94.6% Benign)
- All models achieve high Weighted-F1 due to Benign dominance, which is misleading

## Confusion Matrix Analysis

### Classes Most Often Confused

1. **Shellcode → Exploits**: Both involve code execution attacks; feature distributions
   (IN_BYTES, OUT_BYTES) overlap significantly in NetFlow data
2. **Worms → Exploits/Backdoor**: Only 158 training samples; the feature space is
   almost entirely covered by higher-frequency attack classes
3. **Analysis → Exploits/Fuzzers**: Very low sample count (1,226); traffic patterns
   resemble other low-intensity attack types

### Rare Classes with Low Recall

- **Worms** (158 samples, 0.007%): Recall strongly depends on class_weight;
  without balancing, recall is near zero
- **Analysis** (1,226 samples): Performance improves significantly with class_weight
- **Shellcode** (2,381 samples): Moderate recall improvement with class_weight

## Selected Model Justification

Random Forest was selected as the best model because:
1. Highest Macro-F1 (0.6774) among all evaluated models
2. Fast training time (2 minutes vs 21 minutes for LR)
3. Built-in class_weight support for handling imbalance
4. Tree-based models handle OrdinalEncoder features naturally
5. Feature importances are directly interpretable for Stress C analysis

## Per-Class Results

See `results/per_class_report_RandomForest.csv` for detailed per-class precision,
recall, and F1 scores.
