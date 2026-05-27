# Robust Multi-Class Network Intrusion Detection — Detailed Project Report

> This document is a detailed technical explanation of the project. It is not the final paper-format report, but it is intended to serve as a strong foundation for writing the final course report.

## 1. Project Overview

This project builds and evaluates a machine-learning based Network Intrusion Detection System (NIDS) for multi-class network-flow classification. The main goal is not simply to obtain a high clean-test accuracy number. The project is designed around a more realistic question:

**When a classifier is deployed in a changing network environment, where does it fail, why does it fail, and which practical strategies can reduce those failures?**

The project follows the proposal's core structure:

1. Build baseline multi-class classifiers on the source dataset.
2. Stress test the best classifier under realistic failure modes.
3. Implement robustness strategies and compare them against the baseline.
4. Run ablation studies and failure analysis.
5. Provide a command-line prototype for inference.

The implementation uses the **NF-UQ-NIDS-v3** dataset family. The proposal originally describes NF-UQ-NIDS-v2, but this repository uses the newer v3 data files:

- Source / training domain: `NF-UNSW-NB15-v3`
- Target / distribution-shift domain: `NF-CSE-CIC-IDS2018-v3`
- Optional available domains: `NF-BoT-IoT-v3`, `NF-ToN-IoT-v3`

This is an upgrade rather than a conceptual change. The project still follows the proposal's experimental design: train on UNSW, stress test under held-out classes, cross-domain CICIDS, and feature degradation.

## 2. Repository Structure

The repository is organized into functional modules:

| Folder / File | Purpose |
|---|---|
| `config.py` | Central configuration: paths, feature lists, model parameters, stress-test parameters, seeds |
| `preprocessing/` | Data loading, cleaning, train/validation/test split, imputation, scaling, encoding |
| `models/` | Baseline classifiers and chosen XGBoost model |
| `evaluation/` | Metrics and plotting utilities |
| `robustness/` | Stress tests and robustness strategies |
| `experiments/` | Reproducible experiment scripts |
| `predict.py` | CLI prototype for inference |
| `results/` | CSV output tables |
| `reports/figures_final/` | Final selected report-ready figures |
| `artifacts/` | Saved trained models, preprocessor, label encoder, strategy config |

This structure matches the proposal's expected modular repository design. Each major experiment is reproducible from a script in `experiments/`.

## 3. Dataset and Target Definition

### 3.1 Dataset Version

The project uses NF-UQ-NIDS-v3. The main source dataset is:

```text
data/NF-UNSW-NB15-v3.csv
```

The cross-domain target dataset for Stress Test B is:

```text
data/NF-CSE-CIC-IDS2018-v3.csv
```

The source dataset contains approximately 2.37 million flows, while the CICIDS target domain contains 20,115,529 flows.

### 3.2 Target Columns

The dataset contains two important target columns:

| Column | Meaning | Used For |
|---|---|---|
| `Attack` | Multi-class attack label, including `Benign` | Main multi-class classification task |
| `Label` | Binary label | Binary attack-vs-benign evaluation, especially Stress B |

The main baseline task is a **10-class classification problem**:

```text
Analysis
Backdoor
Benign
DoS
Exploits
Fuzzers
Generic
Reconnaissance
Shellcode
Worms
```

For Stress Test B, the project maps predictions to binary labels because UNSW and CICIDS do not share the same attack-category taxonomy. The mapping is:

```text
Benign -> 0
All other classes -> 1
```

### 3.3 Feature Set

After removing labels, IP addresses, and timestamp columns, the final modeling input has **49 raw features**:

- 41 numeric features
- 6 protocol-specific features
- 2 categorical protocol-related features

This is confirmed by:

```text
artifacts/training_config.json
data/metadata/feature_columns.json
```

The project intentionally drops IP addresses and timestamps because these are identifiers or environment-specific fields. Keeping them could cause leakage or poor generalization.

## 4. Preprocessing Pipeline

Preprocessing is implemented mainly in:

```text
preprocessing/preprocess.py
```

The preprocessing pipeline is one of the most important parts of the project because data leakage would invalidate the reported results.

### 4.1 Loading and Cleaning

The function `load_and_clean_data()`:

1. Loads the CSV using `pandas.read_csv`.
2. Replaces infinite values with `NaN`.
3. Drops non-generalizable identifier columns:

```text
FLOW_START_MILLISECONDS
FLOW_END_MILLISECONDS
IPV4_SRC_ADDR
IPV4_DST_ADDR
```

This keeps the model focused on flow behavior rather than memorizing addresses or collection-time artifacts.

### 4.2 L7_PROTO Cleaning

The v3 dataset contains non-integer values in `L7_PROTO`, such as decimal protocol values. The project handles this with:

```python
fillna(0) -> round() -> astype(int)
```

Why this matters:

- `L7_PROTO` is treated as a categorical protocol-like feature.
- Missing values are interpreted as "no application-layer protocol detected".
- Rounding prevents inconsistent category encoding from small decimal artifacts.

### 4.3 Train / Validation / Test Split

The split is:

```text
70% train
10% validation
20% test
```

The split is stratified by `Attack`, which is necessary because the dataset is highly imbalanced. Without stratification, rare classes such as `Worms` could become underrepresented or disappear from smaller splits.

### 4.4 ColumnTransformer Design

The project uses a `ColumnTransformer` with three paths:

| Feature Group | Processing |
|---|---|
| Numeric features | Median imputation + `StandardScaler` |
| Protocol-specific features | Constant 0 imputation + `StandardScaler` |
| Categorical features | Constant 0 imputation + `OrdinalEncoder` |

The design is appropriate because not all missingness has the same meaning:

- Generic numeric missingness is handled by median imputation.
- Protocol-specific missingness often means "not applicable", so filling with 0 is semantically reasonable.
- Protocol categorical values need encoding before model training.

### 4.5 Leakage Prevention

The preprocessor is fit only on the training split:

```python
preprocessor.fit(X_train)
```

Validation, test, and target-domain data are only transformed:

```python
preprocessor.transform(X_val)
preprocessor.transform(X_test)
preprocessor.transform(X_cicids)
```

This is essential. If the preprocessor were fit on test or CICIDS data, scaling and imputation statistics would leak information from evaluation data into training.

## 5. Class Imbalance Handling

Network intrusion datasets are extremely imbalanced. In this dataset, `Benign` dominates, while rare attack classes such as `Worms`, `Analysis`, and `Shellcode` contain far fewer samples.

The default strategy is **class weighting**, implemented in:

```text
preprocessing/balance.py
```

The project uses:

- `class_weight="balanced"` for Logistic Regression and Random Forest
- sample weights for XGBoost

SMOTE is implemented but used only in ablation experiments. This is a sensible design choice because SMOTE can be unstable or misleading when a class is extremely rare. For example, `Worms` has very limited sample diversity, so synthetic oversampling may not represent true unseen attack variation.

## 6. Baseline Classification

Baseline experiments are implemented in:

```text
experiments/run_baseline.py
models/baseline.py
models/chosen_model.py
```

### 6.1 Models

The project trains four models:

| Model | Purpose |
|---|---|
| Majority Classifier | Floor baseline |
| Logistic Regression | Linear baseline |
| Random Forest | Strong nonlinear baseline |
| XGBoost | Chosen advanced model |

This satisfies the proposal's baseline requirements: majority classifier, Logistic Regression, Random Forest, and one chosen model.

### 6.2 Hyperparameter Selection

Logistic Regression, Random Forest, and XGBoost use hyperparameter search on a stratified tuning subset. The search objective is macro-F1 rather than accuracy.

This is important because accuracy is misleading on imbalanced data. A model can achieve high accuracy by predicting the majority class while failing minority attacks.

### 6.3 Baseline Results

The baseline summary is stored in:

```text
results/baseline_results.csv
```

| Model | Macro-F1 | Weighted-F1 | Train Time (s) | Inference Time (s) |
|---|---:|---:|---:|---:|
| Majority | 0.0972 | 0.9198 | 0.1 | 0.00 |
| Logistic Regression | 0.4211 | 0.9744 | 1298.2 | 0.07 |
| Random Forest | 0.6774 | 0.9863 | 120.2 | 0.77 |
| XGBoost | 0.6398 | 0.9857 | 338.6 | 0.48 |

### 6.4 Baseline Analysis

Random Forest is the best clean-test model by macro-F1:

```text
Random Forest Macro-F1 = 0.6774
```

Weighted-F1 is much higher than macro-F1 for all strong models. This gap indicates that the model performs well on high-support classes but still struggles on minority classes.

The final report-ready figures are:

```text
reports/figures_final/03_baseline_macro_f1_comparison.png
reports/figures_final/04_random_forest_per_class_f1.png
```

The per-class Random Forest result shows that minority attack classes are the main weakness. This supports the project's focus on robustness and failure analysis rather than clean accuracy alone.

### 6.5 Reference Model for Robustness Experiments

The later robustness experiments are not independent model-selection experiments. They mainly use the best Phase 3 baseline model as the reference detector.

The selected reference model is:

```text
Random Forest
```

with the following main settings:

```text
n_estimators = 100
max_depth = None
min_samples_leaf = 2
min_samples_split = 5
class_weight = balanced
random_state = 42
```

This matters because Stress A, Stress B, Stress C, and Strategy 1 should be interpreted as robustness evaluations of the Random Forest detector, not as separate attempts to find a better classifier. Random Forest was chosen because it had the best clean macro-F1 among the baseline models, and macro-F1 is the most appropriate primary metric for this imbalanced multi-class intrusion-detection task.

There are two implementation details:

- For Stress B, Stress C, and Strategy 1, the model represents the full UNSW-trained Random Forest baseline using the UNSW-fitted preprocessing pipeline.
- For Stress A, the Random Forest architecture is reused, but a new known-class-only Random Forest is trained for each held-out-class setting. This is required because the held-out classes must be completely absent from both training and preprocessing.

## 7. Stress Test A: Held-Out Attack Classes

Stress Test A evaluates open-set behavior. The key question is:

**What happens when the model sees attack classes during testing that were completely absent during training?**

This is implemented in:

```text
robustness/stress_tests.py
experiments/run_stress.py
```

### 7.1 Model Used for Stress A

Stress A is based on the same Random Forest model family selected in the baseline phase, but it does not directly reuse the already saved full 10-class model.

Instead, for each held-out group, the experiment trains a new Random Forest on only the known classes:

```text
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)
```

This is the correct design for open-set evaluation. If the original full 10-class Random Forest were reused, it would already have seen the held-out attack classes during training, which would invalidate Stress A. Therefore, Stress A keeps the model type and hyperparameters consistent with the baseline, but retrains the model under a stricter known-class-only condition.

The preprocessing pipeline is also refit only on known-class training data for each held-out group. This means the held-out classes are absent from:

- model fitting
- validation data used during the held-out experiment
- preprocessing statistics such as imputation and scaling

The full test set is still used for evaluation, so the trained known-class Random Forest must make predictions on both known and unknown classes.

### 7.2 Held-Out Groups

The project uses two held-out groups:

| Group | Held-Out Classes |
|---|---|
| Set 1 | `Worms`, `Analysis`, `Shellcode` |
| Set 2 | `Backdoor`, `DoS`, `Fuzzers` |

Using two groups is important because results can depend heavily on which attack classes are withheld. A single held-out group would not be enough evidence.

### 7.3 Strict Preprocessing for Stress A

For each held-out group:

1. Remove held-out classes from train and validation.
2. Fit a new preprocessor only on known-class training data.
3. Transform the full test set, including unknown classes.
4. Train a model only on known classes.
5. Evaluate both known-class performance and unknown-class behavior.

This avoids leakage from unknown classes into preprocessing statistics.

### 7.4 Stress A Metrics

The project measures:

- Known-class macro-F1
- Unknown mean confidence
- Known-correct mean confidence
- Known-wrong mean confidence
- AUROC of confidence as an unknown detector
- Unknown-to-known misclassification mapping
- Full confusion matrix

### 7.5 Stress A Results

Stored in:

```text
results/stress_a_results.csv
```

| Held-Out Classes | Known Macro-F1 | Unknown Mean Confidence | Known Correct Confidence | Known Wrong Confidence | Unknown AUROC |
|---|---:|---:|---:|---:|---:|
| Worms, Analysis, Shellcode | 0.7037 | 0.5442 | 0.9908 | 0.5653 | 0.9870 |
| Backdoor, DoS, Fuzzers | 0.7913 | 0.6321 | 0.9966 | 0.5767 | 0.9832 |

### 7.6 Stress A Interpretation

The model assigns much lower confidence to unknown attacks than to correctly classified known traffic. This is a useful signal for open-set detection.

However, the model still must assign every unknown attack to a known class when no rejection strategy is used. The unknown mapping shows a strong collapse into classes such as `Exploits`, `Reconnaissance`, and `Generic`.

Examples:

- `Analysis` is mostly predicted as `Exploits`.
- `DoS` is mostly predicted as `Exploits`.
- `Fuzzers` is often split between `Exploits` and `Reconnaissance`.

This suggests that the learned feature space groups several unknown or withheld attacks near broad exploitation-like known classes.

Final report figures:

```text
reports/figures_final/05a_stress_a_metrics_and_confidence_with_values.png
reports/figures_final/05b_stress_a_unknown_mapping_with_counts.png
reports/figures_final/06_stress_a_full_confusion_normalized_set1.png
reports/figures_final/07_stress_a_full_confusion_normalized_set2.png
```

## 8. Stress Test B: Cross-Dataset Distribution Shift

Stress Test B evaluates domain shift:

**Train on UNSW, test directly on CICIDS without retraining.**

This is implemented in:

```text
experiments/run_stress.py
experiments/run_stress_b_full.py
robustness/stress_tests.py
```

### 8.1 Model Used for Stress B

Stress B evaluates the full UNSW-trained Random Forest baseline under cross-dataset distribution shift.

The model is trained on the UNSW training split with the same selected Random Forest settings:

```text
n_estimators = 100
min_samples_leaf = 2
min_samples_split = 5
class_weight = balanced
```

The preprocessing pipeline is fitted only on the UNSW training data. During Stress B, the CICIDS data is never used to fit a model, tune a threshold, fit a scaler, or update imputation statistics.

Conceptually, the Stress B pipeline is:

```text
UNSW train data
    -> fit preprocessor
    -> train Random Forest
    -> stream CICIDS chunks
    -> transform CICIDS using UNSW preprocessor
    -> predict with UNSW-trained Random Forest
    -> convert prediction to binary attack/benign
```

This means Stress B tests true transfer from one dataset/domain to another. The poor Stress B result should therefore be interpreted as a domain-generalization failure of the UNSW-trained Random Forest, not as a failure caused by retraining or tuning on CICIDS.

### 8.2 Why Binary Evaluation?

UNSW and CICIDS have different attack taxonomies. Their multi-class labels are not directly comparable. Therefore, Stress B maps both true and predicted labels into:

```text
Benign
Attack
```

This allows evaluation of whether the UNSW-trained detector can still detect malicious traffic in a new domain.

### 8.3 Full Chunked Evaluation

The project originally used a 10% sampled CICIDS evaluation. That has now been removed. The current Stress B uses full target-domain evaluation with chunked reading.

The final Stress B result uses all:

```text
20,115,529 CICIDS flows
```

The chunked pipeline:

1. Read one chunk of CICIDS.
2. Clean it using the same cleaning logic.
3. Transform it with the UNSW-fitted preprocessor.
4. Predict with the UNSW-trained model.
5. Convert predictions to binary attack/benign.
6. Accumulate TP, TN, FP, FN.
7. Repeat until all chunks are processed.

This design avoids loading the entire 20M-row target dataset into memory.

### 8.4 Stress B Results

Stored in:

```text
results/stress_b_full_results.csv
```

| Metric | Value |
|---|---:|
| n_samples | 20,115,529 |
| Accuracy | 0.4745 |
| Precision | 0.1585 |
| Recall | 0.7113 |
| F1 | 0.2593 |
| FPR | 0.5607 |
| FNR | 0.2887 |
| Delta F1 vs UNSW | 0.4181 |

### 8.5 Stress B Interpretation

Stress B reveals severe distribution shift. The model retains moderate recall:

```text
Recall = 0.7113
```

but precision is very low:

```text
Precision = 0.1585
```

This means the model flags many CICIDS benign flows as attacks. The false positive rate is high:

```text
FPR = 0.5607
```

This is an important result. The model does not simply fail by missing attacks; it also becomes too aggressive in the target domain. In a real deployment, this would create alert fatigue.

Final report figures:

```text
reports/figures_final/08_stress_b_full_binary_metrics.png
reports/figures_final/09_stress_b_full_binary_confusion.png
```

## 9. Stress Test C: Feature Degradation

Stress Test C measures sensitivity to corrupted or missing features.

Implemented in:

```text
robustness/stress_tests.py
experiments/run_stress.py
```

### 9.1 Model Used for Stress C

Stress C also uses the full UNSW-trained Random Forest baseline and the UNSW-fitted preprocessing pipeline. It is not a retraining experiment.

The important difference from Stress B is the evaluation domain:

- Stress B changes the dataset domain from UNSW to CICIDS.
- Stress C stays on the UNSW test split but corrupts or removes input features.

The Stress C pipeline is:

```text
UNSW-trained Random Forest
    -> take raw UNSW test features
    -> apply controlled degradation
    -> transform using the original UNSW preprocessor
    -> predict with the same Random Forest
    -> measure macro-F1 drop
```

Using the same Random Forest is important because Stress C is intended to measure feature sensitivity of the selected detector. If a new model were retrained after degradation, the experiment would test adaptation instead of robustness.

### 9.2 Degradation Types

The project tests:

| Type | Levels |
|---|---|
| Gaussian noise | sigma = 0.1, 0.5, 1.0 |
| Random masking | p = 0.1, 0.25, 0.5 |
| Feature dropout, top features | k = 2, 4, 6 |
| Feature dropout, bottom features | k = 2, 4, 6 |

Corruption is applied at the raw feature level, then the data is transformed through the fitted preprocessor. This is realistic because in deployment the raw NetFlow fields are what become noisy or missing.

### 9.3 Stress C Results

Stored in:

```text
results/stress_c_results.csv
```

| Type | Level | Macro-F1 |
|---|---|---:|
| Noise | sigma=0.1 | 0.2688 |
| Noise | sigma=0.5 | 0.2036 |
| Noise | sigma=1.0 | 0.1682 |
| Masking | p=0.1 | 0.6336 |
| Masking | p=0.25 | 0.5214 |
| Masking | p=0.5 | 0.2959 |
| Drop top features | k=2 | 0.5379 |
| Drop top features | k=4 | 0.4151 |
| Drop top features | k=6 | 0.2660 |
| Drop bottom features | k=2 | 0.6774 |
| Drop bottom features | k=4 | 0.6758 |
| Drop bottom features | k=6 | 0.6740 |

### 9.4 Stress C Interpretation

The clean Random Forest macro-F1 is 0.6774. The degradation results show:

- Gaussian noise is highly damaging, even at sigma=0.1.
- Random masking degrades performance gradually.
- Dropping the most important features causes a large performance drop.
- Dropping the least important features has almost no effect.

This indicates that the classifier depends heavily on a subset of high-importance features. This is useful for failure analysis because it identifies which feature groups are most critical to model behavior.

Final report figure:

```text
reports/figures_final/10_stress_c_degradation_by_type.png
```

## 10. Robustness Strategy 1: Confidence Threshold Rejection

Strategy 1 implements a rejection option:

```text
If max predicted probability >= tau:
    accept prediction
else:
    reject / flag for review
```

Implemented in:

```text
robustness/strategies.py
experiments/run_strategies.py
```

### 10.1 Base Model Used for Strategy 1

Strategy 1 is built directly on top of the Phase 3 best model:

```text
artifacts/best_model.joblib
```

In this project, that saved best model is the Random Forest baseline. Strategy 1 does not train a new classifier for the standard clean, Stress B, or Stress C comparisons. It changes only the decision rule applied to the Random Forest probability output.

The baseline Random Forest always produces a hard class label:

```text
prediction = argmax(class probabilities)
```

Strategy 1 adds a confidence gate:

```text
if max_probability >= tau:
    accept Random Forest prediction
else:
    reject / flag for review
```

Therefore, Strategy 1 should be understood as a post-processing uncertainty strategy for the Random Forest, not as a different underlying classifier.

For Strategy 1 under Stress A, the same strict open-set rule is used as in Stress A itself: a known-class-only Random Forest is trained for each held-out group, and the confidence threshold is applied to that known-class-only model. This prevents held-out classes from leaking into the strategy evaluation.

### 10.2 Why This Strategy?

Stress A shows that unknown attacks often receive lower confidence than correctly classified known samples. This suggests that confidence can be used as a simple uncertainty signal.

The goal is not to make the classifier perfectly open-set aware. The goal is to avoid forcing a hard label when the model is uncertain.

### 10.3 Threshold Selection

The threshold is selected on the validation set, not the test set.

The selected threshold is:

```text
tau = 0.99
```

This is stored in:

```text
artifacts/strategy_config.json
```

### 10.4 Strategy 1 Results

From:

```text
results/strategy1_summary.csv
results/strategies_comparison.csv
```

| Metric | Value |
|---|---:|
| Clean full F1, treating reject as error | 0.1157 |
| Coverage | 0.9459 |
| Accepted F1 | 0.8667 |
| Stress A F1 | 0.9574 |
| Rejection rate | 0.0541 |
| Detection AUROC | 0.9851 |

### 10.5 Interpretation

Strategy 1 substantially improves accepted-sample quality. Accepted F1 is much higher than the baseline clean macro-F1:

```text
Accepted F1 = 0.8667
Baseline Macro-F1 = 0.6774
```

However, the full F1 is low when rejected samples are counted as errors. This is expected because rejection changes the task: the model is no longer trying to classify every sample. In deployment, rejected samples would be routed to a human analyst or a secondary detector.

On Stress B and Stress C, Strategy 1 at tau=0.99 is too conservative in the reported final comparison, producing effectively undefined accepted evaluation under those conditions. This is an important limitation, not a failure to hide. It shows that a globally fixed threshold may not transfer across domains.

Final figures:

```text
reports/figures_final/12_strategy1_coverage_accuracy_curve.png
reports/figures_final/14_ablation_threshold_tradeoff.png
```

## 11. Robustness Strategy 2: Ensemble Disagreement Detection

Strategy 2 trains a heterogeneous ensemble:

```text
3 x Random Forest
1 x XGBoost
1 x Logistic Regression
```

Implemented in:

```text
robustness/strategies.py
robustness/streaming_ensemble.py
experiments/run_strategies.py
```

### 11.1 Base Models Used for Strategy 2

Strategy 2 is not based on a single Random Forest. It uses a heterogeneous ensemble with five members:

```text
M = 5
3 Random Forest models with different random seeds
1 XGBoost model
1 Logistic Regression model
```

The ensemble is trained on the same UNSW training split after transformation by the UNSW-fitted preprocessor. In the main streaming implementation, each ensemble member is trained on a stratified subset of the UNSW training data to reduce memory pressure while preserving the full five-member ensemble design.

The prediction rule is majority vote:

```text
final prediction = class chosen by most ensemble members
```

The uncertainty score is disagreement:

```text
disagreement = 1 - fraction of ensemble members voting for the majority class
```

For example, if all five models agree, disagreement is 0. If only three out of five agree, disagreement is 0.4.

For Strategy 2 under Stress B, the ensemble members are trained on UNSW and then evaluated on the full CICIDS dataset using chunked streaming, again without fitting anything on CICIDS. For Strategy 2 under Stress C, the same ensemble predicts on degraded UNSW test features. For Strategy 2 under Stress A, the ensemble is retrained under the known-class-only condition so that held-out classes are not seen during training.

### 11.2 Why This Strategy?

The idea is that difficult or out-of-distribution samples may cause different models to disagree. Disagreement can then act as a proxy for uncertainty.

The disagreement score is:

```text
1 - fraction of ensemble members agreeing with the majority vote
```

### 11.3 Strategy 2 Results

From:

```text
results/strategy2_summary.csv
results/strategy2_stress_b_full_results.csv
```

| Metric | Value |
|---|---:|
| Clean full F1 | 0.6674 |
| Stress A F1 | 0.7353 |
| Stress B F1 | 0.2588 |
| Stress C F1 | 0.2526 |
| Disagreement AUROC | 0.8529 |

### 11.4 Interpretation

Strategy 2 is useful as an uncertainty detector:

```text
Disagreement AUROC = 0.8529
```

This means disagreement is meaningfully correlated with wrong predictions.

However, it does not substantially improve Stress B or Stress C performance. Its Stress B F1 is almost the same as the baseline:

```text
Baseline Stress B F1 = 0.2593
Strategy 2 Stress B F1 = 0.2588
```

This suggests that ensemble disagreement helps identify uncertainty, but it does not automatically fix domain shift. If all models are biased similarly under a target-domain shift, majority vote will not solve the problem.

Final figures:

```text
reports/figures_final/11_strategy_comparison_key_metrics.png
reports/figures_final/13_strategy2_disagreement_auroc.png
```

## 12. Ablation Studies

Ablation studies test whether each design choice actually matters.

Implemented in:

```text
experiments/run_ablation.py
```

### 12.1 Threshold Ablation

The threshold ablation varies tau and tracks:

- Coverage
- Accepted accuracy
- Unknown rejection rate
- Known false rejection rate

The results show the expected trade-off:

- Higher tau increases unknown rejection.
- Higher tau also increases rejection of known samples.

This supports the conclusion that threshold selection is an operating-point decision.

### 12.2 Ensemble Size Ablation

The ensemble size ablation tests:

```text
M = 1, 3, 5, 10
```

Results:

| M | Disagreement AUROC |
|---:|---:|
| 1 | 0.5000 |
| 3 | 0.7326 |
| 5 | 0.8589 |
| 10 | 0.8731 |

The result shows that disagreement becomes useful once multiple models are available. Performance improves strongly from M=1 to M=5, then improves only slightly from M=5 to M=10.

This supports M=5 as a reasonable cost-performance trade-off.

### 12.3 Imbalance Method Ablation

The project compares:

- No balancing
- Class weighting
- SMOTE

Class weighting improves rare-class recall without the extra risk of relying heavily on synthetic samples.

Final ablation figures:

```text
reports/figures_final/14_ablation_threshold_tradeoff.png
reports/figures_final/15_ablation_ensemble_and_imbalance.png
```

## 13. Statistical Significance

The project includes repeated-seed evaluation:

```text
results/statistical_significance.csv
```

The main reported repeated-run results are:

| Condition | Mean Macro-F1 | Std | Runs |
|---|---:|---:|---:|
| Clean | 0.6464 | 0.0036 | 5 |
| Stress A | 0.7215 | 0.0034 | 5 |

The project also includes McNemar tests:

```text
results/mcnemar_results.csv
```

This satisfies the proposal's requirement to avoid relying only on a single run when results are close.

## 14. Failure Analysis

Failure analysis is documented in:

```text
reports/notes/strategy_failure_analysis.md
experiments/run_failure_analysis.py
```

The major failure modes are:

### 14.1 Minority Classes Remain Difficult

Classes such as `Backdoor`, `Analysis`, and `DoS` have low per-class performance compared with high-support classes.

This is visible in:

```text
reports/figures_final/04_random_forest_per_class_f1.png
```

### 14.2 Unknown Classes Collapse Into Known Exploit-Like Classes

Stress A shows that unknown attacks are frequently mapped to `Exploits`, `Reconnaissance`, or `Generic`.

This is visible in:

```text
reports/figures_final/05b_stress_a_unknown_mapping_with_counts.png
```

### 14.3 Cross-Domain Shift Produces Many False Positives

Stress B has high recall but very low precision and high FPR. This means the model over-alerts heavily in the CICIDS target domain.

### 14.4 Confidence Thresholding Does Not Transfer Perfectly

Thresholding is useful on the source/open-set setting, but a fixed tau=0.99 becomes too conservative under some stress conditions.

This suggests that future work should consider calibration or adaptive thresholds.

## 15. CLI Prototype

The CLI is implemented in:

```text
predict.py
```

It supports:

```bash
python predict.py --input data/sample_test.csv
python predict.py --input data/sample_test.csv --strategy confidence_threshold --tau 0.85
python predict.py --input data/sample_test.csv --strategy ensemble
python predict.py --input data/sample_test.csv --strategy ensemble+threshold --tau 0.85
```

The CLI loads:

```text
artifacts/preprocessor.joblib
artifacts/best_model.joblib
artifacts/label_encoder.joblib
```

It does not fit anything during inference. This is correct because inference data must not change preprocessing statistics.

The decision text now distinguishes:

- `NO ALERT` for benign predictions
- `ALERT` for attack predictions
- `FLAG FOR REVIEW` for low-confidence rejected samples
- `UNCERTAIN` for high ensemble disagreement

## 16. Final Report Figures

The final curated figure folder is:

```text
reports/figures_final/
```

The selected report-ready figures are:

| Figure | Purpose |
|---|---|
| `01_class_distribution.png` | Shows class imbalance |
| `02_l7_proto_quality.png` | Shows L7_PROTO cleaning issue |
| `03_baseline_macro_f1_comparison.png` | Baseline model comparison |
| `04_random_forest_per_class_f1.png` | Per-class weakness analysis |
| `05a_stress_a_metrics_and_confidence_with_values.png` | Stress A metrics with concrete values |
| `05b_stress_a_unknown_mapping_with_counts.png` | Unknown-to-known mapping with counts and percentages |
| `06_stress_a_full_confusion_normalized_set1.png` | Stress A Set 1 confusion matrix with row percentages |
| `07_stress_a_full_confusion_normalized_set2.png` | Stress A Set 2 confusion matrix with row percentages |
| `08_stress_b_full_binary_metrics.png` | Full CICIDS binary metrics |
| `09_stress_b_full_binary_confusion.png` | Full CICIDS binary confusion matrix |
| `10_stress_c_degradation_by_type.png` | Feature degradation curves |
| `11_strategy_comparison_key_metrics.png` | Strategy comparison |
| `12_strategy1_coverage_accuracy_curve.png` | Threshold coverage-accuracy trade-off |
| `13_strategy2_disagreement_auroc.png` | Disagreement as wrong-prediction detector |
| `14_ablation_threshold_tradeoff.png` | Threshold ablation |
| `15_ablation_ensemble_and_imbalance.png` | Ensemble size and imbalance ablations |

These figures are clearer than the original raw figures and are better suited for the final written report.

## 17. Data and Submission Notes

The local workspace contains raw dataset CSV files under `data/`. These files are large and should not be committed or submitted as code artifacts.

The repository now includes `.gitignore` entries for:

- raw dataset CSV files
- large joblib model artifacts
- Python cache files

This follows the proposal's instruction that raw data should not be committed.

## 18. Overall Assessment

The project is complete from an experimental-code perspective. It satisfies the proposal's required components:

| Requirement | Status |
|---|---|
| Structured code repository | Complete |
| Preprocessing with leakage prevention | Complete |
| Majority baseline | Complete |
| Logistic Regression baseline | Complete |
| Random Forest baseline | Complete |
| Chosen model | Complete, XGBoost |
| Per-class metrics and confusion matrices | Complete |
| Stress A held-out classes | Complete, two groups |
| Stress B cross-dataset shift | Complete, full CICIDS chunked evaluation |
| Stress C feature degradation | Complete |
| At least two robustness strategies | Complete |
| Ablation study | Complete |
| Statistical significance | Complete |
| Failure analysis | Complete |
| CLI prototype | Complete |

The main remaining task is not code-related: the final paper-style report still needs to be written in the required academic format. This document can be used as the technical base for that final report.

## 19. Key Takeaways

1. Random Forest is the strongest clean baseline, with Macro-F1 = 0.6774.
2. Clean weighted-F1 is high, but macro-F1 reveals minority-class weakness.
3. Stress A shows unknown attacks are often mapped to known exploit-like classes.
4. Confidence is useful for unknown detection under Stress A, with AUROC around 0.983-0.987.
5. Stress B exposes severe cross-domain degradation: F1 = 0.2593 and FPR = 0.5607.
6. Stress C shows high sensitivity to numeric noise and important-feature dropout.
7. Confidence thresholding improves accepted-sample quality but can be too conservative under shift.
8. Ensemble disagreement is useful as an uncertainty signal but does not solve distribution shift by itself.
9. The final system is reproducible, modular, and aligned with the proposal's required evaluation design.
