# Project Proposal
ML Innovation - SP 2026
## Robust. Multi-Class Network Intrusion Detection: Classification, Stress Testing, and Improvement Strategies
Machine Learning and Innovation - Spring 2026
Kean University - Course Project Proposal

### Abstract
Network intrusion detection is a critical and evolving challenge in cybersecurity. While classification accuracy on benchmark datasets has become near-saturated, the robustness of learned classifiers under realistic deployment conditions - including previously unseen attack types, distributional shift, and noisy or incomplete inputs - remains an open and important problem. This project asks students to go beyond standard classification: build a multi-class intrusion detector, systematically expose its failure modes, and implement and evaluate targeted strategies to improve robustness. The work is grounded in a recent, unified benchmark dataset (NF-UQ-NIDS-v2), designed to support cross-dataset generalization experinents. Strong outcomes from this project are directly relevant to current research and can form the basis of a workshop or conference submission.

## Contents
1 Introduction and Motivation 3
2 Research Questions 3
3 Background and Context 4
4 Dataset 5
4.4 Data Preprocessing Requirements......6
5 Methodology
5.1 Component 1: Multi-Class Classifier (Baseline) .......7
5.2 Component 2: Systematic Robustness Evaluation..................7
5.2.1 Stress Test A: Held-Out Attack Classes (Open-Set Condition).......7
5.2.2 Stress Test B: Cross-Dataset Distribution Shift ...............8
5.2.3 Stress Test C: Feature Degradation........ ..........8
5.3 Component 3: Robustness Improvement Strategies ....9
5.3.1 Strategy 1: Confidence Thresholding with Rejection ............9
5.3.2 Strategy 2: Ensemble with Disagreement Detection .............9
5.3.3 Strategy 3: One-Class Classifier as Anomaly Fallback ...........9
5.3.4 Strategy 4: Cost-Sensitive Learning ............10
5.3.5 Strategy 5: Feature Robustness via Augmentation ............10
6 Experimental Design and Evaluation 11
6.1 Metrics …
6.2 Required Baselines ....11
6.3 Ablation Study (Required)......11
6.4 Statistical Significance .........1
7 Deliverables 12
7.1 Code 12
7.2 Report (Required)....12
7.3 Prototype (Required) ......13
8 Project Timeline 14
9 What Makes This Project Publication-Ready 14
10 Common Failure Modes 15
11 Optional Advanced Extensions 15
12 Grading Criteria 16

---

# Introduction and Motivation
Network Intrusion Detection Systems (NIDS) form one of the primary defenses against cyberattacks in modern infrastructure. A NIDS monitors network traffic and raises alerts when traffic patterns match known or inferred attack signatures. Machine learning has become the dominant methodology for building such systems, offering the ability to generalize from labeled examples without manually crafting rules.

Despite significant research activity, current NIDS classifiers face a fundamental limitation that is rarely studied rigorously at the course level: they are evaluated on the same distribution they were trained on. In deployment, this assumption breaks immediately. Networks evolve. Attackers adapt. New attack categories emerge weekly. A classifier that achieves 99% accuracy on a held-out test set from the same dataset can fail silently on traffic it has never encountered, either misclassifying novel attacks as benign or producing unreliable confidence scores that mislead operators.

This failure mode - robustness to distribution shift and unseen classes - is precisely where the gap between academic benchmarks and real-world deployment lies. It is also where current research is most active, with methods drawn from open-set recognition, uncertainty quantification, domain adaptation, and few-shot learning being proposed as solutions.

This project places students at that gap. The task is not to build a classifier that scores well on a standard benchmark. The task is to understand how and why classifiers fail under realistic conditions, and to implement and honestly evaluate strategies to address those failures.

### Why This Problem Matters Now
- Benchmark saturation: top classifiers already exceed 99% accuracy on NSL-KDD and CICIDS2017. Reporting yet another high-accuracy result on these datasets contributes nothing.
- Real-world failures: deployed NIDS systems regularly fail to detect zero-day and low-volume attacks that differ from their training distribution.
- Active research gap: open-set recognition and cross-dataset generalization for NIDS are underexplored relative to their importance.
- Methodological gap: most published NIDS papers do not include any robustness analysis. A paper that does will stand out.

# Research Questions
Each group must address the following research questions in their final report. Your experiments must provide evidence-based answers, not opinions.

**RQ1.** Classification baseline. How accurately can a machine learning classifier distinguish among multiple network attack categories using the NF-UQ-NIDS-v2 dataset? Which attack types are systematically confused, and why?

**RQ2.** Robustness under held-out classes. When the classifier is trained on a subset of attack categories and tested on attack types it has never seen, how does its performance degrade? Does it correctly flag unknown traffic as anomalous, or does it confidently misclassify it?

**RQ3.** Robustness under distribution shift. When trained on traffic from one network environment and tested on traffic from a different environment (within the same feature schema), how much does performance degrade? Which features contribute most to the shift?

**RQ4.** Robustness under feature degradation. How sensitive is the classifier to noise, missing values, and corrupted features? Is degradation graceful or catastrophic?

**RQ5.** Improvement strategies. Which of the robustness improvement strategies you implement provides the best trade-off between detection performance and robustness? Under what conditions does each strategy fail?

### Reviewer Expectation
A reviewer will check that each research question is answered with a dedicated experiment, a clear result table, and a discussion that goes beyond "performance improved." If your experiments do not map directly to these questions, the paper will be rejected for insufficient evaluation.

# Background and Context
This section outlines the conceptual background students must be familiar with before beginning implementation. You are expected to read at least two papers from each area. Do not simply summarize papers - use them to motivate your design choices.

## Multi-Class Intrusion Detection
Standard NIDS classification assigns each network flow to one of K categories: normal traffic or one of \(K-1\) attack types. The learning problem is:
\[f: x \in \mathbb{R}^{d} \to\{0,1, ..., K-1\} \tag{1}\]
where x is a feature vector representing a network flow (packet counts, byte volumes, duration, protocol flags, etc.) and K is the number of classes.

The challenge is that class distributions are highly imbalanced: normal traffic vastly outnumbers any individual attack type, and some attack categories contain orders of magnitude fewer samples than others. This imbalance must be handled explicitly; ignoring it produces classifiers that appear accurate but fail on minority attack classes.

## The Open-Set Recognition Problem
Standard classifiers are closed-set: they assume all test samples belong to one of the K training classes. In practice, novel attack types appear at test time. A closed-set classifier has no choice but to assign a novel attack to whichever training class is "closest," often producing high-confidence wrong predictions.

Open-set recognition extends the classifier to produce a rejection option: given input x , the model can output either a known class or "unknown." This is directly relevant to NIDS: a classifier that flags unknown traffic as suspicious is more useful than one that confidently mislabels it.

## Distribution Shift
A model trained on network traffic from organization A may fail on traffic from organization B, even if both networks face the same attack types. This is because network traffic statistics (byte rates, connection durations, protocol distributions) vary significantly across network environments. The source domain is the training distribution; the target domain is the test distribution. The shift between them degrades model performance in ways that accuracy on the source test set does not reveal.

## Confidence Calibration
A classifier’s confidence score (e.g., softmax probability) should reflect actual accuracy: a sample predicted with 90% confidence should be correct 90% of the time. In practice, modern classifiers are overconfident, especially on out-of-distribution inputs. Calibration techniques (Platt scaling, temperature scaling) adjust the score to be more reliable. In a NIDS context, a well-calibrated confidence score enables a rejection threshold: flag flows below a confidence threshold for human review rather than forcing a hard classification.

# Dataset
## Primary Dataset: NF-UQ-NIDS-v2
- Source: University of Queensland, published 2022.
- Access: https://staff.itee.uq.edu.au/marius/NIDS_datasets/
- Format: CSV, NetFlow-based features (IPFIX/NetFlow v9 standard).

NF-UQ-NIDS-v2 is a unified benchmark that standardizes four independently collected network intrusion datasets into a common NetFlow feature schema. This design enables cross-dataset generalization experiments that are otherwise impossible due to inconsistent feature sets across datasets.

Table 1: NF-UQ-NIDS-v2 subsets and their attack categories.

| Subset Origin | Attack Categories |
| ---- | ---- |
| NF-UNSW-NB15-v2 UNSW Sydney, 2015 | Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms |
| NF-CSE-CIC-IDS2018-v2 CIC Canada, 2018 | Brute Force, Heartbleed, Botnet, DoS, DDoS, Web Attacks, Infiltration |
| NF-BoT-IoT-v2 UNSW Sydney, 2018 | DDoS, DoS, Reconnaissance, Theft |
| NF-ToN-IoT-v2 UNSW Sydney, 2020 | Ransomware, Backdoor, DoS, DDoS, Injection, MITM, Password, Scanning, XSS |

Shared feature schema (12 core NetFlow features):

Table 2: Core NetFlow features in NF-UQ-NIDS-v2.

| Feature | Type | Description |
| --- | --- | --- |
| L4 SRC PORT | Integer | Source port |
| L4 DST PORT | Integer | Destination port |
| PROTOCOL | Categorical | Transport protocol (TCP/UDP/ICMP) |
| L7 PROTO | Categorical | Application layer protocol |
| IN BYTES | Integer | Inbound byte count |
| IN PKTS | Integer | Inbound packet count |
| OUT BYTES | Integer | Outbound byte count |
| OUT PKTS | Integer | Outbound packet count |
| TCP FLAGS | Integer | TCP flag bitmap |
| FLOW DURATION MILLISECONDS | Float | Flow duration in ms |
| Label | Categorical | Attack type (multi-class target) |
| Attack | Binary | 0 = benign, 1 = attack |

### Why This Dataset
1. No feature leakage. Features are derived from NetFlow records, not from reconstructed packet captures, avoiding the leakage issues documented in CIC-IDS2017.
2. Unified schema. All four subsets share the same 12 features, enabling genuine cross-subset generalization experiments without feature alignment preprocessing.
3. Multi-class richness. Across subsets, the dataset contains over 15 distinct attack categories with meaningful semantic differences.
4. Recent and credible. Published in IEEE/Springer venues; actively cited in 2023–2025 research.

### Which Subsets to Use
**Instruction to Students**
- Primary training and evaluation: Use NF-UNSW-NB15-v2. It has 9 attack categories, is the most studied, and provides a clean starting point.
- Distribution shift experiment: Train on NF-UNSW-NB15-v2, test on NF-CSE-CIC-IDS2018-v2. The feature schema is identical; only the traffic distribution changes.
- Optional additional shift: Test the same trained model on NF-ToN-IoT-v2 for a second distribution shift condition.
- Do not merge subsets for training. The experimental value lies in treating them as separate domains.

## Data Preprocessing Requirements
You must document and justify every preprocessing decision. The following steps are required:
1. Missing values: Report the percentage of missing values per feature. Justify your imputation strategy (mean, median, mode, or drop).
2. Normalization: Apply min-max or standard scaling. Fit the scaler on the training set only. Never fit on test data.
3. Class imbalance: Report the class distribution. Apply at least one strategy (SMOTE, class-weighted loss, stratified sampling) and justify your choice.
4. Train/validation/test split: 70/10/20 stratified split. Use a fixed random seed and report it.
5. Categorical encoding: Protocol and L7 PROTO must be encoded (one-hot or label encoding). Justify your choice.

### Common Failure Mode - Read Carefully
Fitting the scaler or imputer on the full dataset (including test data) is a data leakage error. It inflates reported performance and invalidates your results. A reviewer will check this. Report your split sizes and confirm the scaler was fit on training data only.

# Methodology
## Component 1: Multi-Class Classifier (Baseline)
Build a multi-class classifier on NF-UNSW-NB15-v2. Your classifier must handle all 9 attack categories plus normal traffic (10-class problem).

**Required models:**
1. Logistic Regression - linear baseline. Sets the lower bound.
2. Random Forest - strong non-linear baseline. Should perform competitively. This is your primary baseline.
3. Your chosen model - at least one additional model selected and justified by your group. Candidates: Gradient Boosting (XGBoost, LightGBM), MLP neural network, SVM with RBF kernel, or k-NN. Do not choose based on expected accuracy alone. Consider training time, interpretability, and suitability for robustness analysis.

**Instruction to Students**
For each model, report:
- Hyperparameters and how they were selected (grid search, random search, or manual tuning with justification)
- Per-class precision, recall, and F1 score
- Macro-averaged and weighted-averaged F1
- Confusion matrix
- Training time and inference time

Do not report only accuracy. Accuracy is misleading under class imbalance.

## Component 2: Systematic Robustness Evaluation
This is the core scientific contribution of the project. You will subject your best classifier from Component 1 to three controlled stress tests. Each test isolates a specific real-world failure mode.

### 5.2.1 Stress Test A: Held-Out Attack Classes (Open-Set Condition)
**Setup:**
1. Select 3 attack categories from NF-UNSW-NB15-v2 to withhold from training. Call these the unknown classes. The remaining classes form the known set.
2. Train the classifier on the known classes only.
3. At test time, present all 10 classes (including the 3 unknown ones).

**What to measure:**
- For known classes: do metrics degrade from the full-training baseline?
- For unknown classes: what does the classifier predict? What confidence does it assign? Which known class does each unknown class most commonly map to, and is there a semantic explanation?

**Report:** confusion matrix (full 10 classes), per-class confidence distribution for unknown-class predictions, and a qualitative analysis of misclassification patterns.

Repeat this experiment with two different choices of held-out classes to test sensitivity to the specific selection.

### Reviewer Expectation
A reviewer expects a qualitative discussion here, not just numbers. Why does the classifier confuse Shellcode with Exploits? Is there a feature-level explanation? If you only report numbers without analysis, the contribution is incomplete.

### 5.2.2 Stress Test B: Cross-Dataset Distribution Shift
**Setup:**
1. Take the classifier trained on NF-UNSW-NB15-v2 (full training set).
2. Evaluate it directly on NF-CSE-CIC-IDS2018-v2 without any retraining. Map class labels to a common binary scheme (attack / normal) for this experiment since the attack categories differ across subsets.
3. Optionally, evaluate on NF-ToN-IoT-v2 as a second target domain.

**What to measure:**
- Binary detection rate (attack vs. normal) on the target domain.
- False positive rate and false negative rate on the target domain.
- Relative degradation compared to in-distribution performance.
- Feature distribution comparison between source and target domain (report at minimum: mean and variance of each feature per domain, or use a statistical divergence measure such as KL divergence or Maximum Mean Discrepancy if feasible).

### 5.2.3 Stress Test C: Feature Degradation
**Setup:** Introduce controlled corruption to the test set of NF-UNSW-NB15-v2. The classifier is not retrained. Test the following conditions independently:

Table 3: Feature degradation conditions.

| Condition | Description | Levels to Test |
| ---- | ---- | ---- |
| Gaussian noise | Add \(N(0, \sigma^2)\) to numeric features | \(\sigma \in\{0.1, 0.5, 1.0\}\) |
| Random masking | Set p% of feature values to 0 | \(p \in\{10\%, 25\%, 50\%\}\) |
| Feature dropout | Remove k features entirely | \(k \in\{2, 4, 6\}\) |

**What to measure:**
- Macro-F1 at each degradation level.
- Plot: performance degradation curve as a function of degradation intensity.
- Identify which features, when dropped, cause the largest performance drop. This serves as an implicit feature importance analysis.

## Component 3: Robustness Improvement Strategies
After characterizing failure modes in Component 2, each group must implement and evaluate at least two of the following strategies. The choice must be justified relative to your findings in Component 2.

### 5.3.1 Strategy 1: Confidence Thresholding with Rejection
Instead of always outputting a class prediction, the classifier rejects low-confidence inputs and flags them for further review.

**Mechanism:**
\[
\hat{y}=
\begin{cases}
arg max _{k} p_{k}(x) & \text{if } max _{k} p_{k}(x) \geq \tau \\
\chi & \text{otherwise}
\end{cases}
\]
where \(p_{k}(x)\) is the predicted probability for class k and τ is a rejection threshold.

**What to evaluate:**
- Sweep τ from 0.5 to 0.99. Plot coverage (fraction not rejected) vs. accuracy on accepted samples.
- On the held-out-class experiment (Stress Test A), report what fraction of unknown-class inputs are correctly rejected.
- Discuss the operating point: what τ is practical in deployment?

### 5.3.2 Strategy 2: Ensemble with Disagreement Detection
Train an ensemble of M classifiers (e.g., \(M=5\) random forests with different seeds, or a mix of \(RF+GBM+MLP\)) . Measure disagreement among ensemble members as a proxy for umcertainty.

**Mechanism:**
\[
disagreement(x)=1-\frac {1}{M} \sum_{i=1}^{M} \mathbb{1}\left[\hat{y}_{i}(x)=\hat{y}_{majority}(x)\right] \tag{3}
\]

Flag inputs with disagreement above a threshold as uncertain.

**What to evaluate:**
- Does high disagreement correlate with misclassification?
- Does disagreement detect unknown-class inputs (Stress Test A)?
- Report: AUROC of disagreement score as a detector of wrong predictions.

### 5.3.3 Strategy 3: One-Class Classifier as Anomaly Fallback
Train a one-class classifier (e.g., Isolation Forest or One-Class SVM) on normal traffic only. Use it as a pre-filter: inputs flagged as anomalous by the one-class model but rejected or uncertain under the multi-class model are routed to an "unknown attack" category.

**Pipeline:**
\[
output(x)=
\begin{cases}
\hat{y}_{multi-class }(x) & \text{if } \hat{y}_{multi-class } \text{ is high confidence}\\
UNKNOWN ATTACK & \text{if OCC flags anomaly but multi-class is uncertain }\\
NORMAL & \text{if OCC confirms normal}
\end{cases} \tag{4}
\]

**What to evaluate:**
- Does adding the OCC improve detection of held-out attack classes?
- What is the false positive rate introduced by the OCC on normal traffic?
- Report: precision-recall curve for unknown attack detection.

### 5.3.4 Strategy 4: Cost-Sensitive Learning
In cybersecurity, false negatives (missed attacks) are more costly than false positives (false alarms). Train a cost-sensitive classifier where the loss function is weighted to penalize false negatives more heavily.

**Mechanism:** Define a cost matrix C where \(C[i][j]\) is the cost of predicting class j when the true class is i. Set \(C[ attack] [ normal ] \gg C[ normal ][ attack]\).

**What to evaluate:**
- Plot the trade-off curve: false negative rate vs. false positive rate as cost weights vary.
- Identify the cost weight that achieves an acceptable operating point.
- Discuss: what cost weight is justifiable in a real deployment context?

### 5.3.5 Strategy 5: Feature Robustness via Augmentation
To improve resilience to feature degradation (Stress Test C), augment the training set with synthetically corrupted samples.

**Mechanism:** For each training sample x, generate augmented variants by:
- Adding Gaussian noise at randomly sampled σ
- Randomly zeroing out \(p \%\) of features

Mix with original samples during training at a specified augmentation ratio

**What to evaluate:**
- Re-run Stress Test C on the augmented-training classifier.
- Plot: degradation curve comparison (original vs. augmented training).
- Check: does augmentation hurt clean performance? Report both.

# Experimental Design and Evaluation
## Metrics
Table 4: Required evaluation metrics by experiment type.

| Experiment | Metrics Required | Level |
| ---- | ---- | ---- |
| Baseline classification | Per-class Precision / Recall / F1, Macro-F1, Weighted-F1, Confusion Matrix | Per-class + Overall |
| Held-out class (Stress A) | Per-class F1 on known classes, Confidence distribution on unknown classes, False-alarm rate on unknown inputs, AUROC of confidence as unknown detector | Per-class + Overall |
| Distribution shift (Stress B) | Binary precision, recall, F1 on target domain; FPR; FNR; Relative degradation (∆F1) | Overall |
| Feature degradation (Stress C) | Macro-F1 at each degradation level; degradation curve | Overall |
| Improvement strategies | Coverage-accuracy curve or precision-recall curve as appropriate; Comparison vs. baseline under each stress condition | Overall |

## Required Baselines
Every group must compare against all three baselines below. Results without baselines will be treated as incomplete.
1. Majority-class classifier. Always predicts the most frequent class. Sets the floor for any useful result.
2. Logistic Regression. Linear model trained on the same features. Separates the contribution of model complexity from feature quality.
3. Random Forest without any robustness strategy. Standard closed-set classifier. This is the baseline that all robustness strategies must be compared against.

## Ablation Study (Required)
An ablation study systematically removes or disables components of your system to verify that each component contributes. You must include at least one ablation.

**Example ablations:**
- Strategy 1 (thresholding) with \(\tau=0\) vs. \(\tau=0.7\) vs. \(\tau=0.9\) - does threshold choice matter?
- Strategy 2 (ensemble) with \(M=1,3,5,10\) - does ensemble size matter?
- Strategy 5 (augmentation) with augmentation ratio \(\in\{0.1,0.3,0.5\}\) - does augmentation ratio matter?

### Reviewer Expectation
An ablation study is not optional. If you claim that your strategy improves robustness, reviewers expect an ablation showing which component of the strategy drives that improvement. Results without ablations are not trustworthy.

## Statistical Significance
Where results are close (e.g., within 1–2% F1), report confidence intervals or perform a McNemar’s test. Do not claim that one method is better than another based on a single run with no variance estimate. Use at least 5 runs with different random seeds and report mean ± standard deviation.

# Deliverables
## Code (Required)
Your code must be submitted as a structured repository with the following modules:
```
project/
├── data/
├── README.md # Download instructions; do NOT commit raw data
├── preprocessing/
│   ├── preprocess.py
│   └── balance.py # Cleaning , encoding , scaling , splitting # Class imbalance handling
├── models/
│   ├── baseline.py # Logistic Regression , Random Forest
│   └── chosen_model.py # Your selected model
├── robustness/
│   ├── stress_tests.py # Stress Test A, B, C implementations
│   └── strategies.py # Improvement strategy implementations
├── evaluation/
│   ├── metrics.py # All metric computation
│   └── plots.py # All figure generation
├── experiments/
│   ├── run_baseline.py # Reproduces Table [baseline]
│   ├── run_stress.py # Reproduces Table [stress tests]
│   └── run_strategies .py # Reproduces Table [improvement strategies]
├── requirements.txt
└── README.md # Setup and reproduction instructions
```

**Instruction to Students**
Every experiment must be reproducible from a single command. A reviewer or grader will run your code. If it cannot be reproduced, the contribution is not verifiable. Set all random seeds. Document all dependencies in requirements.txt.

## Report (Required)
The report must be written in the style of a research paper (IEEE or ACM format, 8–12 pages excluding references). It must contain the following sections in this order:
1. Abstract (150–200 words): problem, method, key findings, contribution.
2. Introduction: motivation, problem statement, summary of contributions.
3. Related Work: minimum 8 references, organized by theme. Not a list of summaries - synthesize how prior work motivates your approach.
4. Dataset and Preprocessing: describe the dataset, your preprocessing decisions, and class distribution statistics.
5. Methodology: describe all three components. Include equations for any non-trivial decisions.
6. Experiments: all tables and figures from Section 6. Every number in a table must be explained in the text.
7. Ablation Study: dedicated subsection.
8. Failure Analysis: a dedicated section discussing cases where your system fails. Do not hide failures. A good failure analysis is a positive signal.
9. Limitations and Future Work: honest discussion of what your system cannot do and why.
10. Conclusion: concise summary. No new results or claims.
11. References: properly formatted, all cited in text.

### Common Failure Mode - Read Carefully
Reports that do not include a Failure Analysis section will be penalized. Every system fails under some conditions. A report that claims otherwise is not credible.

## Prototype (Required)
A minimal command-line interface that demonstrates the full pipeline:
```bash
# Example usage
python predict.py --input sample_flow.csv --strategy confidence_threshold --tau 0.8
```
**Expected output**
```
Flow ID: 42
Predicted class: Reconnaissance
Confidence: 0.91
Decision: ALERT (above threshold)

Flow ID: 77
Predicted class: [UNKNOWN]
Confidence: 0.43
Decision: FLAG FOR REVIEW (below threshold)
```
The prototype does not need a graphical interface. It must accept a CSV of flows, run the classifier and your chosen robustness strategy, and output a decision per flow.

# Project Timeline
Table 5: Recommended week-by-week timeline.

| Week | Milestone | Deliverable |
| --- | --- | --- |
| 1 | Dataset acquisition and EDA | Class distribution plot, feature statistics, missing value report |
| 2 | Preprocessing + Baseline classifier | Preprocessing pipeline, Logistic Regression and Random Forest results |
| 3 | Chosen model + Stress Test A | Held-out class experiment, confusion matrix, confidence distribution |
| 4 | Stress Tests B and C | Distribution shift results, degradation curves |
| 5 | Improvement Strategies | Two strategies implemented and evaluated |
| 6 | Ablation Study + Failure Analysis | Ablation table, written failure analysis |
| 7 | Report finalization + Prototype | Final report, code repository, CLI prototype |

**Instruction to Students**
Do not compress Weeks 1–3 by skipping exploratory data analysis. EDA findings often change the experimental design. Groups that skip EDA typically discover data issues in Week 5 when it is too late to recover.

# What Makes This Project Publication-Ready
A strong project will produce results that are directly relevant to the current state of the literature. The following criteria separate a course project from a publishable contribution:

## Where Your Contribution Lives
1. Systematic robustness characterization. Most NIDS papers do not include any robustness analysis. A paper that clearly shows how a classifier fails under each of the three stress conditions, with quantified results, fills a real gap.
2. Honest failure analysis. Identifying which attack types are chronically confused, and providing a feature-level explanation, is more valuable than reporting high average F1.
3. Cross-dataset generalization. Training on UNSW-NB15 and testing on CICIDS2018 within the NF-UQ-NIDS-v2 framework is a contribution in itself. Most papers never evaluate outside their training distribution.
4. Practical trade-off analysis. A coverage-accuracy or FPR-FNR trade-off curve with a discussion of deployment implications is exactly what security practitioners need and rarely find in academic papers.
5. Reproducible experiments. A code repository with fixed seeds and reproduction instructions is now expected at most venues. Submit it as supplementary material.

### Reviewer Expectation
A reviewer at a venue such as IEEE S&P workshops, RAID, CSET, or ACM CCS workshops will look for:
- A clearly falsifiable research question answered by the experiments.
- Comparison against meaningful baselines (not just each other).
- Ablation study confirming that reported improvements are not accidental.
- A failure analysis that is honest and specific.
- Metrics that are appropriate for imbalanced classification (F1, not accuracy).
- Acknowledgment of dataset limitations.

A paper that satisfies all of these is competitive at workshop level. A paper that does not is likely to be rejected regardless of results.

# Common Failure Modes
Read this section before you begin. These are the most common ways strong groups underperform.

### Common Failure Mode - Read Carefully
1. Reporting accuracy instead of F1 on imbalanced data. With 90% normal traffic, a classifier that predicts "normal" for everything achieves 90% accuracy. This is not a result. Use macro-F1 and per-class recall.
2. Not separating stress test conditions. Running Stress Tests A, B, and C simultaneously confounds the results. Run each stress condition independently.
3. Fitting the scaler on test data. See Section 4. This inflates results and is detectable.
4. Skipping unknown-class analysis in Stress Test A. Reporting only macro-F1 on known classes misses the entire point of the experiment. The question is what happens to the unknown classes.
5. Choosing the held-out classes to maximize the reported result. You must run Stress Test A with at least two different held-out class selections and report both.
6. Claiming a strategy "works" without comparing to the unmodified baseline. Every improvement strategy must be compared to the same classifier without the strategy under the same stress condition.
7. No failure analysis. A report that only shows successes is not credible. Identify at least three specific failure cases and explain them.

# Optional Advanced Extensions
Groups aiming for a top outcome or seeking a summer research continuation may pursue one of the following extensions. These are not required but are strongly encouraged for groups with remaining capacity after completing the core deliverables.
1. Prototypical Networks for Few-Shot Adaptation. When a new attack type appears, can the classifier adapt from 5–10 labeled examples without full retraining? Implement a prototypical network or metric-learning approach and evaluate it on the held-out classes from Stress Test A.
2. Calibration Analysis. Apply temperature scaling or Platt scaling to calibrate classifier confidence. Plot reliability diagrams and measure Expected Calibration Error (ECE) before and after calibration. Evaluate whether calibrated confidence improves rejection performance.
3. Adaptive Thresholding. Instead of a fixed rejection threshold τ, learn a threshold per class or per cluster. Evaluate whether adaptive thresholds outperform a global threshold on the held-out class experiment.
4. Feature Importance Under Shift. Use SHAP values to compare feature importance on the source domain (NF-UNSW-NB15-v2) and the target domain (NF-CSE-CIC-IDS2018-v2). Identify which features generalize across domains and which do not. This is directly actionable for practitioners choosing features for deployment.

# Grading Criteria
Table 6: Project grading breakdown.

| Component | Weight |
| --- | --- |
| Baseline classifier (Component 1): correctness of training, metrics, and analysis | 20% |
| Robustness evaluation (Component 2): all three stress tests, quality of analysis | 30% |
| Improvement strategies (Component 3): implementation, evaluation, comparison | 20% |
| Ablation study and failure analysis | 15% |
| Code quality and reproducibility | 10% |
| Report quality (clarity, structure, academic writing) | 5% |
| Total | 100% |

Bonus credit (up to 10%) is available for any of the advanced extensions in Section 11 that are implemented, evaluated, and reported at the same standard as the core deliverables.

# Final Note to Students
The goal of this project is not a high accuracy number. A classifier that achieves 99% macro-F1 on the clean test set but has never been stress-tested contributes nothing to the field and nothing to a practitioner trying to deploy a real system.

The goal is understanding: where does the classifier break, why does it break, and what can be done about it?

A group that achieves 88% macro-F1 on the clean set, clearly characterizes three failure modes, implements two strategies that demonstrably reduce those failures under specific conditions, and honestly discusses where the strategies fall short - that group has done real research.

That is what this project asks of you.

Questions about the project should be raised in office hours or via email before Week 2. Late questions about project scope will not be accommodated.