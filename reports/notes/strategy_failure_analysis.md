# Strategy, Ablation, and Failure Analysis Notes

## Failure Case 1: Worms — Very Small Support, Metrics Are Unstable
- **Condition**: Extreme class imbalance (158 training / 32 test samples, 0.007%)
- **Key point**: Worms has very small support, so its per-class metrics are
  unstable and should not be overinterpreted. Although the model achieves
  reasonable F1 on this specific split, the estimate is based on too few
  examples to support a strong conclusion — recall can swing dramatically
  between 0 and 1 across different splits.
- **Ablation C cross-check**: Without class_weight, Worms recall drops to 0.41;
  class_weight raises it to 0.63. This confirms the metric is fragile and
  highly sensitive to both the training configuration and the random split.
- **Ablation C shows**: Without class_weight, Worms recall drops to 0.41;
  class_weight raises it to 0.63. This improvement is real but fragile.
- **Strategy effect**: Strategy 1 (tau=0.99) rejects most low-confidence
  Worms predictions, providing a safety net for deployment.
- **Improvement direction**: Few-shot learning or one-class classifiers are
  more appropriate than standard supervised methods for classes this rare.
- **Practical significance**: Worms propagation patterns are evolving;
  traditional NetFlow features may not capture modern worm behavior.

## Failure Case 2: Strategy 1 Coverage Collapse at High Tau
- **Condition**: tau sweep under strict Stress A (known-only training per held-out group)

  Tau sweep summary (strict open-set):
  | held_out | tau | coverage | unknown_rejection | known_false_rej |
  |----------|-----|----------|-------------------|-----------------|
  | Group 1 (rare) | 0.5 | 0.9807 | 0.7437 | 0.0182 |
  | Group 1 (rare) | 0.7 | 0.9657 | 0.9575 | 0.0328 |
  | Group 1 (rare) | 0.85 | 0.9572 | 0.9947 | 0.0413 |
  | Group 1 (rare) | 0.9 | 0.9530 | 0.9987 | 0.0455 |
  | Group 1 (rare) | 0.95 | 0.9492 | 1.0000 | 0.0493 |
  | Group 2 (medium) | 0.5 | 0.9873 | 0.4508 | 0.0043 |
  | Group 2 (medium) | 0.7 | 0.9754 | 0.8114 | 0.0095 |
  | Group 2 (medium) | 0.85 | 0.9695 | 0.9032 | 0.0138 |
  | Group 2 (medium) | 0.9 | 0.9673 | 0.9226 | 0.0156 |
  | Group 2 (medium) | 0.95 | 0.9642 | 0.9476 | 0.0183 |

- **Finding**: Under strict open-set conditions, Group 1 (rarest classes) shows
  higher unknown rejection at each tau compared to Group 2, because rare-class
  features are less distinguishable from known classes. tau=0.85 already achieves
  90-99% unknown rejection with 96-97% coverage.
- **Recommendation**: The operational tau depends on the security posture.
  - High-security: tau=0.90, unknown rejection 92-100%, coverage 95-97%
  - Balanced: tau=0.85, unknown rejection 90-99%, coverage 96-97%

## Failure Case 3: Stress B Cross-Dataset Recall Collapse
- **Condition**: Model trained on UNSW, tested on full CICIDS2018 (20,115,529 flows, chunked evaluation)
- **Binary F1 on full CICIDS**: 0.2593
- **FPR (Benign -> Alert)**: 0.5607 (56% of normal traffic flagged!)
- **FNR (Attack -> Normal)**: 0.2887
- **Delta F1 (degradation)**: 0.4181
- **Feature-level explanation**: The largest shifts are in retransmission and
  reverse-throughput features: RETRANSMITTED_OUT_BYTES (UNSW=16,793 vs CICIDS=39,
  KS=0.67), RETRANSMITTED_OUT_PKTS (15.8 vs 0.06, KS=0.66), DST_TO_SRC_SECOND_BYTES
  (135 vs 15, KS=0.66), RETRANSMITTED_IN_BYTES (1,581 vs 30), RETRANSMITTED_IN_PKTS
  (5.8 vs 0.5). These features differ strongly between UNSW and CICIDS2018,
  explaining the high false-positive rate and poor cross-dataset F1.
- **Conclusion**: Pure feature-level classifiers cannot solve domain shift.
  Domain adaptation methods or per-network retraining is required for deployment.

## Failure Case 4: SMOTE Limitations for Extremely Rare Classes
- **Condition**: Worms (158 samples), k_neighbors=1

  Imbalance method comparison:
  | method | Worms_recall | Analysis_recall | Shellcode_recall | macro_f1 |
  |--------|-------------|----------------|-----------------|----------|
  | none | 0.4062 | 0.3347 | 0.5147 | 0.5931 |
  | class_weight | 0.6250 | 0.9388 | 0.7290 | 0.6363 |
  | SMOTE | 0.5938 | 0.9388 | 0.7332 | 0.6154 |

- **Finding**: SMOTE's synthetic samples for Worms have very low diversity
  because they are generated from only 158 real examples. The synthetic
  samples do not provide meaningful new information for the classifier.
- **Recommendation**: SMOTE is not recommended for classes with fewer than
  200 samples. Use class_weight or other methods instead.

## Strategy 1: Confidence Thresholding
- **Selected tau**: 0.99 (from validation set)
- **Coverage**: 94.6% at tau=0.99
- **Unknown rejection rate**: 97-100% across both Stress A groups
- **Known false rejection**: 3.6-5.4%
- **Saved config**: artifacts/strategy_config.json

## Strategy 2: Ensemble Disagreement
- **Ensemble size**: M=5 (3xRF + 1xXGBoost + 1xLR)
- **Majority vote Macro-F1**: 0.6674
- **Disagreement AUROC**: 0.8529
- **Key insight**: When disagreement >= 0.4, accuracy drops from 99% to <35%.
  High disagreement is a reliable signal of likely-wrong predictions.
- **Unknown class disagreement**: 0.18-0.19 (elevated vs known-class disagreement of ~0.004-0.006)
- **Models saved**: artifacts/ensemble_models/

### Strategy 2 under Full Stress B

On the full CICIDS target domain (20,115,529 flows, single-pass chunked evaluation),
the five-member ensemble produced nearly identical performance to the baseline
Random Forest:

| Metric | Baseline RF | Strategy 2 Ensemble |
|--------|------------|-------------------|
| F1 | 0.2593 | 0.2588 |
| Precision | 0.1585 | 0.1582 |
| Recall | 0.7113 | 0.7103 |
| FPR | 0.5607 | 0.5613 |
| FNR | 0.2887 | 0.2897 |

This indicates that majority voting alone does not mitigate cross-dataset
distribution shift. Since all ensemble members were trained only on the UNSW
source domain, they shared similar source-domain biases and misclassified many
CICIDS benign flows as attacks. Strategy 2 is therefore more useful as an
uncertainty/disagreement analysis tool than as a direct domain-shift correction
method.

## Overall Strategy Conclusions (for final report)

### Stress B: Full CICIDS cross-dataset evaluation
- The UNSW-trained model was evaluated on the full NF-CSE-CIC-IDS2018-v3 target
  domain (20,115,529 flows) using chunked evaluation without retraining.
- Strategy 2 majority vote F1 (0.2588) is nearly identical to baseline (0.2593).
  Ensemble disagreement provides no meaningful cross-domain improvement.
- Root cause: feature/traffic distribution shift. Confidence and ensembling
  cannot compensate for fundamental domain mismatch. Domain adaptation methods
  or per-network retraining are required for deployment.

### Stress C: Rejection-based selective prediction, not robustness
- Strategy 1's high accepted-only F1 comes from rejecting most noisy predictions,
  drastically reducing coverage. This should be reported as "rejection improves
  accepted-only accuracy at the cost of coverage" — not as improved robustness.

### Metric convention
- Primary metric throughout: Macro-F1. With 94.6% Benign, Weighted-F1 is
  dominated by the majority class and gives misleading scores (~0.98).
  Weighted-F1 is supplementary only.

## Ablation Summary
- Ablation A: Threshold tau sweep (strict open-set) in results/ablation_a_threshold_sensitivity.csv
- Ablation B: Ensemble size sweep results in results/ablation_b_ensemble_size.csv
- Ablation C: Imbalance method comparison in results/ablation_c_imbalance_methods.csv
