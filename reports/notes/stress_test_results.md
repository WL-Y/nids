# Stress Test Results Notes

## Stress Test A: Unknown Attack Classes

### Setup

Two held-out class groups were evaluated:
- **Set 1**: Worms, Analysis, Shellcode (3 rarest classes)
- **Set 2**: Backdoor, DoS, Fuzzers (3 medium-frequency classes)

Model trained on known classes only, evaluated on full test set including unknowns.

### Results

| Held-Out Set | Known Macro-F1 | Unknown Mean Conf | AUROC (Detector) |
|-------------|---------------|-------------------|-------------------|
| Set 1 (rarest) | 0.7037 | 0.5442 | 0.9870 |
| Set 2 (medium) | 0.7913 | 0.6321 | 0.9832 |

### Key Findings

1. **Confidence is an excellent unknown detector**: AUROC > 0.98 across both groups
2. **Exploits is the universal sink**: All unknown classes are primarily misclassified
   as Exploits, the largest attack class (42,748 samples)
3. **Rare classes are harder to detect**: Set 1 mean confidence (0.54) < Set 2 (0.63)
4. **Known-class wrong predictions have moderate confidence**: known_wrong_mean_conf = 0.57–0.58,
   similar to unknown-class confidence (0.54–0.63) and much lower than known-correct
   confidence (0.99)

### Main Failure Modes

- Worms (32 test samples): 69% misclassified as Exploits
- Analysis (245 test samples): 92% misclassified as Exploits
- Backdoor (932 test samples): 66% misclassified as Exploits

## Stress Test B: Cross-Dataset Generalization

### Setup

- **Source**: NF-UNSW-NB15-v3 (training)
- **Target**: NF-CSE-CIC-IDS2018-v3 (full 20,115,529 flows, chunked evaluation)
- **Evaluation**: Binary (Benign vs Attack) due to different attack category systems
- **Method**: Chunked streaming — each chunk is cleaned, transformed via UNSW-fitted preprocessor,
  predicted, and binary TP/TN/FP/FN counts are accumulated without fitting on CICIDS

### Results

| Metric | Value |
|--------|-------|
| n_samples | 20,115,529 |
| Accuracy | 47.45% |
| Binary F1 | 0.2593 |
| Precision | 0.1585 |
| Recall | 0.7113 |
| FPR | 0.5607 |
| FNR | 0.2887 |
| Delta F1 (drop from in-dist) | 0.4181 |

### Top Shifted Features

The largest distribution shifts (by KS statistic) were observed in:
- RETRANSMITTED_OUT_BYTES: UNSW mean=16,793 vs CICIDS mean=39 (KS=0.67)
- RETRANSMITTED_OUT_PKTS: UNSW mean=15.8 vs CICIDS mean=0.06 (KS=0.66)
- DST_TO_SRC_SECOND_BYTES: UNSW mean=135 vs CICIDS mean=15 (KS=0.66)
- RETRANSMITTED_IN_BYTES: UNSW mean=1,581 vs CICIDS mean=30 (KS=0.65)
- RETRANSMITTED_IN_PKTS: UNSW mean=5.8 vs CICIDS mean=0.5 (KS=0.65)

The model fails primarily due to extremely high false positive rate (56%),
indicating that CICIDS normal traffic patterns differ substantially from UNSW.

## Stress Test C: Feature Degradation

### Setup

Three corruption types applied at raw feature level:
- Gaussian noise: sigma in {0.1, 0.5, 1.0} (scaled by feature std)
- Random masking: p in {0.1, 0.25, 0.5}
- Feature dropout: top-k and bottom-k for k in {2, 4, 6}

### Results

| Degradation | Level | Macro-F1 |
|------------|-------|----------|
| Baseline (none) | — | 0.6774 |
| Noise | sigma=0.1 | 0.2688 |
| Noise | sigma=0.5 | 0.2036 |
| Noise | sigma=1.0 | 0.1682 |
| Masking | p=0.1 | 0.6336 |
| Masking | p=0.25 | 0.5214 |
| Masking | p=0.5 | 0.2959 |
| Drop top-2 | ports | 0.5379 |
| Drop top-6 | +byte features | 0.2660 |
| Drop bottom-6 | IAT features | 0.6740 |

### Key Findings

1. Model is **extremely sensitive to Gaussian noise** — even sigma=0.1 causes F1 drop from 0.68 to 0.25
2. Port features (L4_SRC_PORT, L4_DST_PORT) are the most critical
3. IAT min features are almost completely redundant (drop k=6: F1 unchanged at 0.6740)
4. Masking is less destructive than noise at equivalent degradation levels

## Feature-Level Qualitative Analysis

### Which features shift most across domains? (Stress B)

The top-5 most shifted features between UNSW (source) and CICIDS2018 (target) are:

| Feature | UNSW Mean | CICIDS Mean | KS | Interpretation |
|---------|-----------|-------------|-----|----------------|
| RETRANSMITTED_OUT_BYTES | 16,793 | 39 | 0.67 | UNSW has far more retransmission bytes — network conditions differ |
| RETRANSMITTED_OUT_PKTS | 15.8 | 0.06 | 0.66 | Retransmission packet counts nearly zero in CICIDS |
| DST_TO_SRC_SECOND_BYTES | 135 | 15 | 0.66 | Reverse-direction throughput much lower in CICIDS |
| RETRANSMITTED_IN_BYTES | 1,581 | 30 | 0.65 | Same pattern as outbound retransmission |
| RETRANSMITTED_IN_PKTS | 5.8 | 0.5 | 0.65 | Consistently lower retransmission in CICIDS |

The model relies heavily on retransmission and throughput features, which
differ fundamentally between the two collection environments.

### Which features dominate RF importance? (Stress C)

Top-5 most important numeric features (RF feature_importances_):

| Feature | Importance | Drop Effect (k=2) |
|---------|-----------|-------------------|
| L4_DST_PORT | 0.072 | F1 drops to 0.538 |
| L4_SRC_PORT | 0.051 | (included in top-2) |
| IN_BYTES | 0.049 | — |
| MAX_IP_PKT_LEN | 0.047 | — |
| LONGEST_FLOW_PKT | 0.043 | — |

Port numbers are the dominant signal, consistent with port-based attack patterns.
IAT min features are completely redundant (drop k=6: F1 unchanged at 0.674).

### Why do unknown classes map to specific known classes? (Stress A)

All held-out unknown classes are primarily mapped to **Exploits**, the largest
attack class (42,748 samples). This is a frequency-driven "universal sink" effect:

- **Worms → Exploits** (69%): Both involve code propagation; NetFlow-level features
  (byte counts, durations) overlap almost completely.
- **Analysis → Exploits** (92%): Analysis traffic (port scans, probing) has very
  low byte counts, indistinguishable from low-intensity Exploits flows.
- **Shellcode → Exploits/Backdoor**: Shellcode execution and backdoor communication
  share similar port usage patterns (L4_DST_PORT clustering).
- **DoS → Exploits** (76%): DoS volume features overlap with high-intensity Exploits.

This confirms that Exploits has the widest feature coverage and acts as the
default prediction for any traffic pattern the model hasn't explicitly learned.

## Overall Robustness Findings

- **Most challenging condition**: Stress B (cross-dataset) — model fails completely (F1=0.26)
- **Second most challenging**: Stress C Gaussian noise — extreme sensitivity
- **Well-handled**: Stress A unknown detection — AUROC > 0.98 with confidence threshold
- **Recommendation**: Domain adaptation or per-network fine-tuning is essential for deployment
  across different network environments
