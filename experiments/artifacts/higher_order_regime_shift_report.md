# DeltaCore — Higher-Order Baseline & Evaluation-Correction Gate Report

**Date**: 2026-10-07 15:42:28 UTC  
**Benchmark**: `experiments/higher_order_regime_shift.py`  
**Evaluation Seeds**: 20 seeds (0 to 19)  
**Primary Decision**: **FAIL**  

---

## 1. Executive Summary & Hard Decision Gate

> **DECISION**: **FAIL**  
>  
> Online Covariance achieved a higher mean AUROC than DeltaCore Gated on the 20 paired evaluation streams. The mean paired difference, defined as d_i = AUROC_DeltaCore,i - AUROC_Covariance,i, was -0.0265, with a 95% paired bootstrap confidence interval of [-0.0479, -0.0071]. Because this interval excludes zero, the data provide evidence that the mean AUROC difference is negative under the specified benchmark and resampling procedure. The corresponding two-sided sign-test p-value was 0.4000, so the experiment does not provide evidence that DeltaCore loses on a majority of individual seeds. These are different statistical questions and should not be conflated.

---

## 2. Statistical Anomaly Detection & Operating Thresholds

| Algorithm | AUROC (Mean ± Std) | 95% Bootstrap CI | AUPRC | Operational TPR @ tau | Operational FPR @ tau | Operational F1 | State Memory | Median Score Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Centroid** | 0.5753 ± 0.0605 | [0.5497, 0.6025] | 0.3356 | 0.110 | 0.062 | 0.161 | 512 B | 3.7 µs |
| **Online Centroid (Ungated)** | 0.5315 ± 0.0802 | [0.4992, 0.5690] | 0.2953 | 0.118 | 0.106 | 0.157 | 512 B | 3.0 µs |
| **Online Centroid (Gated)** | 0.5330 ± 0.0813 | [0.4984, 0.5695] | 0.2990 | 0.178 | 0.146 | 0.209 | 512 B | 3.0 µs |
| **Robust Huber Centroid** | 0.5351 ± 0.0792 | [0.5025, 0.5728] | 0.2972 | 0.003 | 0.000 | 0.005 | 512 B | 3.0 µs |
| **Static PCA** | 0.5590 ± 0.0895 | [0.5218, 0.5980] | 0.3329 | 0.208 | 0.131 | 0.255 | 4.0 KB | 6.8 µs |
| **Online PCA (Ungated)** | 0.5658 ± 0.0911 | [0.5266, 0.6060] | 0.3351 | 0.190 | 0.111 | 0.248 | 4.0 KB | 5.9 µs |
| **Online PCA (Gated)** | 0.5641 ± 0.0903 | [0.5257, 0.6042] | 0.3339 | 0.202 | 0.124 | 0.254 | 4.0 KB | 6.9 µs |
| **Online Covariance (Mahalanobis)** | 0.5976 ± 0.0667 | [0.5697, 0.6271] | 0.3491 | 0.997 | 1.000 | 0.399 | 64.0 KB | 44.3 µs |
| **Pairwise Correlation** | 0.5379 ± 0.0746 | [0.5079, 0.5707] | 0.2978 | 0.480 | 0.439 | 0.334 | 64.0 KB | 13.0 µs |
| **DeltaCore Continuous** | 0.5709 ± 0.0808 | [0.5362, 0.6080] | 0.3332 | 0.000 | 0.000 | 0.000 | 64.0 KB | 27.8 µs |
| **DeltaCore Gated** | 0.5711 ± 0.0801 | [0.5364, 0.6079] | 0.3315 | 0.000 | 0.000 | 0.000 | 64.0 KB | 25.9 µs |

---

## 3. Paired Statistical Comparison Against DeltaCore Gated

| Comparator Model | Mean Paired Difference | 95% Bootstrap CI | Paired Cohen's d | Sign Test p-value |
| :--- | :---: | :---: | :---: | :---: |
| **DeltaCore Gated vs Static Centroid** | -0.0041 | [-0.0423, 0.0316] | -0.05 | 0.9000 |
| **DeltaCore Gated vs Online Centroid (Ungated)** | +0.0396 | [0.0236, 0.0566] | 1.03 | 0.3000 |
| **DeltaCore Gated vs Online Centroid (Gated)** | +0.0382 | [0.0215, 0.0551] | 1.01 | 0.2000 |
| **DeltaCore Gated vs Robust Huber Centroid** | +0.0360 | [0.0221, 0.0511] | 1.11 | 0.3000 |
| **DeltaCore Gated vs Static PCA** | +0.0121 | [-0.0053, 0.0277] | 0.33 | 0.6000 |
| **DeltaCore Gated vs Online PCA (Ungated)** | +0.0053 | [-0.0091, 0.0190] | 0.17 | 0.9000 |
| **DeltaCore Gated vs Online PCA (Gated)** | +0.0070 | [-0.0079, 0.0205] | 0.22 | 0.9000 |
| **DeltaCore Gated vs Online Covariance (Mahalanobis)** | -0.0265 | [-0.0479, -0.0071] | -0.58 | 0.4000 |
| **DeltaCore Gated vs Pairwise Correlation** | +0.0332 | [-0.0014, 0.0765] | 0.39 | 0.6000 |
| **DeltaCore Gated vs DeltaCore Continuous** | +0.0002 | [-0.0009, 0.0013] | 0.09 | 0.9333 |

### Statistical Interpretation of Paired Differences

> **Online Covariance outperformed DeltaCore in mean paired AUROC by 0.0265 points (DeltaCore − Covariance), with a 95% paired bootstrap confidence interval of [−0.0479, −0.0071]. The interval excludes zero, indicating evidence for a negative mean difference. However, the two-sided sign test was not significant (p = 0.4000), so the experiment does not establish a majority-of-seeds loss.**

- **DeltaCore vs. Online PCA (Gated)**: DeltaCore had a slightly higher mean AUROC than Gated Online PCA (+0.0070), but the 95% paired bootstrap CI included zero ([-0.0079, +0.0205]) and the sign test was non-significant (p = 0.9000). Therefore, the experiment does not establish a reliable performance difference between the two methods.
- **DeltaCore vs. Robust Huber Centroid**: DeltaCore had a higher mean paired AUROC than the Robust Huber Centroid by +0.0360, with a 95% paired bootstrap CI entirely above zero ([+0.0221, +0.0511]). However, the sign test was non-significant (p = 0.3000), so the evidence supports a positive mean performance difference but does not establish that DeltaCore wins on a majority of individual seeds.
- **DeltaCore vs. Gated Online Centroid**: DeltaCore had a higher mean paired AUROC than Gated Online Centroid by +0.0382, with a 95% paired bootstrap CI excluding zero ([+0.0215, +0.0551]). The sign test was non-significant (p = 0.2000), so the result should be interpreted as evidence for a higher mean AUROC rather than proof of a majority-of-seeds win.
- **DeltaCore vs. Static Centroid**: DeltaCore's mean paired AUROC was 0.0041 lower than Static Centroid, but the 95% paired bootstrap CI included zero ([-0.0423, +0.0316]) and the sign test was non-significant (p = 0.9000). No reliable difference was established.

---

## 4. Generator Integrity & Matched Marginals (vs Held-out Phase C)

| Anomaly Family | Max Categorical TVD | TVD <= 0.05 Passed | 3-Way Parity Divergence |
| :--- | :---: | :---: | :---: |
| **H1_correlation_break** | 0.0470 | YES | 0.0000 |
| **H2_higher_order_parity** | 0.0460 | YES | 1.0000 |
| **H3_temporal_sequence** | 0.0490 | YES | 0.0000 |
| **H4_operational_contradiction** | 0.0490 | YES | 0.0000 |
| **H5_conditional_shift** | 0.0460 | YES | 0.0000 |

---

## 5. Representation Ablation

| Representation Mode | DeltaCore Gated AUROC | Online Covariance AUROC | Online Centroid AUROC |
| :--- | :---: | :---: | :---: |
| **UNARY** | 0.5793 | 0.5732 | 0.5201 |
| **PAIRS** | 0.5822 | 0.5828 | 0.5421 |
| **FULL** | 0.6153 | 0.6315 | 0.5657 |
| **LAG** | 0.6028 | 0.5765 | 0.5470 |

---

## 6. RecoveryOS Integration Decision

> **The statistical evidence does not justify DeltaCore as the preferred telemetry detector.** Online Covariance produced a higher mean paired AUROC, and the paired bootstrap confidence interval for the mean difference excluded zero. At the same time, the non-significant sign test indicates that this result should not be characterized as a uniform per-seed failure of DeltaCore. Combined with the substantially higher state cost and the absence of a demonstrated operational-threshold advantage, the current evidence is insufficient to justify integrating DeltaCore into RecoveryOS telemetry regime detection.

---

## 7. Generated Publication Figures

- `plot_1_phase_residual_timeline.png`
- `plot_2_drift_adaptation_trajectories.png`
- `plot_3_frozen_threshold_operating_points.png`
- `plot_4_auroc_comparison.png`
- `plot_5_auprc_comparison.png`
- `plot_6_adaptation_delay_comparison.png`
- `plot_7_anomaly_absorption_decay.png`
- `plot_8_aba_hysteresis_trajectories.png`
- `plot_9_memory_vs_performance.png`
- `plot_10_latency_vs_performance.png`
- `plot_11_anomaly_marginal_validation.png`
- `plot_12_representation_ablation.png`
- `plot_13_second_order_baseline_comparison.png`
- `plot_14_higher_order_anomaly_comparison.png`
- `plot_15_temporal_sequence_comparison.png`
