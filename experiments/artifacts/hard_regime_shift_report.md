# DeltaCore — Hard Regime-Shift Validation Gate Report

**Date**: 2026-10-07 15:01:08 UTC  
**Benchmark**: `experiments/hard_regime_shift.py`  
**Evaluation Seeds**: 20 seeds (0 to 19)  
**Primary Decision**: **CONDITIONAL PASS**  

---

## 1. Executive Summary & Hard Decision Gate

> **DECISION**: **CONDITIONAL PASS**  
>  
> DeltaCore achieves a modest +0.0403 AUROC gain over Online Centroid. Its advantage is confined to complex correlation breaks and temporal contexts (D <= 256).

---

## 2. Statistical Anomaly Detection on Matched-Marginal Anomalies (Phase D)

| Algorithm | AUROC (Mean ± Std) | 95% Confidence Interval | AUPRC | FPR @ 95% TPR | State Memory | Median Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Centroid** | 0.5218 ± 0.0642 | [0.4916, 0.5505] | 0.2921 | 0.9108 | 512 B | 4.1 µs |
| **Online Centroid (Ungated)** | 0.5384 ± 0.0771 | [0.5010, 0.5696] | 0.2886 | 0.9100 | 512 B | 8.5 µs |
| **Online Centroid (Gated)** | 0.5383 ± 0.0733 | [0.5027, 0.5692] | 0.2813 | 0.8992 | 512 B | 8.0 µs |
| **Robust Huber Centroid** | 0.5396 ± 0.0774 | [0.5025, 0.5707] | 0.2861 | 0.8950 | 512 B | 11.0 µs |
| **Static PCA** | 0.5667 ± 0.0814 | [0.5291, 0.6009] | 0.3180 | 0.8708 | 4.0 KB | 7.2 µs |
| **Online PCA (Ungated)** | 0.5713 ± 0.0821 | [0.5346, 0.6043] | 0.3216 | 0.8767 | 4.0 KB | 28.8 µs |
| **Online PCA (Gated)** | 0.5672 ± 0.0825 | [0.5300, 0.6007] | 0.3187 | 0.8750 | 4.0 KB | 29.3 µs |
| **DeltaCore Continuous** | 0.5802 ± 0.0879 | [0.5395, 0.6140] | 0.3292 | 0.8550 | 64.0 KB | 77.9 µs |
| **DeltaCore Gated** | 0.5787 ± 0.0881 | [0.5376, 0.6130] | 0.3239 | 0.8533 | 64.0 KB | 76.5 µs |

---

## 3. Marginal Distribution Matching Verification

- **Categorical Marginal Matching Status**: PASSED (TVD <= 0.05)
- **Max Categorical TVD**: `0.0000`
- **Latency Mean Difference**: `0.00 ms`

---

## 4. Anomaly Absorption over 100 Consecutive Bursts

| Consecutive Anomalies | DeltaCore Ungated Ratio | DeltaCore Gated Ratio | Online Centroid Ungated | Online Centroid Gated |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 1.000 | 1.000 | 1.000 | 1.000 |
| **5** | 0.892 | 0.892 | 0.815 | 0.815 |
| **10** | 0.772 | 0.772 | 0.630 | 0.630 |
| **25** | 0.498 | 0.498 | 0.292 | 0.292 |
| **50** | 0.236 | 0.236 | 0.081 | 0.081 |
| **100** | 0.052 | 0.052 | 0.006 | 0.006 |

---

## 5. Generated Publication Observatory Figures

- `plot_1_residual_timeline.png`
- `plot_2_drift_convergence.png`
- `plot_3_anomaly_score_distributions.png`
- `plot_4_roc_curves.png`
- `plot_5_pr_curves.png`
- `plot_6_adaptation_delay.png`
- `plot_7_anomaly_absorption.png`
- `plot_8_memory_vs_performance.png`
- `plot_9_latency_vs_performance.png`
- `plot_10_marginal_distribution_validation.png`
- `plot_11_interaction_ablation.png`
- `plot_12_temporal_window_ablation.png`
