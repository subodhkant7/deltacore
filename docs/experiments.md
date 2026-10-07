# DeltaCore: Experimental Progression & Benchmark Archive

**Status**: Authoritative Benchmark & Scientific History Archive  
**Version**: `0.2.0`  
**Date**: 2026-10-07  

This document catalogs the complete experimental trajectory of the DeltaCore project, recording all benchmark methodologies, reproduction commands, artifact locations, and empirical findings.

---

## Experimental Trajectory Overview

```mermaid
flowchart TD
    P[Phases 1-18: Core Adaptive Architecture] --> STAB[Stabilization: Canonical API & Telemetry Bridge]
    STAB --> EXP1[Experiment 1: Auto-Associative Regime Shift]
    EXP1 --> EXP2[Experiment 2: Drift-Then-Anomaly]
    EXP2 --> EXP3[Experiment 3: Hard Matched-Marginal Anomalies]
    EXP3 --> EXP4[Experiment 4: Higher-Order Baseline Gate]
    EXP4 --> DEC[Final Decision: Hypothesis Falsified / Research Frozen]
```

---

## 1. Phases 1–18: Core Adaptive-State Research (Foundational)

- **Purpose**: Implement pure-PyTorch mathematical primitives for test-time adaptive associative neural memory ($M_t$), stability controllers (Lyapunov contractive step-size), Hebbian and Delta update rules, parallel affine scans, and 2D classification transfer.
- **Key Result**: Validated local contractive step stability $|1 - \eta_t \|x_t\|^2| \le 1$, deterministic parameter immutability, causal state adaptation, and $4D^2$ byte memory scaling.
- **Tests**: 677 unit tests covering autograd backward passes, numerical edge cases, and affine scan equivalence.

---

## 2. Pre-RecoveryOS Stabilization Phase

- **Purpose**: Establish canonical, frozen public API (`AdaptiveController`, `ControllerConfig`, `DeterministicFeatureHasher`, `save_state`/`load_state`) with atomic state persistence and corruption detection.
- **Verification**: Verified pre-update residual semantics (`score()` non-mutating), state contract, and SHA-256 state hashing.
- **Artifact**: `examples/basic_adaptation.py`.

---

## 3. Experiment 1: Auto-Associative Regime Shift

- **Script**: `experiments/auto_associative_regime_shift.py`
- **Purpose**: First evaluation of auto-associative controller $M_t x_t \approx x_t$ on microservice telemetry under non-stationary regime shift.
- **Key Finding**: DeltaCore adapted to continuous drift and detected severe anomalies. However, baseline comparisons were limited to simple static and online centroids, leaving open whether second-order covariance methods could achieve the same performance.

---

## 4. Experiment 2: Drift-Then-Anomaly Benchmark

- **Script**: `experiments/drift_then_anomaly.py`
- **Purpose**: Evaluate adaptation to gradual distribution drift followed by later anomaly injection across 10 evaluation seeds.
- **Baseline Models**: Static Centroid, Online Centroid, Robust Huber Centroid, Static PCA, Online PCA, DeltaCore Continuous, DeltaCore Gated.
- **Key Finding**: DeltaCore Gated outperformed Online Centroid on synthetic correlation breaks. However, marginal distributions between nominal and anomaly streams were partially conflated due to synthetic token generation.

---

## 5. Experiment 3: Hard Matched-Marginal Regime Shift Benchmark

- **Script**: `experiments/hard_regime_shift.py`
- **Purpose**: Correct benchmark validity defects by requiring matched marginal distributions between nominal and anomalous regimes to prevent token frequency shortcuts.
- **Key Finding**: Confirmed that DeltaCore detects correlation violations without token novelty. However, the evaluation lacked a proper second-order adaptive baseline (Online Covariance / Mahalanobis distance) and an independent held-out Phase-C nominal reference.

---

## 6. Experiment 4: Higher-Order Baseline & Evaluation Gate (Decisive Gate)

- **Script**: `experiments/higher_order_regime_shift.py` (`v3.0.0`)
- **Pre-Registered Config**: `experiments/configs/higher_order_gate.yaml`
- **JSON Artifact**: `experiments/artifacts/higher_order_regime_shift_results.json`
- **Report Artifact**: `experiments/artifacts/higher_order_regime_shift_report.md`
- **Plots Directory**: `experiments/artifacts/higher_order_regime_shift_plots/` (15 publication figures)
- **Execution Command**:
  ```bash
  python experiments/higher_order_regime_shift.py
  ```

### Design & Methodology
1. **Decoupled Reference Distribution**: An independent, held-out Phase-C reference ($N = 1,000$) generated using a dedicated calibration PRNG.
2. **Strict Marginal Matching**: Tested 5 hard anomaly families (H1: Correlation Break, H2: Higher-Order 3-Way XOR Parity, H3: Markov Temporal Inversion, H4: Operational Contradiction, H5: Conditional Shift), all verified to satisfy $\text{TVD} \le 0.05$ against held-out Phase C.
3. **Proper Second-Order Baseline**: Implemented **Online Covariance with regularized Mahalanobis distance** ($\Sigma_t = \beta \Sigma_{t-1} + (1-\beta)(x-\mu)(x-\mu)^T$, $\lambda=10^{-3}$, $d_t = \sqrt{(x-\mu)^T(\Sigma+\lambda I)^{-1}(x-\mu)}$) consuming the same $4D^2$ state memory.
4. **Blind Scoring Safeguard**: Scoring loop executed strictly without access to test labels, validated by regression test `tests/test_benchmark_blind_execution.py`.
5. **Rigorous Paired Statistics**: 20 independent paired seeds with paired percentile bootstrap 95% CIs ($B = 10,000$, seed 42), exact two-sided binomial sign test, and exact paired sign-flip randomization test evaluating all $2^{20} = 1{,}048{,}576$ sign configurations.

### Decisive Empirical Results (20 Seeds)

| Comparison Metric | Value | Statistical Significance |
| :--- | :---: | :--- |
| **DeltaCore Gated Mean AUROC** | 0.5711 $\pm$ 0.0801 | 95% CI: [0.5365, 0.6071] |
| **Online Covariance Mean AUROC** | **0.5976 $\pm$ 0.0667** | 95% CI: [0.5697, 0.6271] |
| **Mean Paired Difference ($\text{DC} - \text{Cov}$)** | **-0.0265** | 95% Bootstrap CI: **[-0.0478, -0.0072]** (excludes 0) |
| **Paired Cohen's $d$** | **-0.58** | Medium negative effect size |
| **Exact Two-Sided Binomial Sign Test** | **$p = 0.0118$** | DeltaCore won on only 4 of 20 seeds ($p < 0.05$) |
| **Exact Paired Sign-Flip Randomization Test** | **$p = 0.0188$** | Evaluated all $2^{20} = 1{,}048{,}576$ sign configurations ($19,728$ extreme, $p = 0.0188$) |
| **DeltaCore vs. Gated Online Centroid** | +0.0382 | 95% CI: [+0.0205, +0.0541], $p_{\text{sign}} = 0.0004$ |
| **DeltaCore vs. Gated Online PCA** | +0.0070 | 95% CI: [-0.0074, 0.0209], $p = 0.8238$ |

### Canonical Benchmark Conclusion

Under the 20-seed non-stationary matched-marginal benchmark, the observed paired difference favored regularized Online Covariance. The exact two-sided sign test rejected its 50/50 directional null at p = 0.0118, and the exact paired sign-flip randomization test rejected its stated exchangeability-of-signs null at p = 0.0188. These tests address different inferential questions and are reported separately; the two p-values are not treated as independent confirmations of a single omnibus hypothesis. DeltaCore Gated did not demonstrate a performance advantage over regularized Online Covariance on the decisive benchmark.

### Multiplicity and Inferential Roles

| Procedure | Question | Null / Framework | Result |
| :--- | :--- | :--- | :--- |
| **Paired percentile bootstrap** | Mean-effect uncertainty | Resampling framework ($B = 10{,}000$, seed 42) | 95% CI `[-0.0478, -0.0072]` |
| **Exact paired sign-flip test** | Paired-effect magnitude | Exchangeability-of-signs null ($2^{20} = 1{,}048{,}576$ assignments) | `p = 0.0188` ($19{,}728$ extreme) |
| **Exact two-sided sign test** | Direction / majority | 50/50 sign null ($H_0: p = 0.5$) | `p = 0.0118` (4 wins / 16 losses) |

Exhaustive enumeration removes Monte Carlo approximation error from the sign-flip calculation; it does not make the inferential procedure assumption-free. The percentile bootstrap interval is a resampling-based uncertainty estimate and is not treated as an exact confidence interval.

**Multiplicity and inferential roles.** The sign test and paired sign-flip test are separately defined inferential procedures addressing different questions. The exact two-sided sign test evaluates directional imbalance under a 50/50 sign null, while the exact paired sign-flip randomization test evaluates the magnitude of the paired mean difference under the stated exchangeability-of-signs null. They are not independent replications of a single hypothesis test. Their p-values are therefore reported separately for their respective null hypotheses and are not interpreted as a combined family-wise-error-controlled omnibus result.

The paired percentile bootstrap serves a different purpose: it provides a resampling-based uncertainty interval for the observed mean paired difference. It is not treated as a third hypothesis test.

### Decisive Benchmark Identity

```text
Benchmark implementation:
experiments/higher_order_regime_shift.py

Benchmark version:
v3.0.0

Streams:
20 independent paired streams

Seeds:
0 through 19

Feature dimension:
D = 128

Feature representation:
unary + pairs + triples + lag

Marginal-invariance constraint:
Categorical TVD <= 0.05 vs Phase C
```

**Git Provenance**:
- Benchmark implementation introduced: commit `deda3eaaef1ebdeb3171198bce2f50070e38f1df` (*"experiments: complete higher-order baseline and evaluation-correction gate"*).
- Exact statistical validation & research freeze: commit `0ac487bf5d85be17267bd97133a5bef55c2e4767` (*"release: close research freeze with exact statistical validation"*).

### Scope of Inference

The evidence is benchmark-specific. It establishes that, on the specified 20-seed non-stationary matched-marginal benchmark with D=128 and the stated feature representation, DeltaCore Gated did not demonstrate an advantage over regularized Online Covariance, while the observed paired difference favored Online Covariance.

The benchmark does not establish universal superiority of Online Covariance, universal inferiority of DeltaCore, or performance ordering under telemetry distributions, dimensions, drift processes, workloads, or deployment conditions not represented by the benchmark.

The result supports terminating this specific DeltaCore research direction for RecoveryOS; it does not establish that adaptive telemetry detection as a broader problem has been solved.

### Practical vs. Statistical Significance

- **Statistical Evidence**: Mean paired AUROC difference = $-0.0265$, Paired Cohen's $d = -0.58$, exact sign $p = 0.0118$, exact randomization $p = 0.0188$, 95% paired bootstrap CI $[-0.0478, -0.0072]$. These metrics quantify the observed benchmark effect and should not be converted into a universal production-performance claim.
- **Engineering Trade-Offs**: DeltaCore incurred $O(D^2)$ state memory ($64\text{ KB}$ at $D=128$) and $8.6\times$ higher latency ($25.9\text{ µs}$ vs $3.0\text{ µs}$) than first-order centroids. Within the evaluated design and benchmark, DeltaCore incurred greater state and latency cost without demonstrating a compensating performance advantage over regularized online covariance.

### Historical Metric Isolation

The values 0.9859 and 0.9594 are Phase 16/17 balanced classification accuracy metrics and are not part of the telemetry regime-detection AUROC record.

---

## 7. Negative Findings Preservation

In accordance with scientific integrity rules, all negative findings are formally preserved:
1. **DeltaCore did not demonstrate an empirical advantage over regularized Online Covariance** on the decisive matched-marginal benchmark.
2. **DeltaCore suffers from anomaly absorption** under continuous adaptation unless external score-gating is enforced.
3. **DeltaCore cannot detect temporal anomalies** from event-only representations without explicit lag features.
4. **Representation interactions (pairs/triples)** explain the majority of cross-feature anomaly detection capability, benefiting classical statistical methods equally or more.
