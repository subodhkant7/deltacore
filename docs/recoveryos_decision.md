# RecoveryOS Architectural Decision Record: DeltaCore Telemetry Evaluation

**Status**: **REJECTED / FROZEN**  
**Component**: DeltaCore Adaptive Associative Controller  
**Evaluated Application**: RecoveryOS Telemetry Regime Detection & Operational Anomaly Tracking  
**Date**: 2026-10-07  
**Decision Authority**: DeltaCore Research & Evaluation Gate Committee  

---

## 1. Context & Purpose

The central research hypothesis evaluated for DeltaCore prior to RecoveryOS integration was:
> *Can DeltaCore's test-time adaptive associative matrix $M_t \in \mathbb{R}^{D \times D}$ provide a decisive, statistically reliable advantage over simpler first- and second-order online statistical baselines when tracking legitimate distribution drift and detecting higher-order operational anomalies?*

This evaluation served as the final falsification gate before any RecoveryOS runtime integration or telemetry dependency could be considered.

---

## 2. Decision

$$\boxed{\textbf{DeltaCore is PERMANENTLY REJECTED & FROZEN for RecoveryOS Telemetry Regime Detection.}}$$

**DeltaCore will NOT be integrated into RecoveryOS.**  
The telemetry integration hypothesis has been tested on the decisive benchmark, rejected, and frozen.

The RecoveryOS integration was rejected because DeltaCore did not demonstrate a sufficient empirical advantage over a simpler regularized online covariance baseline on the decisive benchmark, while incurring greater state complexity and latency. DeltaCore is therefore removed as an active RecoveryOS research dependency.

This is an **engineering use-case decision** based on empirical benchmark results. It does not imply that DeltaCore has no theoretical value as an experimental implementation of adaptive neural associative memory, nor that adaptive telemetry detection as a broader problem has been solved; rather, it establishes that within the evaluated design and benchmark, DeltaCore is not justified as a telemetry anomaly or regime detector in RecoveryOS.

---

## 3. Empirical Reasons for Rejection

The decision is grounded in the findings of the final multi-seed paired benchmark (`experiments/higher_order_regime_shift.py` v3.0.0, 20 independent seeds, matched categorical marginals $\text{TVD} \le 0.05$):

### A. Performance on the Decisive Benchmark
When compared against a proper online second-order baseline (**Online Covariance with regularized Mahalanobis distance**, $4D^2$ state):
- **Under the 20-seed non-stationary matched-marginal benchmark, the observed paired difference favored regularized Online Covariance. The exact two-sided sign test rejected its 50/50 directional null at p = 0.0118, and the exact paired sign-flip randomization test rejected its stated exchangeability-of-signs null at p = 0.0188.**
- **These tests address different inferential questions and are reported separately; the two p-values are not treated as independent confirmations of a single omnibus hypothesis.**
- **DeltaCore Gated did not demonstrate a performance advantage over regularized Online Covariance on the decisive benchmark.**
- **Online Covariance** achieved higher mean discriminative accuracy (**$0.5976 \pm 0.0667$** AUROC) than **DeltaCore Gated** (**$0.5711 \pm 0.0801$** AUROC).
- The mean paired difference ($\text{DeltaCore} - \text{Covariance}$) was **$-0.0265$**, with a **95% paired bootstrap confidence interval of $[-0.0478, -0.0072]$** strictly excluding zero. (Note: the percentile bootstrap interval is a resampling-based uncertainty estimate and is not treated as an exact confidence interval.)
- The exact two-sided binomial sign test yielded **$p = 0.0118$** (DeltaCore won on 4 of 20 evaluation seeds, lost on 16).
- The exact paired sign-flip randomization test yielded **$p = 0.0188$** (evaluating all $2^{20} = 1{,}048{,}576$ sign assignments under the exchangeability null). Exhaustive enumeration removes Monte Carlo approximation error from the sign-flip calculation; it does not make the inferential procedure assumption-free.
- Online Covariance achieved higher benchmark AUROC on this evaluation without requiring non-standard auto-associative controller dynamics.

### B. Computational and Memory Overhead is Unjustified
- DeltaCore requires $4 D^2$ bytes of persistent state memory ($64.0\text{ KB}$ at $D = 128$, $1\text{ MB}$ at $D = 512$, $4\text{ MB}$ at $D = 1024$).
- Standard first-order centroids require only $4 D$ bytes ($512\text{ B}$ at $D = 128$, $128\times$ smaller) and execute in $3.0\text{ µs}$ ($8.6\times$ faster than DeltaCore's $25.9\text{ µs}$).
- While DeltaCore improved upon first-order centroids by $+0.0382$ AUROC, that advantage was entirely matched by standard second-order covariance estimation.
- Within the evaluated design and benchmark, DeltaCore incurred greater state and latency cost without demonstrating a compensating performance advantage over regularized online covariance.

### C. Feature Engineering Explains Higher-Order Performance
Representation tier ablations demonstrated that gains on complex cross-token anomalies were driven primarily by the feature hashing representation (token pairs and triples) rather than DeltaCore's associative matrix:
- On full representations, Online Covariance reached $0.6315$ AUROC vs. DeltaCore's $0.6153$ AUROC.
- Adding pair and triple interaction features benefited classical covariance and centroid models equally or more than DeltaCore.

### D. Persistent Anomaly Absorption Risk
In continuous adaptive operation, un-gated or weakly gated associative updates assimilate persistent anomalies into the memory within 5–25 observations ($r_{100}/r_1 = 0.0529$, absorbing 94.7% of the anomalous signal). In contrast, gated Online Covariance completely froze contamination ($r_{100}/r_1 = 1.0000$, 0% contamination).

### E. Operational Threshold Fragility
Under frozen calibration operating thresholds ($\tau = \mu_{\text{calib}} + 3\sigma_{\text{calib}}$), DeltaCore achieved near-zero operational recall ($0.000$) on several hard anomaly families due to reconstruction residual variance, failing to provide reliable operational alarm triggers.

---

## 4. Architectural Path for RecoveryOS

RecoveryOS will address telemetry monitoring and regime tracking using classical, transparent, and operationally established primitives:
1. **First-Order Tracking**: Lightweight online exponentially weighted moving averages (EWMA) and robust Huber centroids ($O(D)$ state, sub-5 µs latency) for baseline metrics.
2. **Second-Order Covariance Tracking**: Regularized streaming sample covariance and Mahalanobis distances ($O(D^2)$ state) where joint correlation tracking is required.
3. **Explicit Domain Invariants**: Explicit rule tables and transactional state machines for operational contradictions rather than learned associative approximations.

---

## 5. Status of the DeltaCore Codebase

DeltaCore remains an independent, open-source research toolkit for exploring adaptive associative memory dynamics, streaming matrix updates, and test-time learning. Development on DeltaCore is now **frozen**, preserved as a reproducible scientific archive with full benchmark artifacts and negative findings intact.

---

## 6. Scope of Inference

> **The evidence is benchmark-specific. It establishes that, on the specified 20-seed non-stationary matched-marginal benchmark with D=128 and the stated feature representation, DeltaCore Gated did not demonstrate an advantage over regularized Online Covariance, while the observed paired difference favored Online Covariance. The benchmark does not establish universal superiority of Online Covariance, universal inferiority of DeltaCore, or performance ordering under telemetry distributions, dimensions, drift processes, workloads, or deployment conditions not represented by the benchmark. The result supports terminating this specific DeltaCore research direction for RecoveryOS; it does not establish that adaptive telemetry detection as a broader problem has been solved.**
>
> **Any future work on adaptive telemetry detection must be treated as a new independent research effort with a new hypothesis, benchmark design, and preregistered evaluation.**

---

## 7. Historical Metric Isolation

The values 0.9859 and 0.9594 are Phase 16/17 balanced classification accuracy metrics and are not part of the telemetry regime-detection AUROC record.
