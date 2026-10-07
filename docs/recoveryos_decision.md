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

$$\boxed{\textbf{DeltaCore is REJECTED for RecoveryOS Telemetry Regime Detection.}}$$

**DeltaCore will NOT be integrated into RecoveryOS.**  
The telemetry integration hypothesis has been definitively tested, rejected, and frozen.

This is a specific **engineering use-case decision** based on empirical benchmark results. It does not imply that DeltaCore has no theoretical value as an experimental implementation of adaptive neural associative memory; rather, it establishes that DeltaCore is not justified as a telemetry anomaly or regime detector in RecoveryOS.

---

## 3. Empirical Reasons for Rejection

The decision is grounded in the findings of the final multi-seed paired benchmark (`experiments/higher_order_regime_shift.py` v3.0.0, 20 independent seeds, matched categorical marginals $\text{TVD} \le 0.05$):

### A. Online Covariance Matches or Exceeds DeltaCore
When compared against a proper online second-order baseline (**Online Covariance with regularized Mahalanobis distance**, $4D^2$ state):
- **Online Covariance** achieved higher mean discriminative accuracy (**$0.5976 \pm 0.0667$** AUROC) than **DeltaCore Gated** (**$0.5711 \pm 0.0801$** AUROC).
- The mean paired difference ($\text{DeltaCore} - \text{Covariance}$) was **$-0.0265$**, with a **95% paired bootstrap confidence interval of $[-0.0478, -0.0072]$** strictly excluding zero.
- The exact two-sided binomial sign test yielded **$p = 0.0118$** (DeltaCore won on only 4 of 20 evaluation seeds), and the paired permutation test yielded **$p = 0.0192$**.
- Online Covariance demonstrated superior or equivalent discriminative ability without requiring non-standard auto-associative controller dynamics.

### B. Computational and Memory Overhead is Unjustified
- DeltaCore requires $4 D^2$ bytes of persistent state memory ($64.0\text{ KB}$ at $D = 128$, $1\text{ MB}$ at $D = 512$, $4\text{ MB}$ at $D = 1024$).
- Standard first-order centroids require only $4 D$ bytes ($512\text{ B}$ at $D = 128$, $128\times$ smaller) and execute in $3.0\text{ µs}$ ($8.6\times$ faster than DeltaCore's $25.9\text{ µs}$).
- While DeltaCore improved upon first-order centroids by $+0.0382$ AUROC, that advantage was entirely matched by standard second-order covariance estimation.

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

RecoveryOS will address telemetry monitoring and regime tracking using classical, transparent, and operationally proven primitives:
1. **First-Order Tracking**: Lightweight online exponentially weighted moving averages (EWMA) and robust Huber centroids ($O(D)$ state, sub-5 µs latency) for baseline metrics.
2. **Second-Order Covariance Tracking**: Regularized streaming sample covariance and Mahalanobis distances ($O(D^2)$ state) where joint correlation tracking is required.
3. **Explicit Domain Invariants**: Explicit rule tables and transactional state machines for operational contradictions rather than learned associative approximations.

---

## 5. Status of the DeltaCore Codebase

DeltaCore remains an independent, open-source research toolkit for exploring adaptive associative memory dynamics, streaming matrix updates, and test-time learning. Development on DeltaCore is now **frozen**, preserved as a reproducible scientific archive with full benchmark artifacts and negative findings intact.
