# DeltaCore: Scientific Boundaries, Theoretical Assumptions & Limitations

**Status**: Authoritative Epistemic & Experimental Boundary Reference  
**Version**: `0.2.0`  
**Date**: 2026-10-07  

---

## 1. Executive Scientific Stance

DeltaCore is an open-source research implementation of adaptive associative state dynamics. It provides mathematical primitives, Lyapunov contractive step bounds, and reproducible benchmarks for studying test-time adaptation under non-stationary distribution shift.

The experimental evidence establishes specific, bounded capabilities. It does **not** establish universal superiority over classical algorithms, guaranteed anomaly detection, or unconstrained nonlinear representation capacity.

---

## 2. The 10 Primary Empirical Limitations

### 1. Quadratic State Memory Overhead ($O(D^2)$)
Persistent controller state requires an unconstrained square associative matrix $M_t \in \mathbb{R}^{D \times D}$, demanding exactly $4 D^2$ bytes of memory (FP32). At scale:
- $D = 128$: $64\text{ KB}$
- $D = 512$: $1\text{ MB}$
- $D = 1024$: $4\text{ MB}$
- $D = 4096$: $64\text{ MB}$

In contrast, first-order centroid baselines require only $4 D$ bytes ($512\text{ B}$ at $D = 128$, $128\times$ smaller).

### 2. Higher Evaluation Latency Than First-Order Baselines
Scoring and updating an associative matrix requires matrix-vector multiplications ($O(D^2)$ operations). In benchmarking:
- Online Centroid median scoring latency: **$3.0\text{ µs}$**
- DeltaCore Gated median scoring latency: **$25.9\text{ µs}$** ($8.6\times$ slower)

Within the evaluated design and benchmark, DeltaCore incurred substantially greater state/latency cost without demonstrating a compensating performance advantage over regularized online covariance. (This does not imply that $O(D^2)$ state memory is inherently unacceptable in every deployment, but represents an unfavorable trade-off within this benchmark context.)

### 3. No Demonstrated Advantage Over Online Covariance on Decisive Benchmark
Under the predefined 20-seed non-stationary matched-marginal benchmark, regularized Online Covariance significantly outperformed DeltaCore Gated under both the exact paired sign test and exact paired sign-flip randomization test. DeltaCore Gated did not demonstrate a performance advantage over regularized Online Covariance on the decisive benchmark.
- **Online Covariance (Mahalanobis)**: **$0.5976 \pm 0.0667$** AUROC.
- **DeltaCore Gated**: **$0.5711 \pm 0.0801$** AUROC.
- **Mean paired difference**: **$-0.0265$** (95% percentile bootstrap CI: $[-0.0478, -0.0072]$, exact sign test $p = 0.0118$, exact paired sign-flip randomization test $p = 0.0188$).
- **Paired Cohen's $d$**: **$-0.58$**.

These metrics quantify the observed benchmark effect and should not be converted into a universal production-performance claim.

### 4. Representation Dependence
DeltaCore's cross-feature anomaly sensitivity depends overwhelmingly on representation engineering. Representation tier ablations revealed that gains on higher-order joint anomalies were driven by the feature hashing layer (pair and triple interaction hashing) rather than the associative matrix itself. Adding interaction tokens benefited classical covariance and centroid models equally or more.

### 5. Operating Threshold Calibration Sensitivity
Under frozen calibration thresholds ($\tau = \mu_{\text{calib}} + 3\sigma_{\text{calib}}$), DeltaCore exhibited near-zero operational recall ($0.000$) on subtle anomaly families due to residual variance under non-stationary drift. Setting reliable, fixed decision boundaries across arbitrary shifts remains an unsolved challenge.

### 6. Adaptive Anomaly Absorption (Contamination Risk)
Without score-before-update gating (`score()` followed by conditional `step(adapt=False)`), continuous adaptation rapidly assimilates persistent anomalies into the associative matrix within 5–25 steps ($r_{100}/r_1 = 0.0529$, absorbing 94.7% of the anomalous signal). Subsequent anomalies of the same type are reconstructed easily, masking the ongoing regime shift.

### 7. Lack of Inherent Temporal Modeling
From event-only feature representations $x_t$, DeltaCore cannot detect temporal sequence violations (such as inverted Markov event cycles) because the associative matrix treats observations as exchangeable vectors. Temporal detection requires explicit lag feature engineering ($x_t = \text{hash}(\text{event}_t, \text{event}_{t-1})$).

### 8. No Proof of Global Matrix Boundedness
The Lyapunov step-size controller $\eta_t = \min(\eta_0, \rho / (\|x_t\|_2^2 + \epsilon))$ ensures local non-expansion along the active vector direction ($|1 - \eta_t \|x_t\|_2^2| \le 1$). However, this is a local step property; it does **not** constitute a mathematical proof that the Frobenius norm $\|M_t\|_F$ remains globally bounded under arbitrary, adversarial, or non-orthogonal input sequences. Claims of global boundedness are scientifically rejected.

### 9. Synthetic and Controlled Benchmark Limitations
While evaluation streams incorporated non-stationary drift, Markov transitions, and matched marginals ($\text{TVD} \le 0.05$), all telemetry streams were synthetically generated under controlled schema models. Behavior on unstructured real-world distributed traces may exhibit unpredictable token distributions and drift rates.

### 10. No Evidence of Production Superiority
DeltaCore has not been evaluated in live production clusters, high-throughput message brokers, or real-time SRE control loops. It is research software, not a production-hardened telemetry monitoring system.

---

## 3. Scope of Inference

The evidence is benchmark-specific. It establishes that, on the predefined 20-seed non-stationary matched-marginal benchmark with $D=128$ and the specified feature representation, DeltaCore Gated did not demonstrate an advantage over regularized Online Covariance, while the observed paired difference favored Online Covariance.

The benchmark does not establish universal superiority of Online Covariance, universal inferiority of DeltaCore, or performance ordering under telemetry distributions, dimensions, drift processes, workloads, or deployment conditions not represented by the benchmark.

The result therefore supports terminating this specific DeltaCore research direction for RecoveryOS, rather than claiming that adaptive telemetry detection as a broader problem has been solved.

---

## 4. What Was Scientifically Demonstrated

Despite the boundaries above, DeltaCore verified:
- Pure-PyTorch deterministic implementation of test-time associative memory.
- Score-before-update residual semantics preventing mutation during inspection.
- Causal attribution: 100% of adaptation gain in non-stationary tasks stems from $M_t$.
- Safe numerical stability: Zero $NaN$ or $\pm\infty$ blow-ups across all tested seeds and dimensions.
- Reliable state persistence with SHA-256 integrity checks and atomic file replacement.

---

## 5. Prohibited Promotional Language

Contributors and automated agents are strictly prohibited from using the following unproven claims in documentation, code docstrings, and research reports:
- *"guaranteed anomaly detection"*
- *"instant regime detection"*
- *"universal distribution-shift detection"*
- *"proven principal-subspace learning"*
- *"globally stable neural memory"*
- *"production-ready telemetry controller"*
- *"superior to covariance or PCA"*
