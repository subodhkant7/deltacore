# DeltaCore: Scientific Boundaries, Theoretical Assumptions & Limitations

**Status**: Authoritative Epistemic & Experimental Boundary Reference  
**Scope**: DeltaCore Adaptive Associative State Core & Pre-RecoveryOS Evaluation  
**Version**: `0.1.0`  

---

## 1. Executive Scientific Stance

DeltaCore is an experimental framework for studying test-time adaptive associative neural memory. It provides mathematical primitives, Lyapunov contractive bounds, and reproducible streaming benchmarks.

The engineering and experimental evidence establishes specific, bounded capabilities. It does **not** establish universal superiority over classical algorithms, guaranteed anomaly detection, or unconstrained nonlinear representation capacity.

---

## 2. What Has Been Demonstrated

1. **Parameter Immutability ($\Delta\theta = 0$)**:
   Across all synthetic and real-world benchmarks, offline neural/linear parameters $\theta$ remain bit-for-bit immutable during streaming evaluation, verified via cryptographic SHA-256 parameter hashes.
2. **Local Contractive Step Stability**:
   The Lyapunov projection $\eta_t = \min(\eta_{\text{cand}, t}, \rho / (\|x_t\|_2^2 + \epsilon))$ with $\rho < 2$ guarantees local non-expansion of residual error on the active key:
   $$
   |1 - \eta_t \|x_t\|_2^2| \le \max(|1 - \rho|, 1) \le 1
   $$
   This prevents catastrophic numerical runaway ($NaN$ / $\pm\infty$) across tested dimensions $D \in [32, 256]$.
3. **Causal State Mechanism ($M_t \equiv 0$ Ablation)**:
   Ablation experiments across regression and non-stationary classification confirm that $100\%$ of online adaptation gain originates from the evolving associative state $M_t$.
4. **Reproducible Multi-Family Classification Transfer**:
   Under a frozen configuration ($\eta_0=0.015, \rho=1.50, \alpha_{\min}=0.95$), the associative state demonstrates positive adaptation gains over frozen linear baselines under boundary rotation (+5.6 percentage points), translation (+9.8 percentage points), and nonlinear deformation (+9.9 percentage points).
5. **Exact Memory Scaling Law**:
   Persistent state memory scales as exactly $4 D^2$ bytes (FP32) for square associative memory $M_t \in \mathbb{R}^{D \times D}$.

---

## 3. What Has Only Been Tested Synthetically

1. **Multi-Family Regime Transfer**: Evaluated on synthetic 2D/multi-dimensional geometric stream generators with controlled rotation, translation, and deformation dynamics.
2. **Telemetry Novelty & Anomaly Detection**: Evaluated on synthetic structured microservice event streams (e.g. checkout, refund, gateway timeout distributions).
3. **Adversarial Negative Transfer**: Evaluated on synthetic alternating regime sequences ($A \to B \to A$) and mismatched covariance injections.

---

## 4. What Is NOT Established

1. **Auto-Association $\neq$ Proven PCA / Subspace Learning**:
   An unconstrained full-rank matrix $M_t \in \mathbb{R}^{D \times D}$ can learn an identity mapping ($M \to I$). Without an explicit low-rank bottleneck or projection constraint, auto-associative DeltaCore does **not** perform principal component analysis. Claims that $v_t = x_t$ inherently yields a principal-subspace detector are **theoretically false and rejected**.
2. **No Guaranteed Anomaly Detection**:
   Reconstruction residual $r_t = \|x_t - M_{t-1} x_t\|_2$ reflects reconstruction difficulty relative to past associations. In empirical benchmarks on categorical telemetry streams, simpler static baselines (static centroid distance, raw categorical Jaccard dissimilarity, and static PCA reconstruction) perform competitively or superiorly with orders of magnitude smaller memory footprints (e.g. 512 bytes vs. 64 KB–4 MB for DeltaCore).
3. **No Proof of Global Boundedness**:
   Local contractivity $|1 - \eta_t \|x_t\|_2^2| \le 1$ guarantees that the immediate error along direction $x_t$ does not expand. It does **not** constitute a mathematical proof that the matrix norm $\|M_t\|_F$ remains globally bounded under arbitrary non-orthogonal sequence streams.
4. **No Universal Superiority Over Classical Online Learners**:
   In classification benchmarks, `OnlineLogisticRegression` outperformed `SafeAdaptiveDelta` on class-prior shifts (96.6% vs 92.2%). In high-speed atmospheric advection (ECMWF ERA5), `OnlineRidge` achieved lower tracking error ($0.0923$ vs $0.1622$).
5. **No Arbitrary Nonlinear Representational Capacity**:
   Because $M_t$ is a linear matrix mapping $\hat{v} = M x$, it exhibits an empirical performance limitation on complex non-planar decision boundaries without nonlinear kernel lifting or multi-layer architectures.

---

## 5. Identified Failure Modes & Operational Risks

### A. Anomaly Absorption Risk (Score-Before-Update Imperative)
If an anomalous observation $x_{\text{anom}}$ is adapted unconditionally into $M_t$, the associative memory rapidly incorporates the anomaly within 1–3 steps. Consequently, subsequent occurrences of the same anomaly exhibit contracted residuals, masking the ongoing regime shift.
- **Remedy**: Always evaluate novelty via pre-update scoring (`score()`).
- **Gating**: When novelty exceeds an established operating threshold, freeze state adaptation (`adapt=False`) to prevent memory contamination.

### B. Stale-State Negative Transfer Penalty
When the operating environment undergoes an abrupt, incompatible regime shift (e.g. alternating regimes $A \to B$), persistent memory from regime $A$ incurs an empirical penalty ($-1.75$ percentage points overall, $-10.0$ percentage points immediate post-shift) relative to an oracle state reset.

### C. Hash Collision Noise in Categorical Telemetry
Feature hashing via SHA-256 into dimension $D$ produces random bucket collisions across vocabulary tokens. In small dimensions ($D \le 64$), collision noise increases the variance of reconstruction residuals, necessitating calibrated smoothing windows.

### D. Memory Footprint at Large Dimensions
Because persistent memory scales quadratically ($4 D^2$ bytes), high-dimensional settings incur significant memory footprints:
- $D = 256$: $256\text{ KB}$
- $D = 512$: $1\text{ MB}$
- $D = 1024$: $4\text{ MB}$
- $D = 4096$: $64\text{ MB}$

In contrast, first-order centroid or rolling-mean baselines require only $4 D$ bytes ($O(D)$).

---

## 6. Prohibited Promotional Language

Contributors and automated agents are strictly prohibited from using the following unproven claims in documentation, code docstrings, and research reports:
- *"guaranteed anomaly detection"*
- *"instant regime detection"*
- *"universal distribution-shift detection"*
- *"proven principal-subspace learning"*
- *"globally stable neural memory"*
- *"subquadratic associative scaling"*
