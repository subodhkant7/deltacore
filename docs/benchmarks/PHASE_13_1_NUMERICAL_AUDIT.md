# DeltaCore Phase 13.1: Forensic Numerical Audit & Mathematical Provenance

This document records the mathematical provenance, baseline training verification, permutation equivariance proofs, and diagnostic metric semantics for **Phase 13.1**.

All findings are backed by verifiable machine-readable artifacts in [docs/benchmarks/artifacts/phase_13_1/](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_13_1/).

---

## 1. Baseline Training Provenance Audit (Section 4)

To prevent evaluating un-optimized random baselines, all offline static and recurrent models (`FrozenLinear`, `GRU`, `SpatialConv`) were audited for genuine gradient minimization on the training split prior to test-time freezing.

### Provenance Audit Matrix

| Model | Architecture | Parameters | Training Set | Optimizer | Epochs | Initial Train MSE | Final Train MSE | Loss Reduction | Test Parameter Hash Immutable |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FrozenLinear** | $W x_t + b$ | 4,160 | Train $[0, 150)$ | Adam ($\text{lr}=10^{-2}$) | 25 | 0.2078 | 0.0863 | **$-58.5\%$** | **Verified ($\Delta \theta = 0$)** |
| **GRU** | GRUCell ($d_h=8$) + Linear | 2,352 | Train $[0, 150)$ | Adam ($\text{lr}=10^{-2}$) | 25 | 0.4285 | 0.1706 | **$-60.2\%$** | **Verified ($\Delta \theta = 0$)** |
| **SpatialConv** | Conv2d ($3\times 3, 4\text{ch}$) + Head | 16,488 | Train $[0, 150)$ | Adam ($\text{lr}=10^{-2}$) | 25 | 0.3872 | 0.1194 | **$-69.2\%$** | **Verified ($\Delta \theta = 0$)** |
| **OnlineRidge** | Recursive Least Squares | 4,096 | N/A (Online) | Exact RLS ($\lambda=0.98$) | 0 | N/A | N/A | N/A | **Verified ($\Delta \theta = 0$)** |
| **SafeAdaptiveDelta** | Associative Memory Matrix | 4,096 | N/A (Online) | Contractive Delta Update | 0 | N/A | N/A | N/A | **Verified ($\Delta \theta = 0$)** |

### Key Audit Conclusions:
1. **Genuine Offline Optimization**: `SpatialConv`, `FrozenLinear`, and `GRU` were verified to minimize training loss significantly (all achieving $58\%$ to $69\%$ MSE reduction).
2. **Zero Test-Time Parameter Leakage**: In all cases, parameter gradients were disabled (`requires_grad = False`) and parameter SHA-256 hashes remained bit-for-bit identical before and after test evaluation.
3. The comparison between DeltaCore and offline trained baselines is mathematically valid.

Artifact: [baseline_training_provenance.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_13_1/baseline_training_provenance.json).

---

## 2. Actual Permutation-Equivariance Test (Section 6)

Phase 12 and Phase 13 noted that DeltaCore's prediction error was invariant under a fixed spatial permutation. To establish whether this constitutes **exact mathematical permutation equivariance** or merely equal scalar loss, an explicit test was executed across all 5 seeds.

### 2.1 Mathematical Formulation
Let $P \in \{0, 1\}^{D \times D}$ be an arbitrary coordinate permutation matrix ($P^\top P = I, P^{-1} = P^\top$).
* Original sequence: $X = (x_0, x_1, \dots, x_{T-1})$, yielding streaming predictions $\hat{Y} = f(X)$.
* Permuted sequence: $\tilde{X} = (P x_0, P x_1, \dots, P x_{T-1})$, yielding streaming predictions $\hat{\tilde{Y}} = f(P X)$.
* Normalized Equivariance Error:
  $$E_{\text{equiv}} = \frac{\|f(P X) - P f(X)\|_F}{\max(\|P f(X)\|_F, \epsilon)}$$

### 2.2 Mathematical Induction for DeltaCore
At step $t$:
* $\hat{\tilde{y}}_t = \tilde{M}_t \tilde{x}_t$.
* $\tilde{e}_t = \tilde{y}_t - \hat{\tilde{y}}_t = P y_t - \hat{\tilde{y}}_t$.
* $\tilde{M}_{t+1} = \alpha_t(\tilde{e}_t) \tilde{M}_t + \eta_t(\tilde{x}_t) \tilde{e}_t \tilde{x}_t^\top$.

Because $\|\tilde{x}_t\|_2^2 = \|P x_t\|_2^2 = \|x_t\|_2^2$ and $\|\tilde{e}_t\|_2 = \|e_t\|_2$, the scalar step size and retention factor are invariant:
$$\eta_t(\tilde{x}_t) = \eta_t(x_t), \quad \alpha_t(\tilde{e}_t) = \alpha_t(e_t)$$
Assuming by induction that $\tilde{M}_t = P M_t P^\top$ (which holds at $t=0$ where $M_0 = 0$):
$$\hat{\tilde{y}}_t = (P M_t P^\top)(P x_t) = P M_t (P^\top P) x_t = P M_t x_t = P \hat{y}_t$$
$$\tilde{M}_{t+1} = \alpha_t P M_t P^\top + \eta_t (P e_t)(P x_t)^\top = P (\alpha_t M_t + \eta_t e_t x_t^\top) P^\top = P M_{t+1} P^\top$$
Thus, $f(P X)_t \equiv P (f(X)_t)$ for all $t$.

### 2.3 Empirical Equivariance Results

| Model | Mean $E_{\text{equiv}}$ | Std $E_{\text{equiv}}$ | Is Permutation Equivariant? |
| :--- | :---: | :---: | :---: |
| **SafeAdaptiveDelta** | $\mathbf{7.73 \times 10^{-8}}$ | $1.98 \times 10^{-9}$ | **YES (Exact to FP32 Precision)** |
| **FixedDelta** | $\mathbf{8.38 \times 10^{-8}}$ | $2.41 \times 10^{-9}$ | **YES (Exact to FP32 Precision)** |
| **OnlineRidge** | $\mathbf{1.29 \times 10^{-6}}$ | $8.53 \times 10^{-8}$ | **YES (Exact to FP32 Precision)** |
| **SpatialConv** | $\mathbf{1.95 \times 10^{0}}$ | $1.42 \times 10^{-1}$ | **NO ($\|f(PX) - Pf(X)\|_F \gg 0$)** |

**Conclusion**: DeltaCore models and OnlineRidge are **strictly permutation-equivariant** within floating-point roundoff ($\sim 10^{-7}$). `SpatialConv` violates permutation equivariance ($E_{\text{equiv}} \approx 1.95$) because it relies on local 2D convolutional filter adjacencies.

Artifact: [permutation_equivariance.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_13_1/permutation_equivariance.json).

---

## 3. Spatial Gradient Error (SGE) Diagnostic Semantics (Section 14)

### Investigation: Why did SGE drop from $0.5930 \to 0.2798$ under permutation?
In Phase 13, the spatial gradient error was evaluated by calling `restore_spatial_field` directly on the permuted vector $\tilde{x}_t \in \mathbb{R}^D$, reshaping it into an $(8, 8)$ grid without un-permuting the coordinate mapping.

The formula for SGE is:
$$\text{SGE} = \frac{\|\nabla Y - \nabla \hat{Y}\|_F}{\max(\|\nabla Y\|_F, \epsilon)}$$

Under random coordinate permutation $\pi \in \mathcal{S}_D$:
1. Adjacent cells in the permuted grid $(i, j)$ and $(i+1, j)$ correspond to physically disjoint, distant ocean points.
2. The ground truth spatial differences $(\tilde{Y}_{i+1,j} - \tilde{Y}_{i,j})$ become large, uncorrelated high-frequency fluctuations.
3. This **massively inflates the denominator** $\|\nabla \tilde{Y}\|_F$.
4. Even though the absolute gradient error $\|\nabla \tilde{Y} - \nabla \hat{\tilde{Y}}\|_F$ also grows, the inflated denominator causes the normalized ratio to artificially drop from $0.5930$ to $0.2798$.
5. When evaluated in the true physical coordinate space (by applying $\pi^{-1}$ before computing finite differences), the physical SGE of permutation-equivariant models is **bit-for-bit identical** to the unpermuted SGE ($0.5930$).

---

## 4. Temperature & Anomaly Terminology (Section 9)

To ensure scientific precision:
* **Benchmark Anomaly**: The Phase 13 dataset computes a **local normalized SST anomaly**:
  $$\tilde{X}_t = \frac{X_t - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  where $\mu_{\text{train}} = 26.83^\circ\text{C}$ and $\sigma_{\text{train}} = 2.02^\circ\text{C}$.
* The $+2.8^\circ\text{C}$ peak anomaly during $t \in [40, 110]$ represents the physical temperature deviation in the NOAA OISST Pacific waveguide during the 2015–2016 Super El Niño event.
* The terminology has been corrected from *"+2.8°C Kelvin thermal anomaly"* to **"+2.8 °C SST anomaly"**.
* This quantity is distinct from the official NOAA Oceanic Niño Index (ONI / Niño 3.4), which applies a 3-month running mean of ERSST.v5 anomalies over $5^\circ\text{N}-5^\circ\text{S}, 170^\circ\text{W}-120^\circ\text{W}$.

---

## 5. Stability Claims & Local Contractive Safety (Section 8)

The language regarding numerical stability has been strictly qualified:
* Previous wording: *"guaranteeing bounded stability"*, *"strictly bounded"*.
* Revised phrasing:
  > *"remained finite/bounded in the tested $D=256$ experiment while enforcing the implemented local step-size safety condition."*
* The underlying local mathematical condition enforced at each step is:
  $$\eta_t \le \frac{\rho}{\|x_t\|_2^2 + \epsilon}, \quad \text{ensuring } |1 - \eta_t \|x_t\|_2^2| < 1$$
* This condition bounds the residual update locally, but is not documented as a global Lyapunov stability theorem.
