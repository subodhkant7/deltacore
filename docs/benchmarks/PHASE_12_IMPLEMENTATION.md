# DeltaCore Phase 12: Controlled Spatio-Temporal Adaptive State — Implementation Report

This report documents the mathematical formulation, experimental design, and empirical results for **Phase 12: Controlled Spatio-Temporal Adaptive State**.

All experiments were executed with pure PyTorch CPU streaming evaluation across five deterministic random seeds (`seeds = [0, 1, 2, 3, 4]`) under zero test-time parameter gradients (`requires_grad = False`).

---

## 1. Mathematical Data Representation & Regimes

Each observation at timestep $t$ is generated as a spatial field:

$$
X_t \in \mathbb{R}^{H \times W \times C}
$$

and provided to streaming predictors as a flattened vector:

$$
x_t = \operatorname{flatten}(X_t) \in \mathbb{R}^D, \quad D = H \cdot W \cdot C
$$

Primary dimensions evaluated:
* $H=8, W=8, C=1 \implies D=64$
* $H=8, W=8, C=2 \implies D=128$
* $H=16, W=16, C=1 \implies D=256$

The spatial evolution dynamics follow strictly bounded and dissipative differential operators:

$$
X_{t+1} = F_r(X_t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2 I)
$$

### Regimes
1. **Regime A (Horizontal Transport)**: Directional wave translation along horizontal coordinates with dissipative smoothing.
2. **Regime B (Vertical Transport)**: Directional wave translation along vertical coordinates with dissipative smoothing.
3. **Regime C (Diffusion-Vortex)**: 2D Laplacian spatial diffusion coupled with nonlinear localized vortex rotation:
   $$X_{t+1} = \tanh\left(X_t + \kappa \nabla_{2D}^2 X_t\right) \cdot 0.90$$
4. **Regime A2 (Modified Horizontal Transport)**: Related horizontal wave motion with altered spatial frequency and velocity ($shifts=2$).

---

## 2. Retention Mechanisms & Controllers

Phase 12 evaluated five formal retention modes and four causal controls:

### Retention Modes (Section 7)
* **Mode 1 (`retain_high`)**: Fixed high retention ($\alpha_t = 0.99$).
* **Mode 2 (`retain_low`)**: Fixed low retention ($\alpha_t = 0.70$).
* **Mode 3 (`adaptive_retention`)**: Error-conditioned dynamic retention:
  $$\alpha_t = \operatorname{clamp}(1.0 - \gamma \|e_t\|, \alpha_{\min}, 1.0)$$
* **Mode 4 (`adaptive_state_controller`)**: Compact state-conditioned affine controller:
  $$z_t = \left[ \|e_t\|, \|M_t\|_F, \|\Delta M_{t-1}\|_F, \bar{e}_t \right]^\top \in \mathbb{R}^4$$
  $$\alpha_t = \alpha_{\min} + (1.0 - \alpha_{\min}) \cdot \sigma\left(w^\top z_t + b\right)$$
  where $w \in \mathbb{R}^4, b \in \mathbb{R}$ constitute exactly **5 parameters**.
* **Mode 5 (`oracle_retention`)**: Ground-truth boundary detector resetting retention to $\alpha_t = 0.0$ at changepoints and $\alpha_t = 1.0$ otherwise.

### Causal Controls (Section 8)
* **Control A**: Adaptive retention ON vs. OFF (`set_adaptation(True/False)`).
* **Control B**: Continuous adaptive state vs. State reset at shift boundaries.
* **Control C**: Fixed-high vs. Fixed-low vs. Adaptive retention.
* **Control D**: True state-conditioned controller vs. Shuffled decorrelated control preserving identical 5-parameter architecture.

---

## 3. Required Final Tables (Section 26)

### Table 1 — Main Spatio-Temporal Benchmark ($D=64$, Task B: $A \to B \to C \to A$)

Values represent means across seeds `[0, 1, 2, 3, 4]`. State memory represents persistent memory bytes.

| Model | Total Params | Trainable Params | State Memory (Bytes) | Task A / B Mean Error ($E_{\mathrm{rel}}$) | Post-Shift Error ($E_{\mathrm{post},0}$) | Negative Transfer ($E_{\mathrm{cont}} - E_{\mathrm{reset}}$) | Forgetting ($E_{\mathrm{return}} - E_{\mathrm{pre}}$) | Runtime ($\mu\mathrm{s}$/token) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive_persistence** | 0 | 0 | 0 | 0.9893 | 1.0001 | $+6.49 \times 10^{-7}$ | $+8.80 \times 10^{-7}$ | 29.1 |
| **FrozenLinear** | 4,160 | 0 | 0 | 0.9835 | 1.0001 | $0.0$ | $+1.04 \times 10^{-6}$ | 29.0 |
| **FrozenMLP** | 1,096 | 0 | 0 | 4.1756 | 4.2303 | $+1.05 \times 10^{-2}$ | $-2.17 \times 10^{-2}$ | 56.6 |
| **SpatialConvControl** | 16,488 | 0 | 0 | 1.0179 | 0.9999 | $0.0$ | $-1.71 \times 10^{-7}$ | 56.0 |
| **SpatialDownsampleControl** | 1,088 | 0 | 0 | 1.0049 | 1.0000 | $0.0$ | $+1.32 \times 10^{-6}$ | 38.1 |
| **GRU** | 1,160 | 0 | 16 | 7.0547 | 7.2642 | $+6.35 \times 10^{-1}$ | $-2.62 \times 10^{-2}$ | 65.1 |
| **LSTM** | 1,440 | 0 | 32 | 6.1342 | 6.2412 | $-2.63 \times 10^{-1}$ | $-6.41 \times 10^{-2}$ | 80.4 |
| **OnlineRidge** | 4,096 | 0 | 32,768 | 0.9853 | 1.0001 | $+2.32 \times 10^{-6}$ | $+1.16 \times 10^{-6}$ | 50.7 |
| **NonlinearOnlineRidge** | 0 | 0 | 12,288 | 0.9993 | 1.0157 | $-9.75 \times 10^{-3}$ | $-9.75 \times 10^{-4}$ | 75.2 |
| **FixedDelta** | 4,096 | 0 | 16,384 | 0.9887 | 1.0001 | $+1.81 \times 10^{-6}$ | $+7.31 \times 10^{-7}$ | 38.1 |
| **AdaptiveDelta** | 4,096 | 0 | 16,384 | 0.9883 | 1.0001 | $+1.86 \times 10^{-6}$ | $+7.81 \times 10^{-7}$ | 37.2 |
| **SafeAdaptiveDelta** | 4,096 | 0 | 16,384 | 0.9875 | 1.0001 | $+2.07 \times 10^{-6}$ | $+8.52 \times 10^{-7}$ | 40.3 |
| **SelfReferential** | 4,098 | 0 | 16,392 | 0.9872 | 1.0001 | $+2.19 \times 10^{-6}$ | $+8.47 \times 10^{-7}$ | 67.3 |
| **SafeSelfReferential** | 4,098 | 0 | 16,392 | 0.9868 | 1.0001 | $+2.27 \times 10^{-6}$ | $+8.72 \times 10^{-7}$ | 66.4 |
| **Selective_fixed_high** | 0 | 0 | 2,048 | 0.9964 | 1.0000 | $+3.13 \times 10^{-7}$ | $-3.91 \times 10^{-8}$ | 48.8 |
| **Selective_fixed_low** | 0 | 0 | 2,048 | 0.9991 | 1.0000 | $+2.30 \times 10^{-10}$ | $0.0$ | 46.3 |
| **Selective_adaptive** | 0 | 0 | 2,048 | 0.9996 | 1.0000 | $-1.38 \times 10^{-10}$ | $0.0$ | 52.1 |
| **Selective_state_adaptive** | 5 | 0 | 2,064 | 0.9992 | 1.0000 | $-2.29 \times 10^{-10}$ | $0.0$ | 53.4 |
| **Selective_shuffled_control** | 5 | 0 | 2,064 | 0.9991 | 1.0000 | $-3.61 \times 10^{-10}$ | $0.0$ | 70.6 |
| **Selective_oracle** | 0 | 0 | 2,048 | 0.9960 | 1.0000 | $+2.94 \times 10^{-7}$ | $-9.75 \times 10^{-8}$ | 52.8 |

---

### Table 2 — Retention Mode Ablation Analysis

| Variant | Retention Mode ($\alpha_t$) | Mean Post-Shift Error ($E_{\mathrm{post},0}$) | Negative Transfer ($E_{\mathrm{cont}} - E_{\mathrm{reset}}$) | Return Recovery / Forgetting | Cumulative Adaptation Energy ($E_{\mathrm{adapt}}$) | Mean Retention ($\bar{\alpha}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Mode 1** | `retain_high` (Fixed 0.99) | 1.000011 | $+3.13 \times 10^{-7}$ | $-3.91 \times 10^{-8}$ | 0.0884 | 0.9900 |
| **Mode 2** | `retain_low` (Fixed 0.70) | 1.000000 | $+2.30 \times 10^{-10}$ | $0.000000$ | 0.0788 | 0.7000 |
| **Mode 3** | `adaptive_retention` (Error-gated) | 1.000000 | $-1.38 \times 10^{-10}$ | $0.000000$ | 0.0847 | 0.3899 |
| **Mode 4** | `adaptive_state_controller` ($z_t \in \mathbb{R}^4$) | 1.000000 | $-2.29 \times 10^{-10}$ | $0.000000$ | 0.0764 | 0.7991 |
| **Mode 5** | `oracle_retention` (Ground-truth shifts) | 1.000038 | $+2.94 \times 10^{-7}$ | $-9.75 \times 10^{-8}$ | 0.6554 | 0.9941 |
| **Control D** | `shuffled_controller` (Noisy/shuffled $z_t$) | 1.000000 | $-3.61 \times 10^{-10}$ | $0.000000$ | 0.0794 | 0.8305 |

---

### Table 3 — Dimensional Scaling ($D = HWC \in \{64, 128, 256\}$)

| Dimension $D$ | Spatial Config ($H \times W \times C$) | Model | Mean Error ($E_{\mathrm{rel}}$) | Step Latency ($\mu\mathrm{s}$/token) | Persistent State Memory (Bytes) | Memory Ratio vs. OnlineRidge |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **64** | $8 \times 8 \times 1$ | **FrozenLinear** | 0.9835 ± 0.0014 | 27.0 | 0 | 0.00x |
| **64** | $8 \times 8 \times 1$ | **SpatialConvControl** | 1.0179 ± 0.0119 | 44.5 | 0 | 0.00x |
| **64** | $8 \times 8 \times 1$ | **OnlineRidge** | 0.9853 ± 0.0007 | 44.4 | 32,768 | 1.00x |
| **64** | $8 \times 8 \times 1$ | **NonlinearOnlineRidge** | 0.9993 ± 0.0007 | 55.0 | 12,288 | 0.38x |
| **64** | $8 \times 8 \times 1$ | **SafeAdaptiveDelta** | 0.9875 ± 0.0006 | 37.5 | 16,384 | 0.50x |
| **64** | $8 \times 8 \times 1$ | **Selective_adaptive** | 0.9996 ± 0.0001 | 47.7 | 2,048 | 0.06x |
| **64** | $8 \times 8 \times 1$ | **Selective_state_adaptive** | 0.9992 ± 0.0003 | 51.6 | 2,064 | 0.06x |
| **128** | $8 \times 8 \times 2$ | **FrozenLinear** | 0.9888 ± 0.0009 | 37.5 | 0 | 0.00x |
| **128** | $8 \times 8 \times 2$ | **SpatialConvControl** | 1.0154 ± 0.0056 | 48.9 | 0 | 0.00x |
| **128** | $8 \times 8 \times 2$ | **OnlineRidge** | 0.9908 ± 0.0007 | 72.7 | 131,072 | 1.00x |
| **128** | $8 \times 8 \times 2$ | **NonlinearOnlineRidge** | 1.0009 ± 0.0008 | 75.5 | 49,152 | 0.38x |
| **128** | $8 \times 8 \times 2$ | **SafeAdaptiveDelta** | 0.9919 ± 0.0006 | 63.8 | 65,536 | 0.50x |
| **128** | $8 \times 8 \times 2$ | **Selective_adaptive** | 0.9998 ± 0.0001 | 51.7 | 4,096 | 0.03x |
| **128** | $8 \times 8 \times 2$ | **Selective_state_adaptive** | 0.9997 ± 0.0001 | 56.3 | 4,112 | 0.03x |
| **256** | $16 \times 16 \times 1$ | **FrozenLinear** | 0.9917 ± 0.0003 | 56.8 | 0 | 0.00x |
| **256** | $16 \times 16 \times 1$ | **SpatialConvControl** | 1.0094 ± 0.0081 | 64.1 | 0 | 0.00x |
| **256** | $16 \times 16 \times 1$ | **OnlineRidge** | 0.9942 ± 0.0003 | 122.2 | 524,288 | 1.00x |
| **256** | $16 \times 16 \times 1$ | **NonlinearOnlineRidge** | 1.0033 ± 0.0004 | 110.3 | 196,608 | 0.38x |
| **256** | $16 \times 16 \times 1$ | **SafeAdaptiveDelta** | 0.9946 ± 0.0003 | 103.8 | 262,144 | 0.50x |
| **256** | $16 \times 16 \times 1$ | **Selective_adaptive** | 0.9999 ± 0.0000 | 53.5 | 8,192 | 0.016x |
| **256** | $16 \times 16 \times 1$ | **Selective_state_adaptive** | 0.9998 ± 0.0000 | 58.6 | 8,208 | 0.016x |

---

## 4. Fair Parameter and Memory Accounting (Section 10)

For all tested predictors, memory is decomposed strictly across three tiers:
1. **Parameter Memory**: Tensors storing frozen weights/buffers (e.g., $W_{\phi}$, $W_{\mathrm{rff}}$, conv layers).
2. **Persistent State Memory**: Dynamic mutable state carrying historical information ($M_t$, $P_t$, recurrent hidden states).
3. **Temporary Activation Memory**: Transient memory required to execute single-step forward inference and rank-1 outer product updates.

Key observation:
* `OnlineRidge` requires updating both $P_t \in \mathbb{R}^{D \times D}$ and $W_t \in \mathbb{R}^{D \times D}$, occupying $2 D^2 \times 4$ bytes.
* `SafeAdaptiveDelta` requires updating only $M_t \in \mathbb{R}^{D \times D}$, occupying $D^2 \times 4$ bytes (strictly 50% of the linear RLS state budget).
* `SelectiveStateAdaptive` with feature dimension $D_{\mathrm{feat}} = 8$ occupies $8 D \times 4 + 16$ bytes (scaling linearly with $D$ when $D_{\mathrm{feat}}$ is fixed, yielding a 63.8x state memory advantage at $D=256$).
