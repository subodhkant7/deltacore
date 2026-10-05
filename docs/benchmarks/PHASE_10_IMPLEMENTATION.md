# DeltaCore — Phase 10: Streaming Adaptive-State Benchmark Implementation Report

**Document Status**: COMPLETED & EMPIRICALLY VERIFIED  
**Phase**: 10 (Streaming Adaptive-State Benchmark)  
**Author**: Antigravity Assistant & DeltaCore Contributors  
**Date**: October 2026  
**Artifact Directory**: `docs/benchmarks/artifacts/phase_10/`  
**Related Documents**:
- Observatory Report: [docs/benchmarks/PHASE_10_OBSERVATORY_REPORT.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_10_OBSERVATORY_REPORT.md)
- Scientific Interpretation & Epistemic Evaluation: [docs/benchmarks/PHASE_10_INTERPRETATION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_10_INTERPRETATION.md)
- Machine-Readable Summary: [docs/benchmarks/artifacts/phase_10/phase_10_results.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_10/phase_10_results.json)
- Per-Seed Records: [docs/benchmarks/artifacts/phase_10/phase_10_per_seed.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_10/phase_10_per_seed.json)
- Benchmark Manifest: [docs/benchmarks/artifacts/phase_10/phase_10_config.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_10/phase_10_config.json)

---

## 1. Executive Summary & Objective

The explicit scientific objective of **Phase 10** is to establish whether DeltaCore's adaptive memory provides a measurable advantage for **online adaptation under temporal distribution shift**.

This phase is **not a visual-backbone phase**. It does not implement ImageNet, CIFAR, patch embeddings, hierarchical vision stages, attention, Mamba-like architectures, custom CUDA kernels, or VisionHOPE reproduction components.

The scientific question under investigation is:
> *Can an adaptive neural state respond to distribution changes faster or more reliably than a frozen model or conventional recurrent baseline?*

To isolate the causal factors, this benchmark strictly disentangles:
1. **Memory capacity** vs **adaptation speed**
2. **Numerical stability** vs **unbounded divergence**
3. **Information retention** vs **catastrophic forgetting**
4. **Computational cost** (latency per token and parameter overhead)

In strict accordance with the DeltaCore Constitution and Phase 10 specification, **no claim of superiority is made unless supported by empirical measurements**.

---

## 2. Phase 9.2 Mandatory Corrections Verification

Prior to conducting Phase 10 streaming experiments, the historical record for Phase 9.2 was audited and corrected:

1. **C1 — True Shuffled-Coordinate Control**:
   - Implemented matched pair `delta_reference_xy` vs `delta_reference_xy_shuffled` with identical 368p architecture, frozen update dynamics, optimizer, learning rate, epoch count, seed, and coordinate distribution.
   - Evaluated under identical seeds ($0.8863 \pm 0.0315$ vs $1.0015 \pm 0.0007$, difference $\Delta = -0.1152 \pm 0.0315$).
   - Replaced un-matched control in Phase 9.2 documentation and tested in `test_true_shuffled_coordinate_control_c1`.
2. **C2 — MLP Terminology Correction**:
   - Renamed the 274p/304p baseline to `mlp_control` across all Phase 9.2 documentation, explicitly disclaiming parameter equivalence to the 94-parameter single-direction DeltaCore model.
3. **C3 — Epistemic Tone Correction**:
   - Revised causal claims across Phase 9.2 documentation. Specifically changed `EC10: causally attributed` to `SUPPORTED: missing spatial addressability is the leading explanation in the tested configurations`.
   - Removed ungrounded terms (`proves`, `significantly`, `excels`, `inherent mismatch`) where the empirical record did not justify them.

---

## 3. Benchmark Architecture & Task Design

All tasks operate on multivariate temporal streams $x_t \in \mathbb{R}^D$ with $D = 8$, primary sequence length $T = 512$ (with sweeps across $T \in \{128, 512, 1024\}$), evaluated over five independent deterministic seeds:
$$\text{seeds} \in \{0, 1, 2, 3, 4\}$$

### 3.1. Task A — Regime-Switching Prediction
Constructs a piece-wise linear dynamical process with multiple distinct transition regimes:
$$x_{t+1} = A_r x_t + \epsilon_t, \quad r \in \{1, 2, 3\}$$
with cycle:
$$\text{Regime 1} \to \text{Regime 2} \to \text{Regime 3} \to \text{Regime 1}$$
at predetermined change-points $t \in \{T/4, T/2, 3T/4\}$.
- **Stability Guarantee**: Every transition matrix $A_r \in \mathbb{R}^{D \times D}$ is generated deterministically from the seed and normalized to ensure spectral radius $\rho(A_r) \le 0.92 < 1.0$.
- **Information Boundary**: The true regime index $r$ is completely hidden from the model. The model receives $x_t$ and must output $\hat{x}_{t+1}$ online.

### 3.2. Task B — Delayed Context Retrieval
Isolates temporal retention from local smoothing by embedding an informative cue vector at time $t_0 = 10$, followed by irrelevant Gaussian noise context for a controlled delay $d \in \{16, 64, 256\}$, concluding with an orthogonal query prompt that demands outputting the original cue target:
$$\text{cue} \to \text{noise context (delay } d\text{)} \to \text{query} \to \text{target}$$
Measures whether memory degrades as delay $d$ scales up to 256 timesteps.

### 3.3. Task C — Abrupt Distribution Shift ($A \to B \to A$)
Evaluates adaptation, forgetting, and recovery under simultaneous mean, covariance, and dynamical shifts:
- **Phase 1 ($t \in [0, T/3)$)**: Stationary Regime A ($\mu_A = 0$, $A_A$).
- **Phase 2 ($t \in [T/3, 2T/3)$)**: Shifted Regime B ($\mu_B = 1.5 \cdot \mathbf{1}$, scaled covariance $\Sigma_B = 2.0 \cdot I$, distinct transition $A_B$).
- **Phase 3 ($t \in [2T/3, T)$)**: Re-entry into Regime A ($\mu_A = 0$, $A_A$).
Permits simultaneous measurement of:
- **Adaptation**: Mean relative error during Phase 2.
- **Forgetting**: Error degradation on returning to Regime A relative to pre-shift steady-state performance.
- **Recovery**: Speed of re-adaptation after $t = 2T/3$.

---

## 4. Model Matrix & Capacity Matching (~100-Parameter Regime)

To prevent capacity confounds, all models are constrained to a compact ~100-parameter regime with $D = 8$:

| Model Identifier | Model Class | Trainable Params (Offline) | Eval Params Updated | Internal State Representation | State Dimension |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `FrozenLinear` | Static linear regression baseline | 72 | 0 | None (static weights) | 0 |
| `FrozenMLP` | Static 2-layer MLP (hidden dim=4) | 76 | 0 | None (static weights) | 0 |
| `GRU` | Recurrent neural baseline (hidden dim=2) | 96 | 0 | Hidden vector $h_t$ | 2 |
| `LSTM` | Recurrent neural baseline (hidden dim=2) | 120 | 0 | Cell & hidden $(c_t, h_t)$ | 4 |
| `OnlineRidge` | Deterministic Recursive Least Squares (RLS) | 64 | 0 | Inverse covariance $P_t$, weights $W_t$ | $8 \times 8 + 8 \times 8$ |
| `FixedDelta` | Hebbian associative memory ($\eta = 0.15$) | 64 | 0 | Associative matrix $M_t$ | $8 \times 8$ |
| `AdaptiveDelta` | Unconstrained adaptive step-size | 64 | 0 | Associative matrix $M_t$, step controller | $8 \times 8$ |
| `SafeAdaptiveDelta` | Local safe-step contractive controller | 64 | 0 | Associative matrix $M_t$, bound $\eta \le \frac{2-\delta}{\|x\|^2+\epsilon}$ | $8 \times 8$ |
| `SelfReferential` | Coupled content + dynamics memory | 66 | 0 | Content $M_t$, Dynamics $C_t$ | $8 \times 8 + 2 \times 2$ |
| `SafeSelfReferential` | Dual local safe-step contractive memory | 66 | 0 | Content $M_t$, Dynamics $C_t$ (bounded) | $8 \times 8 + 2 \times 2$ |

### 4.1. Strict Online / Offline Separation (EC10.3)
At test time, the model parameters $\theta$ satisfy:
$$\Delta \theta = 0, \quad \nabla_\theta \mathcal{L} = \text{None}$$
Every evaluated model executes in pure evaluation mode (`model.eval()`, `torch.no_grad()`). Only explicitly designated internal dynamical states ($M_t, C_t, h_t, P_t$) are permitted to evolve dynamically via explicit transition functions.

---

## 5. Primary Benchmark Results

### 5.1. Task A: Regime-Switching Prediction ($T = 512, D = 8$)

All metrics are reported as **Mean $\pm$ Standard Deviation across 5 independent seeds**:

| Model | Mean Error $E_{\text{rel}}$ | 1st-Passage Recovery | Sustained Recovery ($K=10$) | Adaptation Energy $E_{\text{adapt}}$ | Latency ($\mu\text{s}/\text{token}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `FrozenLinear` | $1.4005 \pm 0.0902$ | $0.8 \pm 1.2$ steps | $22.0 \pm 0.0$ steps | $0.00 \pm 0.00$ | $28.58$ |
| `FrozenMLP` | $4.2039 \pm 0.6036$ | $0.0 \pm 0.0$ steps | $27.0 \pm 0.0$ steps | $0.00 \pm 0.00$ | $40.56$ |
| `GRU` | $6.1667 \pm 0.7285$ | $0.0 \pm 0.0$ steps | $26.6 \pm 0.8$ steps | $3.78 \pm 0.82$ | $58.80$ |
| `LSTM` | $5.5391 \pm 1.2138$ | $0.0 \pm 0.0$ steps | $26.6 \pm 0.8$ steps | $6.98 \pm 1.45$ | $72.75$ |
| `OnlineRidge` | $\mathbf{0.7652 \pm 0.0291}$ | $5.0 \pm 2.1$ steps | $61.6 \pm 12.4$ steps | $9.71 \pm 0.44$ | $37.45$ |
| `FixedDelta` | $0.9296 \pm 0.0479$ | $0.6 \pm 0.8$ steps | $72.6 \pm 18.2$ steps | $6.58 \pm 0.12$ | $29.65$ |
| `AdaptiveDelta` | $3.7403 \pm 5.3870$ | $2.2 \pm 1.7$ steps | $93.8 \pm 14.9$ steps | $22,603.83 \pm 45,180.20$ | $31.77$ |
| `SafeAdaptiveDelta` | $0.9702 \pm 0.0395$ | $5.8 \pm 1.9$ steps | $101.2 \pm 15.3$ steps | $10.28 \pm 0.16$ | $32.39$ |
| `SelfReferential` | $1.3836 \pm 0.9169$ | $2.4 \pm 2.0$ steps | $88.6 \pm 16.5$ steps | $5.82 \times 10^{18} \pm 1.16 \times 10^{19}$ | $58.33$ |
| `SafeSelfReferential` | $\mathbf{0.9133 \pm 0.0378}$ | $6.0 \pm 2.2$ steps | $73.0 \pm 17.6$ steps | $8.64 \pm 0.14$ | $61.02$ |

*Note on Recovery Steps*: For static models (`FrozenLinear`, `FrozenMLP`), pre-shift error is high ($E_{\text{pre}} > 1.3$), which artificially deflates the first-passage threshold calculation $\tau = \max(1.3 E_{\text{pre}}, 0.25)$, causing apparent 0-step recovery despite failing to track the dynamics.

### 5.2. Task B: Delayed Context Retrieval across Delays $d \in \{16, 64, 256\}$

| Model | Error at $d = 16$ | Error at $d = 64$ | Error at $d = 256$ | Retention Trend ($\Delta_{256 - 16}$) |
| :--- | :---: | :---: | :---: | :---: |
| `FrozenLinear` | $5.5408$ | $5.5408$ | $5.5408$ | $\pm 0.0000$ (No memory) |
| `FrozenMLP` | $2.1334$ | $1.5125$ | $1.9932$ | $-0.1402$ (High variance) |
| `GRU` | $2.0867$ | $2.0645$ | $1.8158$ | $-0.2709$ (Uncalibrated) |
| `LSTM` | $1.7625$ | $1.5616$ | $1.7936$ | $+0.0311$ (Uncalibrated) |
| `OnlineRidge` | $1.0473$ | $0.9932$ | $0.9986$ | $-0.0487$ (Asymptotic mean) |
| `FixedDelta` | $1.0490$ | $1.0364$ | $1.0056$ | $-0.0434$ (Asymptotic mean) |
| `AdaptiveDelta` | $1.0436$ | $1.0423$ | $1.0382$ | $-0.0054$ (Asymptotic mean) |
| `SafeAdaptiveDelta` | $1.0444$ | $1.0427$ | $1.0377$ | $-0.0067$ (Asymptotic mean) |
| `SelfReferential` | $1.0534$ | $1.0352$ | $0.9979$ | $-0.0555$ (Asymptotic mean) |
| `SafeSelfReferential` | $1.0548$ | $1.0350$ | $0.9968$ | $-0.0580$ (Asymptotic mean) |

*Finding*: As delay scales to $d = 256$, DeltaCore and OnlineRidge reach relative error $\approx 1.00$, corresponding to the zero/mean prediction ceiling under intervening temporal noise. They do not selectively preserve an isolated distant cue vector without an explicit addressing or persistent gating mechanism.

### 5.3. Task C: Abrupt Shift ($A \to B \to A$) Adaptation vs. Forgetting

| Model | Phase B Adaptation Error | Forgetting on Re-entry to A ($\Delta E$) | Numerical Stability Status |
| :--- | :---: | :---: | :--- |
| `FrozenLinear` | $0.1258$ | $+0.0046$ | Stable (Static linear map) |
| `FrozenMLP` | $0.9556$ | $-1.9552$ | Stable (High bias) |
| `GRU` | $1.1095$ | $-4.1752$ | Stable (High recurrent bias) |
| `LSTM` | $0.9206$ | $-3.5652$ | Stable (High recurrent bias) |
| `OnlineRidge` | $\mathbf{0.0838}$ | $+0.2883$ | Stable (Fastest adapt, highest forgetting) |
| `FixedDelta` | $\mathbf{NaN}$ | $\mathbf{NaN}$ | **DIVERGED** (Eigenvalue explosion: $\eta \|x_t\|^2 > 2$) |
| `AdaptiveDelta` | $\mathbf{NaN}$ | $\mathbf{NaN}$ | **DIVERGED** (Unconstrained step size explosion) |
| `SafeAdaptiveDelta` | $0.2966$ | $\mathbf{+0.0789}$ | **STABLE** (Local safe-step contraction enforced) |
| `SelfReferential` | $\mathbf{NaN}$ | $\mathbf{NaN}$ | **DIVERGED** (Coupled dynamics feedback explosion) |
| `SafeSelfReferential` | $\mathbf{0.2779}$ | $\mathbf{+0.0829}$ | **STABLE** (Dual local safe-step contraction enforced) |

---

## 6. Primary Causal Controls & Ablations

### 6.1. Adaptive State ON vs. OFF (Primary Causal Control — Section 11)
To isolate whether the state update itself drives performance, DeltaCore models were evaluated with state updates active (`ON`) versus frozen at initialization (`OFF`):

| Model | Adaptive State ON | Adaptive State OFF | Causal Effect ($\Delta_{\text{ON} - \text{OFF}}$) | Empirical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| `FixedDelta` | $0.9296 \pm 0.0479$ | $1.0012 \pm 0.0006$ | $\mathbf{-0.0716}$ | Adaptive state reduces error by 7.2% |
| `AdaptiveDelta` | $3.7403 \pm 5.3870$ | $1.0012 \pm 0.0006$ | $+2.7391$ | Unconstrained controller diverges on some seeds |
| `SafeAdaptiveDelta` | $0.9702 \pm 0.0395$ | $1.0012 \pm 0.0006$ | $\mathbf{-0.0310}$ | Stable adaptation reduces error by 3.1% |
| `SelfReferential` | $1.3836 \pm 0.9169$ | $1.0012 \pm 0.0006$ | $+0.3824$ | Unbounded coupled feedback hurts average error |
| `SafeSelfReferential` | $\mathbf{0.9133 \pm 0.0378}$ | $1.0012 \pm 0.0006$ | $\mathbf{-0.0879}$ | Dual safe adaptation reduces error by 8.8% |

### 6.2. Continuous State vs. State Reset at Shift (Section 12)
Evaluates whether accumulated memory provides forward continuity or induces stale-regime interference across transition boundaries:

| Model | Continuous State | State Reset at Shift | State Differential ($\Delta_{\text{Cont} - \text{Reset}}$) | Empirical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| `FixedDelta` | $0.9296$ | $0.8978$ | $+0.0318$ | Resetting clears stale-regime inertia |
| `AdaptiveDelta` | $3.7403$ | $1.8378$ | $+1.9025$ | Resetting eliminates accumulated instability |
| `SafeAdaptiveDelta` | $0.9702$ | $0.9095$ | $+0.0607$ | Resetting clears old-regime bias |
| `SelfReferential` | $1.3836$ | $1.0476$ | $+0.3360$ | Resetting purges diverged dynamics state |
| `SafeSelfReferential` | $0.9133$ | $0.8679$ | $+0.0454$ | Resetting improves recovery by removing stale state |

*Conclusion*: Across all models, persistent state creates both useful historical memory and measurable stale-regime inertia. Resetting the state at the exact moment of distribution shift consistently lowers post-shift error, confirming that continuous state retention materially alters online dynamics (supporting H10.7).

---

## 7. Computational Profiling & Hardware Environment

All evaluations executed on macOS (Darwin 24.1.0, Apple Silicon ARM64, Python 3.13.0, PyTorch 2.6.0 pure CPU):

| Model | Latency per Timestep | Latency per Token Ratio (vs Linear) | Trainable Parameter Budget |
| :--- | :---: | :---: | :---: |
| `FrozenLinear` | $28.58\ \mu\text{s}$ | $1.00\times$ | 72 |
| `FixedDelta` | $29.65\ \mu\text{s}$ | $1.04\times$ | 64 |
| `AdaptiveDelta` | $31.77\ \mu\text{s}$ | $1.11\times$ | 64 |
| `SafeAdaptiveDelta` | $32.39\ \mu\text{s}$ | $1.13\times$ | 64 |
| `OnlineRidge` | $37.45\ \mu\text{s}$ | $1.31\times$ | 64 |
| `FrozenMLP` | $40.56\ \mu\text{s}$ | $1.42\times$ | 76 |
| `SelfReferential` | $58.33\ \mu\text{s}$ | $2.04\times$ | 66 |
| `GRU` | $58.80\ \mu\text{s}$ | $2.06\times$ | 96 |
| `SafeSelfReferential` | $61.02\ \mu\text{s}$ | $2.13\times$ | 66 |
| `LSTM` | $72.75\ \mu\text{s}$ | $2.55\times$ | 120 |

DeltaCore models incur minimal computational overhead: `SafeAdaptiveDelta` requires only $+3.81\ \mu\text{s}$ (+13%) over a static linear model while updating an $8 \times 8$ state matrix at every timestep; compared to LSTM ($72.75\ \mu\text{s}$), `SafeAdaptiveDelta` ($32.39\ \mu\text{s}$) runs $72.75 / 32.39 \approx 2.25\times$ faster.


---

## 8. Machine-Readable Artifact Manifest

All benchmark results and diagnostic plots have been generated and serialized into `docs/benchmarks/artifacts/phase_10/`:
1. `phase_10_results.json`: Complete aggregated summary metrics across all tasks and models.
2. `phase_10_per_seed.json`: Per-seed raw measurements across seeds `0, 1, 2, 3, 4`.
3. `phase_10_config.json`: Benchmark specification, dimensions, and hyperparameters.
4. `plot_ad_error_over_time.png` (Plot AD)
5. `plot_ae_recovery_distributions.png` (Plot AE)
6. `plot_af_adaptation_vs_forgetting.png` (Plot AF)
7. `plot_ag_performance_vs_delay.png` (Plot AG)
8. `plot_ah_state_norm_trajectories.png` (Plot AH)
9. `plot_ai_adaptation_energy.png` (Plot AI)
10. `plot_aj_performance_vs_params.png` (Plot AJ)
11. `plot_ak_adaptive_on_vs_off.png` (Plot AK)
12. `plot_al_continuous_vs_reset.png` (Plot AL)
13. `plot_am_runtime_per_timestep.png` (Plot AM)

All 483 regression and property tests in the DeltaCore test suite pass cleanly (`python3 -m pytest -q`), and `ruff check .` and `ruff format --check .` report zero violations.
