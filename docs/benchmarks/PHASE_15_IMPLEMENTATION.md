# DeltaCore — Phase 15 Implementation Specification
## Adaptive Regime Transfer & Cross-Domain Robustness

### 1. Architectural Mission & Theoretical Objective

Phase 15 evaluates the core scientific question:
$$\boxed{\text{Can one DeltaCore configuration transfer across slow thermal and fast advective dynamics?}}$$

Following the domain divergence observed between Phase 13 (NOAA OISST sea-surface temperature) and Phase 14 (ERA5 $2\mathrm{m}$ air temperature), Phase 15 conducts a rigorous, multi-faceted transfer and robustness analysis. Phase 15 introduces **no third dataset**, introduces **no larger neural architectures** (no transformers, no learned visual encoders), and relies on pure-PyTorch mathematical primitives.

---

### 2. Pre-Flight Freeze & Mathematical Formalism

#### 2.1 Cryptographic Pre-Flight Baseline
Prior to running Phase 15 experiments, the historical Phase 13 and Phase 14 numerical artifacts were audited and frozen in `docs/benchmarks/PHASE_15_PRE_FLIGHT.md`. SHA-256 digests verify that baseline artifacts remain bit-for-bit immutable.

#### 2.2 Mathematical Definition of SafeAdaptiveDelta
SafeAdaptiveDelta maintains a linear parameter matrix $W_{\mathrm{base}} \in \mathbb{R}^{D \times D}$ and an online associative state memory $M_t \in \mathbb{R}^{D \times D}$.
At each discrete streaming step $t$:
1. **Prediction Step**:
   $$\hat{y}_t = W_{\mathrm{base}} x_t + M_t x_t$$
2. **Prediction Error**:
   $$e_t = y_t - \hat{y}_t$$
3. **Adaptive Step-Size Control**:
   The candidate step size incorporates gradient error magnitude $\gamma$:
   $$\eta_{\mathrm{cand}, t} = \frac{\eta_0}{1 + \gamma \|e_t\|_2}$$
   Subject to the strict contraction stability bound $\rho < 2.0$:
   $$\eta_t = \min\left(\eta_{\mathrm{cand}, t}, \frac{\rho}{\|x_t\|_2^2 + \epsilon}\right)$$
4. **State Retention Decay**:
   $$\alpha_t = \max\left(\alpha_{\min}, 1 - \eta_t \|x_t\|_2^2\right)$$
5. **State Transition Update**:
   $$M_{t+1} = \alpha_t M_t + \eta_t e_t x_t^\top$$

The contraction safety margin is defined as:
$$\text{Margin}_t = 2.0 - \eta_t \|x_t\|_2^2 \ge 2.0 - \rho > 0$$

---

### 3. Protocol for Experiments A through H

#### 3.1 Experiment A: Cross-Domain Parameter Transfer
- **Domain A**: NOAA OISST Equatorial Pacific SST (slow thermal diffusion regime).
- **Domain B**: ERA5 North Atlantic / European $2\mathrm{m}$ air temperature (fast synoptic advection regime).
- Primary SafeAdaptiveDelta configuration calibrated on Domain A validation partition ($\eta_0=0.008, \rho=1.5, \alpha_{\min}=0.95$). Evaluated frozen on Domain B test stream ($t \in [0, 150]$).
- Reciprocal configuration calibrated on Domain B validation partition ($\eta_0=0.015, \rho=1.5, \alpha_{\min}=0.95$). Evaluated frozen on Domain A test stream.

#### 3.2 Experiment B: Zero-Retuning Comparison
Four fixed configurations are benchmarked across both domains with zero retuning on test partitions:
1. **Config 1 (Frozen Phase 13)**: $\eta_0=0.008, \rho=1.9, \alpha_{\min}=0.85$.
2. **Config 2 (Domain B Calibrated)**: $\eta_0=0.015, \rho=1.5, \alpha_{\min}=0.95$.
3. **Config 3 (Pooled Validation Calibrated)**: $\eta_0=0.015, \rho=1.5, \alpha_{\min}=0.95$.
4. **Domain A Calibrated**: $\eta_0=0.008, \rho=1.5, \alpha_{\min}=0.95$.

#### 3.3 Experiment C: Controlled Aggressiveness Sweep
Predetermined validation-only parameter grid:
$$\alpha_{\min} \in \{0.70, 0.85, 0.95\}, \quad \rho \in \{1.5, 1.9\}, \quad \eta_0 \in \{0.008, 0.015\}$$
Evaluated strictly on validation partitions to characterize how adaptation aggressiveness trades off against thermal noise amplification.

#### 3.4 Experiment D: Adaptation / Stability Pareto Frontier
For each sweep configuration, two vector metrics are computed without composite scalar reduction:
1. Multi-objective coordinate: $(E_{\mathrm{shift}}, \text{Numerical Risk})$, where:
   $$\text{Numerical Risk} = \frac{\eta_{\mathrm{eff}} \cdot \max_t \|M_t\|_F}{2}$$
2. Multi-objective coordinate: $(\text{Recovery Time}, E_{\mathrm{adapt}})$, where:
   $$E_{\mathrm{adapt}} = \sum_t \|\Delta M_t\|_F^2$$

#### 3.5 Experiment E: Regime-Switch Transfer Stream
Constructs a controlled continuous stream with two abrupt dynamic transitions:
$$\text{Phase 1: OISST } (t \in [0, 70)) \longrightarrow \text{Phase 2: ERA5 } (t \in [70, 140)) \longrightarrow \text{Phase 3: OISST } (t \in [140, 210))$$
Both domains are normalized to standardized zero-mean, unit-variance coordinates per channel to ensure physically compatible mathematical dimensions ($D=64$) while abruptly alternating the temporal autocorrelation structure.

#### 3.6 Experiment F: State Retention Under Transfer
Ablation of retention modes on both domains:
1. **Continuous State**: Online state memory $M_t$ is preserved continuously throughout the test sequence.
2. **Reset State**: State memory $M_t$ is wiped ($M_{\tau} \leftarrow 0$) at the onset of the regime shift.
3. **Retain-High State**: State memory is preserved with conservative decay ($\alpha=0.95$).

#### 3.7 Experiment G: State Initialization Sensitivity
Evaluates predictor sensitivity to initial state condition $M_0$:
1. $M_0 = 0$ (Zero State).
2. $M_0 \sim \mathcal{N}(0, \sigma^2 I)$ with $\sigma=0.01$ (Small Random State).
3. $M_0 = M_{\mathrm{final}}^{(A)}$ evaluated on Domain B (Transferred State from Domain A).
4. $M_0 = M_{\mathrm{final}}^{(B)}$ evaluated on Domain A (Transferred State from Domain B).

#### 3.8 Experiment H: Safe vs Unsafe Failure Boundary at $D=256$
Tests numerical stability at scaled dimension $D=256$:
- Compares `FixedDeltaPredictor` ($\eta = 0.008$) against `SafeAdaptiveDeltaPredictor` ($\eta_0 = 0.008, \rho = 1.90$).
- Tracks:
  - Divergence flag (`diverged: bool`).
  - Step index of non-finite value ($t_{\mathrm{nonfinite}}$).
  - Maximum Frobenius state norm $\max_t \|M_t\|_F$.
  - Minimum local contraction safety margin $\min_t (2.0 - \eta_t \|x_t\|_2^2)$.

---

### 4. Implementation Source Modules

The Phase 15 implementation is organized across four core modules:
1. `deltacore/streaming/models.py`:
   - Enhanced `SafeAdaptiveDeltaPredictor` supporting configurable `rho`, `alpha_min`, `gamma`, and `set_state(M: torch.Tensor)`.
   - Enhanced `FixedDeltaPredictor` supporting `set_state(M: torch.Tensor)`.
2. `deltacore/streaming/regime_transfer.py`:
   - Dual-domain data loader (`load_cross_domain_datasets`).
   - Validation grid calibrator (`calibrate_validation_grid`).
   - Regime switch stream generator (`generate_regime_switch_stream`).
   - State initialization sensitivity harness (`evaluate_state_initialization`).
   - Numerical failure boundary evaluator (`evaluate_failure_boundary`).
3. `deltacore/streaming/phase_15_benchmark.py`:
   - Orchestrates Experiments A through H across five random seeds `[42, 43, 44, 45, 46]`.
   - Serializes the 6 required JSON artifacts into `docs/benchmarks/artifacts/phase_15/`.
4. `deltacore/observatory/phase_15_plots.py`:
   - Implements rendering routines for all 10 Observatory publication figures CG through CP.

---

### 5. Verification & Testing Protocol

The test suite `tests/test_phase_15_regime_transfer.py` enforces:
1. Zero test leakage during validation calibration.
2. Exact dimensional matching ($D=64$ and $D=256$) across domains.
3. Offline parameter immutability throughout cross-domain streaming.
4. Correct tracking of transferred state initialization.
5. Strict contractive safety margin enforcement ($\text{Margin}_t \ge 0.10$).
6. Schema integrity for all 6 JSON artifacts and 10 publication figures.
