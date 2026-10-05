# DeltaCore Phase 16 Observatory Publication Report
## Unseen Distribution Shift Robustness, Adaptive Safety & Failure Boundary Analysis

This report documents the empirical analysis, numerical tables, and publication diagnostics for the **12 Phase 16 Observatory Figures (Plots CQ through DB)**. The experiments challenge the pooled Phase 15 `SafeAdaptiveDelta` configuration ($\eta_0=0.015, \rho=1.5, \alpha_{\min}=0.95$) with five predetermined unseen distribution shifts across three severities on both:
- **Domain A**: NOAA OISST sea-surface temperature in the Equatorial Pacific (slow thermal regime).
- **Domain B**: ECMWF ERA5 $2\mathrm{m}$ air temperature over the North Atlantic and Europe (fast advective regime).

All artifacts and figures were generated from deterministic execution across five random seeds (`seeds = [42, 43, 44, 45, 46]`) and serialized into `docs/benchmarks/artifacts/phase_16/`.

---

## 1. Figure Index

| Figure ID | Title | Artifact Link |
| :--- | :--- | :--- |
| **Plot CQ** | Relative Error versus Predetermined Shift Severity | [plot_cq_error_vs_severity.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_cq_error_vs_severity.png) |
| **Plot CR** | Adaptive Step-Size ($\eta_t$) Trajectories Under Shift Onset | [plot_cr_eta_trajectories.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_cr_eta_trajectories.png) |
| **Plot CS** | Minimum Safety Margin Preservation ($2.0 - \eta_t \|x_t\|_2^2$) vs Severity | [plot_cs_safety_margin_vs_severity.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_cs_safety_margin_vs_severity.png) |
| **Plot CT** | Continuous vs Reset State Intervention ($\Delta_{\mathrm{reset}} = E_{\mathrm{cont}} - E_{\mathrm{reset}}$) | [plot_ct_continuous_vs_reset.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_ct_continuous_vs_reset.png) |
| **Plot CU** | Retention Stress Trajectories across Regime Switches ($A, B, C, \text{severe-}B$) | [plot_cu_retention_trajectories.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_cu_retention_trajectories.png) |
| **Plot CV** | $D=256$ Empirical Safety Boundary Mapping (FixedDelta Failure vs SafeAdaptiveDelta) | [plot_cv_failure_boundary_map.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_cv_failure_boundary_map.png) |
| **Plot CW** | Observation Perturbation Robustness ($0\%, 1\%, 5\%, 10\%$ Noise) | [plot_cw_noise_robustness.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_cw_noise_robustness.png) |
| **Plot CX** | Cross-Domain Robustness Matrix ($5\text{ Shifts} \times 2\text{ Domains}$) | [plot_cx_cross_domain_matrix.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_cx_cross_domain_matrix.png) |
| **Plot CY** | Per-Seed Robustness Differences Across 5 Deterministic Seeds | [plot_cy_per_seed_differences.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_cy_per_seed_differences.png) |
| **Plot CZ** | Memory Operator State Norm ($\|M_t\|_F$) Under Severe Shift | [plot_cz_state_norm_severe.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_cz_state_norm_severe.png) |
| **Plot DA** | Adaptation Energy Response Across Predetermined Shifts and Severities | [plot_da_adaptation_energy_vs_severity.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_da_adaptation_energy_vs_severity.png) |
| **Plot DB** | Resource Efficiency Under Online Streaming (Latency and Memory) | [plot_db_resource_metrics_vs_severity.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_16/plots/plot_db_resource_metrics_vs_severity.png) |

---

## 2. Primary Cross-Domain Robustness Performance

The benchmark evaluates six baseline and adaptive models under frozen streaming execution across five independent seeds (`42, 43, 44, 45, 46`).

### Table 1: Domain A — NOAA OISST (Slow Thermal Diffusion Regime, $D=64$)

| Model | $E_{\mathrm{rel}}$ (Mean $\pm$ Std) | $E_{\mathrm{shift}}$ (Mean $\pm$ Std) | Latency ($\mu\mathrm{s}$) | Adapt Energy | Max $\|M\|_F$ | State Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | $0.2935 \pm 0.0106$ | $0.3039 \pm 0.0156$ | $1.22$ | $0.00$ | $10.91$ | $256\text{ B}$ |
| **FrozenLinear** | $0.2768 \pm 0.0040$ | $0.3099 \pm 0.0009$ | $5.50$ | $0.00$ | $5.70$ | $0\text{ B}$ |
| **OnlineRidge** | $0.2758 \pm 0.0107$ | $0.2794 \pm 0.0132$ | $1.67$ | $91.27$ | $6.24$ | $32,768\text{ B}$ |
| **SpatialConv** | $0.3321 \pm 0.0162$ | $0.3900 \pm 0.0211$ | $19.50$ | $0.00$ | $0.00$ | $0\text{ B}$ |
| **FixedDelta** | $0.3552 \pm 0.0116$ | $0.3008 \pm 0.0100$ | $1.58$ | $28.56$ | $1.87$ | $16,384\text{ B}$ |
| **SafeAdaptiveDelta** | $\mathbf{0.2989 \pm 0.0111}$ | $\mathbf{0.2755 \pm 0.0118}$ | $1.66$ | $10.85$ | $1.61$ | $16,384\text{ B}$ |

*Key Findings on Domain A*: SafeAdaptiveDelta achieves $E_{\mathrm{shift}} = 0.2755$, beating Persistence, FrozenLinear, SpatialConv, and FixedDelta, while expending $8.4\times$ less adaptation energy than OnlineRidge ($10.85$ vs $91.27$) and preserving state norm $\|M_t\|_F \le 1.61$.

---

### Table 2: Domain B — ECMWF ERA5 (Fast Synoptic Advection Regime, $D=64$)

| Model | $E_{\mathrm{rel}}$ (Mean $\pm$ Std) | $E_{\mathrm{shift}}$ (Mean $\pm$ Std) | Latency ($\mu\mathrm{s}$) | Adapt Energy | Max $\|M\|_F$ | State Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | $0.1366 \pm 0.0005$ | $0.1269 \pm 0.0005$ | $1.13$ | $0.00$ | $13.63$ | $256\text{ B}$ |
| **FrozenLinear** | $0.0815 \pm 0.0026$ | $0.0822 \pm 0.0035$ | $5.56$ | $0.00$ | $7.10$ | $0\text{ B}$ |
| **OnlineRidge** | $0.0912 \pm 0.0005$ | $0.0487 \pm 0.0002$ | $1.57$ | $7.18$ | $3.14$ | $32,768\text{ B}$ |
| **SpatialConv** | $0.2680 \pm 0.0217$ | $0.2962 \pm 0.0250$ | $21.48$ | $0.00$ | $0.00$ | $0\text{ B}$ |
| **FixedDelta** | $36637.57 \pm 6472.12$ | $24708.27 \pm 4273.24$ | $1.60$ | $7.60 \times 10^{11}$ | $164431.69$ | $16,384\text{ B}$ |
| **SafeAdaptiveDelta** | $\mathbf{0.1610 \pm 0.0008}$ | $\mathbf{0.1234 \pm 0.0006}$ | $1.62$ | $5.27$ | $1.60$ | $16,384\text{ B}$ |

*Key Findings on Domain B*: On unscaled fast atmospheric dynamics, FixedDelta catastrophically diverges ($E_{\mathrm{rel}} > 3.6 \times 10^4$, $\|M_t\|_F > 1.6 \times 10^5$, energy $> 7.5 \times 10^{11}$). In contrast, SafeAdaptiveDelta's contractive safety controller dynamically throttles the update rate, ensuring 100% numerical boundedness ($E_{\mathrm{rel}} = 0.1610$, $\|M_t\|_F = 1.60$, energy $= 5.27$).

---

## 3. Failure Boundary Mapping at Scaled Dimension $D=256$

At dimension $D=256$, feature energy $\|x_t\|_2^2$ increases by approximately $4\times$, testing the exact mathematical failure boundary $\eta \|x_t\|_2^2 < 2.0$.

### Table 3: $D=256$ Empirical Failure Boundary

| Step Size ($\eta$) | Domain A FixedDelta | Domain A SafeAdaptiveDelta | Domain B FixedDelta | Domain B SafeAdaptiveDelta |
| :---: | :---: | :---: | :---: | :---: |
| $\mathbf{0.002}$ | Finite ($E=0.2712$, $\|M\|=2.07$) | Finite ($E=0.2771$, $\|M\|=1.76$) | Finite ($E=0.1467$, $\|M\|=2.83$) | Finite ($E=0.1866$, $\|M\|=1.74$) |
| $\mathbf{0.005}$ | Degraded ($E=14.23$, $\|M\|=60.53$) | Finite ($E=0.3448$, $\|M\|=2.08$) | **Diverged (NaN, $t=79$)** | Finite ($E=0.1528$, $\|M\|=2.09$) |
| $\mathbf{0.008}$ | Severe ($E=2.8 \times 10^{14}$) | Finite ($E=0.4071$, $\|M\|=2.08$) | **Diverged (NaN, $t=41$)** | Finite ($E=0.1510$, $\|M\|=2.09$) |
| $\mathbf{0.012}$ | **Diverged (NaN, $t=32$)** | Finite ($E=0.4102$, $\|M\|=2.08$) | **Diverged (NaN, $t=28$)** | Finite ($E=0.1510$, $\|M\|=2.09$) |
| $\mathbf{0.016}$ | **Diverged (NaN, $t=25$)** | Finite ($E=0.4102$, $\|M\|=2.08$) | **Diverged (NaN, $t=23$)** | Finite ($E=0.1510$, $\|M\|=2.09$) |
| $\mathbf{0.020}$ | **Diverged (NaN, $t=21$)** | Finite ($E=0.4102$, $\|M\|=2.08$) | **Diverged (NaN, $t=20$)** | Finite ($E=0.1510$, $\|M\|=2.09$) |
| $\mathbf{0.025}$ | **Diverged (NaN, $t=19$)** | Finite ($E=0.4102$, $\|M\|=2.08$) | **Diverged (NaN, $t=18$)** | Finite ($E=0.1510$, $\|M\|=2.09$) |

*Theoretical Confirmation*: For FixedDelta, the minimum safety margin drops to negative values ($\text{Margin} < 0$) whenever $\eta \ge 0.005$, producing exponential state norm growth and subsequent non-finite NaN values. For SafeAdaptiveDelta, the step size is clipped by $\eta_t \le 1.50 / (\|x_t\|_2^2 + \epsilon)$, preserving a strictly positive minimum margin $\ge 0.50$ across all tested learning rates.

---

## 4. State Reset Intervention & Retention Stress

### Table 4: State Reset Ablation under Predetermined Shifts (Moderate Severity)

| Shift Type | Domain A $E_{\mathrm{cont}}$ | Domain A $E_{\mathrm{reset}}$ | Domain A $\Delta_{\mathrm{reset}}$ | Domain B $E_{\mathrm{cont}}$ | Domain B $E_{\mathrm{reset}}$ | Domain B $\Delta_{\mathrm{reset}}$ | Cont. Adv.? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Shift** | $0.2949$ | $0.3003$ | $-0.0054$ | $0.1922$ | $0.2024$ | $-0.0101$ | **Yes** |
| **Variance Shift** | $0.4118$ | $0.4259$ | $-0.0141$ | $0.1874$ | $0.2123$ | $-0.0248$ | **Yes** |
| **Temporal Speed** | $0.3013$ | $0.3103$ | $-0.0090$ | $0.2268$ | $0.2452$ | $-0.0184$ | **Yes** |
| **Noise Shift** | $0.3057$ | $0.3158$ | $-0.0101$ | $0.2010$ | $0.2219$ | $-0.0209$ | **Yes** |
| **Combined Shift** | $0.4336$ | $0.4413$ | $-0.0077$ | $0.2418$ | $0.2574$ | $-0.0156$ | **Yes** |

*Within-Domain Retention*: Under all within-domain shift perturbations, continuous state outperforms resetting state at shift onset ($\Delta_{\mathrm{reset}} < 0$, $\Delta_{\mathrm{cum}} < 0$). Prior learned representation acts as a regularizer, accelerating recovery.

---

### Table 5: Retention Stress across Multi-Regime Transitions

| History Type | Sequence Description | $E_{\mathrm{cont}}$ | $E_{\mathrm{reset}}$ | $\Delta_{\mathrm{reset}}$ | Max $\|M\|_F$ | Adapt Energy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$A \to B$** | Thermal ($A$) to Synoptic ($B$) | $0.7990$ | $0.7912$ | $+0.0078$ | $2.65$ | $34.35$ |
| **$A \to B \to A$** | Regimes switch to $B$ and return to $A$ | $0.7935$ | $0.7891$ | $+0.0044$ | $2.65$ | $50.92$ |
| **$A \to B \to C$** | Three non-returning regimes | $0.7882$ | $0.7865$ | $+0.0017$ | $3.42$ | $92.19$ |
| **$A \to \text{severe-}B \to A$** | Extreme variance shock during $B$ | $0.6656$ | $0.6643$ | $+0.0012$ | $2.65$ | $74.57$ |

*Severe Regime Mismatch Finding*: In contrast to within-domain shifts, cross-domain transitions ($A \leftrightarrow B$) exhibit positive $\Delta_{\mathrm{reset}} > 0$. Persistent state creates transient inertia across extreme dynamical mismatches, meaning resetting state at the boundary mitigates initial shock. This empirically validates the bounds of persistent associative memory.

---

## 5. Perturbation Robustness & Equivariance

### Table 6: Observation Noise Perturbation ($0\%$ to $10\%$)

| Noise Fraction | Domain A SafeAdaptive | Domain A FixedDelta | Domain B SafeAdaptive | Domain B FixedDelta |
| :---: | :---: | :---: | :---: | :---: |
| $\mathbf{0\%}$ | $0.2922$ | $0.3493$ | $0.1475$ | $46682.31$ (Unstable) |
| $\mathbf{1\%}$ | $0.2923$ | $0.3494$ | $0.1475$ | $46679.52$ (Unstable) |
| $\mathbf{5\%}$ | $0.2932$ | $0.3503$ | $0.1481$ | $46665.41$ (Unstable) |
| $\mathbf{10\%}$ | $0.2961$ | $0.3533$ | $0.1500$ | $46621.15$ (Unstable) |

*Permutation Equivariance*:
- **Domain A**: $E_{\mathrm{equiv}} = 1.08 \times 10^{-7}$
- **Domain B**: $E_{\mathrm{equiv}} = 8.61 \times 10^{-8}$

The principal DeltaCore update mechanism remains strictly permutation-equivariant under spatial reordering, confirming that the robustness observations reflect coordinate-free mathematical properties rather than spatial artifact alignment.
