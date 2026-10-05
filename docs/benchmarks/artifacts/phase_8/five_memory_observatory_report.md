# DeltaCore State Observatory Report: FiveMemory on token_association

## 1. Experiment Metadata & Provenance

| Property | Value |
| :--- | :--- |
| **Model / Baseline** | `FiveMemory` |
| **Benchmark Task** | `token_association` |
| **Random Seed** | `42` |
| **Sequence Length** | 60 |
| **All States Finite** | `YES` |
| **Python Version** | `3.13.0` |
| **PyTorch Version** | `2.6.0` |
| **Platform / OS** | `macOS-26.1-arm64-arm-64bit-Mach-O` |
| **Hardware Device** | `cpu` |
| **Floating-Point Precision** | `float32` |

## 2. Trajectory Summary & Fingerprint

| Metric Axis | Value | Formal Definition / Units |
| :--- | :--- | :--- |
| Mean Error $\bar{E}$ | 1.3101 | Arithmetic mean of $||e_t||_2$ |
| Final Error $E_T$ | 1.5578 | Sequence terminal $||e_T||_2$ |
| Recovery Latency $T_{\text{FP}}$ | 1 | Steps to 50% shock error reduction |
| Sustained Recovery $T_{\text{sust}}$ | 25 | Consecutive window error maintenance |
| Mean Step Size $\bar{\eta}$ | 0.1932 | Mean adaptive rate $\eta_t$ |
| Step Size Variance | 0.003848 | Sample variance $\text{Var}(\eta_t)$ |
| Total Update Energy $U_M$ | 35.8550 | Cumulative $\sum ||\Delta M_t||_F$ |
| Peak Update $\max ||\Delta M_t||_F$ | 3.3501 | Peak single-step state displacement |
| Max State Norm $\max ||M_t||_F$ | 3.0609 | Peak state Frobenius norm |
| Min Stability Margin $\min S_t$ | 0.1000 | Minimum $2 - \eta_t ||k_t||^2$ |
| Divergence Step $t^*$ | None (Finite) | First non-finite step index |

## 3. Detected Trajectory Events

| Step | Event Type | Value | Context |
| :--- | :--- | :--- | :--- |
| 2 | `clipping_event` | 0.1464 | normalized_step=1.899999737739563 |
| 5 | `clipping_event` | 0.2072 | normalized_step=1.899999737739563 |
| 7 | `minimum_stability_margin` | 0.1000 | contractive=True |
| 7 | `clipping_event` | 0.1533 | normalized_step=1.8999998569488525 |
| 13 | `clipping_event` | 0.2670 | normalized_step=1.899999737739563 |
| 14 | `clipping_event` | 0.2091 | normalized_step=1.899999737739563 |
| 18 | `clipping_event` | 0.2187 | normalized_step=1.8999998569488525 |
| 23 | `clipping_event` | 0.1846 | normalized_step=1.8999998569488525 |
| 31 | `maximum_update` | 3.3501 | total_steps=60 |
| 31 | `maximum_state_norm` | 3.0609 | initial_norm=1.2598745822906494 |
| 31 | `clipping_event` | 0.1489 | normalized_step=1.899999737739563 |

## 4. Adaptation Response & Distribution Shift

| Metric | Value | Interpretation Rule |
| :--- | :--- | :--- |
| Pre-Shift Error | N/A | Baseline accuracy prior to perturbation |
| Shock Error | 2.3798 | Immediate post-perturbation error |
| Recovery Slope | 1.5922 | Empirical error reduction rate |
| Post-Shift Energy | 35.8550 | Update budget expended post-shock |
| State Growth Ratio | 2.4295 | Relative state expansion $G_M$ |

## 5. Stability & Contraction Invariants

| Invariant Property | Assessment | Contractive Bound |
| :--- | :--- | :--- |
| Content Margin $\min S_t$ | 0.1000 | Contractive if $S_t \ge 0$ |
| Dynamics Margin $\min S_t^{\text{dyn}}$ | N/A | Contractive if $S_t^{\text{dyn}} \ge 0$ |
| Max Normalized Step $\max \gamma_t$ | 1.9000 | Non-expansive if $\le 2.0$ |
| Active Clip Count | 8 | Steps where safety controller intervened |
| Terminal State Finite | `True` | Strict finiteness check at $t=T-1$ |

## 6. Replay & Provenance Verification

*Replay was not executed for this report.*

## 7. Generated Visual Artifacts

- **Plot A**: [plot_a_error_vs_time.png](/Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_8/plots/plot_a_error_vs_time.png)
- **Plot B**: [plot_b_step_size_vs_time.png](/Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_8/plots/plot_b_step_size_vs_time.png)
- **Plot C**: [plot_c_memory_norm_vs_time.png](/Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_8/plots/plot_c_memory_norm_vs_time.png)
- **Plot D**: [plot_d_update_norm_vs_time.png](/Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_8/plots/plot_d_update_norm_vs_time.png)
- **Plot E**: [plot_e_stability_margin_vs_time.png](/Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_8/plots/plot_e_stability_margin_vs_time.png)

## 8. Epistemic Guardrails & Scientific Limitations

1. **Absence of Proof from Finite Sequences**: Numerical survival over the tested sequence length does not constitute mathematical proof of global Lyapunov stability. Bounds verified here are local sufficient conditions.
2. **Determinism Boundaries**: Exact numerical bit-for-bit equivalence is only guaranteed within identical hardware architecture, PyTorch build, and seed. Cross-platform floating point divergence is expected.
3. **Non-Causal Event Attribution**: Temporal event ordering (e.g. update peak preceded recovery) indicates correlation and sequence order, not proven causal necessity.
4. **No Automated Winner Designations**: Performance involves multi-objective trade-offs. Fast adaptation generally expends higher state update energy and operates with lower stability margins.
