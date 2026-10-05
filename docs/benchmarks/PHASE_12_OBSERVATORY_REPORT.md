# DeltaCore Phase 12 Observatory Publication Report

This report presents the scientific analysis of the **13 Phase 12 Observatory Figures (Plots AX through BJ)** generated from empirical streaming evaluations on the controlled spatio-temporal benchmark.

All figures were generated into [docs/benchmarks/artifacts/phase_12/](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/) using pure PyTorch execution across five deterministic seeds (`seeds = [0, 1, 2, 3, 4]`).

---

## Figure Index

| Figure ID | Title | File Link |
| :--- | :--- | :--- |
| **Plot AX** | Spatio-Temporal Prediction Error Over Time | [plot_ax_error_over_time.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_ax_error_over_time.png) |
| **Plot AY** | Adaptation vs. Negative-Transfer Pareto Frontier | [plot_ay_pareto_frontier.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_ay_pareto_frontier.png) |
| **Plot AZ** | Retention Coefficient Trajectories ($\alpha_t$) | [plot_az_retention_trajectories.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_az_retention_trajectories.png) |
| **Plot BA** | Retention Mode Ablation Analysis | [plot_ba_retention_ablation.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_ba_retention_ablation.png) |
| **Plot BB** | $A_1 \to B \to A_1$ Forgetting and Recovery | [plot_bb_a1_b_a1_recovery.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bb_a1_b_a1_recovery.png) |
| **Plot BC** | Error-Only vs. State-Conditioned Retention Controller | [plot_bc_state_adaptive_comparison.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bc_state_adaptive_comparison.png) |
| **Plot BD** | Spatial Structure Ablation (Order vs. $\mathcal{P}_{\mathrm{spatial}}$) | [plot_bd_spatial_vs_shuffled.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bd_spatial_vs_shuffled.png) |
| **Plot BE** | Prediction Error vs. Dimension $D$ | [plot_be_perf_vs_dim.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_be_perf_vs_dim.png) |
| **Plot BF** | Step Latency vs. Dimension $D$ | [plot_bf_runtime_vs_dim.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bf_runtime_vs_dim.png) |
| **Plot BG** | Persistent State Memory vs. Dimension $D$ | [plot_bg_memory_vs_dim.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bg_memory_vs_dim.png) |
| **Plot BH** | Adaptation Energy vs. Prediction Error | [plot_bh_energy_vs_performance.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bh_energy_vs_performance.png) |
| **Plot BI** | Safe vs. Unsafe Stability Diagnostics | [plot_bi_safe_vs_unsafe.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bi_safe_vs_unsafe.png) |
| **Plot BJ** | Causal Control B: Continuous State vs. State Reset | [plot_bj_continuous_vs_reset.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bj_continuous_vs_reset.png) |

---

## Detailed Visual Analysis

### Plot AX — Spatio-Temporal Prediction Error Over Time ($A \to B \to C \to A$)
* **Location**: [plot_ax_error_over_time.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_ax_error_over_time.png)
* **Description**: Displays the relative error $E_{\mathrm{rel}}(t)$ on a logarithmic scale across $T=512$ streaming steps, punctuated by regime transitions at $t \in \{128, 256, 384\}$.
* **Key Observations**:
  * Recurrent baselines (`GRU`, `LSTM`) and `FrozenMLP` exhibit significant error spikes ($E_{\mathrm{rel}} > 4.0$) and elevated stationary drift throughout the sequence.
  * `OnlineRidge` and `SafeAdaptiveDelta` track stationary dynamics reliably ($E_{\mathrm{rel}} \approx 0.985$), experiencing brief, well-damped transient perturbations during regime shifts.
  * The selective retention models (`Selective_adaptive` and `Selective_state_adaptive`) maintain controlled, bounded error trajectories across all transitions without unbounded divergence.

---

### Plot AY — Adaptation vs. Negative-Transfer Pareto Frontier
* **Location**: [plot_ay_pareto_frontier.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_ay_pareto_frontier.png)
* **Description**: Evaluates trade-offs between immediate post-shift adaptation error $E_{\mathrm{post},0}$ and stale-memory negative transfer $E_{\mathrm{cont}}(B) - E_{\mathrm{reset}}(B)$.
* **Key Observations**:
  * `Selective_state_adaptive` and `Selective_adaptive` occupy the empirical Pareto frontier, achieving non-dominated combinations of minimal negative transfer ($\le -1.3 \times 10^{-10}$) and bounded adaptation error.
  * `Selective_fixed_high` exhibits higher stale-state negative transfer ($+3.13 \times 10^{-7}$), confirming that unyielding historical retention imposes a measurable transfer penalty when underlying spatial transport directions rotate by $90^\circ$.

---

### Plot AZ — Retention Coefficient Trajectories ($\alpha_t$)
* **Location**: [plot_az_retention_trajectories.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_az_retention_trajectories.png)
* **Description**: Tracks the evolution of the retention coefficient $\alpha_t \in [\alpha_{\min}, 1.0]$ over time.
* **Key Observations**:
  * `Selective_fixed_high` ($\alpha = 0.99$) and `Selective_fixed_low` ($\alpha = 0.70$) remain static horizontal lines.
  * `Selective_oracle` maintains $\alpha_t = 1.0$ during stationary periods and drops to $\alpha_t = 0.0$ at changepoint boundaries.
  * `Selective_state_adaptive` dynamically modulates $\alpha_t$, maintaining high average retention ($\bar{\alpha} \approx 0.799$) during stationary field translation, while decreasing retention in response to compound error and state update surges.

---

### Plot BA — Retention Mode Ablation Analysis
* **Location**: [plot_ba_retention_ablation.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_ba_retention_ablation.png)
* **Description**: Dual-panel bar chart comparing post-shift error and negative transfer across all five retention modes.
* **Key Observations**:
  * Negative transfer drops from $+3.13 \times 10^{-7}$ (`retain_high`) to $-2.29 \times 10^{-10}$ (`adaptive_state_controller`), representing an elimination of stale-memory negative transfer across distribution shifts.
  * Both adaptive retention modes effectively eliminate positive transfer penalties without degrading numerical stability.

---

### Plot BB — $A_1 \to B \to A_1$ Forgetting and Recovery
* **Location**: [plot_bb_a1_b_a1_recovery.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bb_a1_b_a1_recovery.png)
* **Description**: Scatter plot of Phase B adaptation error versus forgetting on re-entry ($E_{\mathrm{return},A} - E_{\mathrm{pre},A}$).
* **Key Observations**:
  * Forgetting across DeltaCore models is close to zero ($|E_{\mathrm{forget}}| < 10^{-6}$), demonstrating that re-entering Regime A recovers baseline accuracy without catastrophic interference.
  * Models with higher adaptation capability in Phase B (e.g. `SafeAdaptiveDelta`) recover rapidly upon returning to Phase A1.

---

### Plot BC — Error-Only vs. State-Conditioned Retention Controller
* **Location**: [plot_bc_state_adaptive_comparison.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bc_state_adaptive_comparison.png)
* **Description**: Comparative trajectory analysis between `Selective_adaptive` (error-only gating), `Selective_state_adaptive` (5-parameter compact controller), and `Selective_shuffled_control` (Causal Control D).
* **Key Observations**:
  * `Selective_state_adaptive` achieves lower overall error (0.99916 vs. 0.99955) and lower excess error ($9.86 \times 10^{-8}$ vs. $2.42 \times 10^{-7}$) compared to error-only `Selective_adaptive`.
  * The state-conditioned controller uses context information ($\|M\|_F, \|\Delta M\|_F, \bar{e}$) to make smoother retention transitions, requiring less adaptation energy ($0.0764$ vs. $0.0847$).

---

### Plot BD — Spatial Structure Ablation (Order vs. $\mathcal{P}_{\mathrm{spatial}}$)
* **Location**: [plot_bd_spatial_vs_shuffled.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bd_spatial_vs_shuffled.png)
* **Description**: Compares streaming error under true 2D spatial arrangement versus an identical underlying stream subjected to a fixed spatial coordinate permutation $\mathcal{P}_{\mathrm{spatial}}$.
* **Key Observations**:
  * For `SafeAdaptiveDelta` and `OnlineRidge`, the performance difference between true spatial order and coordinate permutation is negligible ($\Delta \le 8 \times 10^{-10}$).
  * This confirms that linear and matrix-associative adaptive estimators operate as coordinate-permutation covariant operators: outer product state updates $x x^\top$ and $e \phi^\top$ preserve spectral invariants regardless of spatial coordinate indexing.

---

### Plot BE — Prediction Error vs. Dimension $D$
* **Location**: [plot_be_perf_vs_dim.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_be_perf_vs_dim.png)
* **Description**: Evaluates empirical prediction error scaling across $D \in \{64, 128, 256\}$.
* **Key Observations**:
  * At $D=64$: `OnlineRidge` (0.9853) and `SafeAdaptiveDelta` (0.9875) perform comparably.
  * At $D=128$: `OnlineRidge` (0.9908) and `SafeAdaptiveDelta` (0.9919) remain closely aligned, while `NonlinearOnlineRidge` error degrades to 1.0009.
  * At $D=256$: `SafeAdaptiveDelta` (0.9946) matches `OnlineRidge` (0.9942) within 0.0004, while substantially outperforming `NonlinearOnlineRidge` (1.0033).

---

### Plot BF — Step Latency vs. Dimension $D$
* **Location**: [plot_bf_runtime_vs_dim.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bf_runtime_vs_dim.png)
* **Description**: Step latency in microseconds per token ($\mu\mathrm{s}/\mathrm{token}$) as a function of dimension $D$.
* **Key Observations**:
  * At $D=256$, `OnlineRidge` step latency increases to 122.2 $\mu\mathrm{s}$/token due to inverse covariance rank-1 matrix products.
  * `SafeAdaptiveDelta` requires 103.8 $\mu\mathrm{s}$/token (15% faster).
  * `Selective_state_adaptive` achieves 58.6 $\mu\mathrm{s}$/token (>2x faster than `OnlineRidge`).

---

### Plot BG — Persistent State Memory vs. Dimension $D$
* **Location**: [plot_bg_memory_vs_dim.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bg_memory_vs_dim.png)
* **Description**: Memory footprint in bytes of persistent adaptive state across dimensions $D \in \{64, 128, 256\}$.
* **Key Observations**:
  * At $D=256$, `OnlineRidge` occupies 524,288 bytes ($512$ KB) for storing both $P_t$ and $W_t$.
  * `SafeAdaptiveDelta` occupies 262,144 bytes ($256$ KB), achieving a 2.0x constant-factor memory reduction.
  * `Selective_state_adaptive` occupies only 8,208 bytes ($8.2$ KB), providing a **63.8x memory advantage** over `OnlineRidge`.

---

### Plot BH — Adaptation Energy vs. Prediction Error
* **Location**: [plot_bh_energy_vs_performance.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bh_energy_vs_performance.png)
* **Description**: Examines cumulative state adaptation energy $E_{\mathrm{adapt}} = \sum_t \|\Delta S_t\|_F^2$ versus mean prediction error.
* **Key Observations**:
  * DeltaCore models maintain bounded adaptation energy ($E_{\mathrm{adapt}} \in [0.07, 0.22]$).
  * Recurrent and nonlinear RLS baselines expend substantially higher adaptation energy ($E_{\mathrm{adapt}} > 0.60$) without obtaining proportional error reductions.

---

### Plot BI — Safe vs. Unsafe Stability Diagnostics
* **Location**: [plot_bi_safe_vs_unsafe.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bi_safe_vs_unsafe.png)
* **Description**: Compares maximum state norms $\max_t \|S_t\|_F$ and minimum local stability margins $\min_t \mu_t$ between standard and safe Lyapunov-clamped variants.
* **Key Observations**:
  * `SafeAdaptiveDelta` maintains a contractive stability margin ($\min_t \mu_t \ge 1.75 > 0$) across all sequence steps.
  * State growth ratio is bounded across all seeds, and non-finite counts are strictly zero ($0$).

---

### Plot BJ — Causal Control B: Continuous State vs. State Reset
* **Location**: [plot_bj_continuous_vs_reset.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_12/plot_bj_continuous_vs_reset.png)
* **Description**: Evaluates post-shift error under continuous memory propagation versus oracle state reset at regime transitions.
* **Key Observations**:
  * In selective retention models, continuous state propagation matches the performance of oracle state reset, demonstrating that adaptive forgetting dynamically flushes obsolete representations without requiring external reset triggers.
