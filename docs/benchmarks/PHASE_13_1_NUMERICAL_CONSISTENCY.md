# DeltaCore Phase 13.1: Numerical Consistency & Cross-Table Audit

This document provides a forensic audit and reconciliation of every numerical value reported in the **Phase 13 Real-World Spatio-Temporal Adaptive State Benchmark** across all documentation tables and JSON artifacts in [docs/benchmarks/artifacts/phase_13/](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_13/).

---

## 1. Cross-Table & JSON Alignment Audit

Every headline metric in Phase 13 documentation has been cross-referenced against the raw benchmark output files:
* `phase_13_results.json` (5-seed statistical aggregation)
* `phase_13_per_seed.json` (individual seed metrics for seeds 42–46)
* `phase_13_scaling.json` (multi-resolution benchmark on reference seed 42)
* `phase_13_shift_analysis.json` (retention intervention, spatial permutation control, and trajectory data)

### 1.1 Table 1: Main Benchmark Verification ($D=64$)

| Model | Table 1 Rel Error | `phase_13_results.json` | Match | Table 1 Shift Error | `phase_13_results.json` | Match | Table 1 Latency | `phase_13_results.json` | Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.2902 | 0.2901587 | **EXACT** | 0.2991 | 0.2990964 | **EXACT** | $1.09\,\mu\text{s}$ | $1.08635\,\mu\text{s}$ | **EXACT** |
| **FrozenLinear** | 0.2749 | 0.2749262 | **EXACT** | 0.3067 | 0.3067172 | **EXACT** | $5.22\,\mu\text{s}$ | $5.22456\,\mu\text{s}$ | **EXACT** |
| **OnlineRidge** | 0.3003 | 0.3002584 | **EXACT** | 0.3118 | 0.3118099 | **EXACT** | $1.85\,\mu\text{s}$ | $1.85497\,\mu\text{s}$ | **EXACT** |
| **NonlinearOnlineRidge** | 0.4808 | 0.4808041 | **EXACT** | 0.4514 | 0.4514224 | **EXACT** | $8.21\,\mu\text{s}$ | $8.20672\,\mu\text{s}$ | **EXACT** |
| **GRU** | 0.3764 | 0.3763763 | **EXACT** | 0.4370 | 0.4370127 | **EXACT** | $33.46\,\mu\text{s}$ | $33.46028\,\mu\text{s}$ | **EXACT** |
| **FixedDelta** | 0.2692 | 0.2691871 | **EXACT** | 0.2564 | 0.2564103 | **EXACT** | $1.84\,\mu\text{s}$ | $1.84442\,\mu\text{s}$ | **EXACT** |
| **SafeAdaptiveDelta** | **0.2582** | 0.2581615 | **EXACT** | **0.2485** | 0.2484705 | **EXACT** | $2.84\,\mu\text{s}$ | $2.83583\,\mu\text{s}$ | **EXACT** |
| **SelectiveRetention** | 0.2669 | 0.2669430 | **EXACT** | 0.2526 | 0.2526195 | **EXACT** | $2.21\,\mu\text{s}$ | $2.21023\,\mu\text{s}$ | **EXACT** |
| **SpatialConv** | 0.3285 | 0.3285338 | **EXACT** | 0.3863 | 0.3863460 | **EXACT** | $28.69\,\mu\text{s}$ | $28.69018\,\mu\text{s}$ | **EXACT** |

---

## 2. Reconciliation of Table 1 vs. Table 4

The apparent discrepancies identified during pre-audit between Table 1 (Main Benchmark) and Table 4 (Resolution Scaling) arise from differences in **measurement scope, seed aggregation, and training epoch configuration**:

### 2.1 Latency Measurements
* **OnlineRidge**: Table 1 reports **$1.85\,\mu\text{s}$**; Table 4 reports **$1.76\,\mu\text{s}$**.
  * *Root Cause*: Table 1 represents the mean per-token step latency averaged across all **5 seeds** (`seeds = (42, 43, 44, 45, 46)`), where per-seed latencies were $[1.94, 1.83, 1.85, 1.77, 1.88]\,\mu\text{s}$. Table 4 represents the latency measured during the multi-resolution scaling benchmark executed exclusively on reference **seed 42** without spatial field tracking overhead (`track_fields = False`).
  * *Reconciliation*: Both measurements are valid and accurately reflect their specified operational scopes.
* **SafeAdaptiveDelta**: Table 1 reports **$2.84\,\mu\text{s}$**; Table 4 reports **$1.91\,\mu\text{s}$**.
  * *Root Cause*: Table 1 averages step latencies across all 5 seeds during the comprehensive benchmark loop where full field anomaly tracking was active ($[2.96, 2.76, 2.82, 2.79, 2.85]\,\mu\text{s}$). In Table 4, the scaling run was executed with `track_fields = False`, reducing Python execution overhead on single-step inference.

### 2.2 Prediction Error for SpatialConv
* **SpatialConv Relative Error**: Table 1 reports **$0.3285$**; Table 4 reports **$0.3351$**.
  * *Root Cause*: In Table 1, $0.3285 \pm 0.0173$ is the mean across all 5 seeds (per-seed values: seed 42 = $0.3470$, seed 43 = $0.3228$, seed 44 = $0.3493$, seed 45 = $0.3098$, seed 46 = $0.3138$). In the scaling suite (`scaling_results` in `deltacore/streaming/real_benchmark.py`), `SpatialConv` was trained for $20$ epochs on seed 42 (yielding $0.3351$), whereas in the main benchmark loop it was trained for $25$ epochs on seed 42 (yielding $0.3470$).
  * *Resolution*: Table 1 accurately reflects multi-seed aggregation ($N=5$). Table 4 reflects single-seed reference scaling ($N=1$).

---

## 3. Seed-Level Consistency & Win-Rate Verification

Phase 13 claim audit: *"Did SafeAdaptiveDelta win every seed or only on average?"*

Below are the exact per-seed relative prediction errors ($E_{\text{rel}}$) extracted from [phase_13_per_seed.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_13/phase_13_per_seed.json):

| Seed | Persistence | FrozenLinear | OnlineRidge | FixedDelta | SafeAdaptiveDelta | SpatialConv | Lowest Error Model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 0.2902 | 0.2796 | 0.3003 | 0.2692 | **0.2582** | 0.3470 | **SafeAdaptiveDelta** |
| **43** | 0.2902 | 0.2741 | 0.3003 | 0.2692 | **0.2582** | 0.3228 | **SafeAdaptiveDelta** |
| **44** | 0.2902 | 0.2764 | 0.3003 | 0.2692 | **0.2582** | 0.3493 | **SafeAdaptiveDelta** |
| **45** | 0.2902 | 0.2741 | 0.3003 | 0.2692 | **0.2582** | 0.3098 | **SafeAdaptiveDelta** |
| **46** | 0.2902 | 0.2704 | 0.3003 | 0.2692 | **0.2582** | 0.3138 | **SafeAdaptiveDelta** |

### Per-Seed Win Rates:
* `SafeAdaptiveDelta` vs. `FixedDelta`: **5 / 5 wins** ($100\%$)
* `SafeAdaptiveDelta` vs. `OnlineRidge`: **5 / 5 wins** ($100\%$)
* `SafeAdaptiveDelta` vs. `Persistence`: **5 / 5 wins** ($100\%$)
* `SafeAdaptiveDelta` vs. `FrozenLinear`: **5 / 5 wins** ($100\%$)
* `SafeAdaptiveDelta` vs. `SpatialConv`: **5 / 5 wins** ($100\%$)

**Scientific Finding**: `SafeAdaptiveDelta` achieved the lowest prediction error across **100% of tested seeds** ($5/5$). The statement *"achieved the lowest prediction error across all five tested seeds"* is fully verified.

---

## 4. Controlled State-Reset Intervention & the 26% Result

The controlled state-reset intervention at the changepoint boundary ($t=40$) was audited from raw unrounded metrics in [phase_13_shift_analysis.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_13/phase_13_shift_analysis.json):

* **Continuous State Cumulative Excess Error**:
  $$C_{\text{continuous}} = 2.598730149548588$$
* **State Reset Cumulative Excess Error**:
  $$C_{\text{reset}} = 3.5100209005700638$$
* **Exact Difference**:
  $$\Delta_{\text{cumulative}} = C_{\text{continuous}} - C_{\text{reset}} = -0.9112907510214758$$
* **Exact Percentage Reduction**:
  $$\frac{3.5100209005700638 - 2.598730149548588}{3.5100209005700638} \times 100\% = 25.96253\%$$

Rounding to integer precision yields **$26\%$**.
* **Epistemic Qualification**: This result is formally documented as:
  > *"lower cumulative excess error in the tested 2015–2016 shift interval under controlled state-reset intervention."*

---

## 5. D=256 Resource Verification

From [phase_13_scaling.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_13/phase_13_scaling.json) under `medium_D256`:
* **Persistent State Memory**:
  * `OnlineRidge`: $P_t \in \mathbb{R}^{256 \times 256}$ and $W_t \in \mathbb{R}^{256 \times 256} \implies 2 \times 256^2 \times 4\text{ bytes} = 524,288\text{ bytes} = \mathbf{512\text{ KB}}$.
  * `SafeAdaptiveDelta`: $M_t \in \mathbb{R}^{256 \times 256} \implies 256^2 \times 4\text{ bytes} = 262,144\text{ bytes} = \mathbf{256\text{ KB}}$.
  * **Memory Ratio**:
    $$\frac{524,288}{262,144} = 2.000 \implies \mathbf{50.0\%\text{ memory reduction}}$$
* **Step Latency**:
  * `OnlineRidge`: $4.7061956\,\mu\text{s}$ per token.
  * `SafeAdaptiveDelta`: $4.6342133\,\mu\text{s}$ per token.
  * **Latency Ratio**:
    $$\frac{4.7061956}{4.6342133} = 1.0155 \implies \mathbf{1.6\%\text{ lower latency}}$$

Both measurements share an identical timing scope (`track_fields = False`, CPU streaming, reference seed 42).
