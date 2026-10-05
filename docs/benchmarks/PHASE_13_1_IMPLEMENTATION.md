# DeltaCore Phase 13.1: Benchmark Consistency & Scientific Claim Audit — Implementation Report

This report documents the implementation, execution, and methodology for **Phase 13.1: Real-World Benchmark Consistency & Scientific Claim Audit**.

Phase 13.1 introduced **no new model architectures, no additional datasets, and no hyperparameter tuning**. Its sole purpose is to perform a rigorous forensic audit of the empirical, numerical, statistical, and code-integrity foundations of Phase 13.

---

## 1. Audit Scope & Boundary Enforcement

In strict compliance with Phase 13.1 directives:
* **No Dataset Expansion**: Evaluation remained bounded to NOAA OISST v2.1 Equatorial Pacific SST.
* **No Model Additions**: Evaluated solely the existing 9-model matrix (`Persistence`, `FrozenLinear`, `OnlineRidge`, `NonlinearOnlineRidge`, `GRU`, `FixedDelta`, `SafeAdaptiveDelta`, `SelectiveRetention`, `SpatialConv`).
* **No DeltaCore Modification**: `SafeAdaptiveDelta` parameters ($\eta_{\max} = 0.008, \rho = 1.90, \alpha_{\min} = 0.85, \gamma = 0.05$) were frozen.

---

## 2. Audit Suite Architecture

The forensic audit tool was executed across all components and produced six machine-readable JSON artifacts in [docs/benchmarks/artifacts/phase_13_1/](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_13_1/):

1. **`table_consistency.json`**: Cross-referenced every headline metric in Table 1 against Table 4 and individual seed outputs in `phase_13_per_seed.json`, resolving scope and measurement differences.
2. **`baseline_training_provenance.json`**: Audited offline training loss reduction, epochs, optimizers, parameter freeze checkpoints, and parameter immutability hashes for all offline baselines (`FrozenLinear`, `GRU`, `SpatialConv`).
3. **`permutation_equivariance.json`**: Implemented an explicit vector-level permutation equivariance test measuring:
   $$E_{\text{equiv}} = \frac{\|f(P X) - P f(X)\|_F}{\max(\|P f(X)\|_F, \epsilon)}$$
   across all 5 seeds.
4. **`phase_13_1_per_seed_verification.json`**: Evaluated seed-by-seed prediction error win rates for `SafeAdaptiveDelta` against all four primary baselines across seeds $[42, 43, 44, 45, 46]$.
5. **`phase_13_1_audit.json`**: Synthesized complete machine-readable audit verification across retention intervention deltas, unrounded percentage reductions, and spatial gradient error mechanics.
6. **`code_health.json`**: Verified repository-wide static typing and linters (0 Pyright errors, 0 Ruff errors, 571 passing unit tests).

---

## 3. Methodological Findings

### 3.1 Table Reconciliation
* Discrepancies between Table 1 and Table 4 were traced to differences between 5-seed aggregation (`Table 1`, $N=5$, full telemetry) and single-seed reference scaling (`Table 4`, $N=1$, seed 42, `track_fields = False`).

### 3.2 Baseline Optimization Audit
* Confirmed that `SpatialConv`, `FrozenLinear`, and `GRU` were genuinely trained via Adam optimizer on the training split, achieving $58.5\%$ to $69.2\%$ MSE loss reduction before test evaluation, with test-time parameter hashes remaining bit-for-bit immutable ($\Delta \theta = 0$).

### 3.3 Exact Permutation Equivariance
* Proved mathematically and verified empirically that `SafeAdaptiveDelta` and `FixedDelta` are **strictly permutation-equivariant** ($E_{\text{equiv}} \approx 7.7 \times 10^{-8}$, within FP32 roundoff), while `SpatialConv` violates equivariance ($E_{\text{equiv}} \approx 1.95$).

### 3.4 Spatial Gradient Error Mechanics
* Discovered that the apparent drop in SGE under coordinate permutation ($0.5930 \to 0.2798$) was an artifact of computing gradients on permuted coordinates without inverse un-shuffling, which inflated the denominator $\|\nabla \tilde{Y}\|_F$. In physical coordinates, permutation-equivariant models have identical SGE before and after permutation.

### 3.5 Retention Reduction Precision
* Recomputed the cumulative excess error reduction from machine-readable data:
  $$\frac{3.5100209 - 2.5987301}{3.5100209} \times 100\% = 25.96253\% \approx \mathbf{26\%}$$
  verifying the unrounded quantitative accuracy of the reported $26\%$ reduction.
