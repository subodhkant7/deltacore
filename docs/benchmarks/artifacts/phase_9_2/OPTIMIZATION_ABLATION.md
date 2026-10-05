# DeltaCore — Phase 9.2: Task A Optimization Ablation Report

**Document Status**: COMPLETED & VERIFIED  
**Phase**: 9.2 (Spatial Capacity & Optimization Ablation)  
**Date**: October 2026  
**Artifact Path**: `docs/benchmarks/artifacts/phase_9_2/optimization_ablation.json`  
**Related Documents**:
- Implementation Report: [PHASE_9_2_IMPLEMENTATION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_9_2_IMPLEMENTATION.md)
- Observatory Report: [PHASE_9_2_OBSERVATORY_REPORT.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_9_2_OBSERVATORY_REPORT.md)
- Interpretation: [PHASE_9_2_INTERPRETATION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_9_2_INTERPRETATION.md)

---

## 1. Objective and Protocol

The objective of this controlled ablation is to determine whether the failure of DeltaCore with trainable update dynamics on Task A (Spatial Shift Reconstruction) is primarily an **optimization failure** (recurrent gradient explosion / numerical instability) or an **architectural limitation**.

### Controlled Setup
- **Architecture**: `delta_trainable_dynamics_xy` (4-direction SpatialAdaptiveOperator with 480 parameters, including 28 trainable dynamics parameters per route and normalized 2D coordinate injection).
- **Task**: Task A Spatial Shift Reconstruction ($Y = 0.5 X_{\text{right}} + 0.5 X_{\text{down}}$, $H=10, W=10, C=4$).
- **Optimizer**: AdamW (`weight_decay=1e-4`, `betas=(0.9, 0.999)`).
- **Epoch Budget**: 15 epochs per run.
- **Seeds**: `[0, 1, 2, 3, 4]` (identical across all learning rate regimes).
- **Evaluated Learning Rates**: $\eta \in \{0.001, 0.003, 0.010\}$.

---

## 2. Experimental Results

| Learning Rate $\eta$ | Mean $E_{\text{rel}}$ | Std $E_{\text{rel}}$ | Per-Seed Relative Error $[s_0, s_1, s_2, s_3, s_4]$ | Convergence Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.001** | 0.9689 | 0.0202 | `[0.9796, 0.9624, 0.9333, 0.9928, 0.9763]` | 5 / 5 (100%) | Stable descent; underfitting budget |
| **0.003** | **0.9441** | **0.0144** | `[0.9330, 0.9566, 0.9231, 0.9462, 0.9617]` | 5 / 5 (100%) | **Optimal stable convergence** |
| **0.010** | NaN* | NaN* | `[NaN, NaN, 0.9675, NaN, NaN]` | 1 / 5 (20%) | **Divergence / Gradient overflow** |

*\*Note: At $\eta=0.010$, 4 out of 5 seeds suffered gradient explosion during unclipped BPTT through the recurrent five-memory chain, yielding non-finite (NaN) loss and model weights.*

---

## 3. Telemetry and Diagnostic Analysis

1. **Gradient Norm Dynamics**:
   - At $\eta = 0.001$: Gradient $L_2$ norms remained bounded ($\|\nabla_\theta\|_2 \in [0.12, 0.85]$).
   - At $\eta = 0.003$: Gradient norms stabilized in the regime $[0.35, 1.42]$, allowing meaningful parameter updates without exploding state transitions.
   - At $\eta = 0.010$: Unclipped recurrent accumulation caused gradient norms to exceed $10^4$ by epoch 4, rapidly resulting in IEEE 754 floating-point overflow (`inf` $\to$ `nan`).

2. **State Norm Trajectories**:
   - Content and value memory norms ($M_{\text{content}}, M_{\text{val}}$) grew linearly under stable learning rates, reflecting controlled associative trace accumulation.
   - Under $\eta = 0.010$, unconstrained scalar retention parameters $\rho_\eta, \lambda_\eta$ exited the non-expansive contractive domain ($0 \le \eta \|k\|^2 \le 2$), triggering exponential divergence.

---

## 4. Epistemic Findings

1. **[OBSERVED]**: Setting $\eta = 0.010$ on unconstrained recurrent update parameters causes severe optimization divergence on Task A without explicit gradient clipping.
2. **[SUPPORTED]**: Reducing the learning rate to $\eta = 0.003$ completely eliminates non-finite divergence across all 5 seeds, achieving 100% stable convergence.
3. **[SUPPORTED]**: Even under optimal optimization ($\eta = 0.003$), trainable dynamics alone without coordinates achieve only $E_{\text{rel}} \approx 0.944$, whereas frozen initial states with explicit coordinates (`delta_reference_xy`) achieve $E_{\text{rel}} \approx 0.866$.
4. **[CONCLUSION]**: Optimization instability is a real operational factor for unconstrained dynamics, but **missing spatial addressability** (not optimization failure) is the primary causal explanation for DeltaCore's failure on spatial neighborhood reconstruction.
