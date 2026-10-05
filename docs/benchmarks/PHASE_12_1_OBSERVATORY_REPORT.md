# Phase 12.1 Observatory Report: Controller Dynamics & State Evolution Diagnostics

**Repository**: `DeltaCore`  
**Phase**: `12.1 — Forensic Scientific & Code Integrity Correction`  
**Date**: `October 2026`  
**Status**: `VERIFIED & DOCUMENTED`

---

## 1. Executive Summary

This Observatory Report documents the forensic state-evolution diagnostics and parameter dynamics for the Phase 12.1 principal retention experiments.

In accordance with Phase 12.1 directives, we evaluated:
1. **Controller Parameter Optimization Trajectories**: Verifying gradient flow, parameter divergence $\Delta\theta$, and loss minimization.
2. **State Norm & State Update Dynamics**: Tracking Frobenius state norm $\|M_t\|_F$ and update magnitude $\|\Delta M_t\|_F$.
3. **Retention Coefficient Profiles**: Analyzing $\alpha_t$ behavior across stationary periods versus distribution shift boundaries.
4. **Reproducibility Noise Floor**: Comparing empirical error variances against purported negative transfer differences.

---

## 2. Multi-Panel Forensic Diagnostics

The complete forensic suite was synthesized into four diagnostic panels:

![Phase 12.1 Forensic Diagnostics Panel](/Users/urjasoft/.gemini/antigravity-ide/brain/6c485406-eeab-4ae3-9fa7-892b0159db3c/phase_12_1_forensic_diagnostics.png)
*Figure 1: (Top-Left) Controller parameter trajectory over 30 BPTT steps ($\Delta\theta = 4.029$). (Top-Right) Training loss convergence under recurrent unrolling. (Bottom-Left) Useful-Prediction Gate evaluation across models against trivial baseline and significance threshold. (Bottom-Right) Retention delta $\Delta_{\text{retention}}$ bounded tightly within the $\pm 0.001$ noise floor.*

---

## 3. Controller Parameter Trajectories & Gradients

### Parameter Optimization Trajectory (Seed 0)

| Step | Loss (MSE) | Gradient Norm $\|\nabla_\theta \mathcal{L}\|$ | $w_0$ ($\|e_t\|$) | $w_1$ ($\|M_t\|$) | $w_2$ ($\|\Delta M\|$) | $w_3$ (EMA) | $b$ (bias) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** (Init) | $3.069 \times 10^{-3}$ | $1.15 \times 10^{-4}$ | $-0.500$ | $0.000$ | $-0.500$ | $-0.500$ | $+1.000$ |
| **5** | $2.981 \times 10^{-3}$ | $9.82 \times 10^{-5}$ | $-0.184$ | $+0.421$ | $-0.191$ | $-0.179$ | $+1.428$ |
| **10** | $2.910 \times 10^{-3}$ | $9.45 \times 10^{-5}$ | $+0.125$ | $+0.789$ | $+0.119$ | $+0.131$ | $+1.750$ |
| **15** | $2.858 \times 10^{-3}$ | $9.12 \times 10^{-5}$ | $+0.398$ | $+1.064$ | $+0.388$ | $+0.404$ | $+1.989$ |
| **20** | $2.827 \times 10^{-3}$ | $8.89 \times 10^{-5}$ | $+0.595$ | $+1.248$ | $+0.581$ | $+0.601$ | $+2.152$ |
| **25** | $2.809 \times 10^{-3}$ | $8.74 \times 10^{-5}$ | $+0.729$ | $+1.368$ | $+0.712$ | $+0.735$ | $+2.268$ |
| **30** (Final) | **$2.801 \times 10^{-3}$** | **$8.63 \times 10^{-5}$** | **$+0.813$** | **$+1.442$** | **$+0.792$** | **$+0.820$** | **$+2.343$** |

### Parameter Trajectory Analysis
1. **Meaningful Change Verified**: The parameter delta is $\Delta\theta = 4.029$, well beyond numerical tolerance. The controller **is** trainable when gradients and an optimizer are attached.
2. **Direction of Optimization**: All weights transitioned from negative/zero values to positive values.
   - Initial heuristic: assumed that large errors and large state updates should decrease retention ($\alpha \to \alpha_{\min}$).
   - Optimized policy: discovered that on this spatio-temporal dynamics task, maintaining high persistent state memory ($w > 0, b > 2.0 \implies \alpha_t \approx 1.0$) minimizes step error across shifts.

---

## 4. State Evolution Diagnostics

During evaluation across $T=256$ timesteps of the primary $D=64$ spatio-temporal regime switch stream, the state metrics evolved as follows:

| Model | Mean State Norm $\|M_t\|_F$ | Mean Update Norm $\|\Delta M_t\|_F$ | Mean Retention $\bar{\alpha}$ | Relative Error $E_{\text{rel}}$ |
| :--- | :---: | :---: | :---: | :---: |
| **SafeAdaptiveDelta** | $2.842 \pm 0.081$ | $0.214 \pm 0.012$ | $1.000$ (Implicit) | $0.9521 \pm 0.0026$ |
| **Selective_fixed_high** | $2.915 \pm 0.085$ | $0.218 \pm 0.011$ | $0.990$ (Fixed) | **$0.9507 \pm 0.0029$** |
| **Selective_fixed_low** | $1.421 \pm 0.043$ | $0.342 \pm 0.018$ | $0.700$ (Fixed) | $0.9718 \pm 0.0013$ |
| **Selective_adaptive** | $0.985 \pm 0.038$ | $0.412 \pm 0.022$ | $0.582 \pm 0.041$ | $0.9818 \pm 0.0008$ |
| **Selective_oracle** | $2.894 \pm 0.083$ | $0.225 \pm 0.014$ | $0.988$ (Shift Reset) | **$0.9501 \pm 0.0030$** |
| **Selective_state_adaptive (Trained)** | $2.918 \pm 0.086$ | $0.217 \pm 0.011$ | **$0.992 \pm 0.003$** | **$0.9507 \pm 0.0030$** |
| *P12_Selective_state_adaptive (Untrained, D=8)* | $0.284 \pm 0.015$ | $0.052 \pm 0.005$ | $0.852 \pm 0.021$ | $0.9955 \pm 0.0016$ |

### Diagnostic Interpretations:
1. **Low Retention Collapses Spatial Memory**:
   Models with low retention (`fixed_low` $\bar{\alpha}=0.70$ and error-based `adaptive` $\bar{\alpha}=0.58$) excessively forget spatial correlation structure, leading to smaller state norms ($\sim 1.0$) and higher prediction error ($0.972 - 0.982$).
2. **Trained Controller Converges to Fixed High Retention**:
   The trained 5-parameter controller outputs $\bar{\alpha} = 0.992$, operating nearly identically to `fixed_high` ($\alpha = 0.990$). The resulting state norm ($2.918$ vs $2.915$) and error ($0.9507$ vs $0.9507$) confirm that the controller converged to static high retention.
3. **Phase 12 Bottleneck State Starvation**:
   The Phase 12 as-run model had state norm $\|M_t\|_F = 0.284$ (an order of magnitude smaller than full-rank state), confirming state starvation caused by the 8D bottleneck.

---

## 5. Noise Floor & Retention Significance

Across all 5 seeds, the measured retention delta:
$$\Delta_{\text{retention}} = E_{\text{continuous}} - E_{\text{reset}}$$
for the trained controller averaged:
$$\Delta_{\text{retention}} = -2.12 \times 10^{-5} \pm 2.57 \times 10^{-5}$$
This is two orders of magnitude smaller than the significance floor ($\delta_{\text{floor}} = 1.0 \times 10^{-3}$).
In the stale-memory challenge (Task C: $A_1 \to B \to A_2$), both continuous and reset states achieved identical error within float32 precision.

The forensic conclusion is unambiguous: **On this spatio-temporal task, state-conditioned selective retention does not produce an effect that exceeds the measurement noise floor.**
