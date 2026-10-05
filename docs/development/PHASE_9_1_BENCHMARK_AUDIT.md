# Phase 9.1: Spatial Benchmark Audit & Diagnostic Report

## 1. Executive Summary

This document presents a rigorous diagnostic audit of the synthetic spatial pattern benchmark introduced in Phase 9 (`examples/phase_9_spatial_benchmark.py`).

**Central Finding**:
The benchmark in Phase 9 was evaluating an **untrained, freshly initialized operator in a zero-shot forward pass**. No weights were optimized, no loss function was minimized across training iterations, and no train/validation split existed. Consequently, the reported `Fused Error` values ($\approx 28.0$) did **not** measure spatial learning failure or directional capability; rather, they were almost entirely equal to the raw Frobenius norm $\|Z\|_F$ of the synthetic input tensors evaluated against near-zero forward predictions.

---

## 2. Forensic Audit of `examples/phase_9_spatial_benchmark.py`

### 2.1. Exact Input Generation
The benchmark defined six synthetic spatial patterns over $H = 14, W = 14, C = 8$:
1. `horizontal_stripe`: $Z(c, h, w) = \sin(2\pi h / H)$
2. `vertical_stripe`: $Z(c, h, w) = \sin(2\pi w / W)$
3. `diagonal_pattern`: $Z(c, h, w) = \sin(2\pi (h + w) / \max(H, W))$
4. `localized_square`: Centered $4 \times 4$ box with amplitude $2.0$, zeros elsewhere
5. `repeated_checkerboard`: $((h + w) \pmod 2) \in \{0, 1\}$
6. `asymmetric_pattern`: Quadrant step function ($+1$ in top-left, $-1$ in bottom-right)

### 2.2. Exact Target Generation
- In the execution call `res = op_generic(Z)`, the optional argument `targets` was omitted (`targets=None`).
- Inside `FiveMemorySystem.step()` / `BoundaryRefreshChunkScan.scan()`, when `targets=None`, the effective target defaults to:
  $$v_{\text{eff}, t} = v_{\text{gen}, t} = M_{\text{val}} x_t$$
- The benchmark did not supply an external spatial target $Y$; it measured reconstruction error directly against the input feature map:
  $$\text{Fused Error} = \|Y_{\text{fused}} - Z\|_F$$

### 2.3. Initialization State
Because `initial_states=None` was passed, each directional branch called `FiveMemoryState.initialize()`:
- Content memory $M_{\text{content}} \in \mathbb{R}^{V \times K}$: initialized to exact **zeros**.
- Key memory $M_{\text{key}} \in \mathbb{R}^{K \times D_{\text{in}}}$: initialized with Gaussian noise scaled by $0.01$.
- Value memory $M_{\text{val}} \in \mathbb{R}^{V \times D_{\text{in}}}$: initialized with Gaussian noise scaled by $0.01$.
- Meta-controller memories $M_\eta, M_\alpha$: initialized to exact **zeros**.

### 2.4. Why Fused Error Approximated ~28.0
1. At step $t = 0$, content memory is zero:
   $$\hat{v}_0 = M_{\text{content}, 0} k_0 = \mathbf{0}$$
2. The self-generated target $v_{\text{gen}, 0} = M_{\text{val}} x_0$ has magnitude on the order of $\|M_{\text{val}}\| \approx 0.01$.
3. Over $T = 196$ sequence steps with small learning rates ($\eta_{\text{max}} = 0.5, \eta_{\text{val}} = 0.02$), content memory accumulates updates with total magnitude $< 0.05$.
4. The output predictions $\hat{v}_t$ remain close to zero ($\|\hat{v}_t\| \ll 1$).
5. Thus, the restored output $Y_{\text{fused}} \approx \mathbf{0}$, meaning:
   $$\|Y_{\text{fused}} - Z\|_F \approx \|\mathbf{0} - Z\|_F = \|Z\|_F$$
6. For a sinusoidal tensor with $C = 8, H = 14, W = 14$ and peak amplitude $1.0$:
   $$\mathbb{E}[\sin^2(\cdot)] = 0.5 \implies \|Z\|_F^2 \approx 8 \times 14 \times 14 \times 0.5 = 784 \implies \|Z\|_F \approx \sqrt{784} = 28.0$$
   For the localized square ($4 \times 4 \times 8$ elements with value 2.0):
   $$\|Z\|_F^2 = 8 \times 16 \times 4.0 = 512 \implies \|Z\|_F = \sqrt{512} \approx 22.627$$

The reported errors (27.998, 28.002, 22.628) were literally the norms of the input tensors themselves.

### 2.5. Optimization Protocol Audit
- **Parameters Trained**: **0** (Zero parameters were optimized).
- **Optimizer Used**: **None** (Neither Adam, AdamW, SGD, nor any PyTorch optimizer was invoked).
- **Number of Optimization Steps**: **0**.
- **Evaluation Timing**: Evaluated strictly **before** (and without) learning.
- **Loss Function**: No loss was computed, backpropagated, or minimized.

---

## 3. Epistemic Assessment & Rectification

### 3.1. Scientific Clarification
The Phase 9 benchmark successfully verified:
- Dimensional shape preservation through serialization, multi-directional chunk unrolling, restoration, and fusion;
- Numerical runtime on CPU across grid dimensions.

However, **it provided zero evidence regarding spatial representation learning or directional utility**.
Reporting directional errors from an untrained, zero-predicting model and comparing differences on the order of $10^{-4}$ does not reflect semantic specialization.

### 3.2. Mandatory Requirements for Phase 9.1
To transform this into a genuine, scientifically valid learning benchmark:
1. **Meaningful Learnable Tasks**: Must require non-trivial 2D spatial transformations (Task A: spatial neighbor reconstruction/shift) and spatial configuration discrimination (Task B: normalized spatial pattern classification).
2. **Normalized Metrics**: Report both absolute Frobenius error $\|Y - \hat{Y}\|_F$ and relative error $E_{\text{rel}} = \frac{\|Y - \hat{Y}\|_F}{\max(\|Y\|_F, \epsilon)}$.
3. **Rigorous Baselines**: Compare against a Zero-predictor, Static Linear predictor, Non-spatial 1D sequential predictor, 1-direction, 2-direction, and 4-direction DeltaCore.
4. **Supervised Optimization**: Standardized PyTorch training loop (AdamW, train/val splits, fixed seeds, multi-seed statistical aggregation over seeds $[0, 1, 2, 3, 4]$).
5. **No Magnitude Shortcuts**: Inputs must be unit-norm normalized so classification cannot be solved by total tensor energy.
