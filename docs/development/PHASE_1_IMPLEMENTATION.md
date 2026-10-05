# Phase 1: Mathematical Reference Primitives — Implementation Report

**Date**: 2026-10-05  
**Status**: Verified & Complete  
**Reference Document**: [PHASE_1_ASSOCIATIVE_MEMORY.md](file:///Users/urjasoft/Documents/DeltaCore/docs/math/PHASE_1_ASSOCIATIVE_MEMORY.md)

---

## 1. What Was Implemented

Phase 1 established pure-PyTorch, mathematically inspectable reference primitives for linear associative memory and state transitions:

1. **State Container (`AssociativeMemory`)** in [deltacore/memory/associative.py](file:///Users/urjasoft/Documents/DeltaCore/deltacore/memory/associative.py):
   - Explicit state storage $M \in \mathbb{R}^{V \times K}$ (2D) and batched $M \in \mathbb{R}^{B \times V \times K}$ (3D).
   - Factory initializers (`zeros`, `from_tensor`), cloning (`clone`), resetting (`reset`), and precision/device casting (`to`).
   - Strictly decoupled from update logic to prevent architectural conflation.

2. **Read Operation (`read`)** in [deltacore/memory/read.py](file:///Users/urjasoft/Documents/DeltaCore/deltacore/memory/read.py):
   - Linear associative retrieval $\hat{v} = M k$.
   - Supports unbatched (`[V, K] @ [K] -> [V]`) and batched (`[B, K] @ [K, V] -> [B, V]` or `[B, V, K] @ [B, K] -> [B, V]`).
   - Strict dimension, dtype, and device validation with informative error messages.

3. **Hebbian Outer-Product Update (`HebbianRule`)** in [deltacore/updates/hebbian.py](file:///Users/urjasoft/Documents/DeltaCore/deltacore/updates/hebbian.py):
   - Transition: $\Delta M = \eta v k^\top$, $M' = M + \Delta M$.
   - Exposes diagnostic output via `HebbianStepResult` (`outer_product`, `update`, `new_memory`).
   - Pure functional execution without in-place mutation of caller's state.

4. **Delta-Rule Error-Correcting Update (`DeltaRule`)** in [deltacore/updates/delta.py](file:///Users/urjasoft/Documents/DeltaCore/deltacore/updates/delta.py):
   - Transition: $\hat{v} = M k$, $e = v - \hat{v}$, $\Delta M = \eta e k^\top$, $M' = M + \Delta M$.
   - Full diagnostic telemetry exposed via `DeltaStepResult` (`prediction`, `error`, `outer_product`, `update`, `new_memory`).
   - Strict adherence to LMS error feedback; no simplification or omitting of the prior state prediction $M k$.

5. **Sequential State Scan (`sequential_scan`)** in [deltacore/scans/sequential.py](file:///Users/urjasoft/Documents/DeltaCore/deltacore/scans/sequential.py):
   - Unrolled sequential execution over $T$ steps preserving chronological state transitions.
   - Emits step-wise predictions $\hat{v}_t$ before updating state for $t+1$.
   - Compatible with full PyTorch autograd backpropagation through recurrent state trajectories.

6. **Demonstration & Synthetic Benchmark Script** in [examples/phase_1_hebbian_vs_delta.py](file:///Users/urjasoft/Documents/DeltaCore/examples/phase_1_hebbian_vs_delta.py):
   - Head-to-head empirical comparison (20 passes over 3 associations, for 60 total updates) under non-orthogonal keys proving cross-talk accumulation in Hebbian updates versus residual correction convergence in Delta updates.

---

## 2. Mathematical Equations Implemented

| Primitive | Mathematical Formulation | Dimensionality |
| :--- | :--- | :--- |
| **Read** | $\hat{v}_t = M_t k_t$ | $[V, K] \times [K] \to [V]$ |
| **Error** | $e_t = v_t - \hat{v}_t$ | $[V] - [V] \to [V]$ |
| **Hebbian Update** | $M_{t+1} = M_t + \eta v_t k_t^\top$ | $[V, K] + [V, K] \to [V, K]$ |
| **Delta Update** | $M_{t+1} = M_t + \eta (v_t - M_t k_t) k_t^\top$ | $[V, K] + [V, K] \to [V, K]$ |
| **Residual Error Recurrence** | $e_{t+1}(k_t) = (1 - \eta \|k_t\|_2^2) e_t$ | Scalar factor $\gamma = 1 - \eta \|k_t\|_2^2$ |

### Theoretical Invariants & Scope Clarifications
- **Exact One-Step Recall Condition**: $\eta = \frac{1}{\|k_t\|_2^2}$ strictly requires non-zero key $k_t \neq 0$.
- **Negative Gamma vs. API**: The scalar recurrence analysis studies arbitrary $\gamma = 1 - \eta \|k\|^2$ purely as mathematical recurrence equations in tests. The public DeltaCore API strictly rejects negative step sizes ($\eta < 0$).
- **Scope of Contraction**: The immediate-key residual contraction factor $|1 - \eta \|k_t\|_2^2| < 1$ is an immediate-step local property on $k_t$. It does NOT establish global memory convergence, arbitrary-sequence stability, boundedness under arbitrary external inputs, or stability of future self-referential dynamics.

---

## 3. Public API Surface

Exposed at root package `deltacore` and respective subpackages:

```python
from deltacore import (
    AssociativeMemory,
    DeltaRule,
    DeltaStepResult,
    HebbianRule,
    HebbianStepResult,
    SequentialScanResult,
    read,
    sequential_scan,
)
```

Internal helper logic and private methods remain unexported.

---

## 4. Tensor Conventions

- **State**: $M \in \mathbb{R}^{V \times K}$ (`[V, K]` where $V$ is rows/value dim, $K$ is columns/key dim).
- **Key**: $k \in \mathbb{R}^K$ (`[K]`).
- **Target**: $v \in \mathbb{R}^V$ (`[V]`).
- **Batched Sequences**: `keys` in `[B, T, K]`, `targets` in `[B, T, V]`, emitted predictions in `[B, T, V]`.
- All conventions locked in [docs/math/TENSOR_CONVENTIONS.md](file:///Users/urjasoft/Documents/DeltaCore/docs/math/TENSOR_CONVENTIONS.md).

---

## 5. Numerical Considerations & Precision

- **Double Precision Verification (`float64`)**: Analytical error contraction rates ($\gamma = 1 - \eta \|k\|^2$), zero-error invariants ($M k = v \implies \Delta M = 0$), and `torch.autograd.gradcheck` were verified in `float64` within machine precision ($\text{atol} \le 10^{-12}$).
- **Single Precision Verification (`float32`)**: All operations run deterministically in FP32 with tolerance $\text{atol} \le 10^{-6}$.
- **Step Size Bounds**: Negative step sizes ($\eta < 0$) are strictly rejected with `ValueError` to prevent explosive divergence.
- **Autograd Compatibility**: Backward gradients through `read`, `HebbianRule.update`, `DeltaRule.update`, and unrolled `sequential_scan` were verified analytically and numerically.

---

## 6. Known Limitations & Intentionally Deferred Items

In strict adherence to the project constitution:
1. **Multi-Head Associative Topologies**: Multi-head state containers (`[B, H, V, K]`) are deferred to Phase 2+.
2. **Dynamic Step Sizes & Modulation**: Self-referential or learned learning rates $\eta_t = \sigma(\dots)$ are deferred to Phase 3.
3. **Retention & Forgetting Factors**: Time-dependent retention decay $\lambda_t M_t$ is deferred to Phase 4.
4. **Chunked Associative Scans**: Parallel chunked scans (e.g. DeltaNet / associative parallel prefix scans) are deferred to Phase 5.
5. **Hardware Kernels**: No custom CUDA/Triton kernels were introduced; reference pure-PyTorch implementation guarantees complete platform portability.
