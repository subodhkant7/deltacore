# Phase 1 Mathematical Specification: Linear Associative Memory & Update Rules

This document specifies the exact mathematical formulations implemented by the reference primitives in **DeltaCore Phase 1**.

---

## 1. State Formulation & Dimensionality

Let the associative memory state at discrete time $t$ be represented as a 2D matrix:

$$
M_t \in \mathbb{R}^{V \times K}
$$

where:
- $K \in \mathbb{N}^+$ denotes the key (query/input) dimension.
- $V \in \mathbb{N}^+$ denotes the value (target/output) dimension.

A query/key vector is:

$$
k_t \in \mathbb{R}^{K}
$$

A target/value vector is:

$$
v_t \in \mathbb{R}^{V}
$$

---

## 2. Read Operation

The retrieval (read) operation computes the linear projection of the key through the memory matrix:

$$
\hat{v}_t = \text{read}(M_t, k_t) = M_t k_t \in \mathbb{R}^{V}
$$

In index notation:

$$
\hat{v}_{t, i} = \sum_{j=1}^K M_{t, ij} k_{t, j} \quad \text{for } i \in \{1, \dots, V\}
$$

The read operator is linear, deterministic, and non-mutating with respect to $M_t$.

---

## 3. Retrieval Error

The discrepancy between the retrieved value $\hat{v}_t$ and the target value $v_t$ is the prediction error vector:

$$
e_t = v_t - \hat{v}_t = v_t - M_t k_t \in \mathbb{R}^{V}
$$

Sign convention:
- $e_t$ is defined as target minus prediction: $e_t = v_t - \hat{v}_t$.
- Under this sign convention, the gradient of the squared Euclidean error $\mathcal{L}_t = \frac{1}{2} \|v_t - M_t k_t\|_2^2$ with respect to $M_t$ is:

$$
\frac{\partial \mathcal{L}_t}{\partial M_t} = -(v_t - M_t k_t) k_t^\top = -e_t k_t^\top
$$

---

## 4. Hebbian Update Rule

The classical outer-product associative update (Kohonen, 1972) binds key and target vectors without error correction:

$$
\Delta M_t^{\text{Hebbian}} = \eta \, v_t k_t^\top \in \mathbb{R}^{V \times K}
$$

$$
M_{t+1} = M_t + \Delta M_t^{\text{Hebbian}} = M_t + \eta \, v_t k_t^\top
$$

where $\eta \ge 0$ is a scalar step size (learning rate).

### Immediate Recall Under Hebbian Update
Testing retrieval immediately after update on the same key $k_t$:

$$
\hat{v}_{t+1} = M_{t+1} k_t = M_t k_t + \eta \, (k_t^\top k_t) v_t = \hat{v}_t + \eta \|k_t\|_2^2 v_t
$$

Notice that the prior state output $\hat{v}_t = M_t k_t$ is **not** cancelled out. If $M_t \neq 0$, the retrieved vector is a linear mixture of previous memory contents and the new target $v_t$, causing memory cross-talk.

---

## 5. Delta-Rule (Error-Correcting) Update

The delta rule (Widrow & Hoff, 1960; Kohonen, 1984) updates memory proportionally to the *prediction residual* $e_t$:

$$
\Delta M_t^{\text{Delta}} = \eta \, (v_t - M_t k_t) k_t^\top = \eta \, e_t k_t^\top \in \mathbb{R}^{V \times K}
$$

$$
M_{t+1} = M_t + \Delta M_t^{\text{Delta}} = M_t + \eta \, (v_t - M_t k_t) k_t^\top
$$

### Immediate Recall Under Delta Update
Evaluating immediate retrieval on $k_t$:

$$
\hat{v}_{t+1} = M_{t+1} k_t = M_t k_t + \eta (v_t - M_t k_t) (k_t^\top k_t) = \hat{v}_t + \eta \|k_t\|_2^2 e_t
$$

The updated error on key $k_t$ is:

$$
e_{t+1}(k_t) = v_t - \hat{v}_{t+1} = v_t - (\hat{v}_t + \eta \|k_t\|_2^2 e_t) = (1 - \eta \|k_t\|_2^2) e_t
$$

### Contraction & Invariant Properties (Single-Step / Immediate-Key)
From this closed-form recurrence:
1. **Zero Error Invariant**: If $M_t k_t = v_t$ ($e_t = 0$), then $\Delta M_t^{\text{Delta}} = 0$. The memory does not alter bindings that are already correctly recalled.
2. **Exact One-Step Recall**: If $\eta = \frac{1}{\|k_t\|_2^2}$ for non-zero key $k_t \neq 0$, then $e_{t+1}(k_t) = 0$, achieving exact immediate recall in a single step.
3. **Strict Error Contraction**: For any non-zero error $e_t$ and non-zero key $k_t \neq 0$, the Euclidean norm of the immediate residual strictly decreases if and only if:

$$
|1 - \eta \|k_t\|_2^2| < 1 \iff 0 < \eta < \frac{2}{\|k_t\|_2^2}
$$

4. **Distinction: Analytical Recurrence vs. Public DeltaRule API**:
   - The scalar error recurrence $e_{t+1} = \gamma e_t$ with $\gamma = 1 - \eta \|k_t\|_2^2$ is an analytical formula. In theoretical stress testing, arbitrary values of $\gamma$ (including negative $\gamma$, representing overshooting or oscillatory dynamics) may be studied strictly as mathematical recurrence analysis.
   - In contrast, the public DeltaCore `DeltaRule` and `HebbianRule` API strictly enforces $\eta \ge 0$, rejecting negative step sizes ($\eta < 0$) with `ValueError` to prevent divergent anti-Hebbian/unbounded dynamics.

5. **Strict Scope of Contraction**:
   The single-step contraction factor $|1 - \eta \|k_t\|_2^2| < 1$ is an **immediate-key, local residual property** on the single vector $k_t$ just presented. It **does NOT establish**:
   - Global memory convergence across a dataset of associations
   - Arbitrary-sequence stability under non-stationary or non-orthogonal inputs
   - Boundedness of memory $\|M_t\|_F$ under arbitrary external streams
   - Stability of multi-layer or future self-referential dynamics.

---

## 6. Distinctions from Prior and Related Work

To uphold Principle A (Research Correctness) of the DeltaCore Constitution:

| Concept | Historical Formulation | DeltaCore Phase 1 Implementation | Notes & Exclusions |
| :--- | :--- | :--- | :--- |
| **Linear Associative Memory** | Kohonen (1972), Anderson (1972) | $M \in \mathbb{R}^{V \times K}$, $\hat{v} = M k$ | Canonical matrix linear associative memory. |
| **Hebbian Update** | Hebb (1949), Kohonen (1972) | $M + \eta v k^\top$ | Pure correlation learning without error feedback. |
| **Delta Rule** | Widrow & Hoff (1960), Rumelhart et al. (1986) | $M + \eta (v - M k) k^\top$ | Online least-mean-squares (LMS) matrix adaptation. |
| **VisionHOPE Dynamics** | Peng et al. (September 2026) | **NOT IMPLEMENTED in Phase 1** | Involves 5 coupled memories, self-referential $\eta$, spatial patch scans, and retention gating. |
| **Chunked Parallel Scans** | DeltaNet / Mamba-style scans | **NOT IMPLEMENTED in Phase 1** | Sequential unrolling is the sole reference implementation in Phase 1. |
