# DeltaCore Tensor Conventions

This document locks the dimensional conventions, axis ordering, and tensor shapes across all DeltaCore mathematical modules.

---

## 1. Locked Shape Conventions

### 1.1. Simple Reference Memory (Phase 1 Focus)
For unbatched reference operators:

| Tensor | Symbol | Shape | Description |
| :--- | :--- | :--- | :--- |
| **Memory State** | $M$ | `[V, K]` | State matrix mapping key space to value space. |
| **Key / Query** | $k$ | `[K]` | Query or associative key vector. |
| **Value / Target** | $v$ | `[V]` | Associated target or retrieved value vector. |
| **Prediction** | $\hat{v}$ | `[V]` | Output of retrieval: $\hat{v} = M k$. |
| **Prediction Error**| $e$ | `[V]` | Residual: $e = v - \hat{v}$. |
| **Outer Product** | $e k^\top$ | `[V, K]` | Outer product matrix: $(e k^\top)_{ij} = e_i k_j$. |
| **Update Matrix** | $\Delta M$| `[V, K]` | Directional state change: $\eta e k^\top$. |

*Dimensional Rationale*:
Setting $M \in \mathbb{R}^{V \times K}$ aligns directly with standard linear algebra $\hat{v} = M k$, where $M$ acts as a linear transformation from $\mathbb{R}^K \to \mathbb{R}^V$.

---

### 1.2. Sequential Streams & Scans
For sequential streams processed over time:

| Sequence Tensor | Shape | Description |
| :--- | :--- | :--- |
| **Batched Keys** | `[B, T, K]` | $B$ independent sequences of length $T$ with key dimension $K$. |
| **Batched Targets**| `[B, T, V]` | $B$ independent sequences of length $T$ with value dimension $V$. |
| **Batched Predictions**| `[B, T, V]` | Output predictions emitted at each time step. |
| **Batched State** | `[B, V, K]` | Independent memory state per batch element. |
| **Unbatched Keys** | `[T, K]` | Single sequence of keys. |
| **Unbatched Targets**| `[T, V]` | Single sequence of targets. |

---

### 1.3. Multi-Head Representation (Reserved for Future Phases)
In subsequent phases (e.g., Phase 2+, Phase 5), multi-head attention-style associative topologies will extend these conventions:

| Multi-Head Tensor | Shape | Description |
| :--- | :--- | :--- |
| **Multi-Head Memory** | `[B, H, V, K]` | $H$ heads per sequence element. |
| **Multi-Head Keys** | `[B, H, K]` | Head-projected key vectors. |
| **Multi-Head Values**| `[B, H, V]` | Head-projected value vectors. |

> **Constraint for Phase 1**: Multi-head tensors are **not** implemented in Phase 1. The reference implementation focuses strictly on 2D matrix state `[V, K]` and cleanly batched sequences `[B, T, K]`.

---

## 2. Axis Indexing & Semantics

- **Axis -1 (Last Dimension)**:
  - In a key vector $k \in \mathbb{R}^K$: Key feature dimension $K$.
  - In a value vector $v \in \mathbb{R}^V$: Value feature dimension $V$.
  - In a memory matrix $M \in \mathbb{R}^{V \times K}$: Dimension 0 is $V$ (rows), Dimension 1 is $K$ (columns).
- **Matrix Multiplication Alignment**:
  - `M @ k` computes $\sum_{j=1}^K M_{ij} k_j = \hat{v}_i$, producing shape `[V]`.
  - For batched inputs $k \in [B, K]$, `(M @ k.T).T` or `torch.matmul(k, M.T)` or `torch.einsum('vk, bk -> bv', M, k)` produces `[B, V]`.
  - In outer products: `torch.outer(v, k)` or `v.unsqueeze(-1) * k.unsqueeze(-2)` produces shape `[V, K]`.

---

## 3. Strict Shape Validation Rules

Every module boundary must enforce:
1. No silent squeeze or unsqueeze on mismatched dimensions.
2. Tensor dimensionality must be strictly verified (e.g., raise `ValueError` if $M$ is not 2D or $k$ does not have length $K$).
3. Floating point precision must match between $M$, $k$, and $v$ (e.g., do not mix `float32` and `float64` without explicit conversion).
