# Phase 5 Mathematical Specification: Associative & Chunked State Scans

This document establishes the mathematical foundations for representing DeltaCore memory updates as **affine operators**, deriving their **associative composition law**, proving sequential/chunked equivalence, and defining chunked recurrent execution.

---

## 1. Mathematical Objective

The central question of Phase 5 is:
> **Can DeltaCore reduce sequential associative memory recurrence into composable chunk operators without altering the mathematical result?**

Specifically, we investigate:
1. Under what algebraic conditions the error-correcting delta update $M_{t+1} = M_t + \eta_t (v_t - M_t k_t) k_t^\top$ forms an associative operator.
2. The exact composition law for chaining multiple transitions.
3. The boundary states and prefix reconstruction mechanics across chunks.
4. The fundamental theoretical boundary between **known transition coefficients** (parallelizable associative scan) and **self-referential dynamics** (recurrently dependent coefficient generation).

---

## 2. Affine Operator Representation

### 2.1. The DeltaCore Recurrence
Recall the Phase 1–4 content memory update equation:

$$
\hat{v}_t = M_t k_t \in \mathbb{R}^V
$$

$$
e_t = v_t - \hat{v}_t = v_t - M_t k_t \in \mathbb{R}^V
$$

$$
M_{t+1} = M_t + \eta_t e_t k_t^\top = M_t + \eta_t (v_t - M_t k_t) k_t^\top \in \mathbb{R}^{V \times K}
$$

where:
- $M_t \in \mathbb{R}^{V \times K}$ is the associative memory matrix at step $t$.
- $k_t \in \mathbb{R}^K$ is the key vector.
- $v_t \in \mathbb{R}^V$ is the target value vector.
- $\eta_t \in \mathbb{R}_{>0}$ is the step size.

### 2.2. Rearrangement into Affine Form
Expanding the update:

$$
\begin{aligned}
M_{t+1} &= M_t + \eta_t v_t k_t^\top - \eta_t M_t k_t k_t^\top \\
&= M_t \left(I_K - \eta_t k_t k_t^\top\right) + \eta_t v_t k_t^\top
\end{aligned}
$$

Define the **transition matrix** $A_t$ and **affine drive matrix** $B_t$:

$$
A_t = I_K - \eta_t k_t k_t^\top \in \mathbb{R}^{K \times K}
$$

$$
B_t = \eta_t v_t k_t^\top \in \mathbb{R}^{V \times K}
$$

The memory state update is therefore an **affine transformation**:

$$
M_{t+1} = M_t A_t + B_t
$$

### 2.3. Dimension Bookkeeping
The matrix dimensions are strictly:
- Memory state: $M_t \in \mathbb{R}^{V \times K}$
- Right-multiplier operator: $A_t \in \mathbb{R}^{K \times K}$ (acting on the key dimension $K$)
- Translation term: $B_t \in \mathbb{R}^{V \times K}$ (matching the memory dimension $V \times K$)
- Product: $M_t A_t \in \mathbb{R}^{V \times K}$
- Sum: $M_t A_t + B_t \in \mathbb{R}^{V \times K}$

---

## 3. Composition Law of Affine Operators

### 3.1. Two-Step Sequential Composition
Consider two successive affine updates:

$$
F_1(M) = M A_1 + B_1
$$

$$
F_2(M) = M A_2 + B_2
$$

Evaluating the composite map $F_2(F_1(M))$ (applying $F_1$ first, then $F_2$):

$$
\begin{aligned}
F_2(F_1(M)) &= \left(M A_1 + B_1\right) A_2 + B_2 \\
&= M (A_1 A_2) + (B_1 A_2 + B_2)
\end{aligned}
$$

### 3.2. Definition of the Composition Operator $\otimes$
Let an affine operator be represented by the tuple $(A, B)$.
We define the composition operator $\otimes$ such that applying the composed operator $(A_{12}, B_{12}) = (A_1, B_1) \otimes (A_2, B_2)$ to $M$ equals applying $(A_1, B_1)$ followed by $(A_2, B_2)$:

$$
(A_1, B_1) \otimes (A_2, B_2) = \left(A_1 A_2, \; B_1 A_2 + B_2\right)
$$

### 3.3. Proof of Associativity
Let $(A_1, B_1), (A_2, B_2), (A_3, B_3)$ be three affine operators.

#### Left-associated composition:
$$
\begin{aligned}
\left[(A_1, B_1) \otimes (A_2, B_2)\right] \otimes (A_3, B_3) &= (A_1 A_2, \; B_1 A_2 + B_2) \otimes (A_3, B_3) \\
&= \left((A_1 A_2) A_3, \; (B_1 A_2 + B_2) A_3 + B_3\right) \\
&= \left(A_1 A_2 A_3, \; B_1 A_2 A_3 + B_2 A_3 + B_3\right)
\end{aligned}
$$

#### Right-associated composition:
$$
\begin{aligned}
(A_1, B_1) \otimes \left[(A_2, B_2) \otimes (A_3, B_3)\right] &= (A_1, B_1) \otimes (A_2 A_3, \; B_2 A_3 + B_3) \\
&= \left(A_1 (A_2 A_3), \; B_1 (A_2 A_3) + (B_2 A_3 + B_3)\right) \\
&= \left(A_1 A_2 A_3, \; B_1 A_2 A_3 + B_2 A_3 + B_3\right)
\end{aligned}
$$

Both groupings produce identical matrix results:

$$
\left[(A_1, B_1) \otimes (A_2, B_2)\right] \otimes (A_3, B_3) = (A_1, B_1) \otimes \left[(A_2, B_2) \otimes (A_3, B_3)\right] \quad \blacksquare
$$

Thus, the composition algebra forms a **monoid** with associative binary operation $\otimes$.

### 3.4. Identity Element
The affine identity operator is:

$$
I = \left(I_K, \; 0_{V \times K}\right)
$$

Verifying left identity:
$$
I \otimes (A, B) = (I_K A, \; 0 A + B) = (A, B)
$$

Verifying right identity:
$$
(A, B) \otimes I = (A I_K, \; B I_K + 0) = (A, B)
$$

For any memory $M$:
$$
I(M) = M I_K + 0 = M
$$

---

## 4. Chunked Recurrence Formulation

### 4.1. Sequence Decomposition into Chunks
Given a sequence of length $T$ with transition operators $F_0, F_1, \dots, F_{T-1}$ and chunk size $C$:
The sequence is partitioned into $N = \lceil T / C \rceil$ contiguous chunks:

$$
\text{Chunk } k: \quad \{t_k, t_k+1, \dots, t_k + C_k - 1\}
$$

where $C_k = C$ for all $k < N-1$, and $C_{N-1} = T - (N-1)C$ (the non-divisible final chunk).

### 4.2. Chunk Operator Composition
For chunk $k$ spanning time indices $i$ through $j$:

$$
F_{\text{chunk}}^{(k)} = F_i \otimes F_{i+1} \otimes \dots \otimes F_j = (A_{\text{chunk}}^{(k)}, \; B_{\text{chunk}}^{(k)})
$$

### 4.3. Exact Boundary State Transition
Let $M_{t_k}$ be the memory state at the start of chunk $k$. The state at the end of the chunk is given directly by:

$$
M_{t_{k+1}} = F_{\text{chunk}}^{(k)}(M_{t_k}) = M_{t_k} A_{\text{chunk}}^{(k)} + B_{\text{chunk}}^{(k)}
$$

This requires **zero intermediate sequential step evaluations** to advance across chunk boundaries once the chunk operator is composed.

### 4.4. Within-Chunk Prefix Reconstruction
To reconstruct intermediate memory states within chunk $k$:
1. Compute within-chunk prefix operators:
   $$P_\tau^{(k)} = F_i \otimes F_{i+1} \otimes \dots \otimes F_{i+\tau} \quad \text{for } 0 \le \tau < C_k$$
2. Evaluate prefix memories:
   $$M_{i+\tau+1} = P_\tau^{(k)}(M_{t_k}) = M_{t_k} A_{\text{prefix},\tau}^{(k)} + B_{\text{prefix},\tau}^{(k)}$$

---

## 5. Critical Distinction: Transition Scanning vs. Coefficient Generation

> **Mandatory Scientific Qualification**:
> An associative scan enables parallel prefix evaluation **if and only if the transition coefficients $(A_t, B_t)$ are known prior to the scan**.

Consider the computational dependencies across DeltaCore phases:

### Phase 1 (Fixed Step Size) & Phase 2 (Input-Conditioned Adaptive Step Size):
- $k_t$ and $v_t$ are known inputs.
- $\eta_t = \eta_0$ (Phase 1) or $\eta_t = f(x_t)$ (Phase 2 input-conditioned).
- Therefore, all $(A_t, B_t)$ pairs can be computed **in parallel** for all $t \in [0, T-1]$.
- The associative scan can be executed in $O(\log T)$ parallel depth or in parallel chunks.

### Phase 3 (Coupled Self-Referential Adaptive Memory):
- The controller memory $C_t$ co-evolves with $M_t$:
  $$e_t = v_t - M_t k_t \implies z_t = f(e_t, M_t, k_t) \implies r_t = C_t z_t \implies \eta_t = \sigma(r_t)$$
- $\eta_t$ cannot be computed without $e_t$ and $M_t$.
- $M_t$ cannot be advanced without $\eta_t$.
- $C_t$ cannot be updated without $e_t$.
- Therefore, **Phase 3 self-referential dynamics cannot be directly converted into a precomputed associative scan**.

DeltaCore Phase 5 explicitly implements:
1. The **exact associative scan** for known/precomputed transition coefficients $(A_t, B_t)$.
2. A **reference chunked execution interface** that separates chunk boundaries.
3. Explicit documentation that self-referential coefficient generation remains sequential unless an independent analytical decoupling is proven.

---

## 6. Stability Metadata Preservation Under Chunking

When individual token transitions enforce stability bounds:

$$
\gamma_t = \eta_t \|k_t\|_2^2 < 2, \quad \mu_t = 2 - \gamma_t > 0
$$

A chunk operator $F_{\text{chunk}} = (A_{\text{chunk}}, B_{\text{chunk}})$ represents the composition of multiple rank-1 updates.
- $A_{\text{chunk}} = \prod_{\tau=0}^{C-1} (I - \eta_{t+\tau} k_{t+\tau} k_{t+\tau}^\top)$.
- Under contractive token conditions ($\|A_t\|_2 \le 1$), the submultiplicative property guarantees:
  $$\|A_{\text{chunk}}\|_2 \le \prod_{\tau=0}^{C-1} \|A_{t+\tau}\|_2 \le 1$$
- However, the chunk operator does **not** possess a single scalar key norm or scalar step size.
- To prevent silent loss of diagnostic information, chunk operators expose:
  - `min_constituent_stability_margin` = $\min_\tau \mu_{t+\tau}$
  - `max_constituent_normalized_step` = $\max_\tau \gamma_{t+\tau}$
  - `total_clipped_transitions` = $\sum_\tau \mathbb{I}(\text{clipped}_{t+\tau})$

> **Stability Scope Qualification**:
> Dual controllers enforce local non-expansive normalized-step conditions under their stated assumptions. They do not establish globally bounded trajectories under arbitrary affine forcing.

---

## 7. Numerical Precision, Tolerances & Composition Depth

### 7.1. Numerical Associativity of Floating-Point Arithmetic
> **Core Mathematical Qualification**:
> The affine composition algebra is mathematically associative; floating-point evaluations may differ because matrix multiplication is not numerically associative.

If an observed difference is `0.0` in a specific experiment (e.g. certain small integer or rank-1 tests in FP64), this must be reported as an **observed empirical result** for that test vector, not a theorem of bit-exactness.

- In FP64 (`float64`), observed differences between sequential and chunked evaluation remain small ($\|M^{\text{seq}} - M^{\text{chunk}}\|_\infty \lesssim 10^{-12}$).
- In FP32 (`float32`), round-off accumulation across long composed products can reach $\|M^{\text{seq}} - M^{\text{chunk}}\|_\infty \sim 10^{-5}$ to $10^{-7}$.
- Tests must enforce dtype-appropriate tolerances without asserting bit-level equality.

### 7.2. Composition Depth Definition
The diagnostic field `composition_depth` (reported in scan diagnostics) is explicitly defined as:
$$\text{composition\_depth} = \max(C_{\max} - 1, 0)$$
where $C_{\max}$ is the maximum chunk size. This measures the maximum number of sequential binary compositions chained within any single chunk under linear left-to-right composition. It does **not** represent logarithmic tree reduction depth ($O(\log C)$), as the reference Python implementation performs sequential composition within each chunk. Logarithmic depth applies only when an explicit parallel binary reduction tree is executed.

