# Phase 5: Associative & Chunked State Scans — Implementation Report

**Date**: 2026-10-05  
**Status**: Verified & Complete  
**Reference Documents**:
- [PHASE_5_ASSOCIATIVE_SCAN.md](file:///Users/urjasoft/Documents/DeltaCore/docs/math/PHASE_5_ASSOCIATIVE_SCAN.md)
- [PHASE_4_STABILITY.md](file:///Users/urjasoft/Documents/DeltaCore/docs/math/PHASE_4_STABILITY.md)
- [RELATION_TO_VISIONHOPE.md](file:///Users/urjasoft/Documents/DeltaCore/docs/math/RELATION_TO_VISIONHOPE.md)

---

## 1. Executive Summary

Phase 5 introduces a mathematically explicit **associative affine scan** and develops a modular chunked execution layer for DeltaCore's error-correcting associative memory updates.

The core research question investigated is:
> **Can DeltaCore reduce sequential memory recurrence into composable chunk operators without altering the mathematical result?**

### Key Findings & Achievements:
1. **Formal Affine Recurrence**: Formulated the rank-1 Delta update as an affine transformation $M_{t+1} = M_t A_t + B_t$ with transition matrix $A_t = I_K - \eta_t k_t k_t^\top \in \mathbb{R}^{K \times K}$ and affine drive $B_t = \eta_t v_t k_t^\top \in \mathbb{R}^{V \times K}$.
2. **Associative Monoid Algebra**: Derived and formally proved the composition law $(A_1, B_1) \otimes (A_2, B_2) = (A_1 A_2, B_1 A_2 + B_2)$ with two-sided identity $(I_K, 0)$.
3. **Exact Equivalence**: Proved and verified that decomposing sequences into arbitrary chunk sizes $C \ge 1$ (including irregular non-divisible trailing chunks) yields identical final states and reconstructible full state trajectories within machine precision ($10^{-7}$ in FP32, $0.00$ in FP64).
4. **Boundary State API**: Designed an explicit chunk boundary execution pathway (`run_chunk`) consuming `(initial_state, coefficients)` and returning `(final_state, internal_states)`.
5. **Autograd Continuity**: Proved and verified gradient propagation through affine operator application, composition, sequential unrolling, and chunked execution via FP64 `torch.autograd.gradcheck`.
6. **Scientific Honesty & Non-Claims**:
   - Explicitly proved that **associative parallelization requires known transition coefficients**. Self-referential online dynamics ($C_t \to \eta_t \to M_t$) cannot be parallelized without decoupling.
   - Benchmark measurements on CPU confirm that Python-level chunk composition incurs matrix-multiplication and memory allocation overhead relative to inlined rank-1 vector loops; speedups require compiled parallel hardware kernels (CUDA/Triton), which are neither assumed nor falsely reported.

---

## 2. Phase 4 Documentation Corrections

Prior to implementing Phase 5, an audit of Phase 4 documentation and code was performed to ensure rigorous scientific honesty:

1. **Global Stability Claim Narrowed**:
   - *Previous statement*: Dual controllers prevent runaway divergence under adversarial input sequences.
   - *Corrected statement*: *Dual controllers enforce local non-expansive normalized-step conditions under their stated assumptions. They do not establish globally bounded trajectories under arbitrary affine forcing.*
   - *Locations updated*: `docs/math/PHASE_4_STABILITY.md`, `docs/development/PHASE_4_IMPLEMENTATION.md`.
2. **Finite-State Semantics Corrected**:
   - *Previous behavior*: Telemetry reported `finite_state = True` even if historical steps overflowed to `inf` during the trajectory.
   - *Corrected behavior*: Telemetry exposes:
     - `all_states_finite: bool` (true iff every intermediate state in the trajectory is finite)
     - `terminal_state_finite: bool` (true iff the final state is finite)
     - `first_nonfinite_step: Optional[int]` (exact 0-indexed step where finiteness was first violated)
   - *Locations updated*: `deltacore/diagnostics/stability.py`, `examples/phase_4_stability_comparison.py`.
3. **$\epsilon$ Regularization Semantics Clarified**:
   - *Previous statement*: Claimed $\beta = 1$ gives an exact normalized step of 1.
   - *Corrected statement*: Explicitly documented that for any numerical stabilizer $\epsilon > 0$, the attained normalized step satisfies:
     $$\gamma_t = \frac{\beta \|k_t\|^2}{\|k_t\|^2 + \epsilon} < \beta$$
     Documented the strict inequality while preserving the robust implementation.
   - *Locations updated*: `docs/math/PHASE_4_STABILITY.md`, `docs/development/PHASE_4_IMPLEMENTATION.md`.

---

## 3. Mathematical Derivation of the Affine Recurrence

### 3.1. Standard Delta Update
Let memory $M_t \in \mathbb{R}^{V \times K}$, key $k_t \in \mathbb{R}^K$, target $v_t \in \mathbb{R}^V$, and step size $\eta_t \in \mathbb{R}^+$. The Delta update is:
$$M_{t+1} = M_t + \eta_t (v_t - M_t k_t) k_t^\top$$

Distributing terms:
$$M_{t+1} = M_t + \eta_t v_t k_t^\top - \eta_t M_t k_t k_t^\top = M_t (I_K - \eta_t k_t k_t^\top) + \eta_t v_t k_t^\top$$

### 3.2. Affine Normal Form
Define:
$$A_t = I_K - \eta_t k_t k_t^\top \in \mathbb{R}^{K \times K}$$
$$B_t = \eta_t v_t k_t^\top \in \mathbb{R}^{V \times K}$$

Then the recurrence is purely affine:
$$M_{t+1} = M_t A_t + B_t$$

### 3.3. Composition Law
Let $F_1(M) = M A_1 + B_1$ and $F_2(M) = M A_2 + B_2$. Applying $F_1$ followed by $F_2$:
$$F_2(F_1(M)) = (M A_1 + B_1) A_2 + B_2 = M (A_1 A_2) + (B_1 A_2 + B_2)$$

Therefore, composition $(F_1 \otimes F_2)$ is defined by:
$$(A_{12}, B_{12}) = (A_1 A_2, \; B_1 A_2 + B_2)$$

### 3.4. Associativity Proof
Given three sequential affine operators $F_1, F_2, F_3$:
- **Left composition**:
  $$(F_1 \otimes F_2) \otimes F_3 = (A_1 A_2, B_1 A_2 + B_2) \otimes (A_3, B_3) = ((A_1 A_2) A_3, (B_1 A_2 + B_2) A_3 + B_3) = (A_1 A_2 A_3, B_1 A_2 A_3 + B_2 A_3 + B_3)$$
- **Right composition**:
  $$F_1 \otimes (F_2 \otimes F_3) = (A_1, B_1) \otimes (A_2 A_3, B_2 A_3 + B_3) = (A_1 (A_2 A_3), B_1 (A_2 A_3) + B_2 A_3 + B_3)$$
By associativity of matrix multiplication and distributivity, $(F_1 \otimes F_2) \otimes F_3 = F_1 \otimes (F_2 \otimes F_3)$.

### 3.5. Identity Operator
The identity operator $I(M) = M I_K + 0$ satisfies:
$$(I_K, 0) \otimes (A, B) = (I_K A, 0 A + B) = (A, B)$$
$$(A, B) \otimes (I_K, 0) = (A I_K, B I_K + 0) = (A, B)$$

Thus, the set of affine operators under $\otimes$ forms a **monoid**.

---

## 4. Software Architecture & Components

DeltaCore implements this algebra in `deltacore/scans/` and `deltacore/diagnostics/`:

```text
deltacore/
├── scans/
│   ├── affine.py         # AffineScanOperator monoid container & apply/compose
│   ├── delta_affine.py   # delta_to_affine token converter with stability hooks
│   ├── prefix.py         # affine_prefix_scan & prefix_scan_memory
│   ├── chunked.py        # AffineChunk, build_chunks, run_chunk, chunked_scan
│   └── sequential.py     # Reference step-by-step unrolled scan (Phase 1)
└── diagnostics/
    └── scans.py          # ScanDiagnostics telemetry & verification
```

### Module Summary:
- **`AffineScanOperator`**: Dataclass holding $A \in \mathbb{R}^{K \times K}$ (or batched $[B, K, K]$) and $B \in \mathbb{R}^{V \times K}$ (or batched $[B, V, K]$), stability metadata (`min_stability_margin`, `max_normalized_step`, `clipped_transition_count`), providing `.apply(memory)`, `.compose(second)`, and `.identity(k_dim, v_dim)`.
- **`delta_to_affine`**: Converts raw $(k_t, v_t, \eta_t)$ tuples into an `AffineScanOperator`. Supports optional `BaseStabilityController` injection.
- **`AffineChunk`**: Represents an aggregated sequence chunk retaining `start_idx`, `end_idx`, `transition_count`, and the composed `operator`.
- **`build_chunks`**: Aggregates a sequence of affine operators into chunks of specified size $C$. Safely handles arbitrary chunk sizes and non-divisible trailing chunks.
- **`run_chunk`**: Executes chunk boundary state propagation: consumes `initial_state` and `AffineChunk` (or constituent coefficients) and returns `final_state` plus reconstructible `internal_states`.
- **`chunked_scan`**: High-level executor dividing sequences into chunks, running inter-chunk state propagation and optional intra-chunk prefix unrolling.
- **`ScanDiagnostics`**: Records chunk count, mean/max chunk lengths, boundary count, composition tree depth, and maximum absolute difference $\|M_{\text{seq}} - M_{\text{chk}}\|_\infty$.

---

## 5. Critical Distinction: Known Coefficients vs. Self-Reference

A central architectural mandate of Phase 5 is distinguishing between:
1. **State Scanning with Known Coefficients**:
   When $\eta_t, k_t, v_t$ are known or precomputed (e.g. static step size, input-conditioned step size $f_\theta(x_t)$, or precomputed schedule):
   $$A_t = I - \eta_t k_t k_t^\top, \quad B_t = \eta_t v_t k_t^\top$$
   The sequence of operators $\{F_0, F_1, \dots, F_{T-1}\}$ can be precomputed and composed associatively in parallel using tree prefix scans with $O(\log T)$ depth.
2. **Self-Referential Recurrence (Phase 3)**:
   In self-referential nested learning, the controller memory $C_t$ co-evolves with $M_t$:
   $$C_t \to \eta_t(C_t, z_t) \to A_t(C_t), B_t(C_t) \to M_{t+1} \to e_{C, t}(M_t) \to C_{t+1}$$
   Because $A_t$ depends on $C_t$ which depends on $M_{t-1}$, transition coefficients cannot be precomputed prior to state propagation.

**Scientific Finding**: Associative chunking parallelizes **state scanning**, NOT the sequential feedback loop of self-referential dynamics. Any claim that chunking automatically parallelizes full self-reference is mathematically false and rejected in DeltaCore.

---

## 6. Verification & Test Suite Matrix

DeltaCore Phase 5 is verified across 4 dedicated test suites (totaling 43 tests for Phase 5, 215 across the repository):

| Test Suite | File | Tests | Coverage & Invariants Verified |
| :--- | :--- | :--- | :--- |
| **Associativity & Identity** | `tests/test_affine_scan.py` | 9 | - Associativity $(F_1 \otimes F_2) \otimes F_3 == F_1 \otimes (F_2 \otimes F_3)$ in FP32 and FP64.<br>- Application match $(F_1 \otimes F_2)(M) == F_2(F_1(M))$.<br>- Two-sided identity $I \otimes F == F \otimes I == F$.<br>- Batched $[B, V, K]$ operator composition.<br>- Stability metadata aggregation across compositions. |
| **Sequential Equivalence** | `tests/test_affine_equivalence.py` | 11 | - `sequential_scan` vs sequential `AffineScanOperator.apply`.<br>- Verified across FP32 ($< 10^{-6}$) and FP64 ($< 10^{-14}$).<br>- Zero-vector keys ($A = I, B = 0$).<br>- Repeated conflicting keys and highly correlated keys.<br>- Varying $(V, K)$ dimensions ($[4, 8]$, $[16, 16]$, $[32, 64]$).<br>- Trajectory reconstruction matching step-for-step. |
| **Chunked Equivalence** | `tests/test_chunked_equivalence.py` | 18 | - Final memory exact match across chunk sizes $C \in \{1, 2, 3, 4, 7, 16\}$.<br>- Irregular non-divisible sequences (e.g. $T=17, C=5$).<br>- Trajectory reconstruction over all intermediate steps.<br>- Chunk boundary API (`run_chunk`).<br>- Adaptive precomputed step sizes $\eta_t$. |
| **Autograd Continuity** | `tests/test_scan_autograd.py` | 5 | - FP64 `gradcheck` on `apply` w.r.t $M, A, B$.<br>- FP64 `gradcheck` on `compose` w.r.t $A_1, B_1, A_2, B_2$.<br>- Backward gradient equivalence: $\nabla_{M_0, K, V, \eta} \mathcal{L}$ identical between `sequential_scan` and `chunked_scan`. |

All 215 tests in the repository pass cleanly in 1.64s.

---

## 7. Empirical Performance & Numerical Benchmarks

Benchmarks were executed via `examples/phase_5_scan_benchmark.py` on an Apple Silicon M-series processor (PyTorch 2.6.0, Python 3.13.0, CPU device):

### Part 1: Numerical Precision Scaling vs. Sequence Length ($C = 16$)
$$\text{Max Abs Diff} = \max_t \|M_t^{\text{seq}} - M_t^{\text{chk}}\|_\infty$$

| Length ($T$) | Dtype | Chunk Size ($C$) | Chunks ($N$) | Comp Depth | Max Abs Diff ($\|M_{\text{seq}} - M_{\text{chk}}\|_\infty$) | Numerical Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | FP32 | 16 | 1 | 7 | $5.96 \times 10^{-8}$ | Consistent ($< 10^{-4}$) |
| 8 | FP64 | 16 | 1 | 7 | $0.00 \times 10^{00}$ | Exact ($< 10^{-10}$) |
| 16 | FP32 | 16 | 1 | 15 | $3.58 \times 10^{-7}$ | Consistent ($< 10^{-4}$) |
| 16 | FP64 | 16 | 1 | 15 | $0.00 \times 10^{00}$ | Exact ($< 10^{-10}$) |
| 32 | FP32 | 16 | 2 | 15 | $4.77 \times 10^{-7}$ | Consistent ($< 10^{-4}$) |
| 32 | FP64 | 16 | 2 | 15 | $0.00 \times 10^{00}$ | Exact ($< 10^{-10}$) |
| 64 | FP32 | 16 | 4 | 15 | $3.58 \times 10^{-7}$ | Consistent ($< 10^{-4}$) |
| 64 | FP64 | 16 | 4 | 15 | $0.00 \times 10^{00}$ | Exact ($< 10^{-10}$) |
| 128 | FP32 | 16 | 8 | 15 | $4.77 \times 10^{-7}$ | Consistent ($< 10^{-4}$) |
| 128 | FP64 | 16 | 8 | 15 | $0.00 \times 10^{00}$ | Exact ($< 10^{-10}$) |
| 512 | FP32 | 16 | 32 | 15 | $2.38 \times 10^{-7}$ | Consistent ($< 10^{-4}$) |
| 512 | FP64 | 16 | 32 | 15 | $0.00 \times 10^{00}$ | Exact ($< 10^{-10}$) |
| 1024 | FP32 | 16 | 64 | 15 | $1.19 \times 10^{-7}$ | Consistent ($< 10^{-4}$) |
| 1024 | FP64 | 16 | 64 | 15 | $0.00 \times 10^{00}$ | Exact (< 1e-10) |

**Observations & Scientific Notes**:
- **Comp Depth Semantics**: `Comp Depth` is explicitly defined as $\max(C_k - 1, 0)$ under linear left-to-right composition chaining (15 binary compositions for chunk size 16). It represents linear sequential composition depth within the chunk, **not** logarithmic tree reduction depth ($O(\log C)$).
- **FP64 Associativity Semantics**: The affine composition algebra is mathematically associative; floating-point evaluations may differ because matrix multiplication is not numerically associative. In this specific CPU benchmark, the observed FP64 difference was $0.00 \times 10^{00}$, which is an observed empirical result for this sequence, rather than a universal mathematical guarantee of bit-exactness.
- In FP32, rounding errors accumulate to at most $4.77 \times 10^{-7}$, well below typical threshold tolerances.

### Part 2: Runtime Measurement on CPU (Wall-clock Time in ms)

| Length ($T$) | Sequential Scan (ms) | Pure Affine Loop (ms) | Chunk $C=4$ (ms) | Chunk $C=16$ (ms) | Chunk $C=64$ (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64 | 1.187 ms | 0.249 ms | 0.261 ms | 0.295 ms | 0.294 ms |
| 256 | 4.520 ms | 0.517 ms | 1.050 ms | 1.135 ms | 1.141 ms |
| 1024 | 18.216 ms | 2.115 ms | 4.219 ms | 4.551 ms | 4.583 ms |

**Analysis of Results**:
1. `Pure Affine Loop` ($M_{t+1} = M_t A_t + B_t$) takes 2.115 ms at $T=1024$.
2. `Sequential Scan` (Phase 1 `sequential_scan`) includes read computations $e_t = v_t - M_t k_t$, diagnostic trajectory capture, and rank-1 updates, taking 18.216 ms.
3. `Chunked Scan` takes 4.219 ms – 4.583 ms. In single-threaded CPU Python, composing chunks computes $(K \times K) \times (K \times K)$ matrix products sequentially, which costs more FLOPs than pure rank-1 updates.
4. **Conclusion**: The wall-clock benefit of associative chunking stems from **algorithmic parallelism** ($O(\log T)$ parallel depth on multithreaded hardware accelerators like GPUs/TPUs). On a single-threaded CPU runtime, composing matrices sequentially increases total operations. We document this clearly without falsely claiming a Python CPU speedup.

---

## 8. Relation to VisionHOPE & Literature Architecture

As documented in [RELATION_TO_VISIONHOPE.md](file:///Users/urjasoft/Documents/DeltaCore/docs/math/RELATION_TO_VISIONHOPE.md#L87-L108):
- **DeltaCore Phase 5**: Establishes a pure 1D algebraic monoid for affine state updates $(A, B)$ in PyTorch, proving exact sequential/chunk equivalence and separating coefficient generation from state propagation.
- **VisionHOPE (Peng et al., September 2026)**: Operates over 2D visual feature grids using four-directional row/column chunked sweeps, combining intra-chunk self-attention, learned retention decays, and Triton hardware kernels.
- **Epistemic Boundary**: DeltaCore does **not** claim equivalence with VisionHOPE's directional chunk formulation. Spatial operators and directional scans are deferred to Phase 8.

---

## 9. Phase 5 Scientific Honesty Gate

Before completing Phase 5, all negative constraints and claims were audited:
- [x] Did NOT claim that chunking automatically parallelizes self-reference.
- [x] Did NOT claim chunked execution is faster on CPU without benchmark evidence.
- [x] Did NOT claim floating-point results are bitwise identical (reported FP32 $\sim 10^{-7}$ and FP64 exactness).
- [x] Did NOT claim chunking provides global stability (retained local non-expansion definitions).
- [x] Did NOT claim chunk-level non-expansion follows automatically from token-level non-expansion (exposed minimum constituent margins).
- [x] Did NOT begin Phase 6 benchmark tasks or Phase 8 VisionHOPE spatial sweeps.
