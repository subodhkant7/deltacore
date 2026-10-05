# Phase 9: Scientific Interpretation & Epistemic Boundaries

This document establishes the formal epistemic boundaries and scientific interpretations of results obtained in **DeltaCore Phase 9: Generic 2D Spatial Routing & Boundary-Chunk Execution**.

In accordance with DeltaCore Rule 2 (*Never Silently Change Mathematical Definitions*), Rule 8 (*Do Not Call an Implementation "Stable" Unless Stability Has Been Tested*), and Rule 9 (*Do Not Claim Equivalence with a Paper Unless Verified*), each property established in Phase 9 is categorized according to its strict epistemic status.

---

## 1. Classification of Phase 9 Results

| Result / Property | Epistemic Status | Formal Characterization | What It Proves | What It Does NOT Prove |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Route Round-Trip** | **Mathematical Property** | Invertible tensor permutation: $\mathcal{P}_r^{-1}(\mathcal{P}_r(Z)) = Z, \forall (H, W)$ | Serialization and spatial restoration are exact mathematical bijections for arbitrary rectangular dimensions. | Does not prove spatial generalization or vision inductive bias. |
| **$C=1$ Boundary Semantics** | **Reference Equivalence** | Exact equivalence: $\text{BoundaryRefreshScan}(C=1) \equiv \text{SequentialScan}$ | Step-by-step state and representation updates with $C=1$ match the token-indexed reference recurrence token-for-token. | Does not imply that $C > 1$ is equivalent to sequential unrolling. |
| **$C > 1$ Chunk Sensitivity** | **Algorithmic Chunking Effect** | Approximate recurrence with stale boundary generators $(M_b^k, M_b^v, m_b^\eta, m_b^\alpha)$ | Boundary staleness is a real, measurable computational phenomenon; trajectory discrepancy increased monotonically with chunk length in the tested configuration. | Does not represent a numerical precision bug or implementation discrepancy. |
| **Four-Direction Fusion** | **Empirical Result** | Convex or weighted aggregation: $Y = \sum_r \lambda_r \odot Y_r$ | Multi-directional sweeps accumulate gradients across orthogonal spatial axes, improving reconstruction stability over single routes. | Does not prove that the model possesses high-level visual understanding. |
| **Spatial Activity Patterns** | **Diagnostic Observation** | Direction-dependent activity maps $\|\Delta W_r(h, w)\|$ | Distinct spatial patterns produce direction-specific update magnitudes due to 1D scan alignment with pattern axes. | **Does NOT prove semantic specialization** or human-like spatial feature abstraction. |

---

## 2. Detailed Epistemic Delimitations

### 2.1. Exact Route Invertibility is an Algebraic Property
The round-trip identity:
$$\mathcal{P}_r^{-1}(\mathcal{P}_r(Z)) = Z$$
is an algebraic consequence of bijective coordinate permutation and reshape operations over the 4D feature map. It preserves all elements, precisions (FP32/FP64), batch structures, and gradient flows identically. It provides the plumbing for multidirectional processing, but carries no inherent visual semantics.

### 2.2. $C=1$ vs. $C > 1$ Boundary-Refresh Semantics
A central finding of Phase 9 is the mathematical distinction between:
1. **DeltaCore Phase 5 Affine Chunk Composition (`ExactAffineChunkScan`)**:
   Associative scan composition over known affine operators $(A_t, B_t)$ is exact:
   $$\text{FinalState}(C) = \text{FinalState}(C=1), \quad \forall C \ge 1$$
2. **Phase 9 Boundary-Refresh Chunk Execution (`BoundaryRefreshChunkScan`)**:
   Within each chunk $b$, the state generating representations is held fixed:
   $$k_t = M_b^k x_t, \quad v_t = M_b^v x_t, \quad \eta_t = \phi_\eta(m_b^\eta x_t), \quad \alpha_t = \phi_\alpha(m_b^\alpha x_t)$$
   Consequently, the memory state used to evaluate error and generate keys is *stale* relative to the actively evolving working memory.
   - For $C = 1$, $S_b = S_t$ at every step, yielding strict mathematical equivalence to full sequential unrolling.
   - For $C > 1$, boundary staleness induces real, computed differences in the trajectory ($\|Y_C - Y_1\|_F > 0$).
   - As observed in `examples/phase_9_chunk_semantics.py`, trajectory discrepancy increased monotonically with chunk length in the tested configuration ($1.66 \times 10^{-4}$ for $C=2$ up to $1.59 \times 10^{-3}$ for $C=64$).
   - This divergence is **an inherent algorithmic property of boundary-refresh chunking**, not an implementation artifact.

### 2.3. Definition of Paper Alignment Diff
The metric `Paper Alignment Diff` reported in Phase 9 spatial benchmarks is explicitly defined as the Frobenius norm discrepancy between the paper-aligned chunk execution and the token-indexed reference ($C = 1$):
$$\text{Paper Alignment Diff} \equiv \|Y_{\text{paper\_aligned}} - Y_{\text{generic}(C=1)}\|_F$$
where $Y_{\text{paper\_aligned}}$ executes with $C_\rightarrow = W, C_\leftarrow = W, C_\downarrow = H, C_\uparrow = H$, and $Y_{\text{generic}(C=1)}$ executes with uniform $C = 1$. It quantifies the precise numerical perturbation induced by row/column boundary staleness on a given spatial feature map.

### 2.4. Computational Scaling & Runtime Epistemics
During CPU profiling across grid scales $7\times 7, 14\times 14, 14\times 20, 28\times 28$, observed CPU runtime scaled approximately linearly over the tested grid sizes ($\mathcal{O}(N)$ where $N = HW$). This is an empirical scaling observation on the tested CPU architecture, not an asymptotic hardware guarantee.

### 2.5. Directional State Activity is NOT Semantic Specialization
When traversing synthetic test patterns (e.g. horizontal stripes, vertical stripes, checkerboards):
- A horizontal sweep ($\rightarrow$ or $\leftarrow$) along a horizontal stripe encounters constant values within each row, resulting in minimal intra-row error and lower update activity.
- A vertical sweep ($\downarrow$ or $\uparrow$) along a horizontal stripe crosses gradient boundaries at every step, inducing continuous prediction error and high update energy $\|\Delta W_r\|$.

This asymmetry is a direct mechanical consequence of sequence serialization intersecting with spatial gradient orientations.
**It must not be reported as "the model learned to attend to horizontal concepts."**
It is a diagnostic observation of 1D linear recurrence dynamics projected onto a 2D grid.

---

## 3. Exit Status & Phase 9 Conclusions
Phase 9 establishes a verifiable, mathematically rigorous foundation for 2D directional spatial routing in DeltaCore. All core invariants—exact serialization, directional independence, boundary-refresh chunking, multi-route autograd differentiability, and observable diagnostic mapping—have been implemented, tested, and validated without premature GPU optimizations or unverified literature equivalence claims.

---

## 4. Phase 9.2 Ablation Update

Phase 9.2 conducted a causal-diagnosis ablation that materially extends the epistemic understanding of spatial routing established above. The following updates and revisions are documented here for cross-reference; the full analysis resides in [PHASE_9_2_INTERPRETATION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_9_2_INTERPRETATION.md).

### 4.1. Updated Classification of Phase 9 Results

| Result / Property | Original Phase 9 Status | Phase 9.2 Update |
| :--- | :--- | :--- |
| **Spatial Activity Patterns** | Diagnostic Observation | **SUPPORTED as genuine spatial dependence**: Pixel-shuffle control proves DeltaCore relies on intact spatial arrangement, not coordinate-independent statistics (Task B: $42.3\% \to 30.7\%$ under pixel permutation). |
| **Four-Direction Fusion** | Empirical Result (qualitative improvement) | **NOT SUPPORTED as quantitative advantage**: Paired difference $\Delta_{\text{4dir} - \text{1dir}} = +5.33\% \pm 8.78\%$; confidence interval includes zero. Single-direction scanning achieves higher accuracy with fewer parameters. |

### 4.2. New Epistemic Properties from Phase 9.2

| Property | Epistemic Status | Finding |
| :--- | :--- | :--- |
| **Spatial Addressability Necessity** | **SUPPORTED** | Injecting normalized $(x,y)$ coordinates reduces Task A error from $1.002$ to $0.866$; shuffling coordinates reverts to $1.001$. DeltaCore's sequential recurrence requires explicit positional information for spatial tasks. |
| **Recurrent Optimization Instability** | **OBSERVED** | Trainable update dynamics diverge to NaN at $\eta = 0.01$ (4/5 seeds), requiring $\eta \le 0.003$ for stability. Unconstrained recurrent parameters amplify gradients multiplicatively over 100-step BPTT chains. |
| **Task A Architectural Bottleneck** | **SUPPORTED** | The primary Task A failure mode is the absence of a direct 2D local receptive field, not insufficient parameter count. `conv3x3` ($0.349$) vastly outperforms `delta_reference_xy` ($0.866$) despite fewer parameters. |
| **Chunk Size Invariance Under Learning** | **OBSERVED** | When the model is actively learning (coordinate-injected), chunk size has negligible effect on task quality ($E_{\text{rel}} \in [0.849, 0.879]$ across $C \in \{1, 4, 8, \text{full}\}$). |

### 4.3. Phase Gate Outcome
**Outcome B**: DeltaCore demonstrates genuine spatial sensitivity on classification tasks but fails spatial reconstruction competitive with local convolution. The primary bottleneck is architectural, not parametric. See [PHASE_9_2_INTERPRETATION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_9_2_INTERPRETATION.md) for full hypothesis evaluation (H1–H7).
