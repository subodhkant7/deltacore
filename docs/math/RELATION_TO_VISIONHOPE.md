# Relation to VisionHOPE & Literature Architecture

This document clarifies the relationship between **DeltaCore's foundational phases** and published literature on self-referential nested learning (SRNL) and test-time training, specifically **VisionHOPE** (Siran Peng, Tianshuo Zhang, Tianyu Fu, Weisong Zhao, Haoyuan Zhang, Jiankuo Zhao, Minghui Wu, Ping Jiang, Xiangyu Zhu, Chenxu Zhao, Zhen Lei, September 2026; [arXiv:2609.33325](https://arxiv.org/abs/2609.33325); official repository: [https://github.com/PSRben/VisionHOPE](https://github.com/PSRben/VisionHOPE)).

---

## 1. Incremental Abstraction Hierarchy

DeltaCore investigates adaptive neural state through isolated, verifiable mathematical milestones rather than jumping directly to monolithic end-to-end vision models.

```
DeltaCore Phase 1: Pure Associative Memory & Delta-Rule Update
        │
        ▼
DeltaCore Phase 2: Adaptive Update Dynamics (External Controllers)
        │
        ▼
DeltaCore Phase 3: Minimal Self-Referential Adaptive Memory (Two Coupled Memories)
        │
        ▼
[Future Phases 4-7: Stability Guarantees, Parallel Scans, Standardized Benchmarks]
        │
        ▼
DeltaCore Phase 8: Multi-Memory Vision Experiments (VisionHOPE Analysis)
```

### Phase Comparison Table

| Architecture Level | State Components | Update Mechanism | Modulation Source | Scope & Exclusions |
| :--- | :--- | :--- | :--- | :--- |
| **DeltaCore Phase 1** | Single memory matrix: $M_t \in \mathbb{R}^{V \times K}$ | Error-correcting delta rule: $M + \eta (v - M k) k^\top$ | Fixed scalar $\eta_0$ | Reference mathematical foundation. |
| **DeltaCore Phase 2** | Content memory: $M_t \in \mathbb{R}^{V \times K}$ | Delta rule with dynamic scalar: $M + \eta_t e_t k^\top$ | Static external controller: $\eta_t = f_\theta(z_t)$ | Adaptive dynamics; controller parameters $\theta$ do not evolve during inference. |
| **DeltaCore Phase 3** | Two coupled memories: Content $M_t$, Dynamics $C_t$ | Dual delta updates: $\Delta M_t \propto \eta_t e_t k^\top$, $\Delta C_t \propto \rho_t e_{C, t} z^\top$ | Evolving controller memory: $\eta_t = \eta_{\max}\sigma(C_t z_t)$ | **Minimal self-referential state machine**. Controller memory $C_t$ co-evolves online. |
| **VisionHOPE (Full Paper)** | **Five coupled memories**: $M_{\text{content}}, M_{\text{key}}, M_{\text{val}}, M_{\text{lr}}, M_{\text{ret}}$ | Co-evolving associative memories over spatial 2D image patch scans | Self-generated learning rates AND retention factors | Complex vision pipeline with spectral norm clamps, soft injection caps, and multi-head attention. |

---

## 2. What DeltaCore Phase 3 Implements vs. What It Defers

### What Phase 3 Implements
- A **two-memory coupled recurrence**: a content associative memory $M_t$ and a dynamics memory $C_t$.
- True **self-referential state evolution**: $C_{t+1} \neq C_t$; the state controlling the update rate itself adapts via online feedback.
- Pure-PyTorch functional semantics and full autograd differentiability through the coupled recurrence.
- Strict diagnostic telemetry for observing coupling correlation and state growth without premature guards.

### What Phase 3 Intentionally Does NOT Implement
1. **Five Coupled Memories**: VisionHOPE couples five distinct memories ($M_{\text{content}}, M_{\text{key}}, M_{\text{val}}, M_{\text{lr}}, M_{\text{ret}}$). DeltaCore Phase 3 deliberately restricts the system to two memories ($M_t$ and $C_t$) to isolate the minimal dynamics of self-reference.
2. **Self-Generated Retention Factors**: VisionHOPE adapts retention decay factors $\lambda_t$ via memory. Phase 3 focuses strictly on step-size adaptation $\eta_t$; retention dynamics are investigated in Phase 4.
3. **Spectral Stability Controls**: VisionHOPE uses an empirical spectral norm clamp to prevent memory blow-up. DeltaCore Phase 3 intentionally exposes raw, unconstrained dynamics to observe divergence modes before introducing formal controllers in Phase 4.
4. **Spatial Patch Scans & Vision Encoders**: VisionHOPE operates over 2D patch tokens from ViT backends. Spatial operators belong to Phase 8.

---

## 3. Epistemic Status (DeltaCore Constitution Principle A)

- **Source Literature Claim**: VisionHOPE reports that five co-evolving associative memories improve downstream visual adaptation under distribution shift.
- **DeltaCore Phase 3 Implementation**: A minimal, decoupled two-memory state machine implementing online-updated learning rates.
- **DeltaCore Verification**: Proves that self-referential controller evolution can be implemented functionally, differentiated via PyTorch autograd, and benchmarked against fixed and adaptive baselines.
- **Hypothesis Remaining**: Whether self-referential dynamics systematically outperform well-tuned adaptive controllers without requiring aggressive stability clamps remains an open research question to be settled across future phases.

> **Conclusion**: DeltaCore Phase 3 and Phase 4 represent an **abstraction study**, not a reproduction of VisionHOPE.

---

## 4. Phase 4 Stability Construction vs. VisionHOPE Stability Mechanisms

A central focus of Phase 4 is understanding the theoretical mechanisms that prevent runaway state divergence in recurrent adaptive systems.

### 4.1. DeltaCore Phase 4 Formulation
DeltaCore explicitly derives local non-expansion conditions directly from its own mathematical recurrence:
- **Immediate-Key Contraction**: Proves that the prediction error on key $k_t$ satisfies $e_{t+1}^{(k_t)} = (1 - \eta_t \|k_t\|_2^2) e_t$, giving the necessary and sufficient non-expansion condition $0 \le \eta_t \|k_t\|_2^2 \le 2$.
- **Safe Step Size Controller**: Enforces $\eta_t^{\text{safe}} = \min(\eta_t^{\text{raw}}, \frac{\beta}{\|k_t\|_2^2 + \epsilon})$ with $0 < \beta < 2$, leaving raw adaptive dynamics untouched until they approach the expansion boundary.
- **Safe Dynamics Rate Controller**: Enforces $\rho_t^{\text{safe}} = \min(\rho_t^{\text{raw}}, \frac{\beta_C}{\|z_t\|_2^2 + \epsilon})$ with $0 < \beta_C < 2$ on the dynamics memory update.
- **Operator Norm Analysis**: Demonstrates that the homogeneous transition operator $A_t = I - \eta_t k_t k_t^\top$ satisfies $\|A_t\|_2 \le 1$, but proves that the affine drive $B_t = \eta_t v_t k_t^\top$ can still increase $\|M_t\|_F$ when absorbing non-zero targets.

### 4.2. VisionHOPE Paper Construction
As described by Peng et al. (September 2026), VisionHOPE stabilizes its five-memory system via three distinct mechanisms:
1. **Soft Cap on Self-Referential Injection**: Clamps or squashes generated learning rates and retention factors through bounded non-linearities (e.g. sigmoid and scaled tanh) to prevent unbounded update magnitudes.
2. **Spectral Norm Clamp on Retained Transitions**: Projects or rescales the recurrent transition matrix so that its maximum singular value (spectral norm) remains strictly $\le 1$, ensuring non-expansive state propagation across token boundaries.
3. **Non-Expansive Token/Chunk Recurrence**: Structures associative chunk updates so that intra-chunk associative scans maintain bounded energy across deep visual backbones.

### 4.3. Comparison & Mathematical Distinction
- **DeltaCore**: Derives explicit closed-form bounds $\eta_t \le \frac{\beta}{\|k_t\|^2}$ based on the exact rank-1 algebraic properties of the delta rule, without introducing heavy SVD or power-iteration spectral clamps.
- **VisionHOPE**: Operates in higher-dimensional multi-head attention and chunked spatial scans where spectral clamps and soft injection caps are required to stabilize five interconnected memories.
- **No Equivalence Claimed**: DeltaCore's safety controllers and VisionHOPE's spectral clamps are **distinct mechanisms addressing stability at different structural levels**. They are NOT mathematically equivalent.

---

## 5. Phase 5 Associative & Chunked State Scans vs. VisionHOPE Chunk Formulation

Phase 5 addresses associative state scans and chunked execution strategies for recurrent associative memory updates.

### 5.1. DeltaCore Phase 5 Formulation
DeltaCore develops a pure mathematical abstraction of affine associative state scanning:
1. **Generic Affine Monoid**: Formulates the delta-rule transition as an affine mapping $F(M) = M A + B$ where $A_t = I - \eta_t k_t k_t^\top \in \mathbb{R}^{K \times K}$ and $B_t = \eta_t v_t k_t^\top \in \mathbb{R}^{V \times K}$. Proves that composition $(A_1, B_1) \otimes (A_2, B_2) = (A_1 A_2, B_1 A_2 + B_2)$ forms an associative monoid with identity $(I, 0)$.
2. **Exact Equivalence**: Proves and numerically verifies that decomposing a sequence into chunks of arbitrary size $C \ge 1$ (including irregular, non-divisible trailing chunks) reproduces the exact sequential final state and full step trajectory up to floating-point precision ($10^{-7}$ in FP32, machine zero $< 10^{-15}$ in FP64).
3. **Decoupled Architecture**: Strictly separates the **generation of transition coefficients** $(\eta_t, k_t, v_t)$ from the **associative state scan** over $(A_t, B_t)$. The scan is parallelizable if and only if transition coefficients are known prior to or independently of the state scan.

### 5.2. VisionHOPE Formulation
In contrast, VisionHOPE (Peng et al., September 2026; Nested Learning / TTT-inspired):
1. **2D Directional Chunked Scans**: VisionHOPE adapts Nested Learning's chunk formulation specifically for visual tokens arranged on a 2D spatial lattice. It processes feature maps via four directional scans (left-to-right, right-to-left, top-to-bottom, bottom-to-top) across rows and columns.
2. **Specialized Block Decay & Attention**: Integrates intra-chunk self-attention with inter-chunk recurrent state propagation, combining continuous exponential decay or learned retention factors into the chunk representation.
3. **Triton/CUDA Acceleration**: Relies on optimized custom hardware kernels to parallelize chunk-level matrix operations and directional sweeps.

### 5.3. Scientific & Methodological Distinction
- **DeltaCore Phase 5**: A minimal, generic, 1D sequential/chunked algebraic operator in pure PyTorch. It does not implement 2D directional sweeps, image patch folding, or fused hardware kernels.
- **Independence & Non-Equivalence**: DeltaCore does **not** claim to reproduce VisionHOPE's chunk formulation. The generic affine scan $(A, B)$ is the classical monoid for linear/affine recurrence (e.g. Blelloch prefix scans / parallel linear state space models); VisionHOPE's formulation is a specialized vision-domain architecture combining multi-memory nested adaptation, retention decay, and spatial sweeps.

---

## 6. Phase 9: 2D Spatial Routing & Boundary-Chunk Execution vs. VisionHOPE

Phase 9 introduces 2D spatial serialization, directional routing, spatial restoration, and boundary-refresh chunk execution.

### 6.1. Shared Concepts

1. **Four Canonical Directions**:
   Both architectures traverse 2D feature maps along four canonical routes:
   $$\mathcal{R} = \{\text{RIGHT} \ (\rightarrow), \text{LEFT} \ (\leftarrow), \text{DOWN} \ (\downarrow), \text{UP} \ (\uparrow)\}$$
2. **Independent Directional State**:
   Each directional route maintains an entirely independent internal memory state:
   $$\frac{\partial S_t^r}{\partial S_{t'}^{r'}} = 0, \quad \forall r \neq r'$$
   There is zero cross-directional state contamination during sequence traversal.
3. **Row/Column-Aligned Traversal**:
   Horizontal routes ($\rightarrow, \leftarrow$) serialize across rows; vertical routes ($\downarrow, \uparrow$) serialize across columns.
4. **Spatial Restoration**:
   Sequence predictions $[B, N, C]$ are inverted via exact spatial restorations $\mathcal{P}_r^{-1}$ back to the original 2D lattice $[B, C, H, W]$.
5. **Direction-Wise Channel Fusion**:
   Fused spatial output combines directional predictions via channel-wise scaling:
   $$Y = \sum_{r \in \{R, L, D, U\}} \lambda_r \odot Y_r, \quad \lambda_r \in \mathbb{R}^C$$

### 6.2. Paper-Aligned Chunking Specification

VisionHOPE specifies directional chunk sizes aligned to the spatial grid geometry:
$$C_\rightarrow = C_\leftarrow = W, \quad C_\downarrow = C_\uparrow = H$$
Under this geometry, horizontal routes execute one chunk boundary refresh per image row, while vertical routes execute one chunk boundary refresh per image column.

DeltaCore implements this explicitly in `mode="paper_aligned"`:
```python
op = SpatialAdaptiveOperator(config=cfg, mode="paper_aligned")
```
while also supporting `mode="generic"` where users may specify arbitrary uniform or per-route chunk sizes $C \ge 1$.

### 6.3. Fundamental DeltaCore Distinctions & Engineering Differences

| Dimension | VisionHOPE (Peng et al., 2026) | DeltaCore Phase 9 |
| :--- | :--- | :--- |
| **Learner Core** | Monolithic SRNL visual block | Modular, route-agnostic `FiveMemorySystem` |
| **Lattice Geometry** | Fixed square patch grids ($14 \times 14$) | Arbitrary rectangular dimensions $(H, W)$ with exact round-trip identity |
| **Execution Backend** | Custom Triton / CUDA GPU kernels | Pure PyTorch CPU reference implementation |
| **Observability** | Final downstream loss / metrics | Full step-by-step diagnostic trajectories & spatial heatmaps (Plots J–N) |
| **Chunking Semantics** | Hardcoded boundary-refresh chunking in CUDA | Dual execution models: exact associative scan (`ExactAffineChunkScan`) vs. boundary refresh (`BoundaryRefreshChunkScan`) |
| **Vision Backbone** | Deep 12/24-layer ViT with residual blocks | Zero deep backbone; isolated spatial routing layer |
| **Pretrained Weights** | Pretrained checkpoints (ImageNet-1K, COCO, ADE20K) | No pretrained weights; deterministic synthetic benchmark patterns |

### 6.4. Methodological Summary
DeltaCore Phase 9 is an **algorithmic and mathematical study** of directional state routing and boundary chunking semantics. It is **NOT** a reproduction of the VisionHOPE visual backbone, does not include VisionHOPE's residual network blocks, and makes zero claims of ImageNet or benchmark equivalence.

---

## 7. Phase 9.2 Empirical Findings & Lessons for Vision-Adapted Neural State

Phase 9.2 conducted controlled causal ablations across Task A (Spatial Shift Reconstruction) and Task B (Shortcut-Resistant Pattern Classification), establishing several key empirical insights directly relevant to VisionHOPE-style spatial state-space modeling:

### 7.1. Spatial Addressability vs. Pure Directional Routing
* **VisionHOPE Mechanism**: In VisionHOPE, 2D image patches are linearly projected and added to standard 2D position embeddings ($E_{\text{pos}} \in \mathbb{R}^{N \times D}$) prior to directional state traversal.
* **DeltaCore Phase 9.1 Limitation**: Phase 9.1 tested whether directional recurrence alone could reconstruct local spatial neighborhood shifts without coordinates, resulting in total failure ($E_{\text{rel}} \approx 1.002$).
* **Phase 9.2 Empirical Finding**: Injecting explicit normalized 2D coordinates $(x, y) \in [-1, 1]$ directly into input channels (`delta_reference_xy`) reduces relative error from $1.0018$ to $0.8660$ ($\approx 14\%$ reduction), whereas scrambling coordinate positions (`delta_trainable_dynamics_shuffled_xy`) collapses error back to $1.0011$. This empirically proves that directional recurrence requires explicit spatial addressability to localize relative neighborhood operations.

### 7.2. Optimization Stability of Unbounded Recurrent State Transitions
* **VisionHOPE Mechanism**: VisionHOPE encapsulates the five-memory recurrence inside deep ViT blocks stabilized by LayerNorm/RMSNorm, residual connections, and multi-head MLP layers.
* **Phase 9.2 Empirical Finding**: When update-generation parameters ($\eta_k, \eta_v, \lambda_k, \lambda_v$) are trained via unclipped BPTT across $T=100$ spatial steps in an isolated recurrent operator at $\eta = 0.010$, recurrent gradient accumulation causes non-finite overflow (`NaN` in 4 of 5 seeds). Restricting $\eta \le 0.003$ restores monotonic convergence ($E_{\text{rel}} \approx 0.9441$). This demonstrates that adaptive neural state transitions require conservative learning rates or normalization when operated outside deep residual ViT backbones.

### 7.3. Single-Direction vs. Multi-Directional Route Utility
* **VisionHOPE Assumption**: VisionHOPE uses quad-directional routing (Right, Left, Down, Up) throughout all layers, assuming multi-directional fusion is uniformly superior.
* **Phase 9.2 Empirical Finding**: On the controlled Task B classification benchmark, single-direction scanning (`delta_reference 1-Dir`, 94 parameters) achieved $42.33\% \pm 8.66\%$ accuracy, whereas 4-direction routing (`delta_reference 4-Dir`, 302 parameters) achieved $35.00\% \pm 12.25\%$. The paired difference ($\Delta = +5.33\% \pm 8.78\%$) spans zero, demonstrating that in lightweight single-layer regimes without deep residual hierarchies, 4-direction routing does not confer an automatic statistical advantage over single-direction scanning.

### 7.4. Genuine Spatial Reliance via Pixel-Shuffle Controls
* Under pixel permutation (B3 control), DeltaCore accuracy drops from $42.33\%$ to $30.67\%$ (falling to the non-spatial GAP ceiling of $30.0\%$), and 4-direction accuracy drops to $18.33\%$ (chance level $16.67\%$). This rigorously validates that DeltaCore's Task B performance is genuinely spatial and does not rely on non-spatial distribution shortcuts.


