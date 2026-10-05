# DeltaCore — Phase 9.2: Spatial Capacity & Optimization Pre-Audit

**Document Status**: COMPLETED & VERIFIED  
**Phase**: 9.2 (Pre-Audit of Phase 9 & Phase 9.1 Benchmarks)  
**Date**: October 2026  
**Related Documents**:
- Phase 9.1 Implementation Report: [docs/development/PHASE_9_1_IMPLEMENTATION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/development/PHASE_9_1_IMPLEMENTATION.md)
- Phase 9.1 Benchmark Audit: [docs/development/PHASE_9_1_BENCHMARK_AUDIT.md](file:///Users/urjasoft/Documents/DeltaCore/docs/development/PHASE_9_1_BENCHMARK_AUDIT.md)
- Phase 9 Interpretation: [docs/math/PHASE_9_INTERPRETATION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/math/PHASE_9_INTERPRETATION.md)

---

## 1. Forensic Audit of Phase 9 & Phase 9.1 Configuration

Before modifying any implementation code for Phase 9.2, we conducted a rigorous inspection of all parameter definitions, model instantiations, execution pipelines, and training protocols across Phase 9 and Phase 9.1.

### 1.1. Resolution of the Baseline B Parameter-Count Discrepancy

In Phase 9.1, the documentation reported:
```text
Baseline B (Static Linear 3x3) | Params: 148 | Task A Rel Error: 0.3485 | Task B Val Accuracy: 30.00%
```

**Forensic Investigation**:
1. **Task A**: Instantiated `StaticLinearSpatialPredictor`:
   ```python
   self.conv = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=True)
   ```
   - Weights: $C_{\text{out}} \times C_{\text{in}} \times K \times K = 4 \times 4 \times 3 \times 3 = 144$
   - Biases: $C_{\text{out}} = 4$
   - **Total Verified Parameters**: $144 + 4 = 148$ parameters.
2. **Task B**: The benchmark script instantiated `Baseline B (Linear)` as:
   ```python
   nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, 6))
   ```
   - Weights: $4 \times 6 = 24$
   - Biases: $6$
   - **Total Verified Parameters**: $24 + 6 = 30$ parameters.

**Confirmed Discrepancy**:
The Phase 9.1 summary table conflated the 148-parameter `Conv2d` from Task A with the 30-parameter `GAP + Linear` from Task B under the single heading "Baseline B (Static Linear 3x3) | Params: 148".
In reality:
- Task A used a 148-parameter 2D spatial convolution.
- Task B used a 30-parameter Global Average Pooling + Linear classifier (`gap_linear`).
- **No spatial `Conv2d` baseline was evaluated on Task B.**
- Furthermore, for Task B, DeltaCore models incorporated both the operator (64 to 272 parameters) **and** a 30-parameter classification head, totaling 94 to 302 parameters, compared to only 30 parameters in the baseline.

In Phase 9.2, these models are strictly disentangled into separate, runtime-verified baselines:
- `conv3x3`: 148 parameters (Task A) and $148 + 30 = 178$ parameters (Task B).
- `gap_linear`: 30 parameters (Task B).

---

## 2. Runtime Parameter Inventory

Every model was instantiated in Python and measured via:
```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### 2.1. Parameter Breakdown Table

| Model Variant | Component | Total Params | Trainable Params | Mathematical Description / Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline A (Zero)** | Zero predictor | 0 | 0 | Unconditional zero output |
| **`gap_linear` (Task B)** | GAP + Linear | 30 | 30 | $W \in \mathbb{R}^{6 \times 4}, b \in \mathbb{R}^6$ |
| **`conv3x3` (Task A)** | Static Conv2d | 148 | 148 | $W \in \mathbb{R}^{4 \times 4 \times 3 \times 3}, b \in \mathbb{R}^4$ |
| **`conv3x3` (Task B)** | Conv2d + GAP + Head | 178 | 178 | Conv2d (148) + Linear (30) |
| **1-Dir `delta_reference` (Task A)** | Initial States (RIGHT) | 64 | 64 | $M_0^c (16), M_0^k (16), M_0^v (16), M_0^\eta (8), M_0^\alpha (8)$ |
| **1-Dir `delta_reference` (Task B)** | Operator + Head | 94 | 94 | Operator (64) + Linear head (30) |
| **2-Dir `delta_reference` (Task A)** | Initial States (R, L) | 128 | 128 | $2 \times 64 = 128$ |
| **2-Dir `delta_reference` (Task B)** | Operator + Head | 158 | 158 | Operator (128) + Linear head (30) |
| **4-Dir `delta_reference` (Task A, Eq)** | Initial States (4 routes) | 256 | 256 | $4 \times 64 = 256$ |
| **4-Dir `delta_reference` (Task A, Ld)** | Initial States + Fusion | 272 | 272 | Initial states (256) + Fusion weights (16) |
| **4-Dir `delta_reference` (Task B, Ld)** | Operator + Head | 302 | 302 | Operator (272) + Linear head (30) |

### 2.2. Parameter Formulas
For input channel dimension $C$, key dimension $K$, value dimension $V$, and meta-controller dimension $D$:
- Content memory initial state: $V \times K = 4 \times 4 = 16$
- Key memory initial state: $K \times C = 4 \times 4 = 16$
- Value memory initial state: $V \times C = 4 \times 4 = 16$
- Learning rate controller initial state: $D_{\text{lr}} \times D_{\text{feat}} = 2 \times 4 = 8$
- Retention controller initial state: $D_{\text{ret}} \times D_{\text{feat}} = 2 \times 4 = 8$
- **Total initial-state parameters per route**:
  $$P_{\text{route}} = VK + KC + VC + D_{\text{lr}}D_{\text{feat}} + D_{\text{ret}}D_{\text{feat}} = 16 + 16 + 16 + 8 + 8 = 64$$
- Learned Channel Fusion parameters for $R$ routes:
  $$P_{\text{fusion}} = R \times C = 4 \times 4 = 16$$
- Task B Linear classification head:
  $$P_{\text{head}} = C \times \text{num\_classes} + \text{num\_classes} = 4 \times 6 + 6 = 30$$

### 2.3. Frozen Quantities in Phase 9.1
In Phase 9.1, all transition and update dynamics parameters were completely frozen:
- Key adaptation rate $\eta_{\text{key}} = 0.01$ (frozen)
- Key retention factor $\lambda_{\text{key}} = 1.0$ (frozen)
- Value adaptation rate $\eta_{\text{val}} = 0.01$ (frozen)
- Value retention factor $\lambda_{\text{val}} = 1.0$ (frozen)
- LR controller update rate $\rho_\eta = 0.05$ (frozen)
- LR controller retention decay $\lambda_\eta = 0.99$ (frozen)
- Retention controller update rate $\rho_{\text{ret}} = 0.05$ (frozen)
- Retention controller retention decay $\lambda_{\text{ret}} = 0.99$ (frozen)
- Maximum learning rate $\eta_{\text{max}} = 0.5$ (frozen)
- Retention lower bound $\text{ret}_{\text{min}} = 0.1$ (frozen)
- Contraction boundary scale $\beta = 1.9$ (frozen)

---

## 3. Dimensionality, Serialization, and Positional Information

- **Input Dimension**: $B \times C \times H \times W = 16 \times 4 \times 10 \times 10$.
- **Sequence Length**: $T = H \times W = 10 \times 10 = 100$ tokens per sequence.
- **Serialization Order**:
  - `RIGHT`: Row-major $(h, w)$ from $(0, 0) \to (0, W-1), \dots, (H-1, W-1)$.
  - `LEFT`: Row-major reversed $(h, W-1-w)$.
  - `DOWN`: Column-major $(w, h)$ from $(0, 0) \to (H-1, 0), \dots, (H-1, W-1)$.
  - `UP`: Column-major reversed $(w, H-1-h)$.
- **Positional Information**:
  **None**. In Phase 9 and 9.1, no positional embeddings, coordinate channels, or spatial indices were provided to any model.
- **BPTT Sequence Length**: Full sequence length $T = 100$. No truncation was applied.
- **Gradient Clipping**: None was applied in Phase 9.1.

---

## 4. Optimization & Dataset Protocols (Phase 9.1 Audit)

- **Optimizer**: `torch.optim.AdamW`.
- **Learning Rate**: $0.010$.
- **Weight Decay**: $1 \times 10^{-4}$.
- **Epoch Budget**: 15 epochs.
- **Batch Size**: 16.
- **Task A Sample Budget**:
  - $N_{\text{train}} = 64$ samples.
  - $N_{\text{val}} = 32$ samples.
- **Task B Sample Budget**:
  - $N_{\text{train}} = 120$ samples (20 samples per class across 6 classes).
  - $N_{\text{val}} = 60$ samples (10 samples per class across 6 classes).
- **Evaluated Seeds**: $[0, 1, 2, 3, 4]$ (5 independent runs per model).

---

## 5. Scope and Plan for Phase 9.2 Ablations

Based on the pre-audit, Phase 9.2 will systematically decouple and test:
1. **Initial States vs Trainable Dynamics**:
   Introduce `delta_trainable_dynamics` where transition update generators $(\eta_k, \lambda_k, \eta_v, \lambda_v, \rho_\eta, \lambda_\eta, \rho_\alpha, \lambda_\alpha, \dots)$ become trainable parameters.
2. **Spatial Coordinate Injection**:
   Introduce normalized $(x, y) \in [-1, 1]^2$ channels to evaluate whether missing spatial address explains Task A failure (`delta_reference_xy` and `delta_trainable_dynamics_xy`).
3. **Control Shuffling**:
   Evaluate randomized coordinate assignment to verify that coordinate utility stems from true spatial address rather than generic extra feature capacity.
4. **Task B Shortcut Controls**:
   Subject Task B to:
   - Translation stress (random shifts preserving class)
   - Coordinate-statistic matching
   - Pixel-shuffle control (scrambling 2D positions per sample)
   - Independent test templates
5. **Rigorous Capacity-Matched Baselines**:
   Compare against `gap_linear` (30 params), `conv3x3` (178 params), and parameter-matched MLP `mlp_matched` (~274 params).
