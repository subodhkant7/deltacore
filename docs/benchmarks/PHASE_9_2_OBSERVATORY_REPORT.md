# DeltaCore — Phase 9.2: Observatory Report

**Document Status**: COMPLETED  
**Phase**: 9.2 (Spatial Capacity & Optimization Ablation)  
**Date**: October 2026  
**Related Documents**:
- Implementation Report: [PHASE_9_2_IMPLEMENTATION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_9_2_IMPLEMENTATION.md)
- Interpretation & Hypotheses: [PHASE_9_2_INTERPRETATION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_9_2_INTERPRETATION.md)
- Pre-Audit: [PHASE_9_2_PRE_AUDIT.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_9_2_PRE_AUDIT.md)
- Full Results JSON: [phase_9_2_full_results.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_2/phase_9_2_full_results.json)

---

## 1. Observatory Configuration

All diagnostic plots were generated from controlled ablation runs with the following shared configuration:

| Parameter | Value |
| :--- | :--- |
| **Seeds** | `[0, 1, 2, 3, 4]` |
| **Task A Dimensions** | $B=16, H=10, W=10, C=4$ |
| **Task B Dimensions** | $B=16, H=8, W=8, C=4$ |
| **Task B Classes** | 6 (horizontal, vertical, diagonal, checkerboard, localized, asymmetric) |
| **Optimizer** | AdamW (`weight_decay=1e-4`) |
| **Default Learning Rate** | $\eta = 0.010$ |
| **Epochs** | 15 |
| **Translation Stress (B1)** | Max shift $\pm 2$ pixels with circular wrap |
| **Coordinate-Statistic Control (B2)** | Energy normalized ($\|X_i\|_F=1.0$) & center of mass centered |
| **Pixel Shuffle (B3)** | Independent per-sample spatial permutation |

Generated plots reside in `docs/benchmarks/artifacts/phase_9_2/`.

---

## 2. Plot Descriptions & Diagnostic Observations

### Plot U — Task A Validation Relative Error by Model

**File**: [plot_u_task_a_rel_error.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_2/plot_u_task_a_rel_error.png)

**Description**: Bar chart comparing final validation relative error $E_{\text{rel}} = \frac{\|Y - \hat{Y}\|_F}{\max(\|Y\|_F, \epsilon)}$ across all Task A ablation models.

**Key Observations** (OBSERVED):
- `conv3x3` is the only model achieving substantial error reduction ($E_{\text{rel}} = 0.349 \pm 0.013$), confirming that a local 2D receptive field is sufficient for the spatial shift task.
- `delta_reference` and `mlp_matched` both remain at $E_{\text{rel}} \approx 1.001$, indistinguishable from the zero-predictor baseline.
- `delta_reference_xy` achieves $E_{\text{rel}} = 0.866 \pm 0.023$, the strongest DeltaCore result, via coordinate injection.
- `delta_trainable_dynamics` produces NaN entries at $\eta = 0.01$ due to gradient explosion.

---

### Plot V — Task A Train/Validation Curves for Principal DeltaCore Variants

**File**: [plot_v_task_a_train_val_curves.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_2/plot_v_task_a_train_val_curves.png)

**Description**: Training and validation loss trajectories over 15 epochs for `delta_reference`, `delta_reference_xy`, and `delta_trainable_dynamics` (lr=0.003).

**Key Observations** (OBSERVED):
- `delta_reference` train and validation losses remain essentially flat over all 15 epochs, consistent with the operator behaving as a near-zero predictor whose only trainable parameters (initial states) receive insufficient gradient signal through 100-step BPTT.
- `delta_reference_xy` shows steady, monotonic descent in both train and validation loss, with the gap between curves remaining small (minimal overfitting at this data scale).
- `delta_trainable_dynamics` at $\eta = 0.003$ shows descent, but convergence is slower and the final plateau is higher than `delta_reference_xy`.

---

### Plot W — Task A Effect of Normalized (x,y) Coordinates

**File**: [plot_w_coordinate_effect.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_2/plot_w_coordinate_effect.png)

**Description**: Paired comparison of Task A relative error with and without normalized 2D coordinate injection across all 5 seeds.

**Key Observations** (OBSERVED):
- Every seed shows improvement when coordinates are injected.
- Mean improvement: $1.0018 \to 0.8660$ ($-13.6\%$ absolute error reduction).
- The shuffled-coordinate control (`delta_trainable_dynamics_shuffled_xy`) collapses back to $\approx 1.001$, confirming the improvement requires correct spatial correspondence, not merely extra input features.

---

### Plot X — Task B Accuracy by Model and Parameter Count

**File**: [plot_x_task_b_acc_vs_params.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_2/plot_x_task_b_acc_vs_params.png)

**Description**: Scatter plot of Task B standard classification accuracy vs. trainable parameter count for all evaluated models.

**Key Observations** (OBSERVED):
- `gap_linear` (30p) achieves $30.0\%$, setting the non-spatial floor.
- `delta_reference 1-Dir` (94p) achieves the highest mean accuracy ($42.3\%$) with the smallest DeltaCore parameter count.
- `delta_reference 4-Dir` (302p) achieves lower mean accuracy ($35.0\%$) than 1-Dir despite $3.2\times$ more parameters.
- `delta_trainable_dynamics 4-Dir` (414p) and `delta_trainable_dynamics_xy` (438p) both collapse to chance ($16.67\%$), consistent with optimization failure at $\eta = 0.01$.

---

### Plot Y — Task B Accuracy Before/After Translation Stress

**File**: [plot_y_translation_stress.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_2/plot_y_translation_stress.png)

**Description**: Grouped bar chart comparing original accuracy and post-translation accuracy for all Task B models.

**Key Observations** (OBSERVED):
- `delta_reference 1-Dir`: $42.3\% \to 40.3\%$ (retains ~95% of accuracy; robust to spatial translations).
- `conv3x3`: $40.0\% \to 41.7\%$ (slightly improves, consistent with 3×3 kernel translation equivariance).
- `gap_linear`: $30.0\% \to 30.0\%$ (invariant by construction but weak).
- `delta_reference 4-Dir`: $35.0\% \to 27.7\%$ (more sensitive; multi-route fusion may amplify translation perturbation).

---

### Plot Z — Task B Original vs. Pixel-Shuffled Accuracy

**File**: [plot_z_pixel_shuffle_control.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_2/plot_z_pixel_shuffle_control.png)

**Description**: Grouped bar chart comparing original vs. pixel-shuffled Task B accuracy. This is the critical shortcut diagnostic.

**Key Observations** (OBSERVED):
- `gap_linear`: $30.0\% \to 30.0\%$ (zero sensitivity to spatial structure; completely non-spatial).
- `conv3x3`: $40.0\% \to 23.0\%$ (collapses toward chance $16.67\%$; relies entirely on local 2D adjacency).
- `delta_reference 1-Dir`: $42.3\% \to 30.7\%$ ($-11.7\%$ drop, falling to the GAP ceiling; indicates genuine spatial reliance in the tested configuration).
- `delta_reference 4-Dir`: $35.0\% \to 18.3\%$ (collapses to near-chance; 4-Dir is more spatially dependent).
- `mlp_matched`: $36.7\% \to 36.7\%$ (zero sensitivity; consistent with flattened non-spatial processing).

**Interpretation (SUPPORTED)**: DeltaCore's Task B advantage over GAP is not a dataset shortcut. It requires intact spatial arrangement, as demonstrated by the systematic accuracy collapse under pixel permutation.

### Coordinate-Statistic Control (B2 Evaluation)

To ensure pattern classification is not driven by global energy asymmetries or center-of-mass offsets, all samples were processed via `coordinate_statistic_control`, strictly enforcing unit Frobenius norm ($\|X_i\|_F = 1.0000$) and aligning the empirical center of mass $(\text{CoM}_y, \text{CoM}_x)$ to the grid center $((H-1)/2, (W-1)/2)$:
- `conv3x3`: $40.0\% \to 40.0\%$ (invariant)
- `gap_linear`: $30.0\% \to 30.0\%$ (invariant)
- `mlp_matched`: $36.7\% \to 36.7\%$ (invariant)
- `delta_reference 1-Dir`: $42.33\% \to 42.33\%$ (**invariant**)
- `delta_reference 4-Dir`: $35.00\% \to 35.00\%$ (**invariant**)

**Key Diagnostic Finding (SUPPORTED)**:
DeltaCore does not rely on center-of-mass shortcuts or global coordinate statistics. Its classification performance survives coordinate-statistic centering while collapsing specifically under spatial pixel shuffling, isolating true spatial sequence tracking as the operational mechanism.

---

### Plot AA — Directional Accuracy Comparison with Per-Seed Points

**File**: [plot_aa_directional_accuracy_points.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_2/plot_aa_directional_accuracy_points.png)

**Description**: Strip/swarm plot showing per-seed Task B accuracy for each directional route configuration: RIGHT, LEFT, DOWN, UP, Horizontal (R+L), Vertical (D+U), and All 4 Directions.

**Key Observations** (OBSERVED):
- Single-direction routes show high variance across seeds (ranging from $33.3\%$ to $50.0\%$).
- LEFT achieves the highest single-direction mean ($40.0\%$), while DOWN achieves the lowest ($33.3\%$).
- 4-direction fusion does not consistently outperform the best single direction.
- Paired difference $\Delta_{\text{4dir} - \text{1dir\_RIGHT}} = +5.33\% \pm 8.78\%$: the confidence interval includes zero.

---

### Plot AB — Trainable-State Norms and Gradient Norms Over Training

**File**: [plot_ab_state_gradient_norms.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_2/plot_ab_state_gradient_norms.png)

**Description**: Multi-panel time series plot showing per-epoch gradient norm, update magnitude, content-state norm, key-state norm, value-state norm, and dynamics-state norm for the principal DeltaCore variants.

**Key Observations** (OBSERVED):
- `delta_reference`: Gradient norms remain small and stable but produce near-zero updates, consistent with vanishing useful gradients through long BPTT chains.
- `delta_reference_xy`: Gradient norms are larger and sustain meaningful state updates throughout training, with monotonically growing content/value state norms.
- `delta_trainable_dynamics` ($\eta = 0.01$): Shows explosive gradient norm growth in seeds 0, 1, 3, 4 leading to NaN, while seed 2 narrowly avoids divergence.
- `delta_trainable_dynamics` ($\eta = 0.003$): Stable gradient norms with moderate update magnitudes; converges without explosion.

**Diagnostic Finding (OBSERVED)**: The trainable dynamics recurrence at $\eta = 0.01$ exhibits characteristic exponential gradient blow-up during BPTT over $T = 100$ steps. The unconstrained update parameters ($\eta_k, \lambda_k$ etc.) amplify gradients multiplicatively across steps.

---

### Plot AC — Task Performance vs. Parameter Count

**File**: [plot_ac_performance_vs_params.png](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_2/plot_ac_performance_vs_params.png)

**Description**: Dual-panel scatter plot showing (Left) Task A relative error vs. parameter count and (Right) Task B accuracy vs. parameter count for all models.

**Key Observations** (OBSERVED):
- Task A: `conv3x3` (148p) vastly outperforms all models with 2–3× more parameters. The bottleneck is architectural (2D receptive field), not parameter count.
- Task B: `delta_reference 1-Dir` (94p) outperforms `mlp_matched` (304p) and `gap_linear` (30p), suggesting an architecture effect, not merely a parameter-count effect.
- Increasing DeltaCore parameters via trainable dynamics or multi-direction fusion does not consistently improve task performance, ruling out simple underfitting.

---

## 3. Capacity Matching Analysis

### Approximate Capacity Groups

| Group | Model | Trainable Params | Task A $E_{\text{rel}}$ | Task B Acc (%) |
| :---: | :--- | :---: | :---: | :---: |
| **Small (~100p)** | `delta_reference 1-Dir` | 94 | — | **42.3** |
| **Small (~100p)** | `delta_trainable_dynamics 1-Dir` | 122 | — | 32.7 |
| **Medium (~150p)** | `conv3x3` (Task A) | 148 | **0.349** | — |
| **Medium (~300p)** | `delta_reference 4-Dir` | 302 | 1.002 | 35.0 |
| **Medium (~300p)** | `mlp_matched` | 274/304 | 1.001 | 36.7 |
| **Large (~400p)** | `delta_trainable_dynamics 4-Dir` | 414 | NaN | 16.7 |

**Parameter-Count Ratios**:
- `delta_reference 1-Dir` / `gap_linear`: $94/30 = 3.13\times$
- `delta_reference 4-Dir` / `mlp_matched`: $302/304 = 0.99\times$ (near-exact match)
- `conv3x3` (Task A) / `delta_reference` (Task A): $148/272 = 0.54\times$ (conv has fewer params yet vastly outperforms)

**Finding (SUPPORTED)**: At matched parameter budgets (~300p), DeltaCore 4-Dir ($35.0\%$) and MLP ($36.7\%$) perform similarly on Task B, suggesting the multi-direction advantage over GAP ($30.0\%$) may partly reflect additional capacity. However, DeltaCore 1-Dir achieves higher accuracy ($42.3\%$) with fewer parameters ($94$), which is not explainable by parameter count alone.

---

## 4. Chunking Re-evaluation on Strongest Learner

| Chunk Size $C$ | $E_{\text{rel}}$ | $\Delta Y_C = \|Y_C - Y_1\|_F$ | Runtime (ms) |
| :---: | :---: | :---: | :---: |
| $C = 1$ | 0.8766 | 0.0000 | 11526 |
| $C = 4$ | 0.8761 | 0.0322 | 10753 |
| $C = 8$ | 0.8788 | 0.0293 | 10118 |
| $C = \text{full}$ | 0.8485 | 0.0389 | 7897 |

**Finding (OBSERVED)**: Task quality is essentially invariant to chunk size ($E_{\text{rel}} \in [0.849, 0.879]$). Full-sequence chunking provides ~31% faster runtime without degrading reconstruction fidelity. This suggests that boundary staleness at these sequence lengths does not materially affect task performance when the model is actually learning.

---

## 5. Optimization Ablation Summary

### Learning Rate Sweep for `delta_trainable_dynamics`

| $\eta$ | Mean $E_{\text{rel}}$ | Std | Converged Seeds | Status |
| :---: | :---: | :---: | :---: | :--- |
| 0.001 | 0.9689 | 0.0202 | 5/5 | Stable, slow |
| **0.003** | **0.9441** | **0.0144** | **5/5** | **Optimal stable regime** |
| 0.010 | NaN | NaN | 1/5 | Gradient explosion |

**Finding (OBSERVED)**: Trainable dynamics are highly sensitive to learning rate. The viable operating regime is narrow ($\eta \in [0.001, 0.003]$). At the optimal $\eta = 0.003$, trainable dynamics achieve $E_{\text{rel}} = 0.944$, which is inferior to coordinate injection alone ($E_{\text{rel}} = 0.866$).

---

## 6. Reproduction Command

All plots, JSON artifacts, and diagnostic data can be reproduced via:

```bash
python3 examples/phase_9_2_capacity_ablation_benchmark.py
```

All generated artifacts are written to `docs/benchmarks/artifacts/phase_9_2/`.
