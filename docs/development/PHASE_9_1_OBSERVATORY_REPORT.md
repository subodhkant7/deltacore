# DeltaCore State Observatory Report: Phase 9.1 Spatial Learning Validation

See primary artifact in [docs/benchmarks/artifacts/phase_9_1/PHASE_9_1_OBSERVATORY_REPORT.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_9_1/PHASE_9_1_OBSERVATORY_REPORT.md).

## Summary Table

| Model / Architecture | Direction | Chunk Size $C$ | Parameter Count | Task A: Rel Error $E_{\text{rel}}$ | Task B: Val Accuracy (%) | Task A Runtime (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline A (Zero Predictor)** | None | N/A | 0 | $1.0000 \pm 0.0000$ | 16.67% (chance) | 0.0 |
| **Baseline B (Static Linear 3x3)** | None | N/A | 148 | $\mathbf{0.3485 \pm 0.0133}$ | $30.00\% \pm 6.67\%$ | 335.1 |
| **Non-Spatial (1D Sequence)** | 1D raster | 100 | 64 | $1.0002 \pm 0.0002$ | N/A | 2669.0 |
| **1-Dir (RIGHT)** | RIGHT | 1 | 64 | $1.0002 \pm 0.0002$ | $\mathbf{43.33\% \pm 8.16\%}$ | 2377.6 |
| **1-Dir (LEFT)** | LEFT | 1 | 64 | $1.0005 \pm 0.0001$ | N/A | 2369.9 |
| **1-Dir (DOWN)** | DOWN | 1 | 64 | $1.0009 \pm 0.0002$ | N/A | 2388.8 |
| **1-Dir (UP)** | UP | 1 | 64 | $1.0008 \pm 0.0002$ | N/A | 2360.5 |
| **2-Dir (Horizontal R+L)** | RIGHT, LEFT | 1 | 128 | $1.0007 \pm 0.0002$ | $36.67\% \pm 6.67\%$ | 4861.1 |
| **2-Dir (Vertical D+U)** | DOWN, UP | 1 | 128 | $1.0015 \pm 0.0004$ | $39.67\% \pm 6.70\%$ | 4847.5 |
| **4-Dir (All Equal Fusion)** | R, L, D, U | 1 | 256 | $1.0022 \pm 0.0006$ | N/A | 9878.2 |
| **4-Dir (All Learned Fusion)** | R, L, D, U | 1 | 272 | $1.0022 \pm 0.0002$ | $36.67\% \pm 6.67\%$ | 9927.9 |

## Core Diagnostic Plots
- **Plot O**: `plot_o_train_val_loss.png`
- **Plot P**: `plot_p_accuracy_vs_directions.png`
- **Plot Q**: `plot_q_val_error_vs_chunk_size.png`
- **Plot R**: `plot_r_runtime_vs_chunk_size.png`
- **Plot S**: `plot_s_route_comparison.png`
- **Plot T**: `plot_t_absolute_vs_relative_error.png`
