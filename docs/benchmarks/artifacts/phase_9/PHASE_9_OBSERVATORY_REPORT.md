# DeltaCore State Observatory Report: Phase 9 Spatial Routing

## 1. Experiment Metadata & Provenance

| Property | Value |
| :--- | :--- |
| **Operator Model** | `SpatialAdaptiveOperator` |
| **Benchmark Task** | 2D Spatial Diagonal Pattern |
| **Grid Dimensions** | $H = 14, W = 14$ ($N = 196$ tokens) |
| **Channels $C$** | 8 |
| **Chunking Mode** | `paper_aligned` ($C_\rightarrow = W, C_\leftarrow = W, C_\downarrow = H, C_\uparrow = H$) |
| **Spatial Fusion** | `EqualFusion(mode='mean')` |
| **Python Version** | `3.13.0` |
| **PyTorch Version** | `2.6.0` |
| **Compute Device** | CPU |
| **Precision** | Float64 |
| **Fused Reconstruction Error** | 27.9984 |

## 2. Per-Direction Trajectory Summaries

| Direction | Traversal Alignment | Refreshes | Mean Error $\bar{E}$ | Final Directional Error | Final Content Memory Norm |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RIGHT** | Row-major | 14 | 0.0487 | 27.9993 | 0.0091 |
| **LEFT** | Row-major | 14 | 0.0614 | 27.9979 | 0.0088 |
| **DOWN** | Column-major | 14 | 0.0610 | 27.9981 | 0.0086 |
| **UP** | Column-major | 14 | 0.0636 | 27.9982 | 0.0075 |

## 3. Directional Independence & Non-Contamination

- Each direction maintains a strictly independent FiveMemory state instance.
- State transition coupling across directions: $\frac{\partial S_t^r}{\partial S_{t'}^{r'}} = 0, \forall r \neq r'$.
- All four trajectories evolve in isolated state spaces and are combined exclusively via spatial restoration and fusion.

## 4. Associated Diagnostic Visualizations

The following publication-quality diagnostic plots were generated and saved:
- **Plot J: Directional Output Magnitude** (`plot_j_directional_magnitude.png`)
- **Plot K: Directional Learning-Rate Map** (`plot_k_learning_rate_map.png`)
- **Plot L: Directional Memory-Activity Map** (`plot_l_memory_activity.png`)
- **Plot M: Fused Spatial Output** (`plot_m_fused_spatial_output.png`)
- **Plot N: Chunk Size Sensitivity** (`plot_n_chunk_size_difference.png`)

## 5. Epistemic Delimitation

> **Observation Note**: Directional error and state activity differences reflect
> alignment between 1D serialization trajectories and spatial pattern gradients.
> They do **NOT** demonstrate semantic understanding or high-level visual concept specialization.
