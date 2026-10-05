"""DeltaCore Phase 9: Spatial Observatory Report Generator.

Runs a spatial routing experiment using SpatialAdaptiveOperator,
records per-direction state trajectories, and generates a formal
Markdown reproducibility report in docs/benchmarks/artifacts/phase_9/PHASE_9_OBSERVATORY_REPORT.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from deltacore.observatory.schema import StateTrajectory, TrajectoryStep
from deltacore.spatial.fusion import EqualFusion
from deltacore.spatial.operator import SpatialAdaptiveOperator
from deltacore.spatial.routes import ALL_ROUTES
from deltacore.updates.five_memory import FiveMemoryConfig


def generate_spatial_observatory_report() -> Path:
    seed = 42
    torch.manual_seed(seed)
    H, W, C = 14, 14, 8

    cfg = FiveMemoryConfig(
        v_dim=C,
        k_dim=C,
        in_dim=C,
        feat_dim=C,
        lr_dim=2,
        ret_dim=2,
        eta_max=0.5,
        ret_min=0.1,
        apply_stability_control=True,
    )
    fusion = EqualFusion(mode="mean")
    op = SpatialAdaptiveOperator(config=cfg, fusion=fusion, mode="paper_aligned")

    # Diagonal pattern
    diag = (
        torch.sin(
            2.0
            * 3.1415926535
            * (
                torch.arange(H, dtype=torch.float64).view(1, 1, H, 1)
                + torch.arange(W, dtype=torch.float64).view(1, 1, 1, W)
            )
            / max(H, W)
        )
    ).expand(1, C, H, W)

    res = op(diag, record_trajectories=True)
    fused_err = torch.norm(res.fused_output - diag).item()

    # Build per-direction trajectories
    directional_summaries = {}
    for r in ALL_ROUTES:
        d_res = res.directional_results[r.name]
        scan_res = d_res.scan_result
        t_steps = scan_res.predictions.shape[1]

        steps = []
        for t in range(t_steps):
            e_norm = torch.norm(scan_res.errors[0, t, :]).item()
            safe_eta = scan_res.safe_learning_rates[0, t, 0].item()
            safe_ret = scan_res.safe_retentions[0, t, 0].item()
            mem_norm = scan_res.content_memory_norms[0, t].item()
            is_boundary = t in scan_res.chunk_boundaries

            step_obj = TrajectoryStep(
                step=t,
                observed=True,
                error_norm=e_norm,
                step_size=safe_eta,
                memory_norm=mem_norm,
                safe_learning_rate=safe_eta,
                safe_retention=safe_ret,
                route=r.name,
                route_direction=r.name,
                chunk_boundary=is_boundary,
                directional_memory_norm=mem_norm,
                directional_learning_rate=safe_eta,
                directional_retention=safe_ret,
            )
            steps.append(step_obj)

        traj = StateTrajectory(
            model=f"SpatialAdaptiveOperator_{r.name}",
            task="2D_spatial_diagonal_pattern",
            seed=seed,
            steps=steps,
            metadata={"device": "cpu", "dtype": "float64", "route": r.name},
        )
        directional_summaries[r.name] = {
            "trajectory": traj,
            "final_err": torch.norm(d_res.restored_predictions - diag).item(),
            "mean_err": sum(s.error_norm for s in steps) / len(steps),
            "final_mem_norm": torch.norm(d_res.final_state.content).item(),
            "num_refreshes": len(scan_res.boundary_states),
        }

    # Generate Markdown report
    out_dir = Path("docs/benchmarks/artifacts/phase_9")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "PHASE_9_OBSERVATORY_REPORT.md"

    lines = [
        "# DeltaCore State Observatory Report: Phase 9 Spatial Routing",
        "",
        "## 1. Experiment Metadata & Provenance",
        "",
        "| Property | Value |",
        "| :--- | :--- |",
        "| **Operator Model** | `SpatialAdaptiveOperator` |",
        "| **Benchmark Task** | 2D Spatial Diagonal Pattern |",
        "| **Grid Dimensions** | $H = 14, W = 14$ ($N = 196$ tokens) |",
        "| **Channels $C$** | 8 |",
        "| **Chunking Mode** | `paper_aligned` ($C_\\rightarrow = W, C_\\leftarrow = W, C_\\downarrow = H, C_\\uparrow = H$) |",
        "| **Spatial Fusion** | `EqualFusion(mode='mean')` |",
        "| **Python Version** | `" + sys.version.split()[0] + "` |",
        "| **PyTorch Version** | `" + torch.__version__ + "` |",
        "| **Compute Device** | CPU |",
        "| **Precision** | Float64 |",
        "| **Fused Reconstruction Error** | " + f"{fused_err:.4f}" + " |",
        "",
        "## 2. Per-Direction Trajectory Summaries",
        "",
        "| Direction | Traversal Alignment | Refreshes | Mean Error $\\bar{E}$ | Final Directional Error | Final Content Memory Norm |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r_name, summary in directional_summaries.items():
        lines.append(
            f"| **{r_name}** | {'Row-major' if r_name in ('RIGHT', 'LEFT') else 'Column-major'} | "
            f"{summary['num_refreshes']} | {summary['mean_err']:.4f} | {summary['final_err']:.4f} | {summary['final_mem_norm']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Directional Independence & Non-Contamination",
            "",
            "- Each direction maintains a strictly independent FiveMemory state instance.",
            "- State transition coupling across directions: $\\frac{\\partial S_t^r}{\\partial S_{t'}^{r'}} = 0, \\forall r \\neq r'$.",
            "- All four trajectories evolve in isolated state spaces and are combined exclusively via spatial restoration and fusion.",
            "",
            "## 4. Associated Diagnostic Visualizations",
            "",
            "The following publication-quality diagnostic plots were generated and saved:",
            "- **Plot J: Directional Output Magnitude** (`plot_j_directional_magnitude.png`)",
            "- **Plot K: Directional Learning-Rate Map** (`plot_k_learning_rate_map.png`)",
            "- **Plot L: Directional Memory-Activity Map** (`plot_l_memory_activity.png`)",
            "- **Plot M: Fused Spatial Output** (`plot_m_fused_spatial_output.png`)",
            "- **Plot N: Chunk Size Sensitivity** (`plot_n_chunk_size_difference.png`)",
            "",
            "## 5. Epistemic Delimitation",
            "",
            "> **Observation Note**: Directional error and state activity differences reflect",
            "> alignment between 1D serialization trajectories and spatial pattern gradients.",
            "> They do **NOT** demonstrate semantic understanding or high-level visual concept specialization.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Observatory report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_spatial_observatory_report()
