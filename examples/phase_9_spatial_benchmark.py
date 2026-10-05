"""DeltaCore Phase 9: 2D Spatial Routing & Directional Independence Benchmark.

Evaluates four-direction spatial traversal (RIGHT, LEFT, DOWN, UP) and fusion
over standardized synthetic 2D patterns:
1. Horizontal stripe
2. Vertical stripe
3. Diagonal pattern
4. Localized square
5. Repeated checkerboard
6. Asymmetric pattern

Measures and reports:
- Fused reconstruction error ||Y - Z||_F
- Per-direction reconstruction error ||Y_r - Z||_F
- Per-direction state activity and memory norm
- Directional learning-rate statistics
- Chunk-size sensitivity (C = 1, 2, W/H, full)

SCIENTIFIC PRINCIPLE:
Directional state differences reflect route-aligned gradient accumulation
along 1D serialized sequence manifolds. They do NOT imply semantic understanding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from deltacore.observatory.plots import (
    plot_chunk_size_difference,
    plot_directional_learning_rate_map,
    plot_directional_memory_activity_map,
    plot_directional_output_magnitude,
    plot_fused_spatial_output,
)
from deltacore.spatial.fusion import EqualFusion
from deltacore.spatial.operator import SpatialAdaptiveOperator
from deltacore.updates.five_memory import FiveMemoryConfig


def generate_synthetic_patterns(
    height: int = 14, width: int = 14, channels: int = 8, seed: int = 42
) -> dict[str, torch.Tensor]:
    """Generate standardized 2D synthetic spatial patterns of shape [1, C, H, W]."""
    torch.manual_seed(seed)
    patterns = {}

    # 1. Horizontal stripe: variation primarily across rows (height)
    h_idx = torch.arange(height, dtype=torch.float64).view(1, 1, height, 1)
    h_stripe = torch.sin(2.0 * 3.1415926535 * h_idx / height).expand(
        1, channels, height, width
    )
    patterns["horizontal_stripe"] = h_stripe

    # 2. Vertical stripe: variation primarily across columns (width)
    w_idx = torch.arange(width, dtype=torch.float64).view(1, 1, 1, width)
    v_stripe = torch.sin(2.0 * 3.1415926535 * w_idx / width).expand(
        1, channels, height, width
    )
    patterns["vertical_stripe"] = v_stripe

    # 3. Diagonal pattern: coupled variation across both axes
    diag = (
        torch.sin(
            2.0
            * 3.1415926535
            * (
                torch.arange(height, dtype=torch.float64).view(1, 1, height, 1)
                + torch.arange(width, dtype=torch.float64).view(1, 1, 1, width)
            )
            / max(height, width)
        )
    ).expand(1, channels, height, width)
    patterns["diagonal_pattern"] = diag

    # 4. Localized square: centered activation box
    sq = torch.zeros(1, channels, height, width, dtype=torch.float64)
    h_mid, w_mid = height // 2, width // 2
    sq[:, :, h_mid - 2 : h_mid + 2, w_mid - 2 : w_mid + 2] = 2.0
    patterns["localized_square"] = sq

    # 5. Repeated checkerboard: high-frequency spatial alternation
    grid_h = torch.arange(height).view(-1, 1)
    grid_w = torch.arange(width).view(1, -1)
    cb = (
        ((grid_h + grid_w) % 2)
        .to(torch.float64)
        .view(1, 1, height, width)
        .expand(1, channels, height, width)
    )
    patterns["repeated_checkerboard"] = cb

    # 6. Asymmetric pattern: quadrant step function with localized gradient
    asym = torch.zeros(1, channels, height, width, dtype=torch.float64)
    asym[:, :, :h_mid, :w_mid] = 1.0
    asym[:, :, h_mid:, w_mid:] = -1.0
    patterns["asymmetric_pattern"] = asym

    return patterns


def run_spatial_benchmark() -> None:
    print("=" * 86)
    print("DeltaCore Phase 9: 2D Spatial Routing & Directional Benchmark")
    print("=" * 86)

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
        eta_key=0.02,
        eta_val=0.02,
        rho_eta=0.02,
        rho_ret=0.02,
        apply_stability_control=True,
    )

    op_generic = SpatialAdaptiveOperator(
        config=cfg,
        fusion=EqualFusion(mode="mean"),
        mode="generic",
        default_chunk_size=1,
    )
    op_paper = SpatialAdaptiveOperator(
        config=cfg,
        fusion=EqualFusion(mode="mean"),
        mode="paper_aligned",
    )

    patterns = generate_synthetic_patterns(height=H, width=W, channels=C)
    artifacts_dir = Path("docs/benchmarks/artifacts/phase_9")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"{'Pattern':<24} | {'Fused Err':<10} | {'Err RIGHT':<10} | {'Err LEFT':<10} | {'Err DOWN':<10} | {'Err UP':<10}"
    )
    print("-" * 86)

    last_res = None

    for name, Z in patterns.items():
        res = op_generic(Z, record_trajectories=False)
        last_res = res

        fused_err = torch.norm(res.fused_output - Z).item()
        err_r = torch.norm(res.directional_outputs["RIGHT"] - Z).item()
        err_l = torch.norm(res.directional_outputs["LEFT"] - Z).item()
        err_d = torch.norm(res.directional_outputs["DOWN"] - Z).item()
        err_u = torch.norm(res.directional_outputs["UP"] - Z).item()

        print(
            f"{name:<24} | {fused_err:<10.4f} | {err_r:<10.4f} | {err_l:<10.4f} | {err_d:<10.4f} | {err_u:<10.4f}"
        )

    print("-" * 86)
    print("\nPaper-Aligned Mode Evaluation (RIGHT/LEFT chunk = W, DOWN/UP chunk = H):")
    print(
        f"{'Pattern':<24} | {'Paper Fused Err':<16} | {'Generic (C=1) Fused Err':<24} | {'Paper Alignment Diff':<20}"
    )
    print("-" * 86)

    chunk_diffs: dict[Any, float] = {}

    for name, Z in patterns.items():
        res_paper = op_paper(Z)
        res_gen = op_generic(Z)
        p_err = torch.norm(res_paper.fused_output - Z).item()
        g_err = torch.norm(res_gen.fused_output - Z).item()
        diff = torch.norm(res_paper.fused_output - res_gen.fused_output).item()
        chunk_diffs[name[:8]] = diff
        print(f"{name:<24} | {p_err:<16.4f} | {g_err:<24.4f} | {diff:<20.6f}")

    print("-" * 86)
    print("Generating Phase 9 Observatory Spatial Diagnostic Plots...")

    if last_res is not None:
        plot_directional_output_magnitude(
            last_res.directional_outputs,
            artifacts_dir / "plot_j_directional_magnitude.png",
        )
        plot_fused_spatial_output(
            last_res.fused_output,
            artifacts_dir / "plot_m_fused_spatial_output.png",
        )

        # Activity maps based on directional outputs
        plot_directional_memory_activity_map(
            last_res.directional_outputs,
            artifacts_dir / "plot_l_memory_activity.png",
        )
        plot_directional_learning_rate_map(
            last_res.directional_outputs,
            artifacts_dir / "plot_k_learning_rate_map.png",
        )

    # Plot N
    plot_chunk_size_difference(
        chunk_diffs,
        artifacts_dir / "plot_n_chunk_size_difference.png",
    )

    print(f"Observatory plots saved to: {artifacts_dir}")
    print("=" * 86)


if __name__ == "__main__":
    run_spatial_benchmark()
