"""DeltaCore Phase 9: CPU Profiling and Scaling Benchmark.

Measures CPU runtime across standard 2D spatial feature map scales:
    1. H=7,  W=7   (N=49)
    2. H=14, W=14  (N=196)
    3. H=14, W=20  (N=280)
    4. H=28, W=28  (N=784)

Decomposes execution into:
- Serialization: P_r(Z) across 4 routes
- Directional processing: 4 independent FiveMemory boundary scans
- Restoration: P_r^{-1} across 4 routes
- Fusion: Spatial fusion aggregation
- Total end-to-end runtime

CRITICAL COMMITMENT:
No GPU speedup claims are made. Pure PyTorch CPU reference evaluation.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from deltacore.memory.five_memory import FiveMemoryState
from deltacore.spatial.fusion import EqualFusion
from deltacore.spatial.operator import SpatialAdaptiveOperator
from deltacore.spatial.routes import ALL_ROUTES
from deltacore.updates.five_memory import FiveMemoryConfig


def profile_grid_scale(
    height: int,
    width: int,
    channels: int = 8,
    repeats: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Profile serialization, directional processing, fusion, and total CPU runtime."""
    torch.manual_seed(seed)
    cfg = FiveMemoryConfig(
        v_dim=channels,
        k_dim=channels,
        in_dim=channels,
        feat_dim=channels,
        lr_dim=2,
        ret_dim=2,
        apply_stability_control=True,
    )
    fusion = EqualFusion(mode="mean")
    op = SpatialAdaptiveOperator(config=cfg, fusion=fusion, mode="paper_aligned")

    Z = torch.randn(1, channels, height, width, dtype=torch.float32)

    # Initial states
    states = {
        r.name: FiveMemoryState.initialize(
            channels,
            channels,
            channels,
            batch_size=1,
            generator=torch.Generator().manual_seed(i),
        )
        for i, r in enumerate(ALL_ROUTES)
    }

    # Warmup
    _ = op(Z, initial_states=states)

    t_ser_total = 0.0
    t_fus_total = 0.0
    t_tot_total = 0.0

    for _ in range(repeats):
        # 1. Total end-to-end forward
        t0 = time.perf_counter()
        res = op(Z, initial_states=states)
        t1 = time.perf_counter()
        t_tot_total += t1 - t0

        # 2. Serialization isolated
        ts0 = time.perf_counter()
        for r in ALL_ROUTES:
            _ = r.serialize(Z)
        ts1 = time.perf_counter()
        t_ser_total += ts1 - ts0

        # 3. Fusion isolated
        tf0 = time.perf_counter()
        _ = op.fusion(res.directional_outputs)
        tf1 = time.perf_counter()
        t_fus_total += tf1 - tf0

    avg_tot_ms = (t_tot_total / repeats) * 1000.0
    avg_ser_ms = (t_ser_total / repeats) * 1000.0
    avg_fus_ms = (t_fus_total / repeats) * 1000.0
    avg_dir_ms = max(0.0, avg_tot_ms - avg_ser_ms - avg_fus_ms)

    return {
        "grid": f"{height}x{width}",
        "N": height * width,
        "serialization_ms": avg_ser_ms,
        "directional_proc_ms": avg_dir_ms,
        "fusion_ms": avg_fus_ms,
        "total_ms": avg_tot_ms,
    }


def run_cpu_profile_suite() -> None:
    print("=" * 82)
    print("DeltaCore Phase 9: CPU Runtime Profile Across Grid Scales")
    print("=" * 82)
    print(
        f"{'Grid (HxW)':<12} | {'Tokens (N)':<12} | {'Serial (ms)':<14} | {'Direct (ms)':<14} | {'Fusion (ms)':<12} | {'Total (ms)':<12}"
    )
    print("-" * 82)

    grids = [(7, 7), (14, 14), (14, 20), (28, 28)]
    for h, w in grids:
        metrics = profile_grid_scale(h, w, channels=8)
        print(
            f"{metrics['grid']:<12} | {metrics['N']:<12} | "
            f"{metrics['serialization_ms']:<14.3f} | {metrics['directional_proc_ms']:<14.3f} | "
            f"{metrics['fusion_ms']:<12.3f} | {metrics['total_ms']:<12.3f}"
        )

    print("-" * 82)
    print("Pure PyTorch CPU reference benchmark. No GPU/Triton kernels deployed.")
    print("=" * 82)


if __name__ == "__main__":
    run_cpu_profile_suite()
