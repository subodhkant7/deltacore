#!/usr/bin/env python3
"""DeltaCore Phase 16: Unseen Shift Robustness & Adaptive Safety Benchmark Entrypoint.

Executes the complete Phase 16 robustness benchmark pipeline challenging the pooled
SafeAdaptiveDelta configuration with unseen distribution shifts (Mean, Variance,
Temporal Speed, Noise, Combined) across 3 predetermined severities (mild, moderate, severe)
on both NOAA OISST SST (Domain A) and ECMWF ERA5 T2m (Domain B), evaluating continuous
vs reset state, retention stress histories, empirical failure boundaries at D=256,
and generates all 12 Observatory publication figures CQ through DB.

Usage:
    python3 examples/phase_16_robustness_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltacore.observatory.phase_16_plots import generate_all_phase_16_plots
from deltacore.streaming.phase_16_benchmark import run_phase_16_benchmark


def main() -> None:
    print("=" * 88)
    print("DELTACORE PHASE 16: UNSEEN SHIFT ROBUSTNESS & ADAPTIVE SAFETY BENCHMARK")
    print("Evaluating Pooled SafeAdaptiveDelta (eta0=0.015, rho=1.5, alpha_min=0.95)")
    print(
        "Under Unseen Distribution Shifts: Mean, Variance, Temporal Speed, Noise, Combined"
    )
    print("=" * 88)

    _ = run_phase_16_benchmark(seeds=(42, 43, 44, 45, 46))

    print("\n[Observatory] Generating all 12 Phase 16 Publication Figures (CQ - DB)...")
    plots = generate_all_phase_16_plots()
    print(
        f"Generated {len(plots)} publication figures in docs/benchmarks/artifacts/phase_16/plots/"
    )

    print("\n" + "=" * 88)
    print("PHASE 16 BENCHMARK & OBSERVATORY SUITE COMPLETED SUCCESSFULLY.")
    print("=" * 88)


if __name__ == "__main__":
    main()
