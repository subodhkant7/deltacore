#!/usr/bin/env python3
"""DeltaCore Phase 15: Adaptive Regime Transfer & Robustness Benchmark Entrypoint.

Executes the complete Phase 15 cross-domain transfer benchmark pipeline between Domain A
(NOAA OISST Equatorial Pacific SST: slow thermal regime) and Domain B
(ERA5 North Atlantic / European T2m: fast advective regime) across 5 deterministic seeds,
evaluating Experiments A through H, and generates all 10 Observatory publication figures CG-CP.

Usage:
    python3 examples/phase_15_regime_transfer_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltacore.observatory.phase_15_plots import generate_all_phase_15_plots
from deltacore.streaming.phase_15_benchmark import run_phase_15_benchmark


def main() -> None:
    print("=" * 88)
    print("DELTACORE PHASE 15: ADAPTIVE REGIME TRANSFER & ROBUSTNESS BENCHMARK")
    print(
        "Comparing Domain A (OISST Slow Thermal) <---> Domain B (ERA5 Fast Advective)"
    )
    print("=" * 88)

    _ = run_phase_15_benchmark(seeds=(42, 43, 44, 45, 46))

    print("\n[Observatory] Generating all 10 Phase 15 Publication Figures (CG - CP)...")
    generate_all_phase_15_plots()

    print("\n" + "=" * 88)
    print("PHASE 15 BENCHMARK & OBSERVATORY SUITE COMPLETED SUCCESSFULLY.")
    print("=" * 88)


if __name__ == "__main__":
    main()
