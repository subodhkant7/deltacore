#!/usr/bin/env python3
"""DeltaCore Phase 14: Atmospheric Spatio-Temporal Domain Replication Benchmark Entrypoint.

Executes the complete Phase 14 benchmark pipeline on the ERA5 North Atlantic & European
Atmospheric T2m dataset across 5 deterministic seeds and serializes all artifacts.

Usage:
    python3 examples/phase_14_atmospheric_benchmark.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltacore.observatory.phase_14_plots import generate_all_phase_14_plots
from deltacore.streaming.atmospheric_benchmark import run_phase_14_benchmark


def main() -> None:
    print("=" * 88)
    print("DELTACORE PHASE 14: ATMOSPHERIC SPATIO-TEMPORAL REPLICATION BENCHMARK")
    print("=" * 88)
    _ = run_phase_14_benchmark()
    print("\nGenerating Phase 14 Observatory Figures...")
    generate_all_phase_14_plots()
    print("\nBenchmark and Observatory figures completed successfully.")


if __name__ == "__main__":
    main()
