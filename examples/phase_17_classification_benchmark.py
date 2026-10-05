#!/usr/bin/env python3
"""DeltaCore Phase 17: Online Non-Stationary Classification Benchmark Entrypoint.

Executes the complete Phase 17 benchmark pipeline testing whether DeltaCore's adaptive-state
principles generalize to online non-stationary classification:
    - Task A: Stationary classification (useful prediction gate)
    - Task B: Abrupt label-preserving covariate shift (A -> B -> A)
    - Task C: Decision-boundary shift (A -> C -> A hyperplane rotation)
    - Task D: Retention stress & stale state ablation (continuous, reset, fixed-high, fixed-low, adaptive)
    - Dimensional scaling: D in {32, 64, 128, 256}
    - Robustness controls: Label shuffle (negative control) and feature permutation
    - Multi-seed execution across seeds = [42, 43, 44, 45, 46]
    - Serializes all 6 required Phase 17 JSON artifacts
    - Generates all 14 Observatory publication figures DC through DP.

Usage:
    python3 examples/phase_17_classification_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltacore.classification.phase_17_runner import run_phase_17_benchmark
from deltacore.observatory.phase_17_plots import generate_all_phase_17_plots


def main() -> None:
    print("=" * 88)
    print("DELTACORE PHASE 17: ONLINE NON-STATIONARY CLASSIFICATION BENCHMARK")
    print("Evaluating Adaptive Associative State Under Changing Data Distributions")
    print(
        "Testing Parameter Immutability (Δθ = 0), Shift Recovery, and Dimensional Scaling"
    )
    print("=" * 88)

    _ = run_phase_17_benchmark(seeds=(42, 43, 44, 45, 46), dim=32, num_classes=6)

    print("\n[Observatory] Generating all 14 Phase 17 Publication Figures (DC - DP)...")
    generate_all_phase_17_plots()

    print("\n" + "=" * 88)
    print("PHASE 17 BENCHMARK & OBSERVATORY SUITE COMPLETED SUCCESSFULLY.")
    print("=" * 88)


if __name__ == "__main__":
    main()
