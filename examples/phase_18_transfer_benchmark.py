#!/usr/bin/env python3
"""DeltaCore Phase 18: Unseen Classification Regime Transfer & Falsification Entrypoint.

Executes the complete Phase 18 benchmark pipeline testing whether the minimal
SafeAdaptiveDelta associative-state mechanism transfers across genuinely different
non-stationary classification environments without per-task retuning:
    - Family A: Linear boundary rotation (A -> C -> A)
    - Family B: Boundary translation / intercept shift (A -> B_trans -> A)
    - Family C: Nonlinear decision boundary deformation (A -> C_nonlin -> A)
    - Shift dimensions: Covariate, Boundary, Class-prior, Gradual drift, Abrupt, and Strong Mismatch
    - Causal ablations: State ON vs OFF, Continuous vs Reset, FixedDelta vs SafeAdaptiveDelta
    - Negative controls: Shortcut audit, Label-shuffle control, Feature permutation
    - Dimensional scaling: D in {32, 64, 128, 256}
    - Multi-seed execution: seeds = [42, 43, 44, 45, 46]
    - Serializes all 10 Phase 18 JSON artifacts
    - Generates all 18 publication-quality Observatory plots
    - Evaluates hypotheses H18.1 through H18.8 and prints Transfer & Failure Matrices.

Usage:
    python3 examples/phase_18_transfer_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltacore.benchmarks.phase_18.analysis import (
    evaluate_hypotheses_18,
)
from deltacore.benchmarks.phase_18.run import run_phase_18_benchmark
from deltacore.observatory.phase_18_plots import generate_all_phase_18_plots


def main() -> None:
    print("=" * 88)
    print(
        "DELTACORE PHASE 18: UNSEEN CLASSIFICATION TRANSFER & FALSIFICATION BENCHMARK"
    )
    print(
        "Testing Minimal SafeAdaptiveDelta Associative State Across Unseen Shift Families"
    )
    print(
        "Strictly Frozen Configuration: eta0=0.015, rho=1.50, alpha_min=0.95 (No Retuning)"
    )
    print(
        "Parameter Immutability Verified (Delta theta = 0) Bit-for-Bit via Pre/Post SHA-256"
    )
    print("=" * 88)

    artifacts_dir = "docs/benchmarks/artifacts/phase_18"
    _ = run_phase_18_benchmark(
        output_dir=artifacts_dir,
        seeds=[42, 43, 44, 45, 46],
        dimensions=[32, 64, 128, 256],
        verbose=True,
    )

    print("\n[Observatory] Generating all 18 Phase 18 Publication Figures...")
    plot_files = generate_all_phase_18_plots(artifacts_dir=artifacts_dir)
    print(
        f"[Observatory] Successfully generated {len(plot_files)} publication figures in {artifacts_dir}/plots/"
    )

    print("\n[Analysis] Evaluating Pre-Registered Hypotheses H18.1 - H18.8...")
    analysis_res = evaluate_hypotheses_18(artifacts_dir=artifacts_dir)

    print("\n" + "=" * 88)
    print("PHASE 18 HYPOTHESIS EVALUATION SUMMARY")
    print("=" * 88)
    for h_id, h_data in analysis_res["hypotheses"].items():
        print(f"[{h_id}] {h_data['status']}: {h_data['proposition']}")
        print(f"       Evidence: {h_data['evidence']}")

    print("\n" + "=" * 88)
    print(f"FINAL PHASE-GATE OUTCOME: {analysis_res['phase_gate_outcome']}")
    print(f"Rationale: {analysis_res['phase_gate_rationale']}")
    print("=" * 88)


if __name__ == "__main__":
    main()
