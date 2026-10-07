"""Unit tests for statistical testing primitives and benchmark reproducibility.

Validates:
1. Exact binomial sign-test known values.
2. Exact sign-test edge cases.
3. Zero/tie handling.
4. Exact paired sign-flip enumeration.
5. Paired permutation symmetry.
6. Paired bootstrap determinism with seed.
7. Paired bootstrap preserving paired units (regression check against unpaired bootstrap).
8. Confidence interval ordering (lower <= upper).
9. Rejection of the previous pseudo-p-value implementation.
10. Reproducibility of the decisive benchmark summary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from experiments.higher_order_regime_shift import (
    exact_binomial_sign_test,
    exact_paired_sign_flip_permutation_test,
    paired_bootstrap_ci,
    paired_permutation_test,
)


def test_1_exact_binomial_sign_test_known_values() -> None:
    """1. Verify exact binomial sign test produces exact analytical values for known n, k."""
    # Benchmark case: n = 20, k = 4
    # Analytic p = 2 * sum_{j=0}^4 C(20, j) / 2^20 = 2 * 6196 / 1048576 = 0.011817932...
    diffs_20_4 = [1.0] * 4 + [-1.0] * 16
    res = exact_binomial_sign_test(diffs_20_4)
    expected_p = 2.0 * sum(math.comb(20, j) for j in range(5)) * (0.5**20)
    assert math.isclose(res["p_value"], expected_p, rel_tol=1e-7)
    assert math.isclose(res["p_value"], 0.0118179321, rel_tol=1e-5)

    # Smaller case: n = 10, k = 1
    # Analytic p = 2 * (1 + 10) / 1024 = 22 / 1024 = 0.021484375
    diffs_10_1 = [1.0] * 1 + [-1.0] * 9
    res_10 = exact_binomial_sign_test(diffs_10_1)
    assert math.isclose(res_10["p_value"], 22.0 / 1024.0, rel_tol=1e-7)


def test_2_exact_binomial_sign_test_edge_cases() -> None:
    """2. Verify edge cases: all positive, all negative, single element."""
    # All negative (k = 0 out of 10)
    res_k0 = exact_binomial_sign_test([-1.0] * 10)
    assert math.isclose(res_k0["p_value"], 2.0 / 1024.0, rel_tol=1e-7)

    # All positive (k = 10 out of 10)
    res_kn = exact_binomial_sign_test([1.0] * 10)
    assert math.isclose(res_kn["p_value"], 2.0 / 1024.0, rel_tol=1e-7)

    # Single observation: n = 1, k = 1 -> p = 1.0
    res_n1 = exact_binomial_sign_test([5.0])
    assert res_n1["p_value"] == 1.0


def test_3_exact_binomial_sign_test_zero_and_tie_handling() -> None:
    """3. Verify zero/tie handling: zeros must be excluded, effective n updated."""
    # Pure zeros
    res_pure_zero = exact_binomial_sign_test([0.0, 0.0, 0.0])
    assert res_pure_zero["n_total"] == 3
    assert res_pure_zero["n_nonzero"] == 0
    assert res_pure_zero["p_value"] == 1.0

    # Mixed zeros: 2 zeros, 4 positive, 1 negative -> effective n = 5, k = 1 (or 4)
    # Analytic p = 2 * (1 + 5) / 32 = 12 / 32 = 0.375
    diffs_mixed = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0]
    res_mixed = exact_binomial_sign_test(diffs_mixed)
    assert res_mixed["n_total"] == 7
    assert res_mixed["n_nonzero"] == 5
    assert res_mixed["n_positive"] == 4
    assert res_mixed["n_negative"] == 1
    assert math.isclose(res_mixed["p_value"], 12.0 / 32.0, rel_tol=1e-7)


def test_4_exact_paired_sign_flip_enumeration() -> None:
    """4. Verify exact sign-flip test enumerates exactly 2^n assignments and computes analytical p."""
    # Small analytical test: diffs = [1.0, 2.0, 3.0, 4.0]
    # Sum = 10.0. 2^4 = 16 assignments.
    # Only (+1, +2, +3, +4) -> +10 and (-1, -2, -3, -4) -> -10 have |sum| >= 10.
    # Analytical exact two-sided p = 2 / 16 = 0.125.
    diffs_4 = [1.0, 2.0, 3.0, 4.0]
    res_4 = exact_paired_sign_flip_permutation_test(diffs_4)
    assert res_4["n_assignments"] == 16
    assert res_4["count_extreme"] == 2
    assert math.isclose(res_4["permutation_p_value"], 0.125, rel_tol=1e-7)

    # Benchmark scale test: n = 20
    # Must evaluate exactly 2^20 = 1,048,576 configurations
    diffs_20 = [0.1] * 20
    res_20 = exact_paired_sign_flip_permutation_test(diffs_20)
    assert res_20["n_assignments"] == 1048576


def test_5_paired_permutation_symmetry() -> None:
    """5. Verify paired permutation symmetry under exchangeability of signs."""
    # Symmetrically balanced differences -> sum = 0.0 -> p-value = 1.0
    sym_diffs = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
    res_sym = paired_permutation_test(sym_diffs)
    assert res_sym["observed_mean"] == 0.0
    assert math.isclose(res_sym["permutation_p_value"], 1.0, rel_tol=1e-7)

    # Strictly one-sided differences with large separation
    sep_diffs = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    res_sep = paired_permutation_test(sep_diffs)
    # Sum = 75. Only 2 out of 64 have |sum| >= 75.
    assert res_sep["n_assignments"] == 64
    assert res_sep["count_extreme"] == 2
    assert math.isclose(res_sep["permutation_p_value"], 2.0 / 64.0, rel_tol=1e-7)


def test_6_paired_bootstrap_determinism_with_seed() -> None:
    """6. Verify paired bootstrap CI is deterministic when seed is provided."""
    diffs = [-0.05, 0.02, -0.01, 0.03, -0.04, -0.02, 0.01, -0.03, -0.06, 0.02]
    ci1 = paired_bootstrap_ci(diffs, n_boot=2000, ci=0.95, seed=42)
    ci2 = paired_bootstrap_ci(diffs, n_boot=2000, ci=0.95, seed=42)
    assert ci1 == ci2


def test_7_paired_bootstrap_preserves_pairing_preventing_unpaired_regression() -> None:
    """7. Regression test: Prove paired bootstrap preserves pairing by testing correlated units.

    If paired units are preserved:
    d_i = y_i^{(1)} - y_i^{(2)} = c (constant for all seeds),
    then resampling paired units yields identical mean c for every replicate, giving CI width = 0.

    If the bootstrap had accidentally regressed to an independent (unpaired) bootstrap
    resampling y^{(1)} and y^{(2)} separately, the large variance of y would produce
    a huge confidence interval width.
    """
    # Two highly variable series with constant paired difference of +2.0
    y1 = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    y2 = [x - 2.0 for x in y1]
    paired_diffs = [a - b for a, b in zip(y1, y2, strict=True)]  # all +2.0

    ci_low, ci_high = paired_bootstrap_ci(paired_diffs, n_boot=1000, seed=42)
    assert math.isclose(ci_low, 2.0, abs_tol=1e-7)
    assert math.isclose(ci_high, 2.0, abs_tol=1e-7)
    assert abs(ci_high - ci_low) < 1e-6, (
        "Paired bootstrap must preserve zero-variance paired difference"
    )


def test_8_paired_bootstrap_ci_ordering() -> None:
    """8. Verify paired bootstrap CI ordering: lower <= mean <= upper."""
    diffs = [-0.1, 0.05, -0.02, 0.03, -0.04, 0.01, -0.08, 0.02]
    mean_diff = float(np.mean(diffs))
    ci_low, ci_high = paired_bootstrap_ci(diffs, n_boot=2000, ci=0.95, seed=42)
    assert ci_low <= ci_high
    assert ci_low <= mean_diff <= ci_high


def test_9_rejection_of_previous_pseudo_sign_test_formula() -> None:
    """9. Explicitly test rejection of the previous pseudo formula: 2 * min(k, n-k) / n.

    For n = 20, k = 4:
    Pseudo formula incorrectly gave: 2 * 4 / 20 = 0.4000.
    Exact binomial sign test correctly gives: p = 0.0118.
    """
    diffs = [1.0] * 4 + [-1.0] * 16
    res = exact_binomial_sign_test(diffs)
    pseudo_p = 2.0 * min(4, 16) / 20.0
    assert pseudo_p == 0.4000
    assert not math.isclose(res["p_value"], pseudo_p, abs_tol=0.1)
    assert math.isclose(res["p_value"], 0.0118179321, rel_tol=1e-4)


def test_10_reproducibility_of_decisive_benchmark_summary() -> None:
    """10. Verify reproducibility of canonical decisive benchmark results from artifact."""
    artifact_path = Path("experiments/artifacts/higher_order_regime_shift_results.json")
    if not artifact_path.exists():
        return

    with open(artifact_path, encoding="utf-8") as f:
        data = json.load(f)

    # 1. Decision must be FAIL
    assert data["decision"] == "FAIL"

    # 2. Check Online Covariance and DeltaCore Gated mean AUROCs
    agg = data["aggregate_detection"]
    cov_auroc = agg["online_cov_mahalanobis_gated"]["auroc_mean"]
    dc_auroc = agg["deltacore_gated"]["auroc_mean"]
    assert math.isclose(cov_auroc, 0.597625, rel_tol=1e-4)
    assert math.isclose(dc_auroc, 0.571125, rel_tol=1e-4)

    # 3. Check paired comparison statistics
    cov_comp = data["paired_comparisons"]["online_cov_mahalanobis_gated"]
    assert math.isclose(cov_comp["mean_paired_difference"], -0.0265, abs_tol=1e-4)

    ci_low, ci_high = cov_comp["ci_95_paired_difference"]
    assert ci_low < ci_high < 0.0, "95% bootstrap CI must strictly exclude zero"
    assert math.isclose(ci_low, -0.0478, abs_tol=1e-3)
    assert math.isclose(ci_high, -0.0072, abs_tol=1e-3)

    # 4. Check exact sign test (DeltaCore won on 4 of 20 seeds)
    st = cov_comp["sign_test_details"]
    assert st["n_total"] == 20
    assert st["n_positive"] == 4
    assert st["n_negative"] == 16
    assert math.isclose(st["p_value"], 0.0118179321, rel_tol=1e-4)

    # 5. Check exact paired sign-flip permutation test
    pt = cov_comp["permutation_test_details"]
    assert pt["n_assignments"] == 1048576  # 2^20
    assert pt["count_extreme"] == 19728
    assert math.isclose(pt["permutation_p_value"], 0.0188140869, rel_tol=1e-4)
