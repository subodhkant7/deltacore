"""Unit tests for statistical testing primitives (exact sign test, permutation test, bootstrap CI)."""

from __future__ import annotations

import math
from experiments.higher_order_regime_shift import (
    exact_binomial_sign_test,
    paired_bootstrap_ci,
    paired_permutation_test,
)


def test_exact_binomial_sign_test_analytic_cases() -> None:
    # Case 1: All zeros / ties
    res_zero = exact_binomial_sign_test([0.0, 0.0, 0.0])
    assert res_zero["n_nonzero"] == 0
    assert res_zero["p_value"] == 1.0

    # Case 2: n = 10, k = 5 (exact symmetry)
    diffs_sym = [1.0] * 5 + [-1.0] * 5
    res_sym = exact_binomial_sign_test(diffs_sym)
    assert res_sym["n_nonzero"] == 10
    assert res_sym["n_positive"] == 5
    assert res_sym["p_value"] == 1.0

    # Case 3: n = 10, k = 0 (all negative)
    # Analytic p-value = 2 * (0.5)^10 = 2 / 1024 = 0.001953125
    diffs_all_neg = [-1.0] * 10
    res_all_neg = exact_binomial_sign_test(diffs_all_neg)
    assert math.isclose(res_all_neg["p_value"], 2.0 / 1024.0, rel_tol=1e-6)

    # Case 4: n = 20, k = 4 (exact benchmark case)
    # Analytic p-value = 2 * sum_{j=0}^4 C(20, j) / 2^20 = 2 * 6196 / 1048576 = 0.01181793...
    diffs_bench = [1.0] * 4 + [-1.0] * 16
    res_bench = exact_binomial_sign_test(diffs_bench)
    expected_p = 2.0 * sum(math.comb(20, j) for j in range(5)) * (0.5**20)
    assert math.isclose(res_bench["p_value"], expected_p, rel_tol=1e-6)
    assert math.isclose(res_bench["p_value"], 0.0118179321, rel_tol=1e-5)


def test_paired_permutation_test_properties() -> None:
    # Zero mean symmetric differences -> p-value near 1.0
    sym_diffs = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
    res_sym = paired_permutation_test(sym_diffs, n_permutations=10000, seed=42)
    assert res_sym["observed_mean"] == 0.0
    assert math.isclose(res_sym["permutation_p_value"], 1.0, rel_tol=1e-2)

    # Strongly separated one-sided differences -> p-value very small
    sep_diffs = [10.0, 12.0, 11.5, 9.8, 10.2, 11.0, 10.5, 12.1]
    res_sep = paired_permutation_test(sep_diffs, n_permutations=10000, seed=42)
    assert res_sep["permutation_p_value"] < 0.01


def test_paired_bootstrap_ci_coverage() -> None:
    # Deterministic sequence with known mean
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci_low, ci_high = paired_bootstrap_ci(diffs, n_boot=2000, ci=0.95, seed=42)
    # Mean is 3.0, CI should enclose 3.0
    assert ci_low <= 3.0 <= ci_high
    assert ci_low > 1.0
    assert ci_high < 5.0
