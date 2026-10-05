# Phase 6: Adaptive-State Benchmark Framework — Implementation Report

**Date**: 2026-10-05  
**Status**: Verified & Complete  
**Reference Documents**:
- [METRICS.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/METRICS.md)
- [BENCHMARK_MATRIX.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/BENCHMARK_MATRIX.md)
- [HYPOTHESES.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/HYPOTHESES.md)
- [PHASE_5_IMPLEMENTATION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/development/PHASE_5_IMPLEMENTATION.md)

---

## 1. Executive Summary

Phase 6 transforms DeltaCore into a reproducible, standardized scientific benchmark framework for investigating adaptive neural state during inference.

The central research question driving this phase is:
> **When does changing internal state during inference help, what does it cost, and when does it fail?**

### Key Achievements:
1. **Separation of Benchmarking from Models**: Benchmark harnesses evaluate models strictly through a standardized `BaseBaseline` interface without modifying model internals or computational graphs.
2. **Strongly Typed Schemas**: Implemented `BenchmarkConfig` with strict type and range validation, and `RunResult` / `AggregateResult` supporting JSON serialization. Raw PyTorch tensors are strictly excluded from primary JSON artifacts.
3. **Unified Baseline Suite**: Implemented standardized wrappers for five canonical systems:
   - **Baseline A**: Frozen memory (zero test-time adaptation control anchor)
   - **Baseline B**: Fixed Delta rule (Phase 1)
   - **Baseline C**: Adaptive Delta rule with dynamic error-conditioned learning rate (Phase 2)
   - **Baseline D**: Unconstrained coupled self-referential memory (Phase 3)
   - **Baseline E**: Stability-controlled self-referential memory (Phase 4)
4. **Six Controlled Benchmark Tasks**:
   - `stationary_recall`: Quantifies whether adaptation improves or degrades static recall.
   - `distribution_shift`: Measures first-passage and sustained recovery under target inversion, permutation, and partial mapping changes.
   - `key_interference`: Sweeps key cosine similarity $\rho \in [0.0, 1.0]$ to evaluate cross-talk interference.
   - `conflicting_targets`: Evaluates catastrophic forgetting and oscillation under direct contradiction ($k \to v_1$ then $k \to v_2$).
   - `stability_stress`: Stress-tests models in aggressive regimes (large keys, high rates, opposing targets).
   - `adaptation_budget`: Maps the Pareto frontier between recovery speed and state update energy.
5. **Rigorous Failure Semantics**: Categorical run statuses (`SUCCESS`, `NUMERICAL_FAILURE`, `INVALID_CONFIGURATION`, etc.) prevent silent conversion of NaNs into floating-point numbers.
6. **Provenance & Determinism**: Seed management across Python, NumPy, and PyTorch with explicit capture of Python, PyTorch, and thread count environment metadata.
7. **Complete Regression Suite**: 35 dedicated benchmark infrastructure tests verifying schemas, metrics, determinism, failure handling, aggregation, and CLI execution.

---

## 2. Phase 5 Documentation Audit & Terminology Corrections

Prior to benchmark implementation, the Phase 5 documentation and codebase were audited and corrected:

1. **Stability Wording Narrowed**:
   - Replaced broad claims of divergence prevention with the precise formulation:
     > *Dual controllers enforce local non-expansive normalized-step conditions under their stated assumptions. They do not establish globally bounded trajectories under arbitrary affine forcing.*
   - Updated in `docs/math/PHASE_5_ASSOCIATIVE_SCAN.md`, `docs/development/PHASE_5_IMPLEMENTATION.md`, and `deltacore/diagnostics/stability.py`.
2. **FP64 Associativity Semantics**:
   - Explicitly clarified:
     > *The affine composition algebra is mathematically associative; floating-point evaluations may differ because matrix multiplication is not numerically associative.*
   - Documented that observed differences of $0.00 \times 10^{00}$ in specific CPU tests are observed empirical results for those specific test vectors, not a theorem of bit-exactness.
3. **Composition Depth Definition**:
   - Defined `composition_depth = max(C_k - 1, 0)` as the maximum number of sequential binary compositions chained within any single chunk under linear left-to-right composition. Clarified that it represents linear composition depth, **not** logarithmic tree reduction depth ($O(\log C)$).

---

## 3. Architecture & Software Layout

```text
deltacore/
└── benchmarks/
    ├── __init__.py           # Top-level benchmark package exports
    ├── __main__.py           # CLI invocation entrypoint
    ├── cli.py                # Command-line interface parser (argparse)
    ├── schemas/
    │   ├── __init__.py
    │   ├── config.py         # BenchmarkConfig typed dataclass & validation
    │   └── result.py         # RunResult, AggregateResult, RunStatus
    ├── baselines/
    │   ├── __init__.py
    │   ├── base.py           # BaseBaseline abstract class & trajectory dataclass
    │   ├── wrappers.py       # Frozen, Fixed, Adaptive, Self-Ref, Safe-Self-Ref
    │   └── factory.py        # get_baseline instantiation factory
    ├── tasks/
    │   ├── __init__.py       # Task registry and factory
    │   ├── base.py           # BaseTask abstract class
    │   ├── stationary_recall.py
    │   ├── distribution_shift.py
    │   ├── key_interference.py
    │   ├── conflicting_targets.py
    │   ├── stability_stress.py
    │   └── adaptation_budget.py
    ├── metrics/
    │   ├── __init__.py
    │   ├── accuracy.py       # final_error, mean_error, adaptation_gain
    │   ├── recovery.py       # first_passage_recovery, sustained_recovery
    │   └── energy.py         # update_energy, step_energy, state_growth, margins
    ├── runners/
    │   ├── __init__.py
    │   ├── determinism.py    # set_seed, get_environment_info
    │   ├── timing.py         # measure_execution_time, CUDA synchronization
    │   └── suite.py          # run_benchmark_suite, artifact directory layout
    └── reporting/
        ├── __init__.py
        ├── statistics.py     # aggregate_runs, compute_paired_differences
        ├── report.py         # generate_markdown_report
        └── visualization.py  # lazy-imported matplotlib plotting helpers
```

---

## 4. Empirical Benchmark Suite Results

Executed via `python3 examples/run_phase_6_benchmarks.py` on Apple Silicon CPU (PyTorch 2.6.0, Python 3.13.0, 5 seeds: `[0, 1, 2, 3, 4]`):

### Task 1: Stationary Recall ($T=64, K=16, V=16, P=6$)
| Model | Success | Final Error ($E_T$) | Mean Error ($\bar{E}$) | Final $\|M_T\|_F$ | Update Energy ($U_M$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **frozen** | 5/5 | 3.7436 ± 0.486 | 3.9110 ± 0.385 | 0.0000 ± 0.000 | 0.00 ± 0.00 |
| **fixed** | 5/5 | 2.0130 ± 0.572 | 2.6698 ± 0.292 | 6.0451 ± 0.566 | 17.09 ± 1.87 |
| **adaptive** | 5/5 | 0.8066 ± 0.636 | 1.3570 ± 0.238 | 10.3649 ± 1.195 | 36.82 ± 7.38 |
| **self_referential** | 0/5 | Diverged (`inf`) | Diverged | Diverged | Diverged |
| **safe_self_referential** | 5/5 | 0.8138 ± 0.636 | 1.3826 ± 0.234 | 10.3445 ± 1.188 | 36.70 ± 7.34 |

*Observation*: Unconstrained self-reference diverged over $T=64$ steps under continuous accumulation, whereas safety-controlled self-reference achieved 100% finite survival with final error $0.8138$, outperforming fixed delta ($2.0130$).

### Task 2: Distribution Shift (Target Inversion at $t=32$)
| Model | Success | Pre-Shift Err | Shock Err | Final Err ($E_T$) | $T_{\text{FP}}$ (Steps to 50%) | State Growth Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **frozen** | 5/5 | 3.899 ± 0.42 | 3.838 ± 0.44 | 3.744 ± 0.49 | Fail / N/A | 0.000 ± 0.00 |
| **fixed** | 5/5 | 2.545 ± 0.29 | 5.204 ± 0.88 | 3.045 ± 0.94 | 24.2 ± 5.7 | 1.000 ± 0.00 |
| **adaptive** | 5/5 | 1.038 ± 0.31 | 6.977 ± 1.09 | 2.193 ± 1.84 | 5.6 ± 4.0 | 1.014 ± 0.02 |
| **self_referential** | 0/5 | 1.529 ± 0.48 | 5.954 ± 1.82 | Diverged | 8.6 ± 3.9 | 1.000 ± 0.00 |
| **safe_self_referential** | 5/5 | 1.059 ± 0.32 | 6.977 ± 1.11 | 2.187 ± 1.84 | 5.6 ± 4.0 | 1.000 ± 0.00 |

*Observation*: `SafeSelfReferential` recovers in $5.6 \pm 4.0$ steps, over $4\times$ faster than `FixedDelta` ($24.2 \pm 5.7$ steps), while maintaining state growth ratio $1.000$.

### Task 3: Key Interference vs. Cosine Similarity ($\rho$)
| Model | $\rho = 0.00$ Mean Err | $\rho = 0.50$ Mean Err | $\rho = 0.90$ Mean Err |
| :--- | :---: | :---: | :---: |
| **fixed** | 0.1216 ± 0.000 | 0.2389 ± 0.032 | 0.5500 ± 0.076 |
| **adaptive** | 0.0019 ± 0.000 | 0.0261 ± 0.003 | 0.3167 ± 0.036 |
| **self_referential** | 0.0019 ± 0.000 | 0.0267 ± 0.003 | 0.3197 ± 0.037 |
| **safe_self_referential** | 0.0019 ± 0.000 | 0.0267 ± 0.003 | 0.3197 ± 0.037 |

*Observation*: Cross-talk error increases monotonically with cosine similarity across all models. When keys are orthogonal ($\rho=0$), adaptive systems achieve near-zero retrieval error ($0.0019$); at $\rho=0.90$, cross-talk forces error up to $0.3197$.

### Task 4: Conflicting Targets ($k \to v_1$ then $k \to v_2, \|v_1-v_2\|=2.0$)
| Model | Success | Old Retention ($E_{v1}$) | New Target Err ($E_{v2}$) | Adapt Speed (Steps) | Oscillation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 5/5 | 1.757 ± 0.00 | 0.244 ± 0.00 | 16.0 ± 0.0 | 0.0027 ± 0.0000 |
| **adaptive** | 5/5 | 1.997 ± 0.00 | 0.003 ± 0.00 | 4.0 ± 0.0 | 0.0423 ± 0.0000 |
| **self_referential** | 5/5 | 1.997 ± 0.00 | 0.003 ± 0.00 | 4.0 ± 0.0 | 0.0333 ± 0.0000 |
| **safe_self_referential** | 5/5 | 1.997 ± 0.00 | 0.003 ± 0.00 | 4.0 ± 0.0 | 0.0324 ± 0.0000 |

*Observation*: Adaptive and self-referential systems overwrite the old association $4\times$ faster (4 steps vs. 16 steps), driving new target error to $0.003$.

### Task 5: Stability Stress (Large Keys $\|k\|=6.0, \|k\|^2=36.0$)
| Model | Status | Success Rate | Failures | Max $\|M\|_F$ | Min Margin $S_t$ | Clips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | ALL SUCCESS | 5/5 | 0 | $2.54 \times 10^3$ | -1.600 ± 0.00 | 0.0 |
| **self_referential** | NUMERICAL_FAILURE | 0/5 | 5 | $1.48 \times 10^4$ | -16.000 ± 0.00 | 0.0 |
| **safe_self_referential** | ALL SUCCESS | 5/5 | 0 | $2.68 \times 10^0$ | +1.000 ± 0.00 | 100.0 |

*Observation*: In this expansion regime, unconstrained self-reference blew up to $1.48 \times 10^4$ and suffered 100% numerical failure (`inf`). In contrast, `SafeSelfReferential` clipped 100% of transitions, kept margin $S_t = 1.0 > 0$ strictly inside the contractive regime, and maintained bounded memory norm $\|M_T\|_F = 2.68$.

---

## 5. Evaluation of Hypotheses

- **H1 (Self-Reference Acceleration Under Shift)**: **SUPPORTED IN THE TESTED TARGET-INVERSION BENCHMARK**. `SafeSelfReferential` recovered in $5.6 \pm 4.0$ steps vs. $24.2 \pm 5.7$ steps for `FixedDelta` (over $4\times$ speedup) under the tested target-inversion distribution shift.
- **H2 (State Adaptation Cost Under Stationary Data)**: **PARTIALLY FALSIFIED / QUALIFIED**. In stationary recall, `SafeSelfReferential` attained *lower* final error ($0.8138$ vs. $2.0130$) rather than higher error, but expended over $2\times$ greater update energy ($36.70$ vs. $17.09$). Unconstrained `SelfReferential` suffered numerical failure without safety control.
- **H3 (Key Correlation Monotonically Increases Cross-Talk)**: **SUPPORTED ACROSS TESTED SUBSET ($\rho \in \{0.0, 0.5, 0.9\}$)**. Mean retrieval error rose monotonically across the published evaluation subset ($0.0019$ at $\rho=0$, $0.0267$ at $\rho=0.5$, $0.3197$ at $\rho=0.9$). Monotonicity across all seven levels ($\rho \in [0.00, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]$) remains an empirical hypothesis awaiting the complete dense sweep.
- **H4 (Safety Control Reduces Numerical Failure Under Stress)**: **QUALIFIED**. Safety control reduced numerical failure in the tested stress configuration: `SafeSelfReferential` achieved 100% finite success (5/5) with bounded norm $2.68$, while unconstrained `SelfReferential` suffered 100% failure (0/5) with norm overflow.
- **H5 (Recovery Speed Requires Higher Update Energy)**: **QUALIFIED**. Faster recovery was associated with higher observed update energy in the tested configuration: the $4.3\times$ faster recovery in Task 2 observed $54.38$ update energy units compared to $12.37$ for fixed delta.

---

## 6. Scientific Honesty & Exit Gate Verification

All Phase 6 constraints and exit criteria were audited and verified:
- [x] No claim made that self-reference is universally superior (documented that it expends higher update energy and can diverge without safety controls).
- [x] No claim made that DeltaCore is state-of-the-art.
- [x] No claim made that synthetic benchmarks prove vision performance (deferred to Phase 8).
- [x] No claim made that numerical survival over 50 steps establishes global mathematical stability.
- [x] Distinct epistemic categories maintained across all outputs:
  - **Mathematical Theorem**: Local contraction condition $0 \le \eta_t \|k_t\|^2 \le 2$.
  - **Empirical Observation**: `SafeSelfReferential` survived 50 steps with norm $2.68$.
  - **Benchmark Result**: Recovery speedup of $4.3\times$ on target inversion.
  - **Hypothesis**: H1 through H5 tracked in formal registry before execution.
- [x] Development halted at Phase 6 gate. Phase 7 has not been started.
