# DeltaCore Open-Source Release & Research Freeze Checklist

**Target Version**: `0.2.0`  
**Date**: 2026-10-07  
**Branch**: `main`  

---

### Core Science & Algorithm
- [x] **Algorithm Frozen**: `AdaptiveController`, update equations, stability bounds, learning rates, and controller mathematics remain 100% frozen.
- [x] **Statistical Test Corrected**: Replaced pseudo sign-test calculation with exact two-sided binomial sign test ($H_0: p = 0.5$) and added exact paired sign-flip permutation test evaluating all $2^{20} = 1{,}048{,}576$ sign configurations.
- [x] **Bootstrap Verified**: Verified that `paired_bootstrap_ci` resamples paired seed units ($d_i$) rather than algorithms independently ($B = 10,000$, seed 42).
- [x] **Benchmark Artifacts Regenerated**: `higher_order_regime_shift_results.json` and `higher_order_regime_shift_report.md` reflect verified statistical outputs.
- [x] **Statistical Reporting Framework Aligned**: Conflation between mean difference (bootstrap CI) and seed-direction frequency (sign test) eliminated across all documentation.

### Documentation & Scientific Honesty
- [x] **README Accurate**: Clean, conservative positioning as an open-source research implementation of adaptive associative state dynamics; prominent research-status notice included.
- [x] **Research Status Documented**: `docs/research_status.md` records final status as `RESEARCH FROZEN` and summarizes positive and negative findings.
- [x] **RecoveryOS Decision Documented**: `docs/recoveryos_decision.md` formally records that DeltaCore is rejected for RecoveryOS telemetry regime detection.
- [x] **Experimental Trajectory Documented**: `docs/experiments.md` details progression from Phase 1 to Phase 18 and Benchmarks 1 through 4.
- [x] **Limitations Enumerated**: `docs/limitations.md` comprehensively documents the 10 empirical limitations and prohibited claims.
- [x] **Mathematical Contract Accurate**: `docs/mathematical_contract.md` updated to v0.2.0 with explicit conceptual disambiguation.
- [x] **Roadmap Cleaned**: `docs/ROADMAP.md` updated to mark research frozen with no imminent RecoveryOS integration.
- [x] **Contributing Guide Added**: `CONTRIBUTING.md` establishes scientific integrity guidelines and requirements for peer review.
- [x] **Changelog Added**: `CHANGELOG.md` documents release 0.2.0 as the research freeze and archive release.

### Code Health & Reproducibility
- [x] **Reproducibility Instructions Verified**: `docs/reproducibility.md` provides tested instructions and environment specifications.
- [x] **Minimal Example Runs**: `examples/basic_adaptation.py` executes cleanly and passes all assertions.
- [x] **Regression Suite Passes**: All 689 tests in `tests/` pass with zero failures.
- [x] **Type Checker Clean**: Pyright reports 0 errors, 0 warnings.
- [x] **Linters Clean**: Ruff check and format pass 100%.
- [x] **Bytecode Compilation**: `python -m compileall` compiles cleanly across all directories.
- [x] **Version Synchronized**: `pyproject.toml` and `deltacore/__init__.py` synchronized to `0.2.0`.
- [x] **Git Clean**: Working tree clean, committed, and synced with `origin/main`.
