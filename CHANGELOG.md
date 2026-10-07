# Changelog

All notable changes to **DeltaCore** will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-10-07

### Research Freeze & Reproducibility Archive Release

This release marks the formal completion and research freeze of the DeltaCore project. Following conclusive multi-seed evaluation on matched-marginal telemetry distribution shift, the hypothesis that DeltaCore's adaptive associative controller provides an advantage over standard second-order covariance estimation for telemetry regime detection was falsified. The codebase is frozen and preserved as a reproducible open-source research archive.

#### Added
- **Canonical Controller API**: Stabilized `AdaptiveController` and `ControllerConfig` with non-mutating pre-update residual evaluation (`score()`) and explicit update gating (`step(adapt=...)`).
- **Telemetry Feature Hashing**: Added `DeterministicFeatureHasher` with signed SHA-256 token hashing, collision statistics tracking, and interaction feature support (token pairs and triples).
- **Deterministic Persistence**: Added atomic state serialization (`save_state()`, `load_state()`) with SHA-256 checksum integrity verification and corruption detection.
- **Regime Shift Benchmark Suite**: Completed 4 multi-seed non-stationary distribution shift benchmarks, culminating in `higher_order_regime_shift` (v3.0.0) across 20 independent seeds with 11 comparator models.
- **Proper Second-Order Baselines**: Implemented `OnlineCovarianceMahalanobisModel` with regularized Mahalanobis distance scoring and `PairwiseCorrelationModel`.
- **Exact Statistical Hypothesis Tests**: Implemented exact two-sided binomial sign test and paired permutation (sign-flip) randomization test alongside paired percentile bootstrap 95% confidence intervals.
- **Blind Execution Regression Tests**: Added `tests/test_benchmark_blind_execution.py` ensuring score and update signatures execute strictly blind to class labels.
- **Comprehensive Documentation Suite**: Added `docs/research_status.md`, `docs/recoveryos_decision.md`, `docs/experiments.md`, `docs/reproducibility.md`, `docs/release_checklist.md`, and updated `docs/limitations.md`.

#### Changed
- **Statistical Reporting**: Corrected statistical inference reporting to strictly distinguish between mean paired difference uncertainty (bootstrap CI) and seed-direction frequency (sign test).
- **Roadmap & RecoveryOS Status**: Formally recorded that DeltaCore is rejected for RecoveryOS telemetry regime detection and marked all roadmap items as research frozen.
- **Repository Metadata**: Synchronized package version to `0.2.0` across `pyproject.toml` and `deltacore/__init__.py`.

---

## [0.1.0] - 2026-10-04

### Foundational Research Release
- Initial implementation of mathematical primitives across Phases 1–18.
- Associative memory state $M_t \in \mathbb{R}^{V \times K}$ with Hebbian and Delta update rules.
- Lyapunov contractive step-size stability controller $\eta_t = \min(\eta_0, \rho / (\|x\|^2 + \epsilon))$.
- Sequential and chunked parallel affine associative prefix scans.
- 2D classification transfer experiments and diagnostic observatory plotting.
