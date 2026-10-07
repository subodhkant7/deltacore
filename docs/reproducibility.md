# DeltaCore: Reproduction Guide & Environment Specifications

**Status**: Verified Reproducibility Protocol  
**Version**: `0.2.0`  
**Date**: 2026-10-07  

This document provides exact, tested instructions for setting up the DeltaCore environment, running regression suites, and reproducing experimental benchmarks.

---

## 1. Reference Environment Specifications

All reference benchmark results were produced in the following verified environment:

- **Operating System**: macOS 15.0+ (Darwin 24.x) / Linux (Ubuntu 22.04 LTS verified)
- **Python Version**: `3.13.0` (compatible with Python $\ge 3.10$)
- **Core Dependencies**:
  - `torch >= 2.2.0` (tested on `2.6.0`)
  - `numpy >= 1.24.0` (tested on `2.2.3`)
  - `matplotlib >= 3.8.0` (tested on `3.11.2`, running headless `Agg`)
- **Development Tooling**:
  - `pytest >= 8.0.0` (tested on `9.1.1`)
  - `ruff >= 0.4.0` (tested on `0.9.9`)
  - `pyright >= 1.1.350` (tested on `1.1.396`)
- **Deterministic PRNG**:
  - Default benchmark master seed: `42`
  - Multi-seed evaluation range: Seeds `0` through `19`

---

## 2. Installation from Source

Clone the repository and install DeltaCore in editable research mode:

```bash
# 1. Clone repository
git clone https://github.com/subodhkant7/deltacore.git
cd deltacore

# 2. Create isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install package with dev dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:
```bash
python -c "import deltacore; print(deltacore.__name__, deltacore.__version__)"
# Expected output: deltacore 0.2.0
```

---

## 3. Verifying the Test Suite

Execute the complete regression test suite:

```bash
pytest -q
```
**Expected Output**: `689 passed in ~31s` (0 failures, 0 errors).

Run static analysis and code linters:
```bash
# Type checking
pyright deltacore tests examples experiments

# Code style and formatting checks
ruff check .
ruff format --check .

# Bytecode compilation verification
python -m compileall deltacore examples tests experiments
```
All commands must pass with 0 errors and 0 warnings.

---

## 4. Running the Minimal Smoke Example

Execute the canonical adaptive controller example:

```bash
python examples/basic_adaptation.py
```

**Expected Behavior**:
- Generates 150 nominal observations followed by a 20-step gradual distribution drift.
- Logs pre-update reconstruction residuals ($r_t$) adapting smoothly to drift.
- Injects a synthetic anomaly and demonstrates score-before-update gating.
- Serializes controller state to disk, validates SHA-256 integrity, and reloads state.
- Exits with return code `0`.

---

## 5. Reproducing Scientific Benchmarks

### Primary Decisive Benchmark: Higher-Order Regime Shift (Experiment 4)

Reproduce the 20-seed decisive evaluation gate comparing DeltaCore against Online Covariance and other baselines:

```bash
python experiments/higher_order_regime_shift.py
```

**Execution Details**:
- **Wall Time**: ~20 seconds on standard 8-core CPU.
- **Generated Artifacts**:
  - Results JSON: `experiments/artifacts/higher_order_regime_shift_results.json`
  - Markdown Report: `experiments/artifacts/higher_order_regime_shift_report.md`
  - Figures: 15 publication PNG plots in `experiments/artifacts/higher_order_regime_shift_plots/`
- **Expected Results Summary**:
  - `Online Covariance AUROC`: $0.5976 \pm 0.0667$
  - `DeltaCore Gated AUROC`: $0.5711 \pm 0.0801$
  - `Mean Paired Difference`: $-0.0265$ (95% percentile bootstrap CI: $[-0.0478, -0.0072]$, $B=10,000$, seed 42)
  - `Paired Cohen's d`: $-0.58$
  - `Exact Binomial Sign Test`: $p = 0.0118$ (DeltaCore wins 4/20 seeds, 16 losses)
  - `Exact Paired Sign-Flip Randomization Test`: $p = 0.0188$ ($19,728 / 1,048,576$ sign configurations as or more extreme than observed)

**Canonical Conclusion**:
Under the 20-seed non-stationary matched-marginal benchmark, the observed paired difference favored regularized Online Covariance. The exact two-sided sign test rejected its 50/50 directional null at p = 0.0118, and the exact paired sign-flip randomization test rejected its stated exchangeability-of-signs null at p = 0.0188. These tests address different inferential questions and are reported separately; the two p-values are not treated as independent confirmations of a single omnibus hypothesis. DeltaCore Gated did not demonstrate a performance advantage over regularized Online Covariance on the decisive benchmark.

### Multiplicity and Inferential Roles

| Procedure | Question | Null / Framework | Result |
| :--- | :--- | :--- | :--- |
| **Paired percentile bootstrap** | Mean-effect uncertainty | Resampling framework ($B = 10{,}000$, seed 42) | 95% CI `[-0.0478, -0.0072]` |
| **Exact paired sign-flip test** | Paired-effect magnitude | Exchangeability-of-signs null ($2^{20} = 1{,}048{,}576$ assignments) | `p = 0.0188` ($19{,}728$ extreme) |
| **Exact two-sided sign test** | Direction / majority | 50/50 sign null ($H_0: p = 0.5$) | `p = 0.0118` (4 wins / 16 losses) |

Exhaustive enumeration removes Monte Carlo approximation error from the sign-flip calculation; it does not make the inferential procedure assumption-free. The percentile bootstrap interval is a resampling-based uncertainty estimate and is not treated as an exact confidence interval.

**Multiplicity and inferential roles.** The sign test and paired sign-flip test are separately defined inferential procedures addressing different questions. The exact two-sided sign test evaluates directional imbalance under a 50/50 sign null, while the exact paired sign-flip randomization test evaluates the magnitude of the paired mean difference under the stated exchangeability-of-signs null. They are not independent replications of a single hypothesis test. Their p-values are therefore reported separately for their respective null hypotheses and are not interpreted as a combined family-wise-error-controlled omnibus result.

The paired percentile bootstrap serves a different purpose: it provides a resampling-based uncertainty interval for the observed mean paired difference. It is not treated as a third hypothesis test.

### Decisive Benchmark Identity

```text
Benchmark implementation:
experiments/higher_order_regime_shift.py

Benchmark version:
v3.0.0

Streams:
20 independent paired streams

Seeds:
0 through 19

Feature dimension:
D = 128

Feature representation:
unary + pairs + triples + lag

Marginal-invariance constraint:
Categorical TVD <= 0.05 vs Phase C
```

**Git Provenance**:
- Benchmark implementation introduced: commit `deda3eaaef1ebdeb3171198bce2f50070e38f1df` (*"experiments: complete higher-order baseline and evaluation-correction gate"*).
- Exact statistical validation & research freeze: commit `0ac487bf5d85be17267bd97133a5bef55c2e4767` (*"release: close research freeze with exact statistical validation"*).

### Scope of Inference

> **The evidence is benchmark-specific. It establishes that, on the specified 20-seed non-stationary matched-marginal benchmark with D=128 and the stated feature representation, DeltaCore Gated did not demonstrate an advantage over regularized Online Covariance, while the observed paired difference favored Online Covariance. The benchmark does not establish universal superiority of Online Covariance, universal inferiority of DeltaCore, or performance ordering under telemetry distributions, dimensions, drift processes, workloads, or deployment conditions not represented by the benchmark. The result supports terminating this specific DeltaCore research direction for RecoveryOS; it does not establish that adaptive telemetry detection as a broader problem has been solved.**

### Historical Metric Isolation

The values 0.9859 and 0.9594 are Phase 16/17 balanced classification accuracy metrics and are not part of the telemetry regime-detection AUROC record.

### Pre-Registered Configuration Invariance

The evaluation parameters are frozen in `experiments/configs/higher_order_gate.yaml`. Do not modify this file during benchmark reproduction.
