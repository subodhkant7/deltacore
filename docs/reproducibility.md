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
Under the predefined 20-seed non-stationary matched-marginal benchmark, regularized Online Covariance significantly outperformed DeltaCore Gated under both the exact paired sign test and exact paired sign-flip randomization test. DeltaCore Gated did not demonstrate a performance advantage over regularized Online Covariance on the decisive benchmark.

**Statistical Caveats**:
- *Bootstrap*: The 95% percentile bootstrap interval is a resampling-based uncertainty estimate; it is not an assumption-free or mathematically exact confidence interval.
- *Randomization Test*: Exact enumeration over all $2^{20} = 1,048,576$ sign assignments removes Monte Carlo approximation error, but the inferential validity of the test remains conditional on the null and exchangeability assumptions specified by the test.
- *Sign Test*: Directional test evaluating consistency with a 50/50 sign null ($p = 0.0118$).

### Scope of Inference

The evidence is benchmark-specific. It establishes that, on the predefined 20-seed non-stationary matched-marginal benchmark with $D=128$ and the specified feature representation, DeltaCore Gated did not demonstrate an advantage over regularized Online Covariance, while the observed paired difference favored Online Covariance.

The benchmark does not establish universal superiority of Online Covariance, universal inferiority of DeltaCore, or performance ordering under telemetry distributions, dimensions, drift processes, workloads, or deployment conditions not represented by the benchmark.

The result therefore supports terminating this specific DeltaCore research direction for RecoveryOS, rather than claiming that adaptive telemetry detection as a broader problem has been solved.

### Pre-Registered Configuration Invariance

The evaluation parameters are frozen in `experiments/configs/higher_order_gate.yaml`. Do not modify this file during benchmark reproduction.
