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
**Expected Output**: `682 passed in ~35s` (0 failures, 0 errors).

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

Reproduce the 20-seed decisive evaluation gate comparing DeltaCore against Online Covariance and 9 other baselines:

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
  - `Mean Paired Difference`: $-0.0265$ (95% Bootstrap CI: $[-0.0478, -0.0072]$)
  - `Exact Binomial Sign Test`: $p = 0.0118$ (DeltaCore wins 4/20 seeds)
  - `Exact Paired Permutation Test`: $p = 0.0188$ (evaluated over all $2^{20} = 1{,}048{,}576$ sign configurations)
  - `Final Verdict`: `FAIL`

### Pre-Registered Configuration Invariance

The evaluation parameters are frozen in `experiments/configs/higher_order_gate.yaml`. Do not modify this file during benchmark reproduction.
