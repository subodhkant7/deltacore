# Contributing to DeltaCore

DeltaCore is a completed, open-source research archive. While the core algorithm and roadmap are frozen, we welcome bug fixes, documentation corrections, reproducibility enhancements, and independent research experiments.

---

## 1. Prime Directive: Scientific Integrity

DeltaCore operates under strict scientific integrity rules (formalized in [AGENTS.md](file:///Users/urjasoft/Documents/DeltaCore/AGENTS.md)):

1. **No Benchmark Cherry-Picking**: Every evaluation must test across a pre-registered range of random seeds (minimum 10–20 seeds) with matched marginal distributions.
2. **No Silent Mathematical Alterations**: Never silently modify controller equations, stability constraints, learning-rate schedules, or update semantics.
3. **No Unsupported Claims**: Do not use marketing or promotional language ("guaranteed anomaly detection", "optimal", "universal", "superior"). Every empirical claim must be strictly bounded by verified benchmark numbers.
4. **Pre-Specified Criteria**: **New algorithmic claims require reproducible baselines, matched marginal controls, and pre-specified evaluation criteria.** If a baseline matches or outperforms a proposed algorithm, document the finding honestly.

---

## 2. Development Setup

DeltaCore requires Python $\ge 3.10$ and PyTorch $\ge 2.2.0$.

```bash
# Clone the repository
git clone https://github.com/subodhkant7/deltacore.git
cd deltacore

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 3. Quality Standards & Verification Gates

Before submitting any pull request, ensure all verification gates pass:

```bash
# 1. Full test suite (all tests must pass cleanly)
pytest -q

# 2. Type checking (0 errors, 0 warnings)
pyright deltacore tests examples experiments

# 3. Linter and formatting (100% compliance)
ruff check .
ruff format --check .

# 4. Bytecode compilation
python -m compileall deltacore examples tests experiments

# 5. Minimal smoke example
python examples/basic_adaptation.py
```

---

## 4. Proposing New Research Experiments

If you wish to explore new directions (such as low-rank tensor factorization, kernel extensions, or alternative associative rules):
- Do **not** modify existing historical experiment scripts (`experiments/auto_associative_regime_shift.py`, `experiments/drift_then_anomaly.py`, `experiments/hard_regime_shift.py`, `experiments/higher_order_regime_shift.py`).
- Create a new, standalone script in `experiments/` with its own pre-registered configuration and version identifier.
- Include proper second-order and first-order baselines on identical feature representations.
- Report paired difference statistics, bootstrap confidence intervals, and exact sign tests.

---

## 5. Reporting Bugs

To report a bug:
1. Open an issue on GitHub describing the unexpected behavior.
2. Provide a minimal, reproducible example script or unit test that triggers the issue.
3. Include platform details (OS, Python version, PyTorch version, commit hash).
