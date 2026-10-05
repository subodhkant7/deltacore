# DeltaCore Phase 18 Pre-Flight Record: Phase 17 Freeze

This document freezes Phase 17 artifacts, baselines, configurations, and environment state prior to commencing **Phase 18: Unseen Classification Regime Transfer & Falsification**.

---

## 1. Environment & Base Commit

- **Base Git Commit**: `bdd0efa279e901e26542bc4eb5b2dcca0bc8f86f`
- **Python Version**: `3.13.0 (v3.13.0:60403a5409f, Oct  7 2024, 00:37:40)`
- **PyTorch Version**: `2.6.0`
- **NumPy Version**: `2.5.3`
- **Package Version**: `0.1.0` (from `pyproject.toml`)
- **Pytest Status**: `626 passed in 40.89s` (100% passing across 44 test suites)
- **Pyright Status**: `0 errors, 0 warnings, 0 informations`
- **Ruff Linter Status**: `All checks passed!`
- **Ruff Formatter Status**: `176 files already formatted`

---

## 2. Phase 17 Implementation Hashes (SHA-256)

| Component / File | File Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **Classification Models** | `deltacore/classification/models.py` | `18117e2294c1f5f0e28b7fe0f8800193fceab5718581a56c328b0ed9e499b802` |
| **Synthetic Stream Generator** | `deltacore/classification/synthetic_stream.py` | `41b507585c06ac295a534977bfd1951887820499c698b59024c369e9cd05866c` |
| **Classification Metrics** | `deltacore/classification/metrics.py` | `849ecdd51a11ccac870c68e730c93dc42022661c5d36272048a2995288b0f7fe` |
| **Phase 17 Benchmark Runner** | `deltacore/classification/phase_17_runner.py` | `5b11c25c971923579891b9f2e4a175f33c8fb5e366a186c4a610bfcf087bb768` |
| **Phase 17 Observatory Plots** | `deltacore/observatory/phase_17_plots.py` | `67efa8857ac7616cf2d1b56808afa45076fb6495dec8730ba708f25e92c3bfe8` |
| **Benchmark Entrypoint** | `examples/phase_17_classification_benchmark.py` | `efb5ceefbbfece838276f82798e4d9c792aa0c4632837bc2303c72b8344e24eb` |
| **Classification Unit Tests** | `tests/test_phase_17_classification.py` | `118d09618bf77085a56d1efea703fdbbdf3b8fe709a36f7ae9f5e3ba7d4db1c2` |

---

## 3. Phase 17 Serialized Artifact Hashes (SHA-256)

| Artifact | File Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **Aggregate Results** | `docs/benchmarks/artifacts/phase_17/phase_17_results.json` | `45cfcbee956482366abf6c6ce8a7331e3139982a4ee7b82e3f8066bfefafa871` |
| **Benchmark Configuration** | `docs/benchmarks/artifacts/phase_17/phase_17_config.json` | `2c05985a3992fcc7356687ff6c10c772a3ebe33d4e2bb71af7890b3d180edc71` |
| **Per-Seed Detailed Results** | `docs/benchmarks/artifacts/phase_17/phase_17_per_seed.json` | `4f9a543c5e80994da8ee0de67f33e91c9e579bf03155b48914a390e74dcb3b8a` |
| **Shift Severity Results** | `docs/benchmarks/artifacts/phase_17/phase_17_shift_results.json` | `2319ee764d4b426b8d300f668c2f264988c0f271fc8b2168fc22e4a4bc9d7767` |
| **Retention Stress Results** | `docs/benchmarks/artifacts/phase_17/phase_17_retention.json` | `a5b1b4fda011abba43f01f6ed8bc290bb2e890841f6ee40f3740b7cef43abec1` |
| **Dimensional Scaling Results** | `docs/benchmarks/artifacts/phase_17/phase_17_scaling.json` | `4d099a0a6251ef49f153eeadc20f74a3f1167d8fe54da428743647004aebf306` |

---

## 4. Phase 17 Mechanism & Hyperparameter Summary

The exact frozen baseline mechanism inherited for Phase 18 evaluation:

- **Predictor**:
  $$\hat{y}_t = \arg\max_k \left[ W_{\mathrm{head}} (I + M_t) x_t + b_{\mathrm{head}} \right]_k$$
- **Error Backprojection**:
  $$e_{\mathrm{logit}, t} = y_{\mathrm{onehot}, t} - \hat{p}_t, \quad e_{x, t} = W_{\mathrm{head}}^\top e_{\mathrm{logit}, t}$$
- **Contraction Step & Retention**:
  $$\eta_t = \min\left(\frac{\eta_0}{1 + \gamma \|e_{x, t}\|_2}, \frac{\rho}{\|x_t\|_2^2 + \epsilon}\right), \quad \alpha_t = \max(\alpha_{\min}, 1 - \eta_t \|x_t\|_2^2)$$
- **Associative State Update**:
  $$M_{t+1} = \alpha_t M_t + \eta_t e_{x, t} x_t^\top$$
- **Frozen Hyperparameters (Pooled Configuration)**:
  - $\eta_0 = 0.015$ (or nominal development $\eta_0 = 0.015$ to be held constant across all Phase 18 task families without retuning)
  - $\rho = 1.50$
  - $\alpha_{\min} = 0.95$
  - $\gamma = 0.10$
  - $\epsilon = 10^{-6}$

---

## 5. Phase 18 Research Commitment

Phase 18 is an adversarial scientific falsification phase.
No task-specific retuning of $\eta_0, \rho, \alpha_{\min}$ is permitted.
No architecture expansions (no five-memory mechanisms, attention, transformers, or test-time backprop) will be introduced.
Negative results and task failures will be documented explicitly as empirical falsification boundaries.
