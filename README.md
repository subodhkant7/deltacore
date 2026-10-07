# DeltaCore: Adaptive Associative State Research

**A modular, research-oriented toolkit for systems that adapt their internal state during inference.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checker: Pyright](https://img.shields.io/badge/type%20checker-pyright-green.svg)](https://github.com/microsoft/pyright)
[![Research Status: Frozen](https://img.shields.io/badge/research%20status-frozen-red.svg)](docs/research_status.md)

---

## Research Status

> **DeltaCore is research software, not a production anomaly-detection or telemetry-control system.**
>
> The current benchmark program did not establish an advantage over standard regularized online covariance methods for the targeted telemetry regime-detection task.
>
> The telemetry integration hypothesis for RecoveryOS has therefore been **rejected and frozen**.
>
> The repository is preserved as a completed, reproducible research implementation.

---

## 1. What It Is

**DeltaCore** is an open-source research implementation of adaptive associative state dynamics. It provides a canonical controller API, deterministic state persistence, feature hashing utilities, and reproducible experiments for studying online adaptation and reconstruction-based diagnostics under distribution shift.

While conventional models maintain fixed weights during forward passes, adaptive state architectures modify an explicit internal memory matrix ($M_t \in \mathbb{R}^{D \times D}$) dynamically at inference time using error-correcting delta updates:

$$M_t = \alpha_t M_{t-1} + \eta_t (x_t - M_{t-1} x_t) x_t^T$$

DeltaCore isolates these dynamics into clean mathematical abstractions with Lyapunov contractive step bounds, non-mutating pre-update residual diagnostics (`score()`), and verifiable state serialization.

---

## 2. The Central Research Question

> *Can test-time adaptation of an associative matrix $M_t$ provide a decisive, statistically reliable advantage over classical first- and second-order statistical baselines when tracking legitimate distribution drift and detecting higher-order operational anomalies?*

This question was evaluated across 4 multi-seed non-stationary benchmarks, culminating in a 20-seed decisive evaluation gate (`experiments/higher_order_regime_shift.py`).

---

## 3. Current Status & RecoveryOS Decision

Following rigorous multi-seed evaluation with matched marginal distributions ($\text{TVD} \le 0.05$):
- **Core Research Status**: **RESEARCH FROZEN** (see [docs/research_status.md](docs/research_status.md)).
- **RecoveryOS Integration**: **PERMANENTLY REJECTED & FROZEN** (see [docs/recoveryos_decision.md](docs/recoveryos_decision.md)).
- **Reason**: The RecoveryOS integration was rejected because DeltaCore did not demonstrate a sufficient empirical advantage over a simpler regularized online covariance baseline on the decisive benchmark, while incurring greater state complexity and latency. DeltaCore is therefore removed as an active RecoveryOS research dependency.
- **Future Direction**: The repository is archived for scientific reproducibility and peer review. Any future work on adaptive telemetry detection must be treated as a new independent research effort with a new hypothesis, benchmark design, and preregistered evaluation.

---

## 4. What This Project Is / Is Not

### This project IS:
- An open-source research implementation of adaptive associative state dynamics.
- A deterministic, reproducible experimental testbed with pure-PyTorch primitives.
- A reference implementation of Lyapunov step-size controllers and score-before-update semantics.
- An educational and scientific resource documenting positive and negative findings with equal rigor.

### This project is NOT:
- A universal anomaly detector.
- A production SRE control plane or telemetry agent.
- A runtime dependency for RecoveryOS.
- A replacement for classical covariance estimation or Principal Component Analysis (PCA).
- A security monitoring or threat detection product.
- An automated manifold learner.

---

## 5. Key Empirical Findings

Across 20 independent paired streams evaluating 5 hard anomaly families with matched marginals ($\text{TVD} \le 0.05$):

| Comparator Metric | Value | Statistical Inference |
| :--- | :---: | :--- |
| **DeltaCore Gated Mean AUROC** | 0.5711 $\pm$ 0.0801 | 95% Bootstrap CI: [0.5365, 0.6071] |
| **Online Covariance Mean AUROC** | **0.5976 $\pm$ 0.0667** | 95% Bootstrap CI: [0.5697, 0.6271] |
| **Mean Paired Difference ($\text{DC} - \text{Cov}$)** | **-0.0265** | 95% Paired Bootstrap CI: **[-0.0478, -0.0072]** (excludes zero) |
| **Exact Binomial Sign Test** | **$p = 0.0118$** | DeltaCore won on 4 of 20 seeds, lost on 16 ($p < 0.05$) |
| **Exact Paired Sign-Flip Randomization Test** | **$p = 0.0188$** | Exhaustive enumeration across all $2^{20} = 1{,}048{,}576$ sign configurations ($p < 0.05$) |
| **DeltaCore vs. Online Centroid (Gated)** | +0.0382 | 95% Bootstrap CI: [+0.0205, +0.0541], Exact Sign Test $p = 0.0004$ |
| **DeltaCore vs. Online PCA (Gated)** | +0.0070 | 95% Bootstrap CI: [-0.0074, +0.0209] (includes zero), $p = 0.8238$ |

### Scientific Conclusions
1. **DeltaCore outperformed first-order online centroids** (+0.0382 AUROC, $p = 0.0004$), demonstrating sensitivity to linear feature correlations that centroid models cannot track.
2. **Under the 20-seed non-stationary matched-marginal benchmark, the observed paired difference favored regularized Online Covariance. The exact two-sided sign test rejected its 50/50 directional null at p = 0.0118, and the exact paired sign-flip randomization test rejected its stated exchangeability-of-signs null at p = 0.0188.** These tests address different inferential questions and are reported separately; the two p-values are not treated as independent confirmations of a single omnibus hypothesis. **DeltaCore Gated did not demonstrate a performance advantage over regularized Online Covariance on the decisive benchmark.**
3. **Representation engineering explained higher-order detection**: Representation tier ablations revealed that gains on multi-token anomalies were driven by the feature hashing layer (pair and triple interaction tokens), which benefited classical covariance and centroid methods equally or more.
4. **Adaptive Anomaly Absorption**: Continuous un-gated adaptation absorbs repeated anomalies within 5–25 steps ($r_{100}/r_1 = 0.0529$, assimilating 94.7% of the anomaly), confirming that score-before-update gating is strictly necessary.

### Scope of Inference
> **The evidence is benchmark-specific. It establishes that, on the specified 20-seed non-stationary matched-marginal benchmark with D=128 and the stated feature representation, DeltaCore Gated did not demonstrate an advantage over regularized Online Covariance, while the observed paired difference favored Online Covariance. The benchmark does not establish universal superiority of Online Covariance, universal inferiority of DeltaCore, or performance ordering under telemetry distributions, dimensions, drift processes, workloads, or deployment conditions not represented by the benchmark. The result supports terminating this specific DeltaCore research direction for RecoveryOS; it does not establish that adaptive telemetry detection as a broader problem has been solved.**

### Multiplicity and Inferential Roles

| Procedure | Question | Null / Framework | Result |
| :--- | :--- | :--- | :--- |
| **Paired percentile bootstrap** | Mean-effect uncertainty | Resampling framework ($B = 10{,}000$, seed 42) | 95% CI `[-0.0478, -0.0072]` |
| **Exact paired sign-flip test** | Paired-effect magnitude | Exchangeability-of-signs null ($2^{20} = 1{,}048{,}576$ assignments) | `p = 0.0188` ($19{,}728$ extreme) |
| **Exact two-sided sign test** | Direction / majority | 50/50 sign null ($H_0: p = 0.5$) | `p = 0.0118` (4 wins / 16 losses) |

Exhaustive enumeration removes Monte Carlo approximation error from the sign-flip calculation; it does not make the inferential procedure assumption-free. The percentile bootstrap interval is a resampling-based uncertainty estimate and is not treated as an exact confidence interval.

The sign test and paired sign-flip test are separately defined inferential procedures addressing different questions. The exact two-sided sign test evaluates directional imbalance under a 50/50 sign null, while the exact paired sign-flip randomization test evaluates the magnitude of the paired mean difference under the stated exchangeability-of-signs null. They are not independent replications of a single hypothesis test. Their p-values are therefore reported separately for their respective null hypotheses and are not interpreted as a combined family-wise-error-controlled omnibus result. The paired percentile bootstrap serves a different purpose: it provides a resampling-based uncertainty interval for the observed mean paired difference. It is not treated as a third hypothesis test.

### Practical vs. Statistical Significance
Within the evaluated design and benchmark, DeltaCore incurred greater state and latency cost without demonstrating a compensating performance advantage over regularized online covariance (state memory: $O(D^2) = 64\text{ KB}$ at $D=128$; median scoring latency: $25.9\text{ µs}$, $8.6\times$ higher latency relative to first-order centroids at $3.0\text{ µs}$).

### Historical Metric Isolation
The values 0.9859 and 0.9594 are Phase 16/17 balanced classification accuracy metrics and are not part of the telemetry regime-detection AUROC record.

---

## 6. Architecture

```text
Structured Telemetry / Record Stream
                 ↓
     Feature Representation
    (DeterministicFeatureHasher)
  - Signed SHA-256 token hashing
  - Pair & triple interaction terms
  - L2 Euclidean normalization
                 ↓
         AdaptiveController
  ├── score(x)  →  Pre-update reconstruction residual r_t = ||x_t - M_{t-1} x_t||_2
  ├── gating    →  Conditional update control (freeze state if r_t > threshold)
  ├── step(x)   →  Safe Lyapunov step update M_t = alpha_t M_{t-1} + eta_t e_t x_t^T
  └── state     →  Atomic persistence with SHA-256 integrity verification
                 ↓
        Research Diagnostics
  - Reconstruction residuals
  - Frobenius matrix norm ||M_t||_F
  - Lyapunov stability safety margin
```

---

## 7. Installation

DeltaCore requires Python $\ge 3.10$ and PyTorch $\ge 2.2.0$.

```bash
# Clone the repository
git clone https://github.com/subodhkant7/deltacore.git
cd deltacore

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

Verify installation:
```bash
python -c "import deltacore; print(deltacore.__name__, deltacore.__version__)"
# Output: deltacore 0.2.0
```

---

## 8. Minimal Example

```python
import torch
from deltacore import (
    AdaptiveController,
    ControllerConfig,
    DeterministicFeatureHasher,
    TelemetryHasherConfig,
)

# 1. Configure deterministic feature hasher with interaction terms
hasher = DeterministicFeatureHasher(
    TelemetryHasherConfig(
        dim=128,
        normalize=True,
        pair_interactions=(("service", "operation"),),
    )
)

# 2. Instantiate AdaptiveController with contractive Lyapunov bounds
config = ControllerConfig(
    dim=128,
    eta0=0.03,         # Base learning rate
    rho=0.95,          # Contractive margin (rho < 2.0)
    alpha_min=0.98,    # Memory retention factor
    gamma=0.10,        # Error dampening factor
)
controller = AdaptiveController(config)

# 3. Process telemetry event
event = {
    "service": "checkout",
    "operation": "submit_order",
    "http_status": 200,
    "latency_ms": 42.5,
}
x_t = hasher.encode(event)

# 4. Score-before-update: evaluate residual without mutating state
pre_score = controller.score(x_t)
print(f"Pre-update reconstruction residual: {pre_score.reconstruction_residual:.4f}")

# 5. Safe state update with contamination gating
threshold = 0.50
should_adapt = pre_score.reconstruction_residual < threshold
step_res = controller.step(x_t, adapt=should_adapt)
print(f"Post-step state norm: {step_res.state_norm:.4f}, safety margin: {step_res.stability_margin:.4f}")
```

Run the verified smoke script:
```bash
python examples/basic_adaptation.py
```

---

## 9. Public API Reference

DeltaCore exposes a strictly typed, small public API:

### `AdaptiveController`
- `AdaptiveController(config: ControllerConfig)`: Initializes controller with square associative matrix $M_0 \in \mathbb{R}^{D \times D}$.
- `score(x: torch.Tensor, v: torch.Tensor | None = None) -> ControllerStepResult`: Evaluates reconstruction residual $r_t = \|v_t - M_{t-1} x_t\|_2$ **strictly non-mutating**.
- `step(x: torch.Tensor, v: torch.Tensor | None = None, adapt: bool = True) -> ControllerStepResult`: Computes prediction, residual, and executes safe Lyapunov state transition if `adapt=True`.
- `reset() -> None`: Restores state to initial scale without memory reallocation.
- `save_state(file_path: str | Path) -> None`: Atomically persists state, config, and SHA-256 checksum to disk.
- `load_state(file_path: str | Path, verify_checksum: bool = True) -> None`: Restores state with strict schema and cryptographic integrity verification.

### `DeterministicFeatureHasher`
- `DeterministicFeatureHasher(config: TelemetryHasherConfig)`: Configures signed SHA-256 hashing.
- `encode(data: Mapping[str, Any]) -> torch.Tensor`: Maps dictionary to deterministic 1D float32 vector with optional L2 normalization.
- `collision_stats() -> CollisionStats`: Inspects hash bucket utilization and collision frequency.

---

## 10. Mathematical Model & Stability Contract

### Update Dynamics
1. **Pre-Update Association**: $\hat{v}_t = M_{t-1} x_t$
2. **Reconstruction Residual**: $e_t = v_t - \hat{v}_t$ (in auto-associative mode, $v_t \equiv x_t$)
3. **Lyapunov Step-Size Projection**:
   $$\eta_t = \min\left(\frac{\eta_0}{1 + \gamma \|e_t\|_2}, \frac{\rho}{\|x_t\|_2^2 + \epsilon}\right)$$
   Guaranteeing local step non-expansion: $|1 - \eta_t \|x_t\|_2^2| \le \max(|1 - \rho|, 1) \le 1$.
4. **State Transition**:
   $$M_t = \alpha_t M_{t-1} + \eta_t (e_t x_t^T)$$

See [docs/mathematical_contract.md](docs/mathematical_contract.md) for complete proofs and tensor specifications.

---

## 11. Reproducing Experiments

To reproduce all benchmarks and regression tests:

```bash
# 1. Run full regression test suite (682 tests)
pytest -q

# 2. Run static type checking and linters
pyright deltacore tests examples experiments
ruff check .
ruff format --check .

# 3. Reproduce the decisive 20-seed Higher-Order Regime Shift benchmark
python experiments/higher_order_regime_shift.py
```

Generated plots and results will be saved to `experiments/artifacts/`. See [docs/reproducibility.md](docs/reproducibility.md) for full reproduction guides.

---

## 12. Limitations

A complete enumeration of empirical boundaries is maintained in [docs/limitations.md](docs/limitations.md):
1. **Quadratic State Memory ($O(D^2)$)**: $4 D^2$ bytes ($64\text{ KB}$ at $D=128$, $1\text{ MB}$ at $D=512$, $4\text{ MB}$ at $D=1024$).
2. **Evaluation Latency**: $25.9\text{ µs}$ median score latency ($8.6\times$ slower than online centroids).
3. **No Demonstrated Advantage Over Online Covariance**: Statistically outperformed by regularized covariance on matched marginal benchmarks.
4. **Representation Dependence**: Higher-order anomaly detection depends primarily on feature interaction hashing.
5. **Operating Threshold Sensitivity**: Fixed calibration thresholds show high sensitivity to residual drift.
6. **Adaptive Anomaly Absorption**: Continuous adaptation without gating assimilates anomalies within 5–25 steps.
7. **No Inherent Temporal Detection**: Event-only representations cannot detect Markov transition anomalies without lag features.
8. **No Proof of Global Boundedness**: Local contractive step bounds do not guarantee global matrix norm boundedness under arbitrary non-orthogonal streams.
9. **Controlled Benchmark Scope**: Tested on synthetic non-stationary streams rather than live distributed infrastructure.
10. **Not Production Software**: Research archive not suitable for live production deployment.

---

## 13. Repository Structure

```text
deltacore/
├── controller.py       # Canonical AdaptiveController implementation
├── memory/             # Associative memory topologies (square, five-memory)
├── telemetry/          # DeterministicFeatureHasher & collision diagnostics
├── updates/            # Hebbian, Delta, and adaptive step operators
├── stability/          # Lyapunov step-size controllers & contractive guards
└── scans/              # Sequential and parallel affine associative scans

docs/
├── research_status.md         # Final research status & scientific conclusion
├── recoveryos_decision.md     # Architectural decision record (REJECTED/FROZEN)
├── experiments.md             # Benchmark history and experimental progression
├── reproducibility.md         # Verified reproduction environment specifications
├── mathematical_contract.md   # Mathematical equations and conceptual disambiguation
├── limitations.md             # The 10 empirical limitations and failure modes
├── release_checklist.md       # Pre-release verification checklist
└── ROADMAP.md                 # Project roadmap archive

examples/
└── basic_adaptation.py        # Minimal runnable demonstration

experiments/
├── configs/                   # Pre-registered benchmark configuration files
├── artifacts/                 # Serialized JSON results and publication figures
├── higher_order_regime_shift.py # Primary decisive 20-seed benchmark (v3.0.0)
├── hard_regime_shift.py       # Hard matched-marginal anomaly benchmark
├── drift_then_anomaly.py      # Non-stationary drift adaptation benchmark
└── auto_associative_regime_shift.py # Initial auto-associative telemetry benchmark

tests/
├── test_adaptive_controller.py       # Controller API and persistence tests
├── test_benchmark_blind_execution.py # Label-blind scoring regression tests
├── test_statistical_testing.py       # Exact sign test and permutation test tests
└── ...                               # 682 comprehensive unit tests
```

---

## 14. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and scientific integrity rules. While the core algorithm is frozen, bug fixes, reproducibility improvements, and independent research experiments are welcome.

---

## 15. License

DeltaCore is open-source software licensed under the [MIT License](LICENSE).
