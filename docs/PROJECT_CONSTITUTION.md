# DeltaCore Project Constitution

This document establishes the binding philosophical and engineering principles governing the **DeltaCore** research codebase. Every contribution, module design, experiment, and pull request must adhere to these tenets.

---

## Principle A — Research Correctness

Scientific integrity precedes implementation speed or performance metrics. Every mathematical mechanism and formulation implemented within DeltaCore must maintain unambiguous distinction across four epistemic levels:

1. **What the source literature claims**: Theoretical assumptions, published bounds, or empirical assertions documented in foundational papers (e.g., HOPE, nested learning, test-time training, VisionHOPE).
2. **What DeltaCore implements**: The exact mathematical and algorithmic instantiation realized in code, including discrete approximations, tensor shapes, and boundary handling.
3. **What DeltaCore experimentally verifies**: Concrete empirical evidence gathered under controlled, reproducible conditions within DeltaCore test suites and benchmarks.
4. **What remains a hypothesis**: Unverified conjectures regarding convergence, capacity, scaling behavior, or stability under distribution shifts.

> **Rule**: Never present an empirical finding or implementation artifact as a mathematical theorem. Document all deviations from canonical formulations in module docstrings and research notes.

---

## Principle B — Modular Research Primitives

DeltaCore investigates adaptive neural systems whose internal states evolve during inference. To prevent conflation of distinct computational dynamics, the architecture strictly separates the following primitives:

- **Representation**: Maps inputs into query, key, value, and feature manifolds.
- **Memory**: Persistent or recurrent state structures holding learned or adapted associative representations.
- **Update Rule**: Mathematical operators governing step-wise modifications to the memory state (e.g., Hebbian, delta rule, meta-gradient steps).
- **Retention**: Mechanisms regulating memory decay, forgetting rates, leaky integration, or gating factors over sequence steps.
- **Adaptive Dynamics**: Feedback loops, self-referential modulation, or state-conditioned parameter adaptation.
- **Stability Control**: Active bounded norm projections, spectral constraints, gradient/update clipping, and regularization operators.
- **Scan / Execution Strategy**: Algorithmic scheduling of updates (e.g., sequential token-by-token recurrence, parallel chunked associative scans).
- **Diagnostics**: Non-intrusive runtime telemetry extracting state trajectories, eigenvalues, and gradient metrics.
- **Benchmarks**: Controlled synthetic and real-world evaluation workloads evaluating associative capacity, recall, generalization, and stability.

> **Rule**: These primitives must never be fused into an opaque, monolithic architecture. Each component must be independently instantiated, swapped, tested, and benchmarked.

---

## Principle C — Stability is First-Class

In systems where parameters or hidden states adapt during inference, stability failure is not an edge case—it is a fundamental mode of failure. DeltaCore treats stability as an active research target and first-class metric rather than an afterthought.

The architecture must provide observable telemetry and testing hooks for:
- **State Norm**: Frobenius norm, trace, and element-wise bounds of the evolving memory matrix over time.
- **Update Magnitude**: $\|\Delta M_t\|$ relative to current state $\|M_t\|$.
- **Effective Step Size**: Dynamic learning rates or modulation scales driving state transitions.
- **Retention Dynamics**: Decay trajectory and eigenvalue contraction of transition operators.
- **Spectral / Operator Norm**: Leading singular values and spectral radius where mathematically applicable.
- **Numerical Stability**: Condition number tracking, precision edge conditions, and sub-normal floating point behavior.
- **NaN / Inf Detection**: Immediate interception and diagnostic dumping at the exact step and operation where non-finite values manifest.
- **Adaptation Trajectory**: Phase-space trajectories of internal state under repeated or adversarial inputs.

> **Rule**: No module or update rule may be described as "stable" unless supported by automated, deterministic numerical verification under stress tests.

---

## Principle D — Reproducibility

Research code that cannot be independently reproduced is unscientific. Every benchmark run, convergence test, and empirical claim within DeltaCore must log a complete experimental manifest:

- **Randomness**: Explicit PRNG seeds for global environments, PyTorch CPU/CUDA, and data generators.
- **Model Configuration**: Fully serialized dataclass or JSON config capturing all hyper-parameters without hidden defaults.
- **Dataset / Task**: Deterministic generation scripts, data checksums, split definitions, and sequence lengths.
- **Update / Optimizer Configuration**: Explicit schedules, learning rates, decay terms, and clipping thresholds.
- **Hardware Profile**: Microarchitecture, accelerator model, compute precision (FP32, FP16, BF16), and memory environment.
- **Software Dependencies**: Exact package versions (Python, PyTorch, NumPy, OS runtime).
- **Result Artifacts**: Raw metrics, trajectory logs, and configuration snapshots stored in parseable formats (JSON, CSV).

> **Rule**: Any benchmark without a complete, reproducible configuration manifest is considered invalid and must not be accepted into the repository.

---

## Principle E — No Fake Benchmarks

Evaluation in DeltaCore must reflect actual numerical computations and genuine experimental dynamics.

- **Zero Hardcoding**: Never write a benchmark whose outputs, scores, or convergence curves are hardcoded or fabricated to meet an expected outcome.
- **Genuine Divergence**: Failure cases, numerical overflows, catastrophic forgetting, and divergence modes must be reproduced as genuine numerical experiments running against real models.
- **Fair Comparisons**: Baselines must be run under equivalent compute, parameter, and information budgets.

---

## Principle F — Minimal Dependency Surface

To maximize portability, debuggability, and long-term research longevity, DeltaCore enforces strict dependency hygiene:

- **Core Runtime**: Standard Python (>=3.10), PyTorch, NumPy.
- **Testing & Tooling**: Pytest, Ruff.
- **Dependency Veto**: Do not add heavy machine learning frameworks, distributed orchestration packages, or domain-specific libraries into the core package unless a concrete, approved research phase establishes an indispensable need.
- **Isolation**: Hardware-specific kernels (e.g., custom CUDA, Triton) or specialized dataset loaders must remain optional plugins or isolated backends, preserving an accessible pure-PyTorch CPU reference implementation at all times.
