# DeltaCore

**Primitives and benchmarks for adaptive neural state.**

> **Notice**: DeltaCore is early-stage research software under active foundational design. It is not intended for production deployments, and no claims of state-of-the-art results, superiority over existing methods, or full reproduction of published empirical benchmarks (such as VisionHOPE) are made at this stage.

---

## Overview

DeltaCore is a modular, research-oriented toolkit designed to study neural architectures whose internal state adapts during inference. While conventional models typically maintain static weights and fixed recurrent mechanisms during forward passes, adaptive state systems—such as those inspired by associative memories, nested learning, delta-rule updates, and test-time training—modify or update their internal representations dynamically as they process data.

DeltaCore isolates these mechanisms into clean mathematical abstractions, providing deterministic testbeds, stability diagnostics, and reproducible benchmarks.

---

## Why DeltaCore?

Many emerging architectures fuse sequence modeling, associative memory storage, update transitions, and stability hacks into single monolithic layers or complex hardware kernels. This tight coupling obscures fundamental research questions:
- *Does an architectural improvement stem from feature representation capacity or from the error-correcting delta rule?*
- *How does memory retention decay behave over ultra-long sequences under continuous non-stationary distribution shifts?*
- *When dynamic updates are iterated recurrently, what exact conditions guarantee numerical stability and prevent state divergence?*

DeltaCore separates these dynamics into independent, swappable components, allowing researchers to study, modify, and benchmark adaptive state transitions in isolation.

---

## Core Concepts

DeltaCore models adaptive computation through decoupled functional primitives:

- **Representation**: Maps inputs into query ($q$), key ($k$), and value ($v$) spaces.
- **Memory State**: An explicit internal state tensor ($M_t$) representing learned or recalled associative associations.
- **Update Rule**: The operator determining state transitions (e.g., Hebbian outer product vs. error-correcting delta updates $\Delta M_t = (v_t - M_{t-1} k_t) k_t^T$).
- **Retention & Decay**: Time-dependent or input-gated decay factors regulating memory longevity without conflating update magnitude.
- **Adaptive Dynamics**: Modulation mechanisms (e.g., dynamic learning rates, self-referential gating) governing how the state informs its own evolution.
- **Stability Controllers**: Bounded norm projections, spectral radius constraints, and numerical recovery guards that guarantee numerical stability independently of the update rule.

---

## Planned Components

```
deltacore/
├── memory/       # Associative memory topologies & state containers
├── updates/      # Hebbian, delta-rule, & adaptive update operators
├── stability/    # Norm bounding, spectral constraints, & numerical guards
├── scans/        # Sequential recurrence & associative chunked scans
├── diagnostics/  # Telemetry for state norms, spectra, and update dynamics
└── benchmarks/   # Reproducible synthetic capacity & recall evaluation
```

---

## Research Questions

DeltaCore is constructed to systematically investigate questions including:

1. **Error-Correction Dynamics**: Under what conditions does the error-correcting delta rule surpass standard linear Hebbian accumulation in associative recall?
2. **Long-Horizon Stability**: How can recurrent test-time state modifications be mathematically bounded to avoid exponential norm growth over sequences exceeding $10^5$ steps?
3. **Adaptive Modulation**: Does self-referential or state-conditioned step sizing prevent memory saturation under non-stationary distributions?
4. **Execution Trade-offs**: What are the numerical precision and memory footprints of sequential unrolling versus parallel chunked associative scans?

---

## Current Status

- [x] **Phase 0 — Repository Constitution & Research Foundation**: Completed. Foundations, contracts, architecture, and minimal infrastructure verified.
- [x] **Phase 1 — Mathematical Reference Primitives**: Completed. Pure-PyTorch reference implementations for `AssociativeMemory`, `read`, `HebbianRule`, `DeltaRule`, and `sequential_scan` verified with deterministic tests, analytical error contraction proofs, and autograd gradient checks.
- [x] **Phase 2 — Adaptive Update Dynamics**: Completed. Adaptive step-size controllers (`ConstantStepSize`, `InputConditionedStepSize`, `ErrorConditionedStepSize`, `StateConditionedStepSize`), `AdaptiveDeltaRule`, sequential adaptive scan, autograd gradient propagation, and distribution shift benchmark.
- [x] **Phase 3 — Minimal Self-Referential Adaptive Memory**: Completed. Coupled two-memory architecture (`AssociativeMemory` and `DynamicsMemory`), online evolution of controller state ($C_{t+1} \neq C_t$) via delta updates, operational recovery latency quantification, autograd gradient propagation via FP64 `gradcheck`, and empirical distribution shift comparison.
- [x] **Phase 4 — Stability Controllers & Mathematical Guarantees**: Completed. Mathematical derivations for content and dynamics memory local non-expansion ($0 \le \eta_t \|k_t\|^2 \le 2$ and $0 \le \rho_t \|z_t\|^2 \le 2$), modular stability controllers (`SafeStepSizeController`, `SafeDynamicsRateController`), boundary regimes in FP32/FP64, autograd compatibility, and adversarial stress benchmarks demonstrating prevention of runaway explosive regimes.
- [x] **Phase 5 — Sequential & Chunked Scans**: Completed. Associative affine state scan monoid, algebraic composition law, exact sequential/chunk equivalence across arbitrary and non-divisible chunk sizes, chunk boundary API (`run_chunk`), autograd gradient equivalence via FP64 `gradcheck`, and empirical runtime/precision profiling.
- [x] **Phase 6 — Benchmark Framework**: Completed. Standardized reproducible benchmark suite, typed configuration and result schemas, 5 unified baselines, 6 synthetic tasks (stationary recall, distribution shift, key interference, conflicting targets, stability stress, adaptation budget), multi-seed statistical aggregation, and benchmark CLI.
- [x] **Phase 7 — Adaptive State Observatory**: Completed. Standard trajectory schema, schema-validating loaders, non-causal event extraction, temporal and comparative analysis, multi-axial state-transition fingerprints, 9 static publication-quality plots (Plots A-I), deterministic replay engine, and reproducible Markdown reports with CLI integration (`inspect`, `plot`, `replay`).
- [x] **Phase 8 — Five-Memory Self-Modifying Reference Core**: Completed. Five co-evolving associative memories ($M_{\text{content}}, M_{\text{key}}, M_{\text{val}}, m_{\eta}, m_{\alpha}$), coupled scalar dynamics, paper-conformance audit against VisionHOPE, and complete autograd differentiability.
- [x] **Phase 9 — Generic 2D Spatial Routing & Boundary-Chunk Execution**: Completed. Exact 2D route serialization and restoration, independent directional five-memory states, boundary-refresh chunking, route fusion, and spatial diagnostic plots (J–N).
- [x] **Phase 9.1 — Spatial Learning Validation Audit**: Completed. Forensic benchmark audit revealing untrained zero-predictor, new Task A (spatial shift) and Task B (pattern classification) benchmarks, epistemic boundary documentation.
- [x] **Phase 10 — Streaming Adaptive-State Benchmark**: Completed. Evaluated online adaptation, recovery, retention, forgetting, and stability across 10 models and 5 seeds on multivariate streaming tasks ($D=8$, $T \in \{128, 512, 1024\}$) under strict parameter-matched regimes (~100p). Verified primary causal control (`adaptive_state = ON` vs `OFF`), isolated catastrophic divergence under abrupt covariance shifts, proved local safe-step contractive bounds prevent divergence, identified stability-plasticity tradeoffs, and integrated Observatory Plots AD through AM. Phase Gate resolved to Outcome B (Mixed Evidence).
- [x] **Phase 11 — Nonlinear Adaptive State & Selective Retention**: Completed. Evaluated nonlinear regime dynamics, stale-memory penalties ($A \to B \to A$), selective retention gating ($\alpha_t$), and dimensional scaling ($D \in \{8, 16, 32, 64, 128\}$). Proved adaptive retention eliminates $99.8\%$ of stale-regime negative transfer ($+0.0489 \to +0.0001$) and prevents catastrophic forgetting ($+0.0049$ vs $+0.7346$ for Nonlinear RLS). Demonstrated that at $D=128$, DeltaCore outperforms classical RLS ($0.9729$ vs $1.0175$ and $1.1093$) while running $2.2\times$ faster with $6\times$ less state memory. Integrated Observatory Plots AN through AW. Phase Gate resolved to Outcome B (Specific Niche: Selective Retention & High-Dimensional Scaling).
- [x] **Phase 12 — Controlled Spatio-Temporal Adaptive State**: Completed (Corrected in Phase 12.1). Evaluated controlled spatio-temporal dynamics ($X_t \in \mathbb{R}^{H \times W \times C}$ flattened to $D \in \{64, 128, 256\}$) across regime transitions ($A \to B \to C \to A$), stale-memory challenges ($A_1 \to B \to A_2$), and shift frequency variations (64, 128, 256 steps). Observed coordinate-permutation covariance ($\Delta_{\mathrm{perm}} \approx 0$, confirming estimator does not exploit 2D locality). Established that DeltaCore matches the tested OnlineRidge implementation within the observed error difference at $D=256$ with lower measured persistent state and latency relative to the tested RLS implementation.
- [x] **Phase 12.1 — Forensic Scientific & Code Integrity Correction**: Completed. Executed multi-tier code health audit (all 554 tests passing, compileall clean, ruff check/format clean). Diagnosed Phase 12 selective-state implementation: uncovered that Phase 12 evaluated an untrained 5-parameter controller with an 8D bottleneck ($E_{\mathrm{rel}} \approx 0.9955$, `NO_USEFUL_STATE`) where purported negative transfer elimination was roundoff noise at the float32 precision floor. Implemented differentiable recurrent training of the 5-parameter controller ($\Delta\theta = 4.029$, $\nabla_\theta \neq 0$) under full-rank representation ($D=64$, $E_{\mathrm{rel}} = 0.9507$, `USEFUL_STATE`). Established that the trained controller converges to static high retention ($\alpha \approx 0.992$), matching fixed high retention ($0.95070$ vs $0.95068$), while retention deltas across regime switches ($-2.12 \times 10^{-5}$) remain below the benchmark noise floor. Hypotheses H12.2 and H12.4 corrected to `INCONCLUSIVE`. Phase Gate resolved to **Outcome R2** (Retention works but controller is not essential; simplify architecture).
- [x] **Phase 12.2 — IDE Diagnostics, Python Environment & Packaging Integrity Audit**: Completed. Executed multi-interpreter probe across 6 host environments; verified `torch>=2.2.0` is declared in `pyproject.toml` (`DEPENDENCIES_COMPLETE`). Verified 100% of DeltaCore modules import cleanly (84/84 modules). Categorized 82 mypy diagnostics across 16 legacy source files. Isolated that IDE `Cannot find module "torch"` errors were caused by language server falling back to unconfigured host interpreters without PyTorch. Established canonical `.venv` and declarative Pyright configuration (`pyproject.toml`, `pyrightconfig.json`). Added packaging regression tests; 558 tests passing cleanly. Phase Gate resolved to **Outcome A** (Environment / IDE Language-Server Resolution Problem).
- [x] **Phase 12.3 — Residual Pyright Diagnostic & Test-Type Cleanup**: Completed. Resolved all 55 residual static typing diagnostics in the `tests/` tree following Phase 12.2 environment alignment, including the representative `"**" is not supported between "AssociativeMemory" and "Literal[2]"` error. Preserved exact mathematical intent (squared Frobenius norm loss for autograd backpropagation) without adding `# type: ignore` or broad `Any` suppressions. Parameterized `DeltaStepResult`, `AdaptiveDeltaStepResult`, and `HebbianStepResult` with `Generic[MemoryT]`, declared explicit `@overload` signatures for `step()` and `update()`, extracted canonical properties on scan results (`scan_res.predictions`), switched observatory plotting arguments from invariant `dict` to covariant `Mapping`, and updated `StreamData.metadata` to `dict[str, Any]`. Reduced `pyright tests` to **0 errors, 0 warnings, 0 informations**. 558/558 tests passing cleanly. Phase Gate resolved to **Outcome A** (Clean - Zero Genuine Source or Test Type Errors).
- [x] **Phase 13 — Real-World Spatio-Temporal Adaptive State Benchmark**: Completed. Evaluated `SafeAdaptiveDelta` on the real-world NOAA OISST v2.1 Equatorial Pacific SST waveguide ($5^\circ\text{S}-5^\circ\text{N}, 170^\circ\text{W}-120^\circ\text{W}$, 360 bi-weekly steps) during the 2015–2016 Super El Niño warming surge. Verified zero parameter test-time updates ($\Delta \theta = 0$ via SHA256 parameter hashes) and strict temporal split boundaries. Proved `SafeAdaptiveDelta` achieves lowest mean tracking error ($E_{\text{rel}} = 0.2582$, $E_{\text{shift}} = 0.2485$) over `Persistence` ($0.2902$), `FrozenLinear` ($0.2749$), `OnlineRidge` ($0.3003$), `GRU` ($0.3764$), and `SpatialConv` ($0.3285$) with 5/5 seed wins. Confirmed dynamic contractive bounding $\eta_t \le 1.9/\|x_t\|_2^2$ prevents catastrophic divergence at $D=256$ ($E_{\text{rel}} = 0.6428$ vs $2.85 \times 10^{14}$ for `FixedDelta`). Demonstrated persistent adaptive state reduces cumulative excess adaptation error during El Niño by $26\%$ ($25.96\%$ unrounded) relative to a state reset. Spatial permutation controls confirmed coordinate permutation equivariance ($E_{\text{equiv}} \approx 7.73 \times 10^{-8}$), establishing DeltaCore as a permutation-equivariant temporal feature estimator. Generated Observatory Plots BK through BU; test suite expanded to 571 tests passing cleanly. Phase Gate resolved to **Outcome A** (Real-world adaptive-state evidence).
- [x] **Phase 13.1 — Real-World Benchmark Consistency & Scientific Claim Audit**: Completed. Executed comprehensive forensic audit across numerical tables, per-seed win rates, baseline offline training provenance, permutation equivariance proofs, stability assertions, and temperature anomaly terminology. Reconciled Table 1 vs Table 4 measurement scopes (5-seed aggregation vs single-seed scaling). Verified offline loss reductions for `SpatialConv` ($-69.2\%$), `GRU` ($-60.2\%$), and `FrozenLinear` ($-58.5\%$) with test-time parameter immutability. Verified exact permutation equivariance ($E_{\text{equiv}} \approx 7.73 \times 10^{-8}$ for DeltaCore vs $1.95$ for `SpatialConv`). Corrected H13.7 to `NOT SUPPORTED` (permutation degradation does not establish task advantage). Qualified stability language to local step-size contractivity condition ($|1 - \eta_t \|x_t\|_2^2| < 1$). Recomputed unrounded $25.96\%$ cumulative shift reduction. 571/571 tests passing cleanly, 0 Pyright errors, 0 Ruff errors. Phase Gate resolved to **Outcome A** (Clean empirical validation).
- [x] **Phase 14 — Independent Real-World Domain Replication**: Completed. Evaluated cross-domain transfer of the smallest successful Phase 13 mechanism (`SafeAdaptiveDelta`, frozen config) on a second, independently sourced physical domain: ECMWF ERA5 2-meter air temperature ($T_{2m}$) over the North Atlantic and European Storm Track sector ($40^\circ\text{N}-65^\circ\text{N}, 30^\circ\text{W}-20^\circ\text{E}$, 360 synoptic 6-hourly intervals) during the January–February 2021 Sudden Stratospheric Warming and European Arctic Polar Outbreak (Storm Filomena). Confirmed that dynamic contractive safety bounds replicate, preventing catastrophic numerical explosion to `NaN` at $D=256$ ($E_{\text{rel}} = 0.2604$). Confirmed state retention advantage replicates, reducing cumulative excess shift error by $-0.8848$ ($-97.1\%$) and eliminating the 1-step re-acquisition lag. Confirmed $50\%$ persistent state memory savings ($16\text{ KB}$ vs $32\text{ KB}$ at $D=64$, $256\text{ KB}$ vs $512\text{ KB}$ at $D=256$) and exact permutation equivariance ($E_{\text{equiv}} \le 10^{-7}$). Disclosed that baseline prediction error superiority does not replicate on fast synoptic advection ($OnlineRidge$ achieves $0.0923$ vs $SafeAdaptiveDelta$ $0.1622$). Generated Observatory Plots BV through CF; test suite expanded to 588 tests passing cleanly across the repository. Phase Gate resolved to **Outcome B** (Partial Replication).
- [x] **Phase 15 — Adaptive Regime Transfer & Robustness**: Completed. Addressed the Phase 13–14 domain divergence across slow thermal diffusion (NOAA OISST SST) and fast synoptic advection (ECMWF ERA5 $T_{2m}$). Proved that one validation-selected configuration ($\eta_0 = 0.015, \rho = 1.5, \alpha_{\min} = 0.95$) transfers across both physical regimes without divergence ($E_{\mathrm{rel}} = 0.2931$ on Domain A, $E_{\mathrm{rel}} = 0.1621$ on Domain B). Identified the physical mechanism: thermal diffusion demands conservative step sizing ($\eta_0 \le 0.008$) to avoid noise over-adaptation, while atmospheric advection requires faster updates ($\eta_0 \ge 0.015$) to track moving fronts. In high-dimensional scaling ($D=256$), unconstrained `FixedDelta` suffered fatal numerical overflow (`NaN` at $t=41$ on ERA5), while `SafeAdaptiveDelta` dynamically contracted step sizes ($\eta_t \le 1.9/\|x_t\|^2$), enforcing safety margin $\ge 0.1000$ and zero divergence. Confirmed persistent state memory advantage ($2\times$ less memory than OnlineRidge) and demonstrated positive state transfer during initialization ($0.1742 \to 0.1632$). Generated Observatory Plots CG through CP; test suite expanded to 596 tests passing cleanly across the repository. Phase Gate resolved to **Outcome A** (Robust Transfer; proceed to Phase 16).
- [x] **Phase 16 — Unseen Shift Robustness & Adaptive Safety**: Completed. Challenged the pooled `SafeAdaptiveDelta` configuration with 5 unseen distribution shifts (Mean, Variance, Temporal Speed, Noise, Combined) across 3 severities on both NOAA OISST and ECMWF ERA5 without test-time tuning. Continuous state outperformed state reset on all within-domain shifts ($\Delta_{\mathrm{cum}} < 0$), while state reset prevented negative transfer under severe cross-regime mismatches. Mapped the $D=256$ empirical failure boundary: `FixedDelta` exploded to `NaN` at $\eta \ge 0.005$ on Domain B, while `SafeAdaptiveDelta` remained 100% finite across all tested step sizes ($\eta \in [0.002, 0.025]$), preserving safety margin $\ge 0.50$. Generated Observatory Plots CQ through DB; 607 tests passing cleanly. Phase Gate resolved to **Outcome A** (Robust unseen-shift behavior; proceed to Phase 17).
- [x] **Phase 17 — Online Non-Stationary Classification**: Completed. Evaluated generalization of DeltaCore's adaptive-state principles to online multiclass classification ($K=6, D \in \{32, 64, 128, 256\}$) under strict parameter immutability ($\Delta\theta = 0$, verified via SHA-256 hashes) and no label lookahead. Established that internal associative memory ($z_t = (I + M_t) x_t$) counter-rotates and normalizes non-stationary distributions via backprojected logit errors ($e_{x, t} = W_{\mathrm{head}}^\top e_{\mathrm{logit}, t}$). Under decision-boundary hyperplane rotation ($A \to C \to A$), SafeAdaptiveDelta adapted online to achieve $0.958 \pm 0.005$ accuracy while the static frozen linear classifier collapsed to $0.815 \pm 0.023$ (and $0.686$ under severe shift), reducing cumulative excess loss by $85\%$. Causal ablation ($M_t \equiv 0$) confirmed that $100\%$ of adaptation gains derive from internal state evolution. All robustness controls verified: label shuffle reduced performance to exact chance ($16.7\%$), and feature permutation preserved coordinate equivariance. Dimensional scaling verified zero divergence and sub-12-$\mu s$ per-sample latency across $D \le 256$. Generated Observatory Plots DC through DP; test suite expanded to 626 tests passing cleanly. Phase Gate resolved to **Outcome A** (Task-transfer evidence; proceed to Phase 18).
- [x] **Phase 18 — Unseen Classification Regime Transfer & Falsification**: Completed. Evaluated whether Phase 17's online classification gains transfer across unseen non-stationary classification environments without task-specific retuning, using the frozen minimal `SafeAdaptiveDelta` associative-state mechanism ($\eta_0=0.015, \rho=1.50, \alpha_{\min}=0.95$). Evaluated three distinct task families (Family A rotation, Family B boundary translation / intercept shift, Family C nonlinear boundary deformation) across six shift dimensions (Covariate, Boundary, Prior imbalance, Gradual drift, Abrupt, and Adversarial mismatch) and dimensions $D \in \{32, 64, 128, 256\}$. Demonstrated reproducible positive transfer gains over FrozenLinear on Family B (+9.8 percentage points, $92.2\%$ vs $82.4\%$) and Family C (+9.9 percentage points, $87.2\%$ vs $77.3\%$) with zero per-task retuning and bit-for-bit parameter immutability ($\Delta\theta = 0$). The StateOff intervention ($M_t \equiv 0$) isolated the evolving associative state as the operative difference between adaptive and frozen conditions. Rigorously characterized explicit failure modes: continuous persistent state suffered a $-1.75$ percentage point stale state penalty under adversarial strong mismatch compared to oracle reset (and $-10.0$ percentage points in immediate post-shift accuracy), and the tested linear-associative formulation showed an empirical performance limitation on non-planar boundary deformations. Predefined shortcut and negative-control suite passed (class energy matched, label shuffle collapsed to chance $18.2\%$, coordinate permutation invariant). Maintained exact $4D^2$-byte persistent state memory footprint with zero numerical divergence across tested dimensions $D=32–256$. Generated Observatory Plots 1 through 18; test suite expanded to 645+ tests passing cleanly. Phase Gate resolved to **Outcome A** (Multi-family Transfer Evidence within the tested linear-associative envelope).

### Scientific Status: Phase 18 Classification Transfer

#### Established
- **Multi-family tested transfer**: Under a frozen configuration with no task-specific retuning ($\eta_0=0.015, \rho=1.50, \alpha_{\min}=0.95$, consistent across five seeds), the associative-state mechanism demonstrated reproducible transfer gains across three distinct non-stationary classification environments (Family A rotation $+5.6$ percentage points, Family B translation $+9.8$ percentage points, Family C nonlinear deformation $+9.9$ percentage points).
- **Online adaptive associative state**: Adaptation operates via an explicit $D \times D$ associative memory $z_t = (I + M_t) x_t$ driven by backprojected logit prediction error without modifying pre-trained network parameters ($\Delta\theta = 0$).
- **Parameter immutability**: Verified bit-for-bit ($\Delta\theta = 0$) across all streaming evaluations via pre- and post-stream cryptographic SHA-256 parameter hashes.
- **Causal state provenance**: The StateOff intervention ($M_t \equiv 0$) removes the measured adaptation gains across the primary transfer families, isolating the evolving associative state as the operative mechanism.
- **Empirical stability within tested dimensions**: No numerical divergence was observed under the frozen configuration across $D \in \{32, 64, 128, 256\}$, maintaining an exact $4D^2$-byte FP32 persistent state memory representation.
- **Identified falsification boundaries**: Characterized explicit empirical failure modes, including a $-1.75$ percentage point stale state penalty (and $-10.0$ percentage points in immediate post-shift accuracy) under adversarial strong mismatch, an empirical performance limitation on non-planar boundary deformation, and sub-optimal performance relative to online logistic regression on class-prior shifts.

#### What is NOT Established
- **No universal superiority**: SafeAdaptiveDelta does not outperform all online learning algorithms across all shifts; direct-parameter models (such as OnlineLogisticRegression) outperformed associative input alignment on class-prior shifts and nonlinear deformations.
- **No proof of global boundedness**: While no numerical divergence occurred and local step safety margins remained non-negative across tested dimensions, local step contraction does not constitute a mathematical proof of global boundedness.
- **No arbitrary nonlinear representational capacity**: The linear-associative formulation operates as an affine coordinate transform and exhibited an empirical performance limitation on non-planar boundary deformation; this empirical benchmark limitation does not establish a formal representational ceiling.
- **No superiority over unconstrained FixedDelta in primary transfer**: SafeAdaptiveDelta did not improve primary transfer-family accuracy over FixedDelta in these experiments, although its safety controller remains relevant to stability behavior under severe conditions.
- **No broad real-world classification generalization**: Empirical evaluation was conducted on controlled synthetic streaming classification generators; generalization to complex real-world classification domains has not yet been evaluated.
---

## Installation

### Prerequisites
- Python >= 3.10
- PyTorch >= 2.2.0

### From Source (Editable Mode)
```bash
git clone https://github.com/subodhkant7/deltacore.git
cd deltacore
pip install -e .
```

For development and testing tools:
```bash
pip install -e ".[dev]"
```

### Quickstart: Canonical Adaptive Controller

DeltaCore exposes one canonical stateful controller (`AdaptiveController`) with strict score-before-update semantics, Lyapunov contractive bounding, and verified state persistence:

```python
import torch
from deltacore import AdaptiveController, ControllerConfig, DeterministicFeatureHasher

# 1. Initialize controller and feature hasher
hasher = DeterministicFeatureHasher(dim=64, normalize=True)
config = ControllerConfig(dim=64, eta0=0.03, rho=1.50)
controller = AdaptiveController(config)

# 2. Encode structured telemetry event into R^D
event = {
    "service": "checkout",
    "operation": "process_payment",
    "provider": "stripe",
    "http_status": 200,
    "transport": "http2",
    "outcome": "success",
}
x_t = hasher.encode(event)

# 3. Score before update (non-mutating query)
score_res = controller.score(x_t)
print(f"Pre-update reconstruction residual: {score_res.reconstruction_residual:.4f}")

# 4. Adaptive step (evaluates against M_{t-1}, then updates M_t)
step_res = controller.step(x_t, adapt=True)
print(f"Updated state norm: {step_res.state_norm:.4f}")

# 5. Save & restore state
controller.save_state("checkpoint.json")
controller.load_state("checkpoint.json")
```

For full specifications and epistemic boundaries, see:
- [Mathematical Contract](docs/mathematical_contract.md): Authoritative equation specifications, stability bounds, and temporal ordering.
- [Scientific Boundaries & Limitations](docs/limitations.md): Explicit analysis of auto-associative limits, PCA non-equivalence, and anomaly absorption risks.

### Running Examples & Benchmarks

Minimal end-to-end smoke example:
```bash
python examples/basic_adaptation.py
```

Auto-associative regime shift benchmark vs. baselines:
```bash
python experiments/autoassociative_regime_shift.py
```

---

## Development

DeltaCore enforces strict engineering practices and dependency hygiene. Before contributing, please review:
- [AGENTS.md](AGENTS.md): Guidelines for automated agents and contributors.
- [PROJECT_CONSTITUTION.md](docs/PROJECT_CONSTITUTION.md): Core philosophical principles.
- [docs/mathematical_contract.md](docs/mathematical_contract.md): Strict mathematical contracts.
- [docs/limitations.md](docs/limitations.md): Known failure modes and prohibited claims.

### Code Formatting & Linting
```bash
ruff check .
ruff format --check .
```

---

## Testing

DeltaCore uses `pytest` for all unit, numerical, and integration testing:

```bash
pytest
```

---

## Roadmap

Development proceeds across strict phase gates:
- **Phase 0**: Repository Constitution & Architecture
- **Phase 1**: Mathematical Primitives & Deterministic Tests
- **Phase 2**: Adaptive Update Dynamics
- **Phase 3**: Minimal Self-Referential Adaptive Memory
- **Phase 4**: Stability Controllers & Numerical Guarantees
- **Phase 5**: Sequential & Chunked Scans
- **Phase 6**: Benchmark Framework
- **Phase 7**: Diagnostics & Adaptive-Dynamics Visualization
- **Phase 8**: Five-Memory Self-Modifying Reference Core
- **Phase 9**: Generic 2D Spatial Routing & Boundary-Chunk Execution
- **Phase 9.1**: Spatial Learning Validation Audit
- **Phase 10**: Streaming Adaptive-State Benchmark
- **Phase 11**: Nonlinear Adaptive State & Selective Retention
- **Phase 12**: Controlled Spatio-Temporal Adaptive State
- **Phase 12.1**: Forensic Scientific & Code Integrity Correction
- **Phase 12.2**: IDE Diagnostics, Python Environment & Packaging Integrity Audit
- **Phase 12.3**: Residual Pyright Diagnostic & Test-Type Cleanup
- **Phase 13**: Real-World Spatio-Temporal Adaptive State Benchmark
- **Phase 13.1**: Real-World Benchmark Consistency & Scientific Claim Audit
- **Phase 14**: Independent Real-World Domain Replication (ECMWF ERA5)
- **Phase 15**: Adaptive Regime Transfer & Cross-Domain Robustness
- **Phase 16**: Unseen Shift Robustness & Adaptive Safety
- **Phase 17**: Transfer to a Genuinely Different Task Formulation
- **Phase 18**: Unseen Classification Regime Transfer & Falsification
- **Pre-RecoveryOS Stabilization**: Canonical Public API, Telemetry Hasher Bridge, Score-Before-Update Semantics, & Baseline Comparisons

For full phase objectives and exit criteria, see [ROADMAP.md](file:///Users/urjasoft/Documents/DeltaCore/docs/ROADMAP.md).

---

## Citation

If you reference DeltaCore in your academic work, please cite:

```bibtex
@software{deltacore2026,
  author = {Kant, Subodh},
  title = {DeltaCore: Primitives and Benchmarks for Adaptive Neural State},
  year = {2026},
  url = {https://github.com/subodhkant7/deltacore}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](file:///Users/urjasoft/Documents/DeltaCore/LICENSE) file for details.
