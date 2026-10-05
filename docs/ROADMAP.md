# DeltaCore Research & Development Roadmap

This roadmap defines the sequential, phase-gated execution plan for **DeltaCore**. Progress between phases is governed strictly by verifiable exit criteria. No phase begins until the preceding phase's exit criteria are formally satisfied and validated.

---

## Phase Overview

```
Phase 0: Repository Constitution & Architecture
   │
   ▼
Phase 1: Mathematical Primitives & Deterministic Tests
   │
   ▼
Phase 2: Associative Memory & Delta-Rule Updates
   │
   ▼
Phase 3: Adaptive & Self-Referential Memory Dynamics
   │
   ▼
Phase 4: Stability Controllers & Numerical Guarantees
   │
   ▼
Phase 5: Sequential & Chunked Scans
   │
   ▼
Phase 6: Benchmark Framework
   │
   ▼
Phase 7: Diagnostics & Adaptive-Dynamics Visualization
   │
   ▼
Phase 8: Vision Experiments
   │
   ▼
Phase 9: Temporal / Video & Distribution-Shift Experiments
   │
   ▼
Phase 10: Packaging, Reproducibility & Publication-Quality Release
   │
   ▼
Phase 11: Nonlinear Adaptive State & Selective Retention
   │
   ▼
Phase 12: Controlled Spatio-Temporal Adaptive State
   │
   ▼
Phase 12.1: Forensic Scientific & Code Integrity Correction
   │
   ▼
Phase 12.2: IDE Diagnostics, Python Environment & Packaging Integrity Audit
   │
   ▼
Phase 12.3: Residual Pyright Diagnostic & Test-Type Cleanup
   │
   ▼
Phase 13: Real-World Spatio-Temporal Adaptive State Benchmark
   │
   ▼
Phase 13.1: Real-World Benchmark Consistency & Scientific Claim Audit
   │
   ▼
Phase 14: Independent Real-World Domain Replication
   │
   ▼
Phase 15: Adaptive Regime Transfer & Robustness
   │
   ▼
Phase 16: Unseen Shift Robustness & Adaptive Safety
   │
   ▼
Phase 17: Online Non-Stationary Classification
   │
   ▼
Phase 18: Unseen Classification Regime Transfer & Falsification
   │
   ▼
Phase 19: Autonomous Coherence Gating & Nonlinear State Transfer
```


---

## Detailed Phases

### Phase 0: Repository Constitution & Architecture
- **Objective**: Establish the repository foundation, development contracts, architectural abstractions, minimal dependency profile, and infrastructure test suite.
- **Concrete Exit Criterion**:
  - `docs/PROJECT_CONSTITUTION.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `AGENTS.md`, and `docs/development/PHASE_0_ENVIRONMENT.md` authored and committed.
  - Package skeleton created with clean namespace definitions under `deltacore/`.
  - Minimal `pyproject.toml` configured and package installs in editable mode (`pip install -e .`).
  - Infrastructure tests pass 100% via `pytest` with zero warnings or errors.
  - No speculative model implementations or unvetted dependencies added.

---

### Phase 1: Mathematical Primitives & Deterministic Tests [COMPLETED]
- **Objective**: Implement core pure-PyTorch tensor operations, projection operators, vector algebra, and reference matrix math required by memory updates with zero ambient dependencies.
- **Concrete Exit Criterion**:
  - [x] Deterministic mathematical operations implemented under `deltacore/memory/`, `deltacore/updates/`, and `deltacore/scans/` with explicit tensor dimension verification.
  - [x] 100% branch test coverage across all mathematical primitives (57 tests passing in `tests/`).
  - [x] Numerical tests comparing forward pass computations against analytically derived reference values in double precision (`float64`) in `tests/test_delta_learning_behavior.py`.
  - [x] Zero non-finite ($NaN$, $Inf$) output handling verified across degenerate edge cases (zero inputs, orthogonal inputs, tiny step sizes).
  - [x] Autograd backward propagation verified through read, updates, and unrolled scans with `torch.autograd.gradcheck`.

---

### Phase 2: Adaptive Update Dynamics [COMPLETED]
- **Objective**: Introduce explicit adaptive step-size dynamics $\eta_t = f_\theta(z_t)$ without monolithic coupling, establishing swappable step-size controllers, autograd compatibility, and distribution-shift benchmark telemetry.
- **Concrete Exit Criterion**:
  - [x] Explicit controller abstraction (`StepSizeController`) implemented with `ConstantStepSize`, `InputConditionedStepSize`, `ErrorConditionedStepSize`, and `StateConditionedStepSize`.
  - [x] Compositional `AdaptiveDeltaRule` decoupling update formulation ($e k^\top$) from step magnitude ($\eta_t$).
  - [x] Backward-compatibility verified: `AdaptiveDeltaRule(ConstantStepSize(eta0))` is identical to `DeltaRule(eta0)` within machine precision ($10^{-14}$).
  - [x] Autograd differentiability verified through controller parameters, input, and memory with `torch.autograd.gradcheck`.
  - [x] Distribution shift benchmark executed reporting real measurements for pre-shift, post-shift, recovery, and adaptation gain.
  - [x] Full test suite (73 tests) passing with 0 failures and 0 warnings.

---

### Phase 3: Minimal Self-Referential Adaptive Memory [COMPLETED]
- **Objective**: Implement dynamic, self-referential nested learning dynamics where the memory state controls its own learning rate through an evolving controller memory ($C_{t+1} \neq C_t$) using DeltaCore delta-rule mechanics.
- **Concrete Exit Criterion**:
  - [x] Coupled two-memory architecture (`AssociativeMemory` $M_t$ and `DynamicsMemory` $C_t$) implemented with modular abstractions.
  - [x] Dynamics memory online evolution: $C_{t+1} = C_t + \rho_t (c_t^{\text{target}} - C_t z_t) z_t^\top$ with transparent, modular target generator.
  - [x] Invariants verified: Properties A through H (controller evolution, frozen equivalence, zero content error, zero key, determinism, non-mutation, state dependence, controller isolation).
  - [x] Autograd continuous graph verified via FP64 `torch.autograd.gradcheck` and scan unrolling.
  - [x] Operational recovery latency metric measured across Mode A (stationary) and Mode B (distribution shift) benchmarks.
  - [x] Full test suite passing (89 tests, 0 failures, 0 warnings).

---

### Phase 4: Stability Controllers & Mathematical Guarantees [COMPLETED]
- **Objective**: Derive explicit mathematical sufficient conditions for local residual non-expansion, implement modular stability controllers enforcing those conditions, and experimentally verify stability invariants under adversarial regimes.
- **Concrete Exit Criterion**:
  - [x] Mathematical derivations for content-memory and dynamics-memory local non-expansion ($0 \le \eta_t \|k_t\|^2 \le 2$ and $0 \le \rho_t \|z_t\|^2 \le 2$) documented in `docs/math/PHASE_4_STABILITY.md`.
  - [x] Modular stability controllers implemented under `deltacore/stability/` (`SafeStepSizeController`, `SafeDynamicsRateController`, `UnconstrainedController`).
  - [x] Boundary conditions verified ($0$, strictly inside, exact boundary with sign flip, beyond boundary expansion) across FP32 and FP64.
  - [x] Autograd compatibility verified with FP64 `torch.autograd.gradcheck` on controllers and end-to-end recurrent scan backpropagation.
  - [x] Adversarial stress benchmarks executed (`examples/phase_4_stability_comparison.py`), demonstrating dual safety controls prevent runaway explosive regimes (unconstrained norm > 595,000 $\to$ NaN vs safe norm = 15.98).
  - [x] Stability telemetry and diagnostic observers implemented under `deltacore/diagnostics/stability.py`.
  - [x] Full test suite (172 tests) passing cleanly with 0 failures and 0 warnings.

---

### Phase 5: Sequential & Chunked Scans [COMPLETED]
- **Objective**: Implement efficient execution strategies for associative state updates, including sequential recurrent scans and associative chunked parallel scans.
- **Concrete Exit Criterion**:
  - [x] Mathematical derivation of the affine state recurrence $M_{t+1} = M_t A_t + B_t$ and associative composition law $(A_1, B_1) \otimes (A_2, B_2) = (A_1 A_2, B_1 A_2 + B_2)$ documented in `docs/math/PHASE_5_ASSOCIATIVE_SCAN.md`.
  - [x] Mathematical equivalence verified between sequential step-by-step unrolling and chunked associative scan formulations within numerical tolerance ($\max \|M_{\text{seq}} - M_{\text{chk}}\|_\infty < 10^{-6}$ in FP32, exact $0.0$ in FP64).
  - [x] Arbitrary chunk sizes $C \in \{1, 2, 3, 4, 7, 16\}$ and irregular non-divisible trailing chunks verified.
  - [x] Autograd differentiability verified with FP64 `torch.autograd.gradcheck` on affine operators and end-to-end gradient equivalence between sequential and chunked executions.
  - [x] Runtime and numerical precision benchmark profiles measured and documented across sequence lengths ($T \in [8, 1024]$) in `examples/phase_5_scan_benchmark.py`.
  - [x] Pure-PyTorch reference scan operational without requiring non-portable compilation toolchains.
  - [x] Distinction between known transition coefficient scanning and self-referential dynamic generation explicitly formalized.
  - [x] Full test suite (215 tests) passing cleanly with 0 failures and 0 warnings.

---

### Phase 6: Benchmark Framework [COMPLETED]
- **Objective**: Construct a standardized, reproducible benchmark suite measuring associative memory capacity, multi-query associative recall (MQAR), in-context retention, tracking speed, and stability.
- **Concrete Exit Criterion**:
  - [x] Strongly typed, validated `BenchmarkConfig` schema and machine-readable `RunResult` / `AggregateResult` JSON serialization.
  - [x] Standardized baseline wrappers (`BaseBaseline`) evaluating Frozen, Fixed Delta, Adaptive Delta, Self-Referential Delta, and Safe Self-Referential Delta through a common interface.
  - [x] Six synthetic benchmark tasks implemented (`stationary_recall`, `distribution_shift`, `key_interference`, `conflicting_targets`, `stability_stress`, `adaptation_budget`).
  - [x] Mathematical metrics formalized in `docs/benchmarks/METRICS.md` (first-passage recovery, sustained recovery, adaptation gain, state growth, update/step energy, stability margins).
  - [x] Repeated-seed statistical aggregation and paired-difference analysis in `deltacore/benchmarks/reporting/statistics.py`.
  - [x] First-class failure semantics (`RunStatus.NUMERICAL_FAILURE`, non-finite step index capture).
  - [x] Automated benchmark CLI (`python3 -m deltacore.benchmarks run <task>`).
  - [x] Benchmark matrix and hypothesis registry formalized in `docs/benchmarks/BENCHMARK_MATRIX.md` and `docs/benchmarks/HYPOTHESES.md`.
  - [x] Dedicated benchmark regression test suite (35 tests) passing cleanly (250 tests across repository).

---

### Phase 7: Adaptive State Observatory [COMPLETED]
- **Objective**: Build a reproducible scientific analysis and diagnostic layer for inspecting adaptive neural-state trajectories, state-transition fingerprints, deterministic replay, and multi-system comparisons without interactive dashboards.
- **Concrete Exit Criterion**:
  - [x] Standard trajectory schema (`StateTrajectory`, `TrajectoryStep`) with strict missing data semantics.
  - [x] Schema-validating loaders for trajectories, `RunResult`, and `AggregateResult` JSON artifacts.
  - [x] Non-causal event extraction (shift boundaries, first threshold crossings, sustained recoveries, non-finite steps, clipping events, regime changes).
  - [x] Temporal, shift response, and stability analysis functions conforming to benchmark definitions.
  - [x] Pairwise run comparisons and 12-dimensional state-transition fingerprints.
  - [x] Nine static publication-quality plots (Plots A through I) with non-mutating event annotations.
  - [x] Deterministic replay engine verifying metric fidelity (`exact match`, `within numerical tolerance`, `different`).
  - [x] Reproducibility reports in Markdown and CLI extension (`inspect`, `plot`, `replay`).
  - [x] Full regression test suite (21 tests in `tests/observatory/`, 271 tests across repository passing cleanly).

---

### Phase 8: Five-Memory Self-Modifying Reference Core
- **Objective**: Implement and verify a reference five-memory self-modifying system inspired by VisionHOPE / Nested Learning literature, determining whether DeltaCore's composable abstractions can express the co-evolution of five distinct memory states without loss of inspectability or differentiability.
- **Concrete Exit Criterion**:
  - [x] Corrected Phase 7 post-failure observation semantics (`observed = True, finite_state = False` vs `observed = False, finite_state = None`) and updated VisionHOPE author citations (Peng et al., September 2026).
  - [x] Formal paper alignment and mathematical specifications (`PHASE_8_PAPER_ALIGNMENT.md`, `PHASE_8_FIVE_MEMORY_SYSTEM.md`, `PHASE_8_STABILITY.md`).
  - [x] Immutable state container `FiveMemoryState` with full shape validation and dtype/device preservation.
  - [x] `FiveMemorySystem`, `FiveMemoryStepResult`, and `FiveMemoryScanResult` executing coupled transitions across $M_{\text{content}}, M_{\text{key}}, M_{\text{val}}, M_\eta, M_{\text{ret}}$.
  - [x] Integration with Phase 4 stability controllers (rank-1 contraction bounds and spectral clamp).
  - [x] Full invariant test suite (Properties A–J in `tests/test_five_memory.py`).
  - [x] FP64 gradcheck and sequential recurrent backpropagation in `tests/test_five_memory_autograd.py`.
  - [x] Special-case reductions to simpler DeltaCore primitives verified in `tests/test_five_memory_equivalence.py`.
  - [x] Standardized Observatory schema and fingerprint extended with Phase 8 metrics and verified.
  - [x] Controlled synthetic experiments comparing Fixed Delta, Adaptive Delta, Self-Referential, Safe Self-Referential, and Five-Memory across stationary and distribution-shift regimes (`examples/phase_8_five_memory.py`).
  - [x] **Phase 8.1 Conformance Audit**: Rigorous source-of-truth mathematical audit against VisionHOPE equations (`PHASE_8_CONFORMANCE_AUDIT.md`), reclassifying claims under strict allowed labels (`EXACT`, `CONCEPTUALLY ALIGNED`, `GENERALIZED`, `ENGINEERING DIFFERENCE`, `NOT IMPLEMENTED`), correcting Lyapunov non-existence and causal language, and introducing paper-specific regression fixtures (`tests/test_visionhope_paper_conformance.py`).

---

### Phase 9: Generic 2D Spatial Routing & Boundary-Chunk Execution [COMPLETED]
- **Objective**: Introduce a generic 2D spatial routing layer applying DeltaCore's five-memory reference learner along multiple spatial traversal routes (RIGHT, LEFT, DOWN, UP) with exact restoration and boundary-chunk execution.
- **Concrete Exit Criterion**:
  - [x] Mathematical specification of 2D route serialization $\mathcal{P}_r$, directional scans, exact restoration $\mathcal{P}_r^{-1}$, and route fusion in `docs/math/PHASE_9_SPATIAL_ROUTING.md`.
  - [x] Route serialization and restoration round-trip identity verified across arbitrary rectangular dimensions $(H, W)$ with zero data loss.
  - [x] Independent directional five-memory state trajectories maintained per route.
  - [x] Boundary-refresh chunking semantics implemented and verified against exact unrolled recurrence across chunk sizes $C \ge 1$.
  - [x] Publication-quality spatial diagnostic plots (Plots J through N) generated in the Observatory.
  - [x] Complete test coverage across spatial routes, autograd differentiation, and boundary chunking semantics (`tests/test_spatial_routes.py`, `tests/test_spatial_operator.py`, `tests/test_boundary_chunk_semantics.py`).

---

### Phase 9.1: Spatial Learning Validation Audit [COMPLETED]
- **Objective**: Rigorous empirical and forensic benchmark audit evaluating whether the Phase 9 spatial operator performs meaningful learnable spatial tasks, auditing the Phase 9 pattern benchmark, establishing genuine baselines, and documenting epistemic boundaries.
- **Concrete Exit Criterion**:
  - [x] Pre-benchmark forensic audit of Phase 9 code identifying untrained zero-predictor baseline and documenting root cause of uniform ~28 errors (`PHASE_9_1_BENCHMARK_AUDIT.md`).
  - [x] Two standardized synthetic spatial benchmarks formalized: Task A (spatial shift reconstruction) and Task B (shortcut-resistant pattern classification with $\|X_i\|_F = 1$).
  - [x] Static linear Conv2d and non-spatial reference baselines evaluated alongside DeltaCore under identical seeds and budgets.
  - [x] Publication-quality diagnostic plots (Plots O through T) generated.
  - [x] Strict epistemic boundary documentation acknowledging Task A failure ($E_{\text{rel}} \approx 1.002$) and Task B advantage (~48.3% vs ~33.3%).

---

### Phase 9.2: Spatial Capacity & Optimization Ablation [COMPLETED]
- **Objective**: Conduct a causal-diagnosis ablation across 8 architectural variants and 5 seeds to determine why DeltaCore failed Task A (capacity, dynamics parameters, spatial coordinates, optimization divergence) and whether Task B survives shortcut controls (translation stress, coordinate-statistic control, pixel shuffle, parameter matching).
- **Concrete Exit Criterion**:
  - [x] Runtime parameter verification resolving the Phase 9.1 Conv2d parameter-count discrepancy (148p Task A, 178p Task B, 30p GAP).
  - [x] Trainable dynamics ablation establishing that making update-generation parameters trainable without coordinates is insufficient ($E_{\text{rel}} \approx 0.944$ at $\eta=0.003$) and prone to divergence at $\eta=0.010$.
  - [x] Spatial coordinate ablation proving explicit $(x, y)$ coordinates provide essential spatial addressability ($E_{\text{rel}}$ drops to $0.866$, while shuffled coordinates collapse to $1.001$).
  - [x] Shortcut controls (translation stress, coordinate-statistic control, pixel-shuffle control) validating that Task B performance is genuinely spatial (collapsing by -11.7% to non-spatial GAP ceiling under pixel permutation).
  - [x] Directional analysis proving 4-direction routing does not confer a statistically significant advantage over single-direction routing ($\Delta = +5.33\% \pm 8.78\%$) in lightweight single-layer settings.
  - [x] Observatory expanded with Plots U through AC.
  - [x] Hypotheses H1–H7 evaluated under strict epistemic labels, satisfying Phase Gate Outcome B (narrowly defined spatial classification pathway).
  - [x] Full regression suite (433 tests) passing cleanly with ruff lint and format verification.

---

### Phase 10: Streaming Adaptive-State Benchmark [COMPLETED]
- **Objective**: Establish whether DeltaCore's adaptive memory state provides a measurable advantage for online adaptation under temporal distribution shift across memory capacity, adaptation speed, numerical stability, retention, forgetting, and computational latency.
- **Concrete Exit Criterion**:
  - [x] Phase 9.2 causal-control corrections completed and documented (C1 True shuffled-coordinate control, C2 MLP terminology `mlp_control`, C3 Epistemic corrections).
  - [x] Primary adaptive-state causal control (`adaptive_state = ON` vs `OFF`) implemented and evaluated across all DeltaCore models.
  - [x] Strict online/offline training protocol verified: parameter updates during streaming evaluation = 0 (`requires_grad = False`).
  - [x] Three independent temporal distribution shift tasks implemented and evaluated: Task A (Regime-Switching), Task B (Delayed Context Retrieval $d \in \{16, 64, 256\}$), and Task C (Abrupt Shift $A \to B \to A$).
  - [x] Long-delay retrieval evaluated across horizons $d \in \{16, 64, 256\}$ and sequence lengths $T \in \{128, 512, 1024\}$.
  - [x] $A \to B \to A$ forgetting, adaptation, and recovery evaluated with explicit stability-plasticity tradeoff accounting.
  - [x] Deterministic non-neural online statistical baseline (`OnlineRidge` / Recursive Least Squares) evaluated alongside neural baselines.
  - [x] Capacity-matched comparison group (~100 parameters) established and evaluated across 10 models.
  - [x] Five-seed results reported with individual seed values, means, and standard deviations in machine-readable JSON artifacts.
  - [x] State norm trajectories, Lyapunov stability margins, non-finite tracking, and adaptation update energy ($E_{\text{adapt}} = \sum_t \|\Delta S_t\|_F^2$) documented.
  - [x] Strict epistemic boundary maintained: no ungrounded claims of general intelligence, continual learning superiority, or state-of-the-art.
  - [x] All 7 hypotheses (H10.1–H10.7) evaluated with explicit epistemic status (`SUPPORTED` or `NOT SUPPORTED`), satisfying Phase Gate Outcome B (Mixed Evidence).
  - [x] Observatory expanded with Plots AD through AM (10 publication-quality diagnostic figures).
  - [x] Comprehensive test suite expanded to 483 tests passing cleanly (`python3 -m pytest -q`).
  - [x] Full codebase verified with `ruff check .` and `ruff format --check .`.
  - [x] Benchmark 100% reproducible via single command `python3 examples/phase_10_streaming_benchmark.py`.

---

### Phase 11: Nonlinear Adaptive State & Selective Retention [COMPLETED]
- **Objective**: Determine whether DeltaCore's adaptive state provides a useful niche beyond linear online estimation under nonlinear regime shifts, obsolete state penalties, and scaling constraints ($D \in \{8, 16, 32, 64, 128\}$).
- **Concrete Exit Criterion**:
  - [x] Phase 10 mandatory corrections completed and documented (C1 H10.1 to NOT SUPPORTED, C2 H10.5 to NOT SUPPORTED for general capacity matching, C3 Lyapunov language replaced with local safe-step contractive bounds, C4 LSTM latency comparison corrected to $2.25\times$ naming `SafeAdaptiveDelta`, C5 persistent state inertia recognized).
  - [x] Primary Task A (Nonlinear Regime Dynamics $A \to B \to C \to A$) implemented with strictly bounded, dissipative polynomial-tanh transitions.
  - [x] Strong nonlinear classical baseline (`NonlinearOnlineRidge` / Random Fourier Features RLS) mathematically documented and evaluated.
  - [x] Primary Task B (Regime Switching with Stale-Memory Penalty $A \to B \to A$) constructed to quantify adaptation, forgetting, return recovery, and negative transfer.
  - [x] Selective retention mechanism implemented (`SelectiveRetentionPredictor`) comparing fixed high ($\alpha=0.99$), fixed low ($\alpha=0.70$), error-conditioned adaptive retention, and oracle retention.
  - [x] Task C (Delayed Context Retrieval with Nonlinear Target $y = g(c)$) evaluated across delays $d \in \{16, 64, 256\}$.
  - [x] Dimensional scaling sweep executed across $D \in \{8, 16, 32, 64, 128\}$, demonstrating DeltaCore scales at $1.99\times$ ($67.1\ \mu\text{s}$) with $65.5$ KB state memory vs Nonlinear RLS at $2.80\times$ ($146.9\ \mu\text{s}$) with $393.2$ KB state memory.
  - [x] Adaptation efficiency metric $\Delta E / (E_{\text{adapt}} + \epsilon)$ documented across all models.
  - [x] Causal ablations A through E evaluated (Adaptive State ON/OFF, Retention ON/OFF, Continuous/Reset, Fixed/Adaptive/Oracle Retention, Nonlinear Features ON/OFF).
  - [x] Hypotheses H11.1–H11.7 evaluated under strict epistemic taxonomy (`OBSERVED`, `SUPPORTED`, `NOT SUPPORTED`, `INCONCLUSIVE`), resolving to Phase Gate Outcome B (Specific Niche: Selective Retention & High-Dimensional Scaling).
  - [x] Observatory expanded with Plots AN through AW (10 publication-quality diagnostic figures).
  - [x] Test suite expanded to 497 tests passing cleanly (`python3 -m pytest -q`).
  - [x] Full codebase verified with `ruff check .` and `ruff format --check .`.
  - [x] Benchmark 100% reproducible via single command `python3 examples/phase_11_nonlinear_benchmark.py`.

---

### Phase 12: Controlled Spatio-Temporal Adaptive State [COMPLETED - CORRECTED IN PHASE 12.1]
- **Objective**: Implement and scientifically evaluate the first controlled spatio-temporal benchmark for DeltaCore, investigating adaptive neural state and selective retention under spatially structured, temporally evolving, distribution-shifting data ($D \in \{64, 128, 256\}$).
- **Concrete Exit Criterion**:
  - [x] Mandatory Phase 11 record corrections completed and documented.
  - [x] Standardized flattened spatial field representation $X_t \in \mathbb{R}^{H \times W \times C} \implies x_t \in \mathbb{R}^D$ explicitly reporting $D=HWC$.
  - [x] Task A (Moving Spatial Field Prediction) and Task B (Spatio-Temporal Regime Switching $A \to B \to C \to A$) implemented with bounded, dissipative physical dynamics.
  - [x] Task C (Stale-Memory Challenge $A_1 \to B \to A_2$) and $A_1 \to B \to A_1$ return recovery evaluated.
  - [x] Retention Modes 1 through 5 evaluated, including Mode 4 compact 5-parameter state-conditioned controller (`SelectiveStateAdaptivePredictor`).
  - [x] Critical causal controls A through D evaluated (Retention ON/OFF, Continuous/Reset, Fixed/Adaptive, True vs. Shuffled controller).
  - [x] Spatial structure ablation ($\mathcal{P}_{\mathrm{spatial}}$ coordinate permutation) executed, establishing coordinate-permutation covariance ($\Delta_{\mathrm{perm}} \approx 0$, classified as `OBSERVED`).
  - [x] Spatial representation controls evaluated (`SpatialConvControl`, `SpatialDownsampleControl`).
  - [x] Shift frequency ablation executed across frequent (64), medium (128), and infrequent (256) intervals.
  - [x] Dimensional scaling sweep evaluated across $D \in \{64, 128, 256\}$: DeltaCore matches tested OnlineRidge within observed error at $D=256$ with lower measured persistent state and latency relative to tested RLS.
  - [x] Five-seed protocol with paired seed differences, Pareto frontiers, and stability diagnostics documented.
  - [x] Hypotheses corrected in Phase 12.1: H12.2 $\to$ `INCONCLUSIVE`, H12.4 $\to$ `INCONCLUSIVE`, H12.5 qualified observation, H12.6 implementation-level resource observation.
  - [x] Observatory expanded with Plots AX through BJ (13 publication-quality diagnostic figures).
  - [x] Test suite expanded to 554 tests passing cleanly (`python3 -m pytest -q`).
  - [x] Full codebase verified with `ruff check .` and `ruff format --check .`.
  - [x] Benchmark 100% reproducible via single command `python3 examples/phase_12_spatiotemporal_benchmark.py`.

---

### Phase 12.1: Forensic Scientific & Code Integrity Correction [COMPLETED]
- **Objective**: Perform a rigorous forensic audit of Phase 12 code health, diagnose the 5-parameter state-conditioned controller implementation, establish the Useful-Prediction Gate, verify trainable parameter optimization, and rerun the principal retention experiment across 5 seeds.
- **Concrete Exit Criterion**:
  - [x] Multi-tier code health audit executed: `compileall`, `pytest` (554/554 passing), `ruff check`, `ruff format`, and `mypy` diagnostics verified ([PHASE_12_1_CODE_HEALTH_AUDIT.md](docs/benchmarks/PHASE_12_1_CODE_HEALTH_AUDIT.md)).
  - [x] Controller parameter audit conducted: resolved why Phase 12 had 5 parameters but 0 trainable parameters (`requires_grad=False`, no optimizer, no BPTT).
  - [x] Bottleneck defect diagnosed: Phase 12 default `feat_dim=8` compressed 64D spatial fields into an 8D bottleneck, failing the Useful-Prediction Gate ($E_{\mathrm{rel}} \approx 0.9955$, `NO_USEFUL_STATE`).
  - [x] Useful-Prediction Gate mathematically defined ($\text{Gate}(\mathcal{M}) = \mathbf{USEFUL\_STATE} \iff E_{\mathrm{rel}} \le 0.985$, with significance floor $\delta_{\mathrm{floor}} = 1.0 \times 10^{-3}$).
  - [x] Exact 5-parameter controller trained via recurrent unrolled BPTT: verified non-zero gradients ($\|\nabla_\theta \mathcal{L}\| = 9.20 \times 10^{-5}$), verified parameter delta $\Delta\theta = 4.029 \neq 0$, and loss minimization.
  - [x] Principal retention experiment rerun on primary $D=64$ spatio-temporal task across seeds $[0, 1, 2, 3, 4]$ with full-rank state ($D_{\mathrm{feat}}=64$, $E_{\mathrm{rel}} = 0.9507$, `USEFUL_STATE`).
  - [x] Continuous state vs reset state evaluated: $\Delta_{\mathrm{retention}} = -2.12 \times 10^{-5} \pm 2.57 \times 10^{-5}$ confirmed below the measurement noise floor ($10^{-3}$).
  - [x] Architecture comparison: trained controller converged to static high retention ($\alpha \approx 0.992$), matching `Selective_fixed_high` ($0.95068$ vs $0.95070$).
  - [x] Spatial permutation boundary condition formally retained as `OBSERVED`: coordinate-permutation covariant, without 2D locality exploitation.
  - [x] Hypotheses updated: H12.2 $\to$ `INCONCLUSIVE`, H12.4 $\to$ `INCONCLUSIVE`, H12.5 and H12.6 wording corrected.
  - [x] Phase Gate resolved to **Outcome R2** (Retention works but controller is not essential; simplify architecture to fixed/scalar bounds).
  - [x] All 4 documentation reports and 4 JSON artifacts generated and persisted.

---

### Phase 12.2: IDE Diagnostics, Python Environment & Packaging Integrity Audit [COMPLETED]
- **Objective**: Conduct a forensic audit of DeltaCore's packaging metadata, Python interpreters, language-server configuration, and IDE diagnostics to resolve editor errors such as `Cannot find module "torch"` on `deltacore/updates/delta.py`.
- **Concrete Exit Criterion**:
  - [x] Multi-interpreter probe executed across all 6 host Python environments; isolated that only `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` contains PyTorch 2.6.0.
  - [x] Packaging metadata audited: verified `torch>=2.2.0` is declared under `[project].dependencies` in `pyproject.toml` (`DEPENDENCIES_COMPLETE`).
  - [x] Dynamic 84-module import sweep executed: 100% of DeltaCore modules import cleanly with zero errors (`module_import_audit.json`).
  - [x] 82 mypy diagnostics across 16 legacy source files classified by root cause (`typecheck_audit.json`); 0 real runtime defects.
  - [x] Specific audit of `deltacore/updates/delta.py` completed: confirmed functional execution and type resolution.
  - [x] Canonical `.venv` established in workspace root and declarative Pyright configuration added to `pyproject.toml` and `pyrightconfig.json`.
  - [x] Regression tests added (`tests/test_packaging_and_environment.py`); test suite expanded to 558 passing tests cleanly (`pytest -q`).
  - [x] Compileall and ruff check/format verified 100% clean across 148 files.
  - [x] Phase Gate resolved to **Outcome A** (Environment / IDE Language-Server Resolution Problem).

---

### Phase 12.3: Residual Pyright Diagnostic & Test-Type Cleanup [COMPLETED]
- **Objective**: Resolve all residual static typing diagnostics in the `tests/` tree following Phase 12.2 environment alignment, including the representative `"**" is not supported between "AssociativeMemory" and "Literal[2]"` diagnostic, without using `# type: ignore` or broad `Any` suppressions.
- **Concrete Exit Criterion**:
  - [x] Identified and cataloged all 55 Pyright diagnostics across 14 test files (`red_diagnostics_inventory.json`, `pyright_diagnostics.json`).
  - [x] Reproduced the exact `AssociativeMemory` / `Literal[2]` diagnostic and confirmed intended mathematical operation ($\|M_{\mathrm{new}}\|_F^2$ squared Frobenius norm loss).
  - [x] Parameterized `DeltaStepResult`, `AdaptiveDeltaStepResult`, and `HebbianStepResult` with `Generic[MemoryT]`.
  - [x] Added `@overload` signatures for `step()` and `update()` in `DeltaRule`, `AdaptiveDeltaRule`, and `HebbianRule` to preserve exact static types when `torch.Tensor` is passed.
  - [x] Replaced tuple unpacking in autograd tests with direct canonical `.predictions` property access.
  - [x] Replaced invariant `dict[str, Sequence[float]]` with covariant `Mapping[str, Sequence[float]]` in observatory plotting functions.
  - [x] Updated `StreamData.metadata` from `dict[str, object]` to `dict[str, Any]` to allow tensor indexing.
  - [x] Resolved all diagnostics without adding `# type: ignore` or broad `cast(Any, ...)`.
  - [x] `pyright tests` reduced from 55 errors to exactly **0 errors, 0 warnings, 0 informations**.
  - [x] Multi-tier verification clean: `compileall`, `ruff check .`, `ruff format --check .` (148 files), and `pytest -q` (558/558 passed).
  - [x] Phase Gate resolved to **Outcome A** (Clean - Zero Genuine Source or Test Type Errors).

---

### Phase 13: Real-World Spatio-Temporal Adaptive State Benchmark [COMPLETED]
- **Objective**: Evaluate the smallest validated DeltaCore mechanism (`SafeAdaptiveDelta`) on a real-world spatio-temporal dataset (NOAA OISST v2.1 Equatorial Pacific SST waveguide) under temporal distribution change (2015–2016 Super El Niño event), determining whether DeltaCore's adaptive state provides useful online adaptation or retention behavior under real environmental shift.
- **Concrete Exit Criterion**:
  - [x] Preflight repository health check executed: 0 Pyright errors, 0 warnings across `deltacore`, `examples`, `tests`; `compileall` clean; `ruff` clean; 558/558 historical tests passing (`PHASE_13_PREFLIGHT.md`).
  - [x] Primary real-world dataset selected, ingested, and documented: NOAA OISST v2.1 Equatorial Pacific SST ($5^\circ\text{S}-5^\circ\text{N}, 170^\circ\text{W}-120^\circ\text{W}$, 360 bi-weekly steps) with deterministic ingestion pipeline and dataset card (`PHASE_13_DATASET_CARD.md`).
  - [x] Strict temporal splitting and leakage prevention audit completed: disjoint timestamps between train $[0, 150)$, val $[150, 200)$, and test $[200, 360)$; train-only normalization ($\mu=26.83^\circ\text{C}, \sigma=2.02^\circ\text{C}$); sequential target disclosure preventing future target leakage (`PHASE_13_DATA_INTEGRITY.md`, `dataset_config.json`).
  - [x] Test-time model parameter immutability verified: SHA256 parameter hashes bit-for-bit identical before and after test evaluation ($\Delta \theta = 0$).
  - [x] 9-model matrix evaluated under identical streaming conditions: `Persistence`, `FrozenLinear`, `OnlineRidge`, `NonlinearOnlineRidge`, `GRU`, `FixedDelta`, `SafeAdaptiveDelta`, `SelectiveRetention`, and `SpatialConv`.
  - [x] SafeAdaptiveDelta vs FixedDelta evaluated: `SafeAdaptiveDelta` achieves lower relative error ($0.2582$ vs $0.2692$ at $D=64$) and prevents catastrophic numerical divergence at $D=256$ ($E_{\text{rel}} = 0.6428$ vs $2.85 \times 10^{14}$) through dynamic contractive bounding $\eta_t \le 1.9/\|x_t\|_2^2$.
  - [x] Causal Control 1 (State Reset Control) evaluated: retaining continuous historical state during the 2015–2016 Super El Niño shift reduces cumulative excess adaptation error by $25.96\%$ ($26\%$ rounded) (excess error $2.5987$ vs $3.5100$; first-passage recovery $0$ steps vs $2$ steps).
  - [x] Causal Control 2 (Spatial Permutation Control) evaluated: DeltaCore and OnlineRidge exhibit exact permutation equivariance ($E_{\text{equiv}} \approx 7.73 \times 10^{-8}$), confirming the architectural boundary that DeltaCore operates as a permutation-equivariant temporal feature estimator, while `SpatialConv` degrades by $+641\%$ under coordinate shuffling.
  - [x] Resource accounting completed: `SafeAdaptiveDelta` achieves a 50% persistent state memory reduction over `OnlineRidge` ($16\text{ KB}$ vs $32\text{ KB}$ at $D=64$; $256\text{ KB}$ vs $512\text{ KB}$ at $D=256$) with comparable or lower per-token latency ($4.63\,\mu\text{s}$ vs $4.71\,\mu\text{s}$ at $D=256$).
  - [x] Multi-resolution scaling sweep evaluated across $D=64$ ($8 \times 8$) and $D=256$ ($16 \times 16$).
  - [x] Statistical protocol completed across 5 independent random seeds (`seeds = [42, 43, 44, 45, 46]`), recording per-seed means, standard deviations, and shift analysis (`phase_13_per_seed.json`, `phase_13_results.json`).
  - [x] Observatory expanded with Plots BK through BU (11 publication-quality diagnostic figures in `docs/benchmarks/artifacts/phase_13/plots/`).
  - [x] Hypotheses evaluated under strict epistemic taxonomy: H13.1–H13.6 `SUPPORTED`; H13.7 corrected to `NOT SUPPORTED` (spatial convolution permutation sensitivity does not establish task-performance advantage over DeltaCore).
  - [x] Phase Gate resolved to **Outcome A** (Real-world adaptive-state evidence: DeltaCore produces reproducible benefit during real temporal shift while achieving lowest mean prediction error across tested models).
  - [x] Test suite expanded with 13 comprehensive Phase 13 tests in `tests/test_phase_13_real_spatiotemporal.py` (571/571 passing cleanly).
  - [x] Entire benchmark reproducible via single command `python3 examples/phase_13_real_spatiotemporal_benchmark.py`.

---

### Phase 13.1: Real-World Benchmark Consistency & Scientific Claim Audit [COMPLETED]
- **Objective**: Conduct a forensic numerical, statistical, baseline training, and mathematical claim audit of Phase 13 before Phase 14 authorization, establishing that all reported tables, artifacts, and claims are audit-proof and internally consistent.
- **Concrete Exit Criterion**:
  - [x] Preflight checks 100% clean: 0 Pyright errors, 0 runtime failures, 571 pytest passing, ruff check/format clean across 153 files.
  - [x] Table discrepancies reconciled: Table 1 vs Table 4 differences formally traced to multi-seed ($N=5$) aggregation vs single-seed ($N=1$, seed 42) scaling scope (`PHASE_13_1_NUMERICAL_CONSISTENCY.md`, `table_consistency.json`).
  - [x] Baseline offline training provenance verified: `SpatialConv`, `FrozenLinear`, and `GRU` confirmed trained via Adam optimizer on the train split, minimizing MSE by $58.5\%$ to $69.2\%$ before freezing (`baseline_training_provenance.json`).
  - [x] Direct permutation equivariance tested: verified $E_{\text{equiv}} \approx 7.73 \times 10^{-8}$ for DeltaCore (exact to FP32 roundoff), proving permutation equivariance while `SpatialConv` violates equivariance ($E_{\text{equiv}} \approx 1.95$) (`permutation_equivariance.json`).
  - [x] H13.7 corrected to `NOT SUPPORTED`: documented that permutation sensitivity alone does not establish task advantage.
  - [x] H13.6 updated: exact permutation equivariance confirmed.
  - [x] Stability claims qualified to local contractive safety condition ($|1 - \eta_t \|x_t\|_2^2| < 1$).
  - [x] Temperature terminology corrected to "+2.8 °C SST anomaly" and distinguished from official NOAA Niño 3.4 index.
  - [x] Per-seed win rate verified: `SafeAdaptiveDelta` achieves lower prediction error than all baselines across 5 of 5 tested seeds ($100\%$ win rate) (`phase_13_1_per_seed_verification.json`).
  - [x] Controlled state-reset intervention recomputed: unrounded $25.9625\%$ ($26\%$ rounded) cumulative excess error reduction verified from raw data (`phase_13_1_audit.json`).
  - [x] Spatial gradient error semantics documented: coordinate-scrambling denominator inflation artifact explained.
  - [x] Scientific language audit completed: unhedged words removed across all reports.
  - [x] Phase Gate resolved to **Outcome A** (Clean empirical validation).

---

### Phase 14: Independent Real-World Domain Replication [COMPLETED]
- **Objective**: Test whether the Phase 13 real-world findings transfer to a second, independently sourced real-world spatio-temporal domain (ECMWF ERA5 2-meter air temperature $T_{2m}$ over the North Atlantic and European Storm Track sector, $40^\circ\text{N}-65^\circ\text{N}, 30^\circ\text{W}-20^\circ\text{E}$, 360 synoptic 6-hourly intervals) during the January–February 2021 Sudden Stratospheric Warming and European Arctic Polar Outbreak (Storm Filomena), using the smallest successful Phase 13 mechanism (`SafeAdaptiveDelta`) with frozen hyperparameters.
- **Concrete Exit Criterion**:
  - [x] **EC14.1**: Second real-world domain objectively selected and documented before seeing model results (`PHASE_14_DATASET_SELECTION.md`).
  - [x] **EC14.2**: Dataset integrity and chronological temporal leakage audit pass with zero future lookahead (`PHASE_14_DATASET_CARD.md`, `PHASE_14_DATA_INTEGRITY.md`).
  - [x] **EC14.3**: All baseline training provenance documented with strict offline/online parameter separation (`PHASE_14_IMPLEMENTATION.md` §3).
  - [x] **EC14.4**: `SafeAdaptiveDelta` and `FixedDelta` evaluated under identical streaming conditions across 5 seeds (`phase_14_results.json`).
  - [x] **EC14.5**: `OnlineRidge` evaluated under identical streaming conditions.
  - [x] **EC14.6**: Trained small `SpatialConv` evaluated under original and permuted spatial orderings.
  - [x] **EC14.7**: Real temporal distribution shift independently defined and evaluated (January–February 2021 SSW / Storm Filomena cold anomaly, $t \in [45, 115]$, peak $t=75$, $-7.5^\circ\text{C}$ anomaly, `shift_definition.json`).
  - [x] **EC14.8**: Continuous-versus-reset state retention intervention completed: continuous state reduces cumulative excess shift error by $-0.8848$ ($-97.1\%$) and eliminates the 1-step re-acquisition lag.
  - [x] **EC14.9**: Spatial permutation error delta and explicit vector output equivariance evaluated: DeltaCore and OnlineRidge confirm exact equivariance ($E_{\text{equiv}} \le 10^{-7}$), while `SpatialConv` degrades by $+536\%$ ($E_{\text{equiv}} = 1.78$).
  - [x] **EC14.10**: Multi-resolution resource scaling measured across $D=64$ and $D=256$: `FixedDelta` explodes to `NaN` at $D=256$ due to contractive bound violation ($\eta \|x_t\|^2 > 2.0$), while `SafeAdaptiveDelta` dynamically contracts ($\eta_t \le 1.9/\|x_t\|^2$) and remains bounded and stable ($E_{\text{rel}} = 0.2604$).
  - [x] **EC14.11**: Five-seed evaluation completed (`seeds = [42, 43, 44, 45, 46]`) in `phase_14_per_seed.json`.
  - [x] **EC14.12**: Phase 13 versus Phase 14 cross-domain replication table completed (`PHASE_14_CROSS_DOMAIN_REPLICATION.md`).
  - [x] **EC14.13**: All primary hypotheses receive explicit epistemic status: H14.2, H14.3, H14.5, H14.6 `SUPPORTED`; H14.1, H14.4 `NOT SUPPORTED` (OnlineRidge achieves lower error on fast linear advection field); H14.7 `NOT SUPPORTED` for full replication (`PHASE_14_INTERPRETATION.md`).
  - [x] **EC14.14**: Full repository code-health gate passes: 0 Pyright errors, 0 Ruff errors, 588 pytest tests passing cleanly across the repository.
  - [x] **EC14.15**: All benchmark artifacts and Observatory plots (BV through CF) reproducible via single command `python3 examples/phase_14_atmospheric_benchmark.py`.
  - [x] Phase Gate resolved to **Outcome B** (Partial Replication: Dynamic contractive safety, continuous state retention advantage, persistent state memory efficiency, and permutation equivariance transfer robustly; prediction error superiority over OnlineRidge does not transfer on fast synoptic advection).

---

### Phase 15: Adaptive Regime Transfer & Robustness [COMPLETED]
- **Objective**: Determine whether the Phase 13–14 domain divergence is caused by different physical regimes (slow thermal diffusion vs fast synoptic advection) and whether DeltaCore can adapt its update aggressiveness while preserving numerical safety.
- **Concrete Exit Criterion**:
  - [x] **EC15.1**: Phase 14 frozen with corrected claim wording (`docs/benchmarks/PHASE_15_PRE_FLIGHT.md`, `PHASE_14_INTERPRETATION.md`).
  - [x] **EC15.2**: $A \to B$ and $B \to A$ parameter-transfer experiments complete (`phase_15_transfer.json`, Plot CH).
  - [x] **EC15.3**: No-retuning evaluation complete across Phase 13, Phase 14, and pooled validation configurations (`phase_15_transfer.json`, Plot CH).
  - [x] **EC15.4**: Validation-only aggressiveness sweep complete ($\alpha_{\min} \in \{0.70, 0.85, 0.95\}$, $\rho \in \{1.5, 1.9\}$, $\eta_0 \in \{0.008, 0.015\}$) (`phase_15_safety.json`, Plot CI).
  - [x] **EC15.5**: Adaptation / stability Pareto analysis complete without reducing to single composite score (`phase_15_safety.json`, Plot CI).
  - [x] **EC15.6**: Cross-domain state-transfer experiments complete (regime-switch stream OISST $\to$ ERA5 $\to$ OISST, Plots CJ & CL).
  - [x] **EC15.7**: Continuous / reset retention experiments complete: continuous state delivers $26.0\%$ cumulative excess error reduction on OISST and $97.1\%$ on ERA5 (`phase_15_retention.json`, Plot CK).
  - [x] **EC15.8**: $D=256$ safe/unsafe failure boundary measured on both domains: `FixedDelta` explodes to non-finite (`NaN`) at $t=41$ on ERA5, while `SafeAdaptiveDelta` dynamically contracts ($\eta_t \le 1.9/\|x_t\|^2$), enforcing safety margin $\ge 0.1000$ (`phase_15_safety.json`, Plot CM).
  - [x] **EC15.9**: Five-seed results reported for all models (`seeds = [42, 43, 44, 45, 46]`) in `phase_15_per_seed.json`.
  - [x] **EC15.10**: No raw cross-domain physical-unit comparison used as headline statistic; relative error $E_{\mathrm{rel}}$ strictly utilized.
  - [x] **EC15.11**: All hypotheses receive epistemic status: H15.1 through H15.7 all `SUPPORTED` (`PHASE_15_INTERPRETATION.md`).
  - [x] **EC15.12**: Code-health gate passes: 0 Pyright errors, 0 compile errors, 596 pytest tests passing cleanly, 0 Ruff errors.
  - [x] **EC15.13**: All benchmark artifacts and Observatory plots (CG through CP) reproducible via single command `python3 examples/phase_15_regime_transfer_benchmark.py`.
  - [x] Phase Gate resolved to **Outcome A** (Robust transfer: A single pooled validation configuration transfers between domains while preserving useful adaptation and bounded numerical stability; proceed to Phase 16).

---

### Phase 16: Unseen Shift Robustness & Adaptive Safety [COMPLETED]
- **Objective**: Challenge the Phase 15 pooled `SafeAdaptiveDelta` configuration ($\eta_0=0.015, \rho=1.50, \alpha_{\min}=0.95$) with five predetermined unseen distribution shifts (Mean, Variance, Temporal Speed, Noise, Combined) across three predetermined severities (`mild`, `moderate`, `severe`) on both NOAA OISST SST (Domain A) and ECMWF ERA5 $T_{2m}$ (Domain B) without test-time retuning. Characterize the empirical operating envelope and map the exact failure boundary where the current DeltaCore mechanism stops working.
- **Concrete Exit Criterion**:
  - [x] **EC16.1**: Phase 15 claims narrowed and historically frozen (`docs/benchmarks/PHASE_16_PRE_FLIGHT.md`, `PHASE_15_INTERPRETATION.md`).
  - [x] **EC16.2**: Primary pooled configuration used without test-time tuning ($\eta_0=0.015, \rho=1.50, \alpha_{\min}=0.95$).
  - [x] **EC16.3**: Five predefined shift types evaluated: Mean, Variance, Temporal Speed, Noise, and Combined (`phase_16_shift_matrix.json`).
  - [x] **EC16.4**: Three predetermined severities evaluated (`mild`, `moderate`, `severe`).
  - [x] **EC16.5**: Domain A (OISST) and Domain B (ERA5) both evaluated across 5 seeds (`phase_16_per_seed.json`).
  - [x] **EC16.6**: Continuous/reset state intervention completed: continuous state outperforms state reset on all within-domain shifts ($\Delta_{\mathrm{cum}} < 0$) (`phase_16_retention.json`, Plot CT).
  - [x] **EC16.7**: Retention stress evaluated across multi-regime transitions ($A \to B, A \to B \to A, A \to B \to C, A \to \text{severe-}B \to A$); identified positive reset advantage across severe cross-regime mismatches ($\Delta_{\mathrm{reset}} > 0$) (`phase_16_retention.json`, Plot CU).
  - [x] **EC16.8**: $D=256$ empirical failure boundary mapped: `FixedDelta` diverges to NaN at step sizes $\eta \ge 0.005$ on Domain B and $\eta \ge 0.012$ on Domain A, while `SafeAdaptiveDelta` remains 100% finite across all tested step sizes ($\eta \in [0.002, 0.025]$), preserving safety margin $\ge 0.50$ (`phase_16_safety_boundary.json`, Plot CV).
  - [x] **EC16.9**: Five-seed evaluation completed (`seeds = [42, 43, 44, 45, 46]`) in `phase_16_per_seed.json` (Plot CY).
  - [x] **EC16.10**: Cross-domain robustness matrix completed and documented (`PHASE_16_ROBUSTNESS_MATRIX.md`, Plot CX).
  - [x] **EC16.11**: All primary hypotheses receive explicit epistemic status: H16.1 through H16.7 all `SUPPORTED` (`PHASE_16_INTERPRETATION.md`).
  - [x] **EC16.12**: Full repository code-health gate passes: 0 Pyright errors, 0 compile errors, 607 pytest tests passing cleanly, 0 Ruff errors.
  - [x] **EC16.13**: All benchmark artifacts and Observatory publication plots (CQ through DB) reproducible via single command `python3 examples/phase_16_robustness_benchmark.py`.
  - [x] Phase Gate resolved to **Outcome A** (Robust unseen-shift behavior: Pooled configuration remains stable and provides useful adaptation across multiple unseen shifts on both domains; proceed to Phase 17).

---

### Phase 17: Online Non-Stationary Classification [COMPLETED]
- **Objective**: Test whether DeltaCore's adaptive-state principles generalize from continuous spatio-temporal regression to online non-stationary classification ($K=6, D \in \{32, 64, 128, 256\}$), evaluating adaptive associative state under changing distributions without test-time parameter learning ($\Delta\theta = 0$).
- **Concrete Exit Criterion**:
  - [x] **EC17.1**: Phase 16 claims and artifacts historically frozen with cryptographic SHA-256 digests (`PHASE_17_PRE_FLIGHT.md`).
  - [x] **EC17.2**: Online multiclass classification stream created ($K=6, D=32$) with strict no-label-lookahead evaluation: $\hat{y}_t = f(x_t, S_{t-1})$.
  - [x] **EC17.3**: Three non-stationary regimes constructed ($A \to B \to C \to A$): Directional Means, Covariance Structure Shift, and Decision-Boundary Rotation.
  - [x] **EC17.4**: Formal shortcut audit completed: matched class energy ($\|x_t\|_2 = \sqrt{D}$), matched class variances, no single feature leaks label ($< 41\% < 60\%$), passed audit.
  - [x] **EC17.5**: Task A stationary classification evaluated: SafeAdaptiveDelta achieves $1.000$ accuracy, decisively passing the useful-prediction gate.
  - [x] **EC17.6**: Task B abrupt covariate shift ($A \to B \to A$) evaluated: stable retention and low excess loss under coordinate shearing.
  - [x] **EC17.7**: Task C decision-boundary shift ($A \to C \to A$) evaluated: SafeAdaptiveDelta achieves $0.958 \pm 0.005$ accuracy while FrozenLinear collapses to $0.815 \pm 0.023$ (and $0.686$ under severe shift), cutting excess loss by $85\%$.
  - [x] **EC17.8**: Task D stale-state ablation evaluated: continuous state aids returning regimes ($0.992$), while state reset eliminates negative transfer under incompatible regime shifts.
  - [x] **EC17.9**: Causal State-Off ablation ($M_t \equiv 0$) confirms that $100\%$ of adaptation gains are causally driven by associative memory $M_t$.
  - [x] **EC17.10**: Parameter immutability ($\Delta\theta = 0$) verified bit-for-bit via pre/post SHA-256 parameter hashes.
  - [x] **EC17.11**: Robustness controls evaluated: Label shuffle reduces all models to theoretical chance ($16.7\%$), and feature permutation confirms coordinate equivariance.
  - [x] **EC17.12**: Dimensional scaling evaluated across $D \in \{32, 64, 128, 256\}$: 0 divergences, strictly positive safety margin ($\mu_t > 0.50$), latency $6-11\ \mu s$, memory exact $4D^2$ bytes.
  - [x] **EC17.13**: Five-seed evaluation completed (`seeds = [42, 43, 44, 45, 46]`) in `phase_17_per_seed.json`.
  - [x] **EC17.14**: All 8 hypotheses receive explicit epistemic status: H17.1 through H17.8 all `SUPPORTED` (`PHASE_17_INTERPRETATION.md`).
  - [x] **EC17.15**: Code-health gate passes: 0 Pyright errors, 0 compile errors, 626 pytest tests passing cleanly, 0 Ruff errors.
  - [x] **EC17.16**: All benchmark artifacts and Observatory plots (DC through DP) reproducible via single command `python3 examples/phase_17_classification_benchmark.py`.
  - [x] Phase Gate resolved to **Outcome A** (Task-transfer evidence: DeltaCore demonstrates useful classification performance and reproducible adaptation benefits under non-stationary classification shifts while remaining stable. Proceed to Phase 18 using the smallest successful mechanism).

---

### Phase 18: Unseen Classification Regime Transfer & Falsification [COMPLETED]
- **Status**: Completed: multi-family classification transfer and falsification study.
- **Objective**: Determine whether Phase 17's online classification gains transfer to genuinely different non-stationary classification generators without per-task retuning, using the frozen minimal `SafeAdaptiveDelta` associative-state mechanism ($\eta_0=0.015, \rho=1.50, \alpha_{\min}=0.95$). Systematically characterize empirical operating envelopes, failure modes, and falsification cases across multiple shift families and temporal dynamics.
- **Core Findings & Recorded Limitations**:
  - *Primary Positive Finding*: Reproducible transfer gains over FrozenLinear on boundary rotation (+5.6 percentage points, $88.0\%$ vs $82.4\%$), boundary translation (+9.8 percentage points, $92.2\%$ vs $82.4\%$), and nonlinear deformation (+9.9 percentage points, $87.2\%$ vs $77.3\%$) under a frozen configuration without per-task retuning. The StateOff intervention ($M_t \equiv 0$) isolated the evolving associative state as the operative difference between adaptive and frozen conditions. SafeAdaptiveDelta did not improve primary transfer-family accuracy over FixedDelta ($88.5\%$ on A, $93.7\%$ on B, $87.8\%$ on C), though its safety controller remains relevant to stability behavior.
  - *Negative-Transfer Finding*: Continuous persistent state incurred a $-1.75$ percentage point overall accuracy penalty ($60.125\%$ vs $61.875\%$) and a $-10.0$ percentage point immediate post-shift accuracy penalty ($30.0\%$ vs $40.0\%$) under adversarial strong mismatch compared to oracle reset.
  - *Representation Limitation*: The tested linear-associative formulation showed an empirical performance limitation on the nonlinear deformation benchmark ($87.2\%$ vs $95.5\%$ for OnlineLogisticRegression); the experiment does not establish a formal representational ceiling.
  - *Classical-Baseline Limitation*: The associative-state mechanism was not competitive with online logistic regression on class-prior shifts ($92.2\%$ vs $96.6\%$).
  - *Stability Result*: 0 diverged runs occurred across $D \in \{32, 64, 128, 256\}$ with exact $4D^2$-byte FP32 persistent state memory. No numerical divergence was observed under the frozen configuration, although local safety margins approached the controller boundary in some evaluated steps (min margin $\sim 3.9 \times 10^{-9}$ at $D=256$). Local step safety does not constitute a proof of global boundedness.
- **Concrete Exit Criterion**:
  - [x] **EC18.1**: Phase 17 claims and artifacts historically frozen with cryptographic SHA-256 digests (`PHASE_18_PRE_FLIGHT.md`).
  - [x] **EC18.2**: Minimal associative mechanism frozen without architectural expansion: $\hat{y}_t = \arg\max_k [W_{\mathrm{head}} (I + M_t) x_t + b_{\mathrm{head}}]_k$, $\eta_t = \min(\eta_0 / (1 + \gamma \|e_x\|^2), \rho / (\|x\|^2 + \epsilon))$, $\alpha_t = \max(\alpha_{\min}, 1 - \eta_t \|x\|^2)$.
  - [x] **EC18.3**: Hyperparameter freeze strictly maintained across all tasks: $\eta_0=0.015, \rho=1.50, \alpha_{\min}=0.95$ with zero per-task retuning.
  - [x] **EC18.4**: Three genuinely distinct non-stationary task families constructed: Family A (subspace rotation), Family B (boundary translation / intercept shift), Family C (nonlinear quadratic/radial boundary warp).
  - [x] **EC18.5**: Six shift dimensions evaluated: Covariate, Boundary, Class-Prior Imbalance, Gradual Continuous Drift, Abrupt Shift, and Adversarial Strong Mismatch.
  - [x] **EC18.6**: Strict online causal protocol enforced with nanosecond assertions: $t_{\mathrm{pred}} \le t_{\mathrm{reveal}} \le t_{\mathrm{adapt}}$, no future-label lookahead.
  - [x] **EC18.7**: Model matrix evaluated across FrozenLinear, AdaptiveStateOFF, OnlineLogisticRegression, OnlineRidge, OnlineMulticlassLinear, FixedDelta, and SafeAdaptiveDelta.
  - [x] **EC18.8**: Causal ablations completed: StateOff intervention removes the measured adaptation gains (+5.6 percentage points on Family A, +9.8 percentage points on Family B, +9.9 percentage points on Family C), isolating the associative memory $M_t$.
  - [x] **EC18.9**: Retention tradeoff characterized: Continuous persistent state maintains $100.0\%$ return accuracy on Family A, while incurring a $-1.75$ percentage point stale state penalty under adversarial strong mismatch compared to oracle reset (`phase_18_retention.json`, Plot 6).
  - [x] **EC18.10**: Predefined shortcut audit and negative controls passed: class energy matched ($\|x\|_2 = \sqrt{D}$), label shuffle collapses to chance ($18.2\% \approx 16.7\%$), coordinate permutation confirms performance invariance (`phase_18_controls.json`, Plots 16 & 17).
  - [x] **EC18.11**: Parameter immutability ($\Delta\theta = 0$) verified bit-for-bit via pre/post SHA-256 parameter hashes across all runs (`phase_18_hashes.json`).
  - [x] **EC18.12**: Dimensional scaling evaluated across $D \in \{32, 64, 128, 256\}$: 0 divergences, strictly non-negative safety margins within tested operating envelope, latency $1.5-16.3\ \mu s$, memory exact $4D^2$ bytes (`phase_18_scaling.json`, Plots 13–15).
  - [x] **EC18.13**: Five-seed evaluation completed (`seeds = [42, 43, 44, 45, 46]`) in `phase_18_per_seed.json`.
  - [x] **EC18.14**: All 8 pre-registered hypotheses receive explicit epistemic status: H18.1, H18.2, H18.3, H18.7 `SUPPORTED`; H18.4 `SUPPORTED EMPIRICALLY IN TESTED REGIMES`; H18.5 `SUPPORTED FOR TESTED OPERATING ENVELOPE`; H18.6 `SUPPORTED IN A LIMITED / CONFIGURATION-SPECIFIC SENSE`; H18.8 `SUPPORTED FOR D=32–256` (`PHASE_18_INTERPRETATION.md`).
  - [x] **EC18.15**: Code-health gate passes: 0 Pyright errors, 0 compile errors, 656 pytest tests passing cleanly, 0 Ruff errors.
  - [x] **EC18.16**: All 10 serialized JSON artifacts and 18 Observatory publication plots reproducible via single command `python3 examples/phase_18_transfer_benchmark.py`.
  - [x] Phase Gate resolved to **Outcome A** (Multi-family Transfer Evidence: The minimal SafeAdaptiveDelta mechanism transfers across multiple distinct tested non-stationary classification environments without per-task retuning within the tested linear-associative operating envelope, while explicitly characterizing failure boundaries; proceed to Phase 19).


---

### Phase 19: Autonomous Coherence Gating & Nonlinear State Transfer [PLANNED]
- **Objective**: Address the primary failure modes discovered in Phase 18 by developing: (1) an **autonomous coherence gating mechanism** that dynamically dampens or resets stale associative state $M_t$ under incompatible distribution shifts without an oracle signal, and (2) structured feature embeddings that expand representation capacity beyond the linear-head ceiling without requiring test-time backpropagation.


