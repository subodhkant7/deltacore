# DeltaCore Adaptive State Observatory Architecture

This document establishes the architectural specification for the **DeltaCore Adaptive State Observatory**, a reproducible scientific analysis and diagnostic layer for inspecting adaptive neural-state trajectories during sequence processing.

---

## 1. Scientific Objective

The Adaptive State Observatory answers:
> *What changed in the internal state, when did it change, why did the update magnitude change, and what happened immediately before and after a distribution shift or numerical instability?*

The Observatory does **not** provide a live web server, GUI dashboard, or interactive framework. It is an offline, headless-first, deterministic analytical pipeline producing:
1. Typed, validated trajectory representations with explicit missing-data semantics.
2. Automated event extraction (distribution shifts, threshold crossings, recovery latencies, non-finite states, clipping events).
3. Mathematical diagnostics (energy budgets, contraction margins, stability regimes).
4. Publication-quality static plots (Matplotlib-based).
5. Self-contained, provenance-tracked Markdown reproducibility reports.

---

## 2. Pipeline Architecture

The Observatory strictly separates data loading, mathematical analysis, visualization, and reporting:

```text
Benchmark Result (JSON artifact / RunResult / AggregateResult)
      │
      ▼
Trajectory Loader (deltacore.observatory.loader)
      │  - Schema validation
      │  - Trajectory extraction
      │  - Strict missing-value preservation
      ▼
Normalized Trajectory (deltacore.observatory.schema.StateTrajectory)
      │
      ├────────────────────────────────────────┬────────────────────────────────────────┐
      ▼                                        ▼                                        ▼
Temporal Analysis                        Stability Analysis                       Event Extraction
(deltacore.observatory.analysis)         (deltacore.observatory.analysis)         (deltacore.observatory.events)
  - Error trajectory E_t                   - Stability margin S_t                   - Shift boundaries
  - State norm ||M_t||_F                   - Dynamics margin S_t^{dyn}              - First-passage recovery
  - Update norm ||ΔM_t||_F                 - Normalized step bounds                 - Sustained recovery
  - Step size η_t                          - Finite state verification              - First non-finite step
  - Shift response metrics                 - Clip event quantification              - Max update / state norm
      │                                        │                                        │
      └────────────────────────────────────────┴────────────────────────────────────────┘
                                               │
                                               ▼
                               Comparative Analysis & Fingerprints
                               (deltacore.observatory.analysis / fingerprint)
                                 - Cross-model trajectory deltas
                                 - Compact numerical run descriptors
                                               │
                                               ├────────────────────────────────────────┐
                                               ▼                                        ▼
                                         Static Plots                         Reproducible Report
                                  (deltacore.observatory.plots)           (deltacore.observatory.report)
                                    - Plot A: Error vs. Time                - Metadata & environment
                                    - Plot B: Step Size vs. Time            - Trajectory summary
                                    - Plot C: Memory Norm vs. Time          - Shift response & stability
                                    - Plot D: Update Norm vs. Time          - Replay verification status
                                    - Plot E: Stability Margin vs. Time     - Embedded plot artifacts
                                    - Plot F: Comparative Recovery          - Epistemic limitations
                                    - Plot G: Key Correlation vs. Error
                                    - Plot H: Recovery vs. Update Energy
                                    - Plot I: Failure Step Distribution
```

---

## 3. Core Architectural Invariants

1. **Decoupled Analysis and Visualization**:
   Mathematical and event extraction functions must never import plotting libraries or depend on graphic contexts. Analysis functions return pure Python dataclasses and numerical arrays.
2. **Explicit Missing Data Semantics**:
   - `None`: Metric is not applicable, unsupported, or never achieved (e.g. unachieved recovery latency, frozen model step size).
   - `float('nan')`: Computation encountered an undefined numerical operation.
   - `0.0`: The mathematical value is measured or derived as exact zero.
   Missing metrics must never be substituted with `0.0` or arbitrary sentinel numbers like `9999`.
3. **Failure Visibility & Unobserved Post-Failure States**:
   If a trajectory experiences numerical failure (e.g., at step $t^*$), the trajectory must flag `all_states_finite = False`, record `first_nonfinite_step = t^*`, and preserve pre-failure states.
   - For the diverged step $t^*$: `observed = True, finite_state = False`.
   - For all steps strictly after termination ($t > t^*$): `observed = False, finite_state = None`. A step after execution terminated must never be described as an observed non-finite state. Metrics for unobserved steps are strictly `None`.
   Under no circumstances may an intermediate failure be silently disguised as a healthy terminal run.
4. **No Unjustified Scalar Aggregations**:
   The Observatory never computes arbitrary "overall quality scores" or automated "winner" tags. Multi-objective trade-offs (e.g. recovery speed vs. update energy vs. contractive margin) are presented as distinct dimensions.
5. **Headless Execution**:
   All plotting uses standard Matplotlib headless backends (`Agg`). No GUI windows, display servers, or web services are required.
