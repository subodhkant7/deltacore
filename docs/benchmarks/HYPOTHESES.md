# DeltaCore Scientific Hypothesis Registry

This document records the formal empirical hypotheses established prior to executing large-scale benchmark sweeps. In accordance with DeltaCore scientific integrity principles, hypotheses specify explicit independent variables, dependent metrics, reference baselines, and pre-declared falsification conditions. Hypotheses are never rewritten post hoc to match observations.

---

## Hypothesis 1 (H1): Self-Reference Acceleration Under Distribution Shift

- **Statement**: Online co-evolution of controller memory $C_t$ enables faster post-shock recovery than fixed-rate adaptation when distribution shifts occur.
- **Independent Variable**: Model architecture (`SelfReferential` / `SafeSelfReferential` vs. `FixedDelta`).
- **Dependent Metric**: First-passage recovery latency $T_{\text{FP}}$ (steps to 50% error reduction).
- **Reference Baseline**: `FixedDelta` ($\eta_0 = 0.1$).
- **Falsification Condition**: H1 is falsified if `FixedDelta` achieves lower or equal mean $T_{\text{FP}}$ compared to `SafeSelfReferential` across $\ge 60\%$ of tested seeds under standard target inversion.

---

## Hypothesis 2 (H2): State Adaptation Cost Under Stationary Data

- **Statement**: In a stationary associative recall task where keys and targets are time-invariant, unconstrained self-referential dynamics induce unnecessary state drift and higher final error than fixed or frozen baselines.
- **Independent Variable**: Model architecture (`SelfReferential` vs. `FixedDelta` / `Frozen`).
- **Dependent Metric**: Final retrieval error $E_T$ and state update energy $U_M = \sum_t \|\Delta M_t\|_F$.
- **Reference Baseline**: `FixedDelta` and `Frozen`.
- **Falsification Condition**: H2 is falsified if `SelfReferential` achieves lower final error $E_T$ and lower update energy than `FixedDelta` in stationary recall.

---

## Hypothesis 3 (H3): Geometric Key Correlation Monotonically Increases Cross-Talk

- **Statement**: Cross-talk retrieval error between two competing associations increases monotonically as their key cosine similarity $\rho = \langle k_1, k_2 \rangle$ increases from $0.0$ to $1.0$.
- **Independent Variable**: Key cosine similarity $\rho \in [0.0, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]$.
- **Dependent Metric**: Mean retrieval error $\bar{E} = \frac{1}{2} (\|M k_1 - v_1\|_2 + \|M k_2 - v_2\|_2)$.
- **Reference Baseline**: Orthogonal baseline ($\rho = 0.0$).
- **Falsification Condition**: H3 is falsified if $\bar{E}(\rho_2) < \bar{E}(\rho_1)$ for any pair $\rho_2 > \rho_1 + 0.1$ across repeated seeds.

---

## Hypothesis 4 (H4): Safety Control Reduces Numerical Failure Under Stress

- **Statement**: Enforcing closed-form local non-expansion bounds $\eta_t \le \frac{\beta}{\|k_t\|^2 + \epsilon}$ reduces numerical failure (`inf`/`nan`) in adversarial regimes where unconstrained dynamics diverge.
- **Independent Variable**: Stability controller mode (`SafeSelfReferential` vs. `SelfReferential`).
- **Dependent Metric**: Numerical success rate ($\%$ `SUCCESS` vs. `NUMERICAL_FAILURE`) and `all_states_finite`.
- **Reference Baseline**: `SelfReferential` with `UnconstrainedController`.
- **Falsification Condition**: H4 is falsified if `SafeSelfReferential` encounters any `NUMERICAL_FAILURE` or non-finite state within the tested sequence horizon under the standard stability stress task.

---

## Hypothesis 5 (H5): Recovery Speed Associated with Update Energy

- **Statement**: Faster recovery following an abrupt shock was associated with higher observed update energy in the tested configuration: systems that achieve lower $T_{\text{FP}}$ expend higher update energy $U_{\text{rec}} = \sum_{t=t_s}^{t_s + T_{\text{FP}}} \|\Delta M_t\|_F$.
- **Independent Variable**: Model adaptation aggressiveness (step sizes and controller adaptation rates).
- **Dependent Metric**: Cumulative update energy to recovery $U_{\text{rec}}$ versus recovery latency $T_{\text{FP}}$.
- **Reference Baseline**: `FixedDelta`.
- **Falsification Condition**: H5 is falsified if an adaptive configuration achieves strictly faster recovery ($T_{\text{FP}}^{\text{eval}} < T_{\text{FP}}^{\text{base}}$) while consuming lower cumulative update energy ($U_{\text{rec}}^{\text{eval}} < U_{\text{rec}}^{\text{base}}$).
