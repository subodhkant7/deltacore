# Standard Metric Definitions for DeltaCore Benchmarks

This document formalizes the mathematical definitions, measurement windows, normalizations, edge-case conventions, and directional objectives for all metrics in DeltaCore benchmark evaluation.

---

## 1. Retrieval & Prediction Error

### 1.1. Final Error ($E_T$)
- **Mathematical Definition**:
  $$E_T = \|e_T\|_2 = \|v_T - \hat{v}_T\|_2 = \|v_T - M_T k_T\|_2$$
- **Window**: Measured strictly at the terminal step $t = T-1$.
- **Units**: Euclidean distance in target space $\mathbb{R}^V$.
- **Direction**: Lower is better ($\ge 0$).
- **Edge-case**: If non-finite values occur, the run is flagged as `NUMERICAL_FAILURE`.

### 1.2. Mean Error ($\bar{E}_{[t_1, t_2]}$)
- **Mathematical Definition**:
  $$\bar{E}_{[t_1, t_2]} = \frac{1}{t_2 - t_1} \sum_{t=t_1}^{t_2-1} \|e_t\|_2$$
- **Window**: Specified interval $[t_1, t_2]$ (e.g. pre-shift $[0, t_s]$, post-shift $[t_s, T]$).
- **Units**: Euclidean distance.
- **Direction**: Lower is better.

---

## 2. Recovery Dynamics

### 2.1. First-Passage Recovery ($T_{\text{FP}}$)
- **Mathematical Definition**:
  $$T_{\text{FP}} = \min \{ j \ge 0 : \|e_{t_s + j}\|_2 \le \tau \|e_{t_s}\|_2 \}$$
- **Window**: Evaluated for steps $t \ge t_s$ following an abrupt shock at $t_s$.
- **Parameters**: $\tau \in (0, 1)$ (default $\tau = 0.5$, representing 50% error reduction relative to immediate shock error).
- **Units**: Integer steps ($j \ge 0$).
- **Direction**: Lower is better.
- **Edge-case**: If the threshold is never attained within the remainder of the sequence, $T_{\text{FP}} = \text{None}$ (recorded as a recovery failure; not replaced with an arbitrary large number).

### 2.2. Sustained Recovery ($T_{\text{sustained}}(W)$)
- **Mathematical Definition**:
  $$T_{\text{sustained}}(W) = \min \left\{ j \ge 0 : \|e_{t_s + j + w}\|_2 \le \tau \|e_{t_s}\|_2, \; \forall w \in [0, W-1] \right\}$$
- **Window**: Evaluated over sliding window of length $W$ post-shift.
- **Parameters**: Window length $W \ge 1$ (default $W = 4$), threshold fraction $\tau$ (default $0.5$).
- **Units**: Integer steps.
- **Direction**: Lower is better.
- **Edge-case**: If error drops below threshold transiently but oscillates above it, or if $T - t_s - j < W$, returns $\text{None}$.

---

## 3. Comparative Adaptation

### 3.1. Relative Adaptation Gain ($\text{Gain}$)
- **Mathematical Definition**:
  $$\text{Gain}(M, \text{Baseline}) = \frac{E_{\text{baseline}} - E_{\text{eval}}}{\max(E_{\text{baseline}}, \epsilon)}$$
- **Sign Convention**:
  - $\text{Gain} > 0$: Improvement over baseline (eval has lower error).
  - $\text{Gain} < 0$: Degradation relative to baseline.
  - $\text{Gain} = 0$: Identical error.
- **Units**: Dimensionless fraction (e.g. $+0.25$ indicates 25% lower error).
- **Direction**: Higher is better.

---

## 4. State Growth & Energy Budgets

### 4.1. State Growth Ratio ($G_M$)
- **Mathematical Definition**:
  $$G_M = \frac{\max_{t \ge t_s} \|M_t\|_F}{\max(\|M_{t_s}\|_F, \epsilon)}$$
- **Window**: Steps $t \ge t_s$.
- **Units**: Dimensionless ratio.
- **Direction**: Lower is better (bounded near 1.0 indicates stable memory footprint).

### 4.2. Update Energy ($U_M$)
- **Mathematical Definition**:
  $$U_M = \sum_{t=0}^{T-1} \|\Delta M_t\|_F$$
- **Units**: Cumulative Frobenius norm.
- **Direction**: Context-dependent; quantifies the total state modification expended to achieve adaptation.

### 4.3. Step Energy ($U_\eta$)
- **Mathematical Definition**:
  $$U_\eta = \sum_{t=0}^{T-1} \eta_t$$
- **Units**: Cumulative step size.
- **Direction**: Context-dependent; measures total learning rate budget.

---

## 5. Numerical Stability Invariants

### 5.1. Stability Margin ($S_t$)
- **Mathematical Definition**:
  $$S_t = 2 - \eta_t \|k_t\|_2^2$$
- **Contract**: For local non-expansion of the immediate key residual, $S_t \in [0, 2]$. $S_t > 0$ implies strict contraction.
- **Summary Metric**: Minimum stability margin $\min_t S_t$.
- **Direction**: Higher is safer ($\ge 0$ required for local non-expansion).

### 5.2. Maximum Normalized Step ($\gamma_{\max}$)
- **Mathematical Definition**:
  $$\gamma_{\max} = \max_t \left( \eta_t \|k_t\|_2^2 \right)$$
- **Contract**: Must remain $\le 2$ for local residual non-expansion.
- **Direction**: Lower is safer.

### 5.3. Finite-State Semantics
- `all_states_finite`: Boolean flag indicating whether every intermediate state $M_t, C_t$ and prediction $\hat{v}_t$ in the trajectory is finite.
- `terminal_state_finite`: Boolean flag indicating whether the final state $M_T$ is finite.
- `first_nonfinite_step`: Optional integer recording the exact step $t$ where non-finiteness first occurred.
