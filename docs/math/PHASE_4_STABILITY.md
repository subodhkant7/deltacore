# Phase 4 Mathematical Specification: Stability Controllers & Non-Expansion Guarantees

This document establishes the rigorous mathematical foundation for the stability controls introduced in **DeltaCore Phase 4**.

---

## 1. Prime Directive: Derivation Before Implementation

The objective of Phase 4 is **NOT** merely to "prevent NaN values" through ad-hoc clipping.

The objective is:
> **Derive explicit sufficient conditions under which the relevant recurrent memory transitions are non-expansive, implement modular controllers that enforce those conditions, and experimentally verify the stated mathematical invariants.**

---

## 2. Content Memory Local Residual Derivation

### 2.1. The Recurrence
From Phase 1–3, the content memory state $M_t \in \mathbb{R}^{V \times K}$ updates via the error-correcting delta rule:

$$
\hat{v}_t = M_t k_t \in \mathbb{R}^V
$$

$$
e_t = v_t - \hat{v}_t = v_t - M_t k_t \in \mathbb{R}^V
$$

$$
M_{t+1} = M_t + \eta_t e_t k_t^\top \in \mathbb{R}^{V \times K}
$$

where $k_t \in \mathbb{R}^K$ is the query key, $v_t \in \mathbb{R}^V$ is the target value, and $\eta_t \in \mathbb{R}_{>0}$ is the step size.

---

### 2.2. Immediate-Key Residual Contraction Theorem

#### Theorem 1 (Immediate-Key Residual Transition)
Let $e_{t+1}^{(k_t)} = v_t - M_{t+1} k_t$ denote the prediction error of the updated memory $M_{t+1}$ evaluated on the **same key** $k_t$ and target $v_t$. Then:

$$
e_{t+1}^{(k_t)} = \left(1 - \eta_t \|k_t\|_2^2\right) e_t
$$

#### Proof:
By substitution of the update equation:

$$
M_{t+1} k_t = \left(M_t + \eta_t e_t k_t^\top\right) k_t = M_t k_t + \eta_t e_t (k_t^\top k_t) = \hat{v}_t + \eta_t \|k_t\|_2^2 e_t
$$

Subtracting from target $v_t$:

$$
e_{t+1}^{(k_t)} = v_t - M_{t+1} k_t = v_t - \hat{v}_t - \eta_t \|k_t\|_2^2 e_t = e_t - \eta_t \|k_t\|_2^2 e_t = \left(1 - \eta_t \|k_t\|_2^2\right) e_t \quad \blacksquare
$$

---

### 2.3. Necessary and Sufficient Local Non-Expansion Condition

#### Corollary 1.1 (Local Non-Expansion)
The scalar residual magnitude contracts or remains non-expansive ($\|e_{t+1}^{(k_t)}\|_2 \le \|e_t\|_2$) if and only if:

$$
\left|1 - \eta_t \|k_t\|_2^2\right| \le 1
$$

Solving this inequality:

$$
-1 \le 1 - \eta_t \|k_t\|_2^2 \le 1 \iff 0 \le \eta_t \|k_t\|_2^2 \le 2
$$

#### Corollary 1.2 (Strict Local Contraction)
For any chosen tolerance $\epsilon \in (0, 1)$, strict contraction ($\|e_{t+1}^{(k_t)}\|_2 \le (1 - \epsilon) \|e_t\|_2$) is guaranteed if and only if:

$$
\epsilon \le \eta_t \|k_t\|_2^2 \le 2 - \epsilon
$$

#### Four Distinct Dynamical Regimes:
1. **Under-Correction ($\eta_t \|k_t\|_2^2 < 1$)**: Residual contracts monotonically without sign reversal ($0 < 1 - \eta_t \|k_t\|_2^2 < 1$).
2. **Exact One-Step Recall ($\eta_t \|k_t\|_2^2 = 1$)**: Residual is eliminated in a single step ($e_{t+1}^{(k_t)} = 0$).
3. **Over-Correction with Contraction ($1 < \eta_t \|k_t\|_2^2 < 2$)**: Residual contracts in magnitude but flips sign (oscillatory damping).
4. **Expansion Boundary ($\eta_t \|k_t\|_2^2 = 2$)**: Residual flips sign with identical magnitude ($e_{t+1}^{(k_t)} = -e_t$).
5. **Divergence Regime ($\eta_t \|k_t\|_2^2 > 2$)**: Residual magnitude expands geometrically ($|1 - \eta_t \|k_t\|_2^2| > 1$).

---

### 2.4. Explicit Local Assumptions & Scope Limitations
> **Critical Scientific Limitation**:
> Theorem 1 is an **immediate-key local result**.
> It establishes what happens to the residual along the 1D subspace spanned by $k_t$.
> It does **NOT** imply:
> 1. That error on an arbitrary future key $k_{t+1} \neq k_t$ decreases.
> 2. That the matrix norm $\|M_t\|_F$ is monotonically non-increasing over arbitrary sequence inputs.
> 3. Global asymptotic stability of the coupled recurrent state machine.

---

## 3. Dynamics Memory Local Residual Derivation

### 3.1. The Recurrence
From Phase 3, the dynamics memory $C_t \in \mathbb{R}^{1 \times D_c}$ updates via the delta rule on the control target $c_t^{\text{target}} \in \mathbb{R}$:

$$
q_t = C_t z_t \in \mathbb{R}
$$

$$
d_t = c_t^{\text{target}} - q_t = c_t^{\text{target}} - C_t z_t \in \mathbb{R}
$$

$$
C_{t+1} = C_t + \rho_t d_t z_t^\top \in \mathbb{R}^{1 \times D_c}
$$

where $z_t \in \mathbb{R}^{D_c}$ is the control feature vector and $\rho_t \in \mathbb{R}_{>0}$ is the dynamics learning rate.

---

### 3.2. Immediate-Feature Residual Contraction Theorem

#### Theorem 2 (Immediate-Feature Dynamics Residual Transition)
Let $d_{t+1}^{(z_t)} = c_t^{\text{target}} - C_{t+1} z_t$ denote the dynamics prediction error of the updated controller $C_{t+1}$ evaluated on the **same feature vector** $z_t$. Then:

$$
d_{t+1}^{(z_t)} = \left(1 - \rho_t \|z_t\|_2^2\right) d_t
$$

#### Proof:
Analogous to Theorem 1:

$$
C_{t+1} z_t = \left(C_t + \rho_t d_t z_t^\top\right) z_t = C_t z_t + \rho_t d_t (z_t^\top z_t) = q_t + \rho_t \|z_t\|_2^2 d_t
$$

Subtracting from target $c_t^{\text{target}}$:

$$
d_{t+1}^{(z_t)} = c_t^{\text{target}} - C_{t+1} z_t = c_t^{\text{target}} - q_t - \rho_t \|z_t\|_2^2 d_t = \left(1 - \rho_t \|z_t\|_2^2\right) d_t \quad \blacksquare
$$

#### Corollary 2.1 (Dynamics Local Non-Expansion Condition)
The dynamics residual is non-expansive ($|d_{t+1}^{(z_t)}| \le |d_t|$) if and only if:

$$
0 \le \rho_t \|z_t\|_2^2 \le 2
$$

Strict contraction is enforced by bounding:

$$
0 \le \rho_t \|z_t\|_2^2 \le \beta_C < 2
$$

---

## 4. Operator Norm Formulation & Global Matrix Dynamics

To understand global matrix growth beyond the immediate key, we formulate the content memory transition as an affine discrete-time linear dynamical system.

### 4.1. Affine Operator Decomposition
Rewrite the content update:

$$
M_{t+1} = M_t + \eta_t (v_t - M_t k_t) k_t^\top = M_t \left(I_K - \eta_t k_t k_t^\top\right) + \eta_t v_t k_t^\top
$$

Define:
- **Homogeneous Transition Operator**:
  $$A_t = I_K - \eta_t k_t k_t^\top \in \mathbb{R}^{K \times K}$$
- **Non-Homogeneous Affine Drive**:
  $$B_t = \eta_t v_t k_t^\top \in \mathbb{R}^{V \times K}$$

The state transition is:

$$
M_{t+1} = M_t A_t + B_t
$$

---

### 4.2. Spectral Analysis of the Transition Operator $A_t$

#### Theorem 3 (Operator Norm of $A_t$)
Let $k_t \neq 0$ and $\eta_t > 0$. The symmetric matrix $A_t = I_K - \eta_t k_t k_t^\top$ has eigenvalues:

$$
\lambda_1 = 1 - \eta_t \|k_t\|_2^2 \quad \text{with eigenvector } k_t
$$

$$
\lambda_j = 1 \quad \text{for all } j \in \{2, \dots, K\} \text{ orthogonal to } k_t
$$

Consequently, the spectral norm (induced matrix 2-norm) of $A_t$ is:

$$
\|A_t\|_2 = \max\left(\left|1 - \eta_t \|k_t\|_2^2\right|, 1\right)
$$

#### Corollary 3.1 (Non-Expansiveness of $A_t$)
The operator $A_t$ is non-expansive ($\|A_t\|_2 \le 1$) if and only if:

$$
0 \le \eta_t \|k_t\|_2^2 \le 2
$$

Under this condition:

$$
\|A_t\|_2 = 1
$$

and for any memory state $M_t$:

$$
\|M_t A_t\|_2 \le \|M_t\|_2 \|A_t\|_2 = \|M_t\|_2
$$

$$
\|M_t A_t\|_F \le \|M_t\|_F \|A_t\|_2 = \|M_t\|_F
$$

---

### 4.3. What This Proves vs What It Does NOT Prove

#### What is Proved:
- The autonomous/homogeneous part of the memory evolution ($M \mapsto M A_t$) **never amplifies the existing memory norm**:
  $$\|M_t A_t\|_F \le \|M_t\|_F$$
  It is strictly contractive along the 1D subspace of $k_t$ (when $0 < \eta_t \|k_t\|_2^2 < 2$) and isometric on the orthogonal complement.

#### What is NOT Proved (The Affine Drive Limitation):
- Because $B_t = \eta_t v_t k_t^\top$ is added at each step:
  $$\|M_{t+1}\|_F \le \|M_t A_t\|_F + \|B_t\|_F \le \|M_t\|_F + \eta_t \|v_t\|_2 \|k_t\|_2$$
  Since $\|B_t\|_F \ge 0$, new external information drives the memory norm upward.
- Therefore, bounding $\eta_t \|k_t\|_2^2 \le 2$ guarantees that the memory **does not internally self-amplify exponentially**, but it does **NOT** guarantee that $\|M_t\|_F$ remains constant or decreases when persistent external targets $v_t$ are absorbed.
- This is a structural property of linear associative memory: storing new information requires updating the matrix.

---

## 5. Stability Controller Contracts

### 5.1. Separation of Raw Adaptation from Safety Control
A core architectural requirement of DeltaCore is that **stability control must not replace adaptive dynamics**. It acts as a supervisory constraint:

```
[ Self-Referential Controller ]
               │
               ▼
        Raw Step Size (\eta_t^{\text{raw}})
               │
               ▼
   [ SafeStepSizeController ]  <── Query Key (k_t)
               │
               ▼
        Safe Step Size (\eta_t^{\text{safe}})
               │
               ▼
      [ Content Delta Update ]
```

---

### 5.2. Content Step Size Safety Controller
Given raw step size $\eta_t^{\text{raw}}$, key $k_t$, safety coefficient $\beta \in (0, 2)$, and numerical floor $\epsilon > 0$:

$$
\eta_t^{\text{bound}} = \frac{\beta}{\|k_t\|_2^2 + \epsilon}
$$

$$
\eta_t^{\text{safe}} = \min\left(\eta_t^{\text{raw}}, \eta_t^{\text{bound}}\right)
$$

- Parameter $\beta$: Configurable safety coefficient ($0 < \beta < 2$). Default $\beta = 1.0$ (target contraction center).
- **Strict Inequality Introduced by $\epsilon$**:
  Notice that under clipping ($\eta_t^{\text{raw}} \ge \eta_t^{\text{bound}}$):
  $$\gamma_t = \eta_t^{\text{safe}} \|k_t\|_2^2 = \frac{\beta \|k_t\|_2^2}{\|k_t\|_2^2 + \epsilon} < \beta \quad \text{for any } \epsilon > 0$$
  Therefore, the actual attained normalized step is strictly less than $\beta$. For example, with $\|k_t\|_2^2 = 9.0$ and $\epsilon = 10^{-8}$, $\gamma_t = \beta \frac{9}{9 + 10^{-8}} = \beta (1 - 1.11 \times 10^{-9}) < \beta$.
- Stability Margin:
  $$\mu_t = 2 - \gamma_t > 2 - \beta > 0$$

---

### 5.3. Dynamics Rate Safety Controller
Given raw dynamics learning rate $\rho_t^{\text{raw}}$, feature vector $z_t$, safety coefficient $\beta_C \in (0, 2)$, and numerical floor $\epsilon > 0$:

$$
\rho_t^{\text{bound}} = \frac{\beta_C}{\|z_t\|_2^2 + \epsilon}
$$

$$
\rho_t^{\text{safe}} = \min\left(\rho_t^{\text{raw}}, \rho_t^{\text{bound}}\right)
$$

- **Strict Inequality Introduced by $\epsilon$**:
  $$\gamma_{C, t} = \rho_t^{\text{safe}} \|z_t\|_2^2 = \frac{\beta_C \|z_t\|_2^2}{\|z_t\|_2^2 + \epsilon} < \beta_C < 2$$
- Dynamics Stability Margin:
  $$\mu_{C, t} = 2 - \gamma_{C, t} > 2 - \beta_C > 0$$

---

## 6. Numerical Safeguards Policy

1. **Zero Key ($k_t = 0$)**:
   - $\|k_t\|_2^2 = 0$. The bound becomes $\frac{\beta}{\epsilon}$.
   - Content update is identically zero ($\Delta M_t = \eta_t e_t 0^\top = 0$).
2. **Zero Features ($z_t = 0$)**:
   - Bounded by $\frac{\beta_C}{\epsilon}$. Dynamics update is zero.
3. **Non-Finite Inputs ($NaN$, $Inf$)**:
   - Non-finite numbers are **NEVER silently sanitized into finite zeros**.
   - Input checks reject $NaN$/$Inf$ with a descriptive `FloatingPointError` or propagate them visibly according to the active configuration.
4. **Precision**:
   - Both controllers support FP32 (`float32`) and FP64 (`float64`) transparently, preserving device and dtype.

---

## 7. Stability Telemetry Contract

Every step under stability control exposes:
- `raw_step_size`: $\eta_t^{\text{raw}}$
- `safe_step_size`: $\eta_t^{\text{safe}}$
- `step_size_clipped`: $\eta_t^{\text{safe}} < \eta_t^{\text{raw}}$
- `normalized_step`: $\gamma_t = \eta_t^{\text{safe}} \|k_t\|_2^2$
- `stability_margin`: $\mu_t = 2 - \gamma_t$
- `raw_dynamics_rate`: $\rho_t^{\text{raw}}$
- `safe_dynamics_rate`: $\rho_t^{\text{safe}}$
- `dynamics_rate_clipped`: $\rho_t^{\text{safe}} < \rho_t^{\text{raw}}$
- `normalized_dynamics_rate`: $\gamma_{C, t} = \rho_t^{\text{safe}} \|z_t\|_2^2$
- `dynamics_stability_margin`: $\mu_{C, t} = 2 - \gamma_{C, t}$
- `all_states_finite`: boolean True iff all states across all steps remain finite.
- `terminal_state_finite`: boolean True iff final state is finite.
- `first_nonfinite_step`: integer index of first non-finite step, or None.
- `finite_state`: backwards-compatible alias to `all_states_finite`.
