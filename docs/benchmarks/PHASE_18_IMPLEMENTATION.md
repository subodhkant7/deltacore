# Phase 18 Implementation: Unseen Classification Regime Transfer & Falsification

This document records the exact mathematical specifications, architecture freezes, streaming protocols, multi-family generators, and algorithmic implementations for **Phase 18: Unseen Classification Regime Transfer & Falsification**.

---

## 1. Scientific Mission & Core Research Questions

Phase 18 tests whether the online adaptive classification gains discovered in Phase 17 transfer to **genuinely different non-stationary classification environments without task-specific hyperparameter retuning**, while preserving the smallest successful associative mechanism.

This is a **scientific falsification phase**, not a capability-expansion phase.

### Primary Research Question
$$
\boxed{
\text{Does the minimal SafeAdaptiveDelta associative-state mechanism transfer across unseen non-stationary classification environments without task-specific retuning?}
}
$$

### Secondary Research Question
$$
\boxed{
\text{Where does persistent adaptive state help, where does it hurt, and what characteristics of a distribution shift determine that outcome?}
}
$$

---

## 2. Mechanism & Mathematical Freeze

The primary mechanism for Phase 18 is strictly frozen to the minimal successful Phase 17 formulation.

### 2.1 Decision Rule
Given streaming feature vector $x_t \in \mathbb{R}^D$ and associative state $M_t \in \mathbb{R}^{D \times D}$:
$$
\hat{y}_t = \arg\max_k \left[ W_{\mathrm{head}} (I + M_t) x_t + b_{\mathrm{head}} \right]_k
$$
where $W_{\mathrm{head}} \in \mathbb{R}^{K \times D}$ and $b_{\mathrm{head}} \in \mathbb{R}^K$ are offline-fit parameters that remain strictly frozen ($\Delta\theta = 0$) during the entire streaming evaluation.

Predicted class probabilities are computed via the softmax operator:
$$
\hat{p}_t = \text{softmax}\left( W_{\mathrm{head}} (I + M_t) x_t + b_{\mathrm{head}} \right)
$$

### 2.2 Online State Transition
Upon revelation of true ground-truth label $y_t \in \{0, \dots, K-1\}$:
1. **One-Hot Target & Logit Prediction Error**:
   $$
   e_{\mathrm{logit}, t} = y_{\mathrm{onehot}, t} - \hat{p}_t \in \mathbb{R}^K
   $$
2. **Feature-Space Backprojected Error**:
   $$
   e_{x, t} = W_{\mathrm{head}}^\top e_{\mathrm{logit}, t} \in \mathbb{R}^D
   $$
3. **Adaptive Step Size with Error Dampening and Contraction Bound**:
   $$
   \eta_t = \min\left( \frac{\eta_0}{1 + \gamma \|e_{x, t}\|_2^2}, \frac{\rho}{\|x_t\|_2^2 + \epsilon} \right)
   $$
4. **Adaptive Retention Factor**:
   $$
   \alpha_t = \max\left( \alpha_{\min}, 1 - \eta_t \|x_t\|_2^2 \right)
   $$
5. **Associative State Update**:
   $$
   M_{t+1} = \alpha_t M_t + \eta_t e_{x, t} x_t^\top
   $$

### 2.3 Explicit Negative Design Constraints
In strict accordance with Phase 18 specifications, the implementation introduces:
- NO multi-memory or VisionHOPE-style mechanisms
- NO neural networks or learned controllers added during evaluation
- NO test-time optimizer steps or backpropagation
- NO meta-learning or hyperparameter retuning
- NO oracle regime boundary signals or future-label lookahead
- NO handcrafted recovery branches.

---

## 3. Hyperparameter Freeze

The evaluation configuration was chosen prior to evaluation and frozen across all tasks:

| Parameter | Value | Role |
| :--- | :---: | :--- |
| $\eta_0$ | `0.015` | Baseline learning rate candidate |
| $\rho$ | `1.50` | Contraction spectral radius bound ($\eta_t \|x_t\|_2^2 \le \rho$) |
| $\alpha_{\min}$ | `0.95` | Minimum retention lower bound |
| $\gamma$ | `0.10` | Squared error dampening coefficient |
| $\epsilon$ | `1e-6` | Denominator numerical stability regularizer |

**Protocol Rule**: Under no circumstances was any hyperparameter retuned per task or per seed.

---

## 4. Multi-Family Task Generation Architecture

Phase 18 introduces **three genuinely distinct task families** and **six shift dimensions**:

### 4.1 Family A — Linear Decision Boundary Rotation
- **Geometry**: The class prototype subspace is rotated by angle $\theta = \pi / 2.5 \approx 72^\circ$ in the canonical span via $Q_{\mathrm{proto}} R_K(\theta) Q_{\mathrm{proto}}^\top + (I - Q Q^\top)$.
- **Regime Structure**: Regime A (120 steps) $\to$ Regime C (120 steps, rotated) $\to$ Regime A (120 steps, exact return).
- **Purpose**: Direct transfer and replication check against Phase 17.

### 4.2 Family B — Boundary Translation / Intercept Shift
- **Geometry**: Rather than rotating feature axes, class centroids undergo translation offsets in the prototype coordinate space:
  $$
  P_B^{(k)} = P_A^{(k)} + \tau \left( P_A^{( (k+1) \bmod K )} - P_A^{(k)} \right)
  $$
  with $\tau = 0.50$.
- **Regime Structure**: Regime A (120 steps) $\to$ Regime B (120 steps, translated) $\to$ Regime A (120 steps, return).
- **Purpose**: Distinguishes adaptation to geometric orientation from adaptation to intercept offsets and shifting class boundaries.

### 4.3 Family C — Nonlinear Decision Boundary Deformation
- **Geometry**: Applies a radial/quadratic nonlinear boundary warp in the prototype coordinate subspace:
  $$
  u_i' = \cos(\theta_0) u_i - \sin(\theta_0) u_{i+1} + \frac{0.5 (u_i^2 - u_{i+1}^2)}{\sqrt{D}}, \quad u_{i+1}' = \sin(\theta_0) u_i + \cos(\theta_0) u_{i+1}
  $$
- **Regime Structure**: Regime A (120 steps) $\to$ Regime C (120 steps, nonlinear warp) $\to$ Regime A (120 steps, return).
- **Purpose**: Tests whether the minimal linear-head + associative-state mechanism can adapt when boundaries are non-planar.

### 4.4 Additional Shift Dimensions
1. **Covariate Shift**: Anisotropic covariance distortion via $Q \Sigma Q^\top$ ($\kappa = 6.0$) with strictly invariant class prototypes and decision boundaries.
2. **Gradual Drift**: Continuous smooth rotation $\theta(t) = \frac{t}{T} \theta_{\max}$ over $T=360$ steps.
3. **Class-Prior Imbalance**: Fixed conditional feature geometry $P(x|y)$ with shifting class frequencies $P(y)$ ($80\%$ probability concentrated in classes 0 and 1).
4. **Strong Mismatch (Adversarial Falsification)**: 120 steps of shifted Regime A (accumulating state $M_A \neq 0$) followed by a short 40-step window of abruptly incompatible opposite rotation ($-\theta_A$), deliberately penalizing persistent state compared to reset.

---

## 5. Strict Online Causal Protocol & Immutability Verification

Execution is governed by [CausalStreamingProtocol](file:///Users/urjasoft/Documents/DeltaCore/deltacore/benchmarks/phase_18/protocol.py#L66):
1. **Prediction Step**: $\hat{y}_t = f(x_t, S_{t-1})$ executed with label $y_t$ completely withheld.
2. **Timestamp Monotonicity**: Programmatic nanosecond assertions enforce:
   $$
   t_{\mathrm{pred\_start}} \le t_{\mathrm{pred\_end}} \le t_{\mathrm{reveal}} \le t_{\mathrm{adapt\_start}} \le t_{\mathrm{adapt\_end}}
   $$
3. **Parameter Immutability ($\Delta\theta = 0$)**: Evaluated bit-for-bit via pre- and post-stream SHA-256 hashes of all parameters. A mismatch immediately raises a runtime failure.

---

## 6. Code Health & Verification Summary

All implementation code conforms strictly to DeltaCore quality standards:
- **Pyright**: 0 errors, 0 warnings, 0 informations
- **Compileall**: 0 errors
- **Pytest**: 656 tests passed (100% passing rate)
- **Ruff Check**: Clean (0 lint violations)
- **Ruff Format**: Clean (186 files formatted)

