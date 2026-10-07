# DeltaCore Mathematical Contract & Formal Specification

**Module**: `deltacore.controller`, `deltacore.updates`, `deltacore.stability`  
**Status**: Authoritative Runtime Mathematical Contract & Conceptual Boundary Reference  
**Version**: `0.2.0`  

This document specifies the exact mathematical equations, tensor contracts, temporal ordering, and stability bounds implemented across the executable DeltaCore codebase. Every equation below corresponds directly to runtime code in `deltacore/controller.py`, `deltacore/updates/delta.py`, `deltacore/updates/adaptive_delta.py`, and `deltacore/stability/controllers.py`.

---

## Conceptual Disambiguation & Primitive Distinctions

To prevent conflation of theoretical and operational concepts, DeltaCore explicitly establishes the following definitions:

1. **Associative Memory**: An accumulated mapping $M \in \mathbb{R}^{V \times K}$ storing associations between key vectors $k \in \mathbb{R}^K$ and target value vectors $v \in \mathbb{R}^V$ via outer products.
2. **Auto-Association**: The special configuration where target values equal key inputs ($v_t \equiv x_t$). The matrix learns to reconstruct inputs: $\hat{x}_t = M x_t$.
3. **Reconstruction Residual**: The instantaneous Euclidean distance $r_t = \|x_t - M_{t-1} x_t\|_2$ evaluating reconstruction discrepancy strictly prior to any state update.
4. **Adaptive Update**: Test-time, online modification of internal matrix state $M_t = \alpha_t M_{t-1} + \eta_t (e_t x_t^T)$ without gradient backpropagation to offline weights.
5. **Feature Hashing**: Static, stateless deterministic mapping from structured telemetry dictionaries into Euclidean space using signed SHA-256 token hashing.
6. **Novelty Diagnostic**: A scalar score indicating whether the current observation is poorly reconstructed under the currently adapted associative memory state.
7. **Regime Shift**: A non-stationary transition in the joint distribution generating observations over time (e.g. parameter drift $A \to B$).
8. **Principal Component Analysis (PCA)**: A classical spectral technique that projects data onto the top-$k$ eigenvectors of the empirical covariance matrix under an explicit rank constraint ($k \ll D$). **Auto-association in a full-rank square matrix $M \in \mathbb{R}^{D \times D}$ is NOT PCA**; full-rank $M$ can trivially represent the identity matrix $I$ and does not learn principal subspaces without rank constraints.
9. **Covariance**: A classical second-order statistic $\Sigma_t = \mathbb{E}[(x-\mu)(x-\mu)^T]$ tracking pairwise feature covariances, enabling scale-invariant Mahalanobis distance scoring $d_t = \sqrt{(x-\mu)^T(\Sigma+\lambda I)^{-1}(x-\mu)}$. DeltaCore's unnormalized outer-product update does not compute a sample covariance or inverse covariance matrix.

---

## 1. State Definition

The primary adaptive state is an explicit real matrix:
$$
M_t \in \mathbb{R}^{V \times K}
$$
where:
- $K \in \mathbb{N}^+$ is the dimensionality of the key / input space.
- $V \in \mathbb{N}^+$ is the dimensionality of the target / associative value space.
- In the canonical auto-associative and coordinate-transform configurations, $V = K = D$, giving a square state:
  $$
  M_t \in \mathbb{R}^{D \times D}
  $$

The matrix $M_t$ represents an accumulated linear associative mapping between keys and values learned online via test-time error-correcting gradient steps.

### Physical Memory Footprint
For a square state $M_t \in \mathbb{R}^{D \times D}$ in IEEE 754 single precision (`torch.float32`), the persistent state memory is exactly:
$$
\text{Memory}_{\text{state}} = 4 D^2 \text{ bytes}
$$
Example footprints:
- $D = 64$: $16\text{ KB}$
- $D = 128$: $64\text{ KB}$
- $D = 256$: $256\text{ KB}$
- $D = 512$: $1\text{ MB}$
- $D = 1024$: $4\text{ MB}$

---

## 2. Input Definition

At each discrete time step $t \in \mathbb{N}^+$:
- **Input Key Vector**:
  $$
  x_t \in \mathbb{R}^K, \quad \|x_t\|_2 < \infty, \quad x_t \text{ strictly finite (no NaN, } \pm\infty\text{)}
  $$
- **Target Value Vector** (hetero-associative mode):
  $$
  v_t \in \mathbb{R}^V, \quad \|v_t\|_2 < \infty, \quad v_t \text{ strictly finite}
  $$
- **Auto-Associative Identity** (auto-associative mode):
  When target $v_t$ is omitted or in pure auto-association:
  $$
  v_t \equiv x_t \in \mathbb{R}^D \quad (V = K = D)
  $$

---

## 3. Update Equation & Associative Dynamics

The associative state transition follows the error-correcting delta rule (Widrow-Hoff / LMS outer-product update):

### Step 1: Pre-Update Association / Prediction
$$
\hat{v}_t = M_{t-1} x_t \in \mathbb{R}^V
$$

### Step 2: Prediction Residual / Error Vector
$$
e_t = v_t - \hat{v}_t \in \mathbb{R}^V
$$
In auto-associative mode where $v_t = x_t$:
$$
e_t = x_t - M_{t-1} x_t = (I - M_{t-1}) x_t
$$

### Step 3: Directional Outer-Product Tensor
$$
\Delta M_t^{\text{raw}} = e_t \otimes x_t = e_t x_t^\top \in \mathbb{R}^{V \times K}
$$

### Step 4: State Transition Law
When adaptation is enabled (`adapt=True`):
$$
M_t = \alpha_t M_{t-1} + \eta_t \, (e_t x_t^\top)
$$
where:
- $\eta_t \ge 0$ is the effective scalar step size.
- $\alpha_t \in [\alpha_{\min}, 1.0]$ is the selective retention factor.

When adaptation is disabled (`adapt=False` or via `score()`):
$$
M_t \equiv M_{t-1}, \quad \Delta M_t = 0
$$

---

## 4. Initialization

Initial state $M_0$ is configured deterministically:
1. **Zero Initialization** (`initial_scale = 0.0`, default):
   $$
   M_0 = \mathbf{0}_{V \times K}
   $$
   Under zero initialization, pre-update prediction at $t=1$ is $\hat{v}_1 = \mathbf{0}$, giving initial residual $e_1 = v_1$.
2. **Scaled Identity Initialization** (`initial_scale = c > 0`):
   $$
   M_0 = c \cdot I_{D \times D}
   $$
   For auto-associative mode with $c = 1.0$, $M_0 x_t = x_t$, yielding zero initial residual until perturbed.

---

## 5. Normalization

1. **Input Normalization**:
   When working with raw telemetry or hashed categorical vectors, input vectors are optionally L2-normalized:
   $$
   \tilde{x}_t = \frac{x_t}{\|x_t\|_2 + \epsilon} \implies \|\tilde{x}_t\|_2 \approx 1.0
   $$
2. **State Norm Monitoring**:
   The active state magnitude is continuously monitored via Frobenius norm:
   $$
   \|M_t\|_F = \sqrt{\sum_{i=1}^V \sum_{j=1}^K M_{t, ij}^2}
   $$

---

## 6. Stability Constraints & Non-Expansion Guarantees

To prevent numerical runaway and explosive matrix amplification over long streaming horizons, DeltaCore enforces Lyapunov contractive bounds on the step size $\eta_t$.

### Nominal Candidate Rate
The nominal step size modulates with prediction error magnitude:
$$
\eta_{\text{cand}, t} = \frac{\eta_0}{1 + \gamma \|e_t\|_2}
$$
where $\eta_0 > 0$ is the base learning rate and $\gamma \ge 0$ regulates error dampening.

### Lyapunov Contractive Bound
For any key $x_t$, the post-update residual on that exact key satisfies:
$$
e_{t+1}^{(x_t)} = v_t - M_t x_t = v_t - (M_{t-1} + \eta_t e_t x_t^\top) x_t = e_t - \eta_t (x_t^\top x_t) e_t = (1 - \eta_t \|x_t\|_2^2) e_t
$$
Strict local non-expansion of residual error $\|e_{t+1}^{(x_t)}\|_2 \le \|e_t\|_2$ requires:
$$
|1 - \eta_t \|x_t\|_2^2| \le 1 \iff 0 \le \eta_t \|x_t\|_2^2 \le 2
$$
DeltaCore enforces the safety constraint parameterized by contractive margin $\rho \in (0, 2)$:
$$
\eta_t = \min\left(\eta_{\text{cand}, t}, \frac{\rho}{\|x_t\|_2^2 + \epsilon}\right)
$$
where $\epsilon > 0$ is a strictly positive numerical guard against division by zero.

### Local Stability Margin Metric
The scalar distance to the divergence boundary is defined as:
$$
\text{margin}_t = 2.0 - \eta_t \|x_t\|_2^2
$$
- $\text{margin}_t > 0$: strictly within contractive operating envelope.
- $\text{margin}_t \ge 2.0 - \rho > 0$: guaranteed safe lower bound.

> **Crucial Epistemic Caveat**: Local step contractivity $|1 - \eta_t \|x_t\|_2^2| \le 1$ bounds instantaneous residual expansion on key $x_t$. It does NOT constitute a mathematical proof of global matrix boundedness under arbitrary non-orthogonal sequence streams.

---

## 7. Score Definition & Auto-Associative Residual

For novelty, anomaly, and regime-shift experimentation, the scalar diagnostic is the **pre-update reconstruction residual**:
$$
r_t = \|e_t\|_2 = \|v_t - M_{t-1} x_t\|_2
$$
In auto-associative mode ($v_t \equiv x_t$):
$$
r_t = \|x_t - M_{t-1} x_t\|_2 = \|(I - M_{t-1}) x_t\|_2
$$

### Rolling Residual & Smoothed Tracking
A sliding window or exponential moving average (EMA) computes smoothed baseline drift:
$$
\bar{r}_t = (1 - \lambda_{\text{ema}}) \bar{r}_{t-1} + \lambda_{\text{ema}} r_t
$$

---

## 8. Strict Temporal Ordering (Score-Before-Update)

To prevent **anomaly absorption** (where an anomalous event immediately updates $M$ and washes out its own anomaly signal), execution follows an inviolable 7-stage order:

```
Stage 1: Input Validation
         Verify x_t (and v_t) is 1D, dimension matches K (and V), finite (no NaN/Inf), float dtype.

Stage 2: Pre-Update Association
         \hat{v}_t = M_{t-1} x_t

Stage 3: Pre-Update Residual Calculation
         e_t = v_t - \hat{v}_t
         r_t = ||e_t||_2  <--- Evaluated strictly against M_{t-1}

Stage 4: Candidate Rate & Stability Projection
         \eta_{cand, t} = \eta_0 / (1 + \gamma r_t)
         \eta_t = min(\eta_{cand, t}, \rho / (||x_t||_2^2 + \epsilon))
         margin_t = 2.0 - \eta_t ||x_t||_2^2

Stage 5: Anomaly Gating Evaluation
         Determine adapt \in {True, False}
         (e.g., if r_t > threshold, gate update to prevent memory contamination)

Stage 6: State Transition (if adapt == True)
         \Delta M_t = \eta_t (e_t x_t^\top)
         M_t = \alpha_t M_{t-1} + \Delta M_t

Stage 7: Return Diagnostic Result
         Return ControllerStepResult(prediction, error, r_t, \eta_t, margin_t, state_norm)
```

---

## 9. Edge Cases & Boundary Handling

1. **Zero Key Vector ($\|x_t\|_2 = 0$)**:
   - Division by zero guarded by $\epsilon > 0$.
   - Pre-update prediction $\hat{v}_t = M_{t-1} \mathbf{0} = \mathbf{0}$.
   - Outer-product $\Delta M_t = \eta_t e_t \mathbf{0}^\top = \mathbf{0}$.
   - State remains unmodified ($M_t = M_{t-1}$).
2. **Exact Recall ($e_t = \mathbf{0}$)**:
   - Prediction matches target perfectly.
   - Outer-product $\Delta M_t = \eta_t \mathbf{0} x_t^\top = \mathbf{0}$.
   - State remains unmodified.
3. **Collinear Repeating Keys**:
   - Repeated updates along direction $x$ contract geometric error exponentially:
     $e_{t+k} = (1 - \eta \|x\|^2)^k e_t \to 0$.
4. **Orthogonal Keys ($x_i^\top x_j = 0$)**:
   - Zero interference across mutually orthogonal key subspaces.

---

## 10. Known Mathematical & Empirical Limitations

1. **Auto-Association $\neq$ PCA / Subspace Learning**:
   An unconstrained full-rank matrix $M \in \mathbb{R}^{D \times D}$ can learn the identity transformation $M = I$. Unlike PCA, which explicitly constrains representation rank $r \ll D$, unconstrained auto-associative DeltaCore does not perform principal component analysis without rank or projection constraints.
2. **Interference Under Non-Orthogonal Keys**:
   When incoming keys are non-orthogonal ($x_i^\top x_j \ne 0$), updates along $x_i$ perturb previously stored recall along $x_j$ proportionally to their inner product.
3. **Anomaly Absorption Vulnerability**:
   If anomalous observations are updated unconditionally without score-before-update gating, $M$ absorbs the anomaly within 1–3 steps, decaying the residual signal.
4. **Linear Representation Ceiling**:
   Because $M_t$ performs a linear coordinate mapping $\hat{v} = M x$, it cannot represent arbitrary nonlinear manifold structures without feature lifting or multi-layer architectures.
