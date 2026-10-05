# Phase 18: Unseen Classification Regime Transfer & Shift Dynamics

This document provides the mathematical and geometric foundation for **unseen classification regime transfer** under DeltaCore's minimal associative-state mechanism.

---

## 1. General Formulation: The Online Associative Alignment Principle

In online classification under distribution shift, streaming observations follow:
$$
x_t \sim P_t(x), \quad y_t \sim P_t(y | x)
$$
where the conditional and marginal distributions change over streaming time $t$.

DeltaCore enforces **parameter immutability** ($\Delta\theta = 0$). The offline-fit linear head parameters $(W_{\mathrm{head}}, b_{\mathrm{head}})$ remain strictly fixed. Instead of adjusting the decision parameters $W$, DeltaCore adapts an internal linear associative operator $M_t \in \mathbb{R}^{D \times D}$ that maps streaming inputs into canonical feature space:
$$
z_t = (I + M_t) x_t
$$
Classification logits are evaluated on this adapted representation:
$$
s_t = W_{\mathrm{head}} z_t + b_{\mathrm{head}}
$$

---

## 2. Geometric Transfer Across Distinct Shift Families

### 2.1 Family A: Subspace Rotation ($A \to C \to A$)
When the feature coordinates rotate by an orthogonal matrix $R \in O(D)$ within the prototype subspace:
$$
x_t = R x_t^{(0)}
$$
The ideal associative operator satisfies:
$$
(I + M^*) R x_t^{(0)} = x_t^{(0)} \implies I + M^* = R^\top \implies M^* = R^\top - I
$$
Because $R^\top - I$ is an exact linear operator, the associative state $M_t$ converges to $R^\top - I$, fully inverting the rotation and restoring canonical classification accuracy ($88.0\%$, with $100.0\%$ return accuracy).

### 2.2 Family B: Boundary Translation & Intercept Offset ($A \to B_{\mathrm{trans}} \to A$)
When class centroids undergo displacement offsets $\Delta \mu^{(k)} = \tau (P^{(k+1)} - P^{(k)})$ along the prototype subspace:
$$
x_t = P^{(y_t)} + \Delta \mu^{(y_t)} + \epsilon_t
$$
The decision boundary shifts by an intercept offset. The associative operator adapts:
$$
(I + M_t) \left( P^{(k)} + \Delta \mu^{(k)} \right) \approx P^{(k)}
$$
Because the displacements $\Delta \mu^{(k)}$ are correlated with class prototype positions $P^{(k)}$, an associative outer product $\sum_k e_{x, k} x_k^\top$ constructs a linear counter-shift:
$$
M^* \approx - \sum_k \Delta \mu^{(k)} (P^{(k)})^+
$$
This counter-shift linearly realigns the translated centroids back towards their canonical positions, elevating accuracy from $82.4\%$ (FrozenLinear) to $92.2\%$ (SafeAdaptiveDelta), establishing a $+9.8$ percentage point transfer gain without retuning.

### 2.3 Family C: Nonlinear Boundary Deformation ($A \to C_{\mathrm{nonlin}} \to A$)
When features undergo a nonlinear coordinate deformation $\phi(x)$:
$$
x_t = \phi(x_t^{(0)}) = x_t^{(0)} + \text{Warp}(x_t^{(0)})
$$
The optimal alignment operator is nonlinear: $z_t = \phi^{-1}(x_t)$.
Because SafeAdaptiveDelta operates via a **linear** associative state $z = (I + M)x$, it approximates $\phi^{-1}$ via its first-order local affine Jacobian:
$$
I + M_t \approx \mathbb{E}\left[ J_\phi^{-1}(x_t) \right]
$$
This affine approximation successfully captures the bulk orientation shift, boosting accuracy from $77.3\%$ to $87.2\%$ (+9.9 percentage points). The tested linear-associative formulation showed an empirical performance limitation on this nonlinear deformation benchmark ($87.2\%$ vs $95.5\%$ for OnlineLogisticRegression), though this experiment does not establish a formal representational ceiling.

---

## 3. Analysis of Shift Dimensions

### 3.1 Covariate Shift ($p(x)$ changes, $p(y|x)$ invariant)
Under anisotropic covariance distortion ($Q \Sigma Q^\top$ with condition number $\kappa = 6.0$), class prototypes remain unrotated and undisturbed. Both FrozenLinear and SafeAdaptiveDelta achieve $100.0\%$ accuracy. Covariate shift produced parity/invariance rather than a measurable adaptive-state advantage in the tested configuration.

### 3.2 Gradual Drift versus Abrupt Shift
- **Abrupt Shift**: Generates an immediate spike in logit prediction error $e_{\mathrm{logit}}$, producing large step updates $\eta_t$ bounded by $\rho / \|x_t\|^2$, driving rapid convergence within 15–20 steps.
- **Gradual Drift**: Generates small continuous prediction errors, tracking the moving boundary continuously and sustaining $86.0\%$ accuracy throughout the drift.

### 3.3 Class-Prior Imbalance
When class frequencies shift ($P(y=0) = P(y=1) = 0.40$, remaining classes $0.05$), conditional feature distributions $P(x|y)$ remain fixed. OnlineLogisticRegression adapts class prior intercepts directly in $b_{\mathrm{head}}$, achieving $96.6\%$, whereas SafeAdaptiveDelta achieves $92.2\%$ (parity with FrozenLinear at $91.8\%$). The associative-state mechanism was not competitive with the online logistic baseline on the tested class-prior shift.

---

## 4. The Dual-Role Hypothesis: Persistence versus Reset

The phase rigorously evaluated the **dual-role hypothesis of persistent associative state**:

$$
\text{Utility of Persistent Memory } M_t =
\begin{cases}
+ \text{Benefit (retained structure)}, & \text{if } \text{alignment}(P_{\mathrm{new}}, P_{\mathrm{old}}) > 0 \\
- \text{Penalty (stale interference)}, & \text{if } \text{alignment}(P_{\mathrm{new}}, P_{\mathrm{old}}) \le 0
\end{cases}
$$

1. **Re-occurring Regimes ($A \to C \to A$)**:
   $M_t$ provides seamless continuity, maintaining $100.0\%$ accuracy upon returning to Regime A.
2. **Incompatible Adversarial Mismatch ($A \to B_{\mathrm{mismatch}}$)**:
   The historical associative state $M_A$ acts as an adversarial prior under opposite rotation ($-\theta_A$), incurring a $-1.75$ percentage point overall ($60.125\%$ vs $61.875\%$) and $-10.0$ percentage point immediate post-shift accuracy penalty ($30.0\%$ vs $40.0\%$) compared to an oracle reset that zeros $M_t$.

---

## 5. Finding: FixedDelta versus SafeAdaptiveDelta

The transfer result is primarily evidence for the associative-state mechanism. SafeAdaptiveDelta did not improve primary transfer-family accuracy over FixedDelta in these experiments (FixedDelta achieved $88.5 \pm 1.4\%$ on Family A, $93.7 \pm 1.1\%$ on Family B, and $87.8 \pm 1.7\%$ on Family C), although its safety controller remains relevant to stability behavior under severe conditions.
