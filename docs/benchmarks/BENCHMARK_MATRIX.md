# DeltaCore Benchmark Matrix

This matrix specifies the evaluated baseline systems across the six core synthetic benchmark tasks in DeltaCore Phase 6.

Every baseline and task included in the suite has an explicit mathematical and architectural rationale. No architecture is included merely because it is fashionable.

---

## 1. Task × Model Evaluation Matrix

| Task Identifier | Frozen Memory | Fixed Delta (Phase 1) | Adaptive Delta (Phase 2) | Self-Referential (Phase 3) | Safe Self-Ref (Phase 4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stationary Recall** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Distribution Shift** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Key Interference** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Conflicting Targets** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Stability Stress** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Adaptation Budget** | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 2. Architectural Rationale for Systems Included

### 1. Frozen Memory (Baseline A)
- **Mathematical Form**: $M_t = M_0 = 0$, update $\Delta M_t = 0$.
- **Scientific Role**: Represents standard static feedforward inference with no test-time adaptation. Serves as the control anchor for measuring whether adaptation helps or degrades performance under stationary vs. non-stationary data.

### 2. Fixed Delta (Baseline B, Phase 1)
- **Mathematical Form**: $M_{t+1} = M_t + \eta_0 (v_t - M_t k_t) k_t^\top$ with constant $\eta_0 > 0$.
- **Scientific Role**: Classical linear associative memory with constant error-correcting gradient steps. Benchmarks the baseline capability of first-order test-time state adaptation.

### 3. Adaptive Delta (Baseline C, Phase 2)
- **Mathematical Form**: $M_{t+1} = M_t + \eta_t (v_t - M_t k_t) k_t^\top$ where $\eta_t = f(e_t)$.
- **Scientific Role**: Evaluates dynamic learning rate modulation where step magnitude increases with error norm without modifying the controller itself online.

### 4. Self-Referential Delta (Baseline D, Phase 3)
- **Mathematical Form**: Joint co-evolution of content memory $M_t \in \mathbb{R}^{V \times K}$ and dynamics memory $C_t \in \mathbb{R}^{1 \times D_c}$, with unconstrained step sizes $\eta_t = \eta_{\max} \sigma(C_t z_t)$ and unconstrained dynamics rates $\rho_t$.
- **Scientific Role**: Minimal self-referential nested learning system. Tests whether internal state that controls its own learning rate co-adapts effectively or suffers from runaway feedback loops.

### 5. Safe Self-Referential Delta (Baseline E, Phase 4)
- **Mathematical Form**: Two-memory self-referential system equipped with closed-form projection controllers:
  $$\eta_t^{\text{safe}} = \min\left(\eta_t^{\text{raw}}, \frac{\beta}{\|k_t\|_2^2 + \epsilon}\right), \quad \rho_t^{\text{safe}} = \min\left(\rho_t^{\text{raw}}, \frac{\beta_C}{\|z_t\|_2^2 + \epsilon}\right)$$
- **Scientific Role**: Tests whether enforcing local non-expansion of immediate residuals prevents catastrophic overflow without impairing adaptive capacity.

---

## 3. Task Objectives Summary

1. **Stationary Recall**: Quantifies retrieval fidelity and memory growth in a static environment.
2. **Distribution Shift**: Measures recovery latency and error reduction after target inversion, permutation, or partial mapping changes.
3. **Key Interference**: Sweeps key cosine similarity $\rho \in [0.0, 1.0]$ to evaluate cross-talk degradation.
4. **Conflicting Targets**: Tests memory behavior under direct contradiction on identical keys ($k \to v_1$ then $k \to v_2$).
5. **Stability Stress**: Pushes models into adversarial parameter regimes (large keys, high rates, opposing targets) to evaluate numerical survival and margin bounds.
6. **Adaptation Budget**: Maps the empirical Pareto frontier between recovery speed and state update energy.
