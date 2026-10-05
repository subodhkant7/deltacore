# DeltaCore Phase 15: Cross-Domain Transfer Synthesis
## Physical Regimes, Adaptation Aggressiveness & State Portability

### 1. Introduction

A fundamental challenge in machine learning for continuous physical systems is whether an online associative state mechanism can generalize across systems governed by different partial differential equations (PDEs). In DeltaCore:
- **Phase 13** established that `SafeAdaptiveDelta` achieved superior prediction and recovery on **NOAA OISST Sea-Surface Temperature (SST)**.
- **Phase 14** established that on **ECMWF ERA5 $2\mathrm{m}$ Air Temperature ($T_{2m}$)**, linear predictors (`FrozenLinear` and `OnlineRidge`) tracked synoptic advection with lower error than DeltaCore's Phase 13 configuration.

Phase 15 resolves this tension by demonstrating that this discrepancy was driven by the underlying **physical transport regimes** rather than a failure of the DeltaCore mechanism itself.

---

### 2. Slow Thermal vs Fast Advective Regimes

The two domains represent contrasting points in fluid dynamical and thermodynamic space:

```
+------------------------------------+------------------------------------+
| Domain A: NOAA OISST (Pacific SST) | Domain B: ECMWF ERA5 (Storm Track) |
+------------------------------------+------------------------------------+
| Physical Regime: Thermal Diffusion | Physical Regime: Synoptic Advection|
| Governing PDE: Thermal diffusion   | Governing PDE: Navier-Stokes       |
|    + large-scale oceanic drift     |    + baroclinic instability wave   |
| Timescale: Months to seasons       | Timescale: Hours to days           |
| Spatial Coherence: Smooth fields   | Spatial Coherence: Sharp fronts,   |
|    with high spatial correlation   |    cyclonic vortices, cold tongues |
| Optimal Learning Rate: eta = 0.008 | Optimal Learning Rate: eta = 0.015 |
| Optimal Retention: alpha = 0.95    | Optimal Retention: alpha = 0.95    |
| Primary Failure Mode: Over-adapt   | Primary Failure Mode: Under-adapt  |
|    (amplifies thermal noise)       |    (lags propagating fronts)       |
+------------------------------------+------------------------------------+
```

#### 2.1 The Mathematics of Over-Adaptation in Thermal Regimes
In sea-surface temperature, day-to-day temperature perturbations are small relative to the total thermal inertia of the mixed layer. When a streaming predictor uses an aggressive learning rate ($\eta \ge 0.025$):
$$M_{t+1} = \alpha_t M_t + \eta_t e_t x_t^\top$$
The update term $\eta_t e_t x_t^\top$ overfits high-frequency observational noise and localized turbulent eddies, rapidly rotating the associative matrix away from the true seasonal trend. In Phase 15 validation sweeps, setting $\eta_0 = 0.025$ on OISST caused relative validation error to surge to **$1.5580$**. Conservative updates ($\eta_0 = 0.008$) preserve the low-rank seasonal structure and allow DeltaCore to outperform OnlineRidge by **$13.5\%$** ($0.2606$ vs $0.3012$).

#### 2.2 The Mathematics of Under-Adaptation in Advective Regimes
In atmospheric air temperature, synoptic systems (such as the January 2021 Sudden Stratospheric Warming and Storm Filomena) transport coherent cold air masses across thousands of kilometers at speeds of $10\text{--}30\text{ m/s}$. When a predictor uses the conservative Phase 13 configuration ($\eta_0 = 0.008, \alpha = 0.85$):
The state update $\eta_t e_t x_t^\top$ is too small to adjust the linear mapping quickly enough to track the rapid advection of the front. Increasing $\eta_0$ to $0.015$ reduces relative test error from **$0.2386$ to $0.1621$** on ERA5.

---

### 3. The Role of Recursive Least Squares vs DeltaCore

A critical finding from Phase 14 and Phase 15 is why `OnlineRidge` (RLS) is highly competitive on atmospheric advection:
1. **Full Precision Matrix Tracking**:
   OnlineRidge explicitly updates the inverse sample covariance matrix:
   $$P_{t+1} = \lambda^{-1} \left( P_t - \frac{P_t x_t x_t^\top P_t}{\lambda + x_t^\top P_t x_t} \right)$$
   For a rigid linear advection field $y_t \approx A x_t$, recursive least squares rapidly converges to the exact operator $A$.
2. **Resource Trade-off**:
   However, maintaining $P_t \in \mathbb{R}^{D \times D}$ alongside the weight matrix $W_t \in \mathbb{R}^{D \times D}$ requires **twice the persistent state memory** of DeltaCore:
   - At $D=64$: $32\text{ KB}$ vs $16\text{ KB}$.
   - At $D=256$: $512\text{ KB}$ vs $256\text{ KB}$.
   Furthermore, RLS requires continuous division by the quadratic form $\lambda + x_t^\top P_t x_t$, whereas SafeAdaptiveDelta computes a single inner product $\|x_t\|_2^2$ and an outer product $e_t x_t^\top$.

---

### 4. Cross-Domain State Portability & Regime Boundaries

Experiment E and Experiment G provide precise boundaries on the portability of learned associative memory:

#### 4.1 Positive Transfer in Initialization
In Experiment G, initializing predictor memory with state pre-adapted on Domain A ($M_{\mathrm{final}}^{(A)}$) and evaluating on Domain B reduced overall relative error from $0.1742$ (zero state) to **$0.1632$**, while reducing early transient error from $0.3587$ to **$0.3242$**. Because both physical fields share low-frequency global spatio-temporal structure (large-scale planetary waves and seasonal trends), a pre-adapted associative matrix acts as a warm-start regularizer that accelerates early adaptation.

#### 4.2 Negative Transfer Across Orthogonal Dynamics
In Experiment E, when the streaming sequence was abruptly toggled from OISST to ERA5:
- Continuous state produced higher error in Phase 2 ($0.4889$) than resetting state at the boundary ($0.4778$).
- Continuous state produced higher transient error bursts ($0.6854$ vs $0.6696$).
This establishes that **when transitioning across fundamentally disparate physical regimes, explicit state-reset interventions or accelerated forgetting rates ($\alpha_t \to 0$) prevent negative interference from stale associative representations**.

---

### 5. Contraction Safety & The High-Dimensional Failure Boundary

At $D=256$, unconstrained `FixedDelta` suffered fatal numerical overflow (`NaN` at $t=41$ on ERA5), because input norm spikes caused:
$$\eta \|x_t\|_2^2 > 2.0 \implies |1 - \eta \|x_t\|_2^2| > 1$$
In contrast, `SafeAdaptiveDelta` dynamically enforces:
$$\eta_t = \min\left(\eta_0, \frac{\rho}{\|x_t\|_2^2 + \epsilon}\right), \quad \text{with } \rho = 1.90$$
This guarantees that:
$$\text{Margin}_t = 2.0 - \eta_t \|x_t\|_2^2 \ge 2.0 - 1.90 = 0.1000 > 0$$
`SafeAdaptiveDelta` remained bounded and finite on both domains without a single numerical overflow across any tested sequence.

---

### 6. Summary Recommendations for Practitioners

1. **For slow thermal or diffusive systems**: Choose conservative step sizes ($\eta_0 \in [0.005, 0.008]$) and high retention ($\alpha_{\min} \ge 0.95$).
2. **For fast convective or advective systems**: Increase step size ($\eta_0 \in [0.015, 0.020]$) while retaining $\alpha_{\min} \ge 0.95$.
3. **For unknown or mixed regimes**: Use the pooled validation configuration ($\eta_0 = 0.015, \rho = 1.5, \alpha_{\min} = 0.95$), which transfers robustly across both physical regimes without divergence.
4. **Always employ the contractive step-size bound ($\rho \le 1.90$)**: Essential at dimensions $D \ge 128$ to avoid the catastrophic divergence observed in FixedDelta.
