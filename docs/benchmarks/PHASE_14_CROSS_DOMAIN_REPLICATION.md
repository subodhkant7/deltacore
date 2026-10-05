# DeltaCore Phase 14: Cross-Domain Replication Report

This report presents the rigorous cross-domain comparative analysis between **Phase 13 (NOAA OISST v2.1 Sea Surface Temperature)** and **Phase 14 (ECMWF ERA5 2-Meter Air Temperature $T_{2m}$)**.

In accordance with Section 18 of the Phase 14 specification:
* The domains remain independently analyzed and interpreted.
* Benchmark scores are **not averaged** into a composite metric.
* The scientific purpose is answering: **Does the Phase 13 mechanism replicate outside sea-surface temperature?**

---

## 1. Domain Characteristics Comparison

| Dimension | Phase 13: Ocean Domain | Phase 14: Atmospheric Domain |
| :--- | :--- | :--- |
| **Dataset** | NOAA OISST v2.1 SST | ECMWF ERA5 2-Meter Air Temperature ($T_{2m}$) |
| **Geographic Region** | Equatorial Pacific ($5^\circ\text{S}-5^\circ\text{N}, 170^\circ\text{W}-120^\circ\text{W}$) | North Atlantic & European Storm Track ($40^\circ\text{N}-65^\circ\text{N}, 30^\circ\text{W}-20^\circ\text{E}$) |
| **Temporal Resolution** | Bi-weekly (14-day intervals) | Synoptic 6-hourly intervals |
| **Physical Regime** | High thermal inertia, slow planetary oceanic wave dynamics | Rapid advective dynamics, baroclinic storm waves, frontogenesis |
| **Dominant Distribution Shift** | 2015–2016 Super El Niño warming surge ($+2.8^\circ\text{C}$ local peak) | Jan–Feb 2021 Sudden Stratospheric Warming & European cold wave ($-7.5^\circ\text{C}$ peak) |
| **Shift Relative Timesteps** | $t \in [40, 110]$ (peak at $t=75$) | $t \in [45, 115]$ (peak at $t=75$) |
| **Base Spatial Dimension** | $8 \times 8$ ($D=64$) | $8 \times 8$ ($D=64$) |
| **Scaled Dimension** | $16 \times 16$ ($D=256$) | $16 \times 16$ ($D=256$) |
| **Train Mean / Std** | $\mu = 26.83^\circ\text{C}, \sigma = 2.02^\circ\text{C}$ | $\mu = 5.37^\circ\text{C}, \sigma = 7.18^\circ\text{C}$ |

---

## 2. Replication Table (Section 19)

The following table documents the empirical transfer of each Phase 13 finding to Phase 14:

| Property | Phase 13 OISST | Phase 14 Domain | Replicated? |
| :--- | :--- | :--- | :---: |
| **SafeAdaptiveDelta beats FixedDelta** | **YES** at $D=64$ ($0.2582$ vs $0.2692$); **YES** at $D=256$ ($0.6428$ vs $2.85 \times 10^{14}$) | **NO** at $D=64$ ($0.1622$ vs $0.1453$); **YES** at $D=256$ ($0.2604$ vs `NaN`) | **PARTIAL** |
| **SafeAdaptiveDelta beats OnlineRidge** | **YES** ($0.2582$ vs $0.3003$, $\Delta = -0.0421$) | **NO** ($0.1622$ vs $0.0923$, $\Delta = +0.0699$) | **NO** |
| **Continuous state helps shift period** | **YES** ($-26\%$ excess error, $0$ vs $2$ step recovery) | **YES** ($-97.1\%$ excess error, $0$ vs $1$ step recovery) | **YES** |
| **Lower persistent state than RLS** | **YES** ($50\%$ memory reduction: $16\text{ KB}$ vs $32\text{ KB}$) | **YES** ($50\%$ memory reduction: $16\text{ KB}$ vs $32\text{ KB}$, $21\%$ faster at $D=256$) | **YES** |
| **Permutation equivariance** | **YES** ($E_{\text{equiv}} \le 10^{-7}$, $\Delta E_{\text{perm}} \approx 0$) | **YES** ($E_{\text{equiv}} = 7.73 \times 10^{-8}$, $\Delta E_{\text{perm}} \approx 0$) | **YES** |
| **SpatialConv permutation sensitivity** | **YES** ($+641\%$ relative error explosion, $E_{\text{equiv}} = 2.12$) | **YES** ($+536\%$ relative error explosion, $E_{\text{equiv}} = 1.78$) | **YES** |

---

## 3. In-Depth Scientific Analysis of Differences

### 3.1 Why Did OnlineRidge Beat DeltaCore on ERA5 $T_{2m}$?
1. **Regime Linearity and Advection Speed**:
   * Sea surface temperature has high thermal inertia and localized, non-stationary convective teleconnections where unconstrained RLS covariance estimation over-adapts and becomes volatile during shifts ($E_{\text{rel}} = 0.3003$, losing to persistence $0.2902$).
   * Atmospheric 2-meter air temperature at 6-hour synoptic intervals is governed by large-scale linear quasi-geostrophic advection. The true physical operator is close to a rotating advective linear map $X_{t+1} \approx A X_t$, which recursive least squares tracks with high precision ($E_{\text{rel}} = 0.0923$).
2. **Conservative Contraction Bounds**:
   * In Phase 14, `SafeAdaptiveDelta` was evaluated strictly frozen with Phase 13 hyperparameters ($\eta_0 = 0.008, \rho = 1.90, \alpha_{\min} = 0.85, \gamma = 0.05$).
   * These bounds prioritize unconditional contractive stability and fast historical forgetting during shifts. On this synoptic atmospheric series, this conservatism causes DeltaCore to under-track the linear advection velocity, whereas full-rank RLS aggressively adjusts all $D^2$ coefficients.

### 3.2 What Replicated Robustly?
1. **Contractive Safety Bound**:
   * At $D=256$, input vector norms expand ($\|x_t\|_2^2 \approx 256$), causing fixed step size $\eta = 0.008$ to violate the contractive bound ($\eta \|x_t\|_2^2 > 2.0$). In both Phase 13 and Phase 14, `FixedDelta` suffers catastrophic numerical instability (exploding to $2.85 \times 10^{14}$ in Phase 13 and `NaN` in Phase 14).
   * In both phases, `SafeAdaptiveDelta` contracts $\eta_t \le 1.9 / \|x_t\|_2^2 \approx 0.0074$ and remains unconditionally stable ($0.6428$ in Phase 13, $0.2604$ in Phase 14).
2. **State Retention Intervention**:
   * Continuous state retention proved even more impactful in Phase 14 than in Phase 13:
     * Phase 13: $-26\%$ cumulative excess shift error reduction.
     * Phase 14: **$-97.1\%$ cumulative excess shift error reduction** ($0.0261$ vs $0.9109$), eliminating the 1-step re-acquisition lag.
3. **State Memory Efficiency**:
   * DeltaCore consistently requires only $D^2$ floats ($16\text{ KB}$ at $D=64$, $256\text{ KB}$ at $D=256$), saving $50\%$ memory over RLS ($2 D^2$ floats), and runs $21\%$ faster per step at $D=256$.
4. **Architectural Representation Boundary**:
   * In both domains, DeltaCore demonstrates exact permutation equivariance ($E_{\text{equiv}} \le 10^{-7}$), confirming it is a 2D-unaware temporal feature associative operator.
   * Concurrently, `SpatialConv` degrades catastrophically under permutation ($+641\%$ in Phase 13, $+536\%$ in Phase 14), proving both datasets have genuine spatial locality that DeltaCore treats via permutation-equivariant feature alignment.

---

## 4. Formal Replication Status Declaration

In accordance with Section 18:
* **Replication**: False (prediction advantage over OnlineRidge did not replicate).
* **Partial Replication**: **TRUE (Outcome B)**.
  * Contractive safety, continuous state retention advantage, persistent state memory efficiency, and permutation equivariance transfer robustly.
* **Failure to Replicate**: False (core mechanisms did not fail; rather, their performance relative to an unconstrained linear filter changed due to physical regime differences).

### Conclusion for Phase 15:
Phase 15 must explore **Real-World Spatio-Temporal Robustness / Transfer**, testing how adaptive mechanisms navigate across differing physical regimes (slow thermal vs fast advective) and evaluating whether adaptive step-size schedules or multi-scale state retention can bridge the tracking gap without sacrificing stability or parameter simplicity.
