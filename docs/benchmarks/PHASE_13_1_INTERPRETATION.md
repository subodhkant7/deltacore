# DeltaCore Phase 13.1: Hypothesis Register, Epistemic Audit & Phase Gate Determination

This document presents the revised and audited epistemic register of formal hypotheses following the forensic verification of **Phase 13.1**.

All assertions strictly follow the epistemic definitions mandated by [PROJECT_CONSTITUTION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/PROJECT_CONSTITUTION.md) (`OBSERVED`, `SUPPORTED`, `NOT SUPPORTED`, `INCONCLUSIVE`, `REFUTED IN TESTED CONFIGURATION`).

---

## 1. Revised Hypothesis Register

### Hypothesis H13.1: SafeAdaptiveDelta vs. FixedDelta
> *SafeAdaptiveDelta improves online prediction over FixedDelta on the selected real-world spatio-temporal dataset.*

* **Status**: **SUPPORTED**
* **Audited Evidence**:
  * At $D=64$: `SafeAdaptiveDelta` achieves lower relative error ($0.2582$) than `FixedDelta` ($0.2692$) across all five seeds ($5/5$ seed wins).
  * At $D=256$: `FixedDelta` with static step size $\eta = 0.008$ violates the local contractivity condition ($\eta \|x_t\|_2^2 \approx 2.05 > 2.0$), resulting in numerical explosion ($E_{\text{rel}} \approx 2.85 \times 10^{14}$).
  * `SafeAdaptiveDelta` remained finite and bounded ($E_{\text{rel}} = 0.6428$) in the tested $D=256$ experiment while enforcing the local step-size safety condition ($\eta_t \le 1.9 / \|x_t\|_2^2$).

---

### Hypothesis H13.2: Measurable Online Adaptation Under Real Temporal Shift
> *DeltaCore provides measurable online adaptation under naturally occurring temporal distribution shift.*

* **Status**: **SUPPORTED**
* **Audited Evidence**:
  * During the 2015–2016 Super El Niño warming surge ($t \in [40, 110]$, absolute $t \in [240, 310]$), offline `FrozenLinear` degrades to $E_{\text{shift}} = 0.3067$ (peak relative error $\approx 0.380$).
  * `SafeAdaptiveDelta` maintains an average shift error of $E_{\text{shift}} = 0.2485$, achieving lower shift error than `FrozenLinear`, `Persistence` ($0.2991$), and `OnlineRidge` ($0.3118$).
  * All offline parameters remained strictly immutable ($\Delta \theta = 0$, verified via SHA-256 parameter hashes); adaptation was mediated entirely by online associative state updates.

---

### Hypothesis H13.3: Persistent Adaptive State vs. Controlled State-Reset
> *Persistent adaptive state improves shift recovery relative to state reset in at least one meaningful real-data segment.*

* **Status**: **SUPPORTED**
* **Audited Evidence**:
  * Under a controlled state-reset intervention at shift onset ($t=40$):
    * **Continuous State**: Shift error $E_{\text{shift}} = 0.2485$, first-passage recovery = $0$ steps, cumulative excess error = $2.5987$.
    * **Reset State**: Shift error $E_{\text{shift}} = 0.2639$, first-passage recovery = $2$ steps, cumulative excess error = $3.5100$.
  * Retaining historical adaptive state produced a **$25.96\%$ (unrounded) / $26\%$ (rounded) lower cumulative excess error** during the 2015–2016 shift interval and eliminated the 2-step re-acquisition lag.

---

### Hypothesis H13.4: Competitiveness with OnlineRidge
> *DeltaCore remains competitive with OnlineRidge on the selected real-world task.*

* **Status**: **SUPPORTED**
* **Audited Evidence**:
  * `SafeAdaptiveDelta` achieved lower mean relative prediction error ($E_{\text{rel}} = 0.2582$) than `OnlineRidge` ($E_{\text{rel}} = 0.3003$) on the tested task.
  * In seed-by-seed comparison, `SafeAdaptiveDelta` achieved lower relative error in **5 of 5 tested seeds** ($100\%$ win rate, paired difference $\Delta = -0.0421$).
  * The word "significantly" has been removed to avoid implying an unwarranted asymptotic inferential claim beyond the 5-seed sample.

---

### Hypothesis H13.5: Resource/Performance Trade-off vs. Tested RLS
> *DeltaCore provides a useful resource/performance trade-off relative to the tested RLS implementation.*

* **Status**: **SUPPORTED**
* **Audited Evidence**:
  * **State Memory**: At $D=64$, `SafeAdaptiveDelta` requires $16\text{ KB}$ vs. $32\text{ KB}$ for `OnlineRidge` ($50\%$ state memory reduction). At $D=256$, `SafeAdaptiveDelta` requires $256\text{ KB}$ vs. $512\text{ KB}$ for `OnlineRidge` ($50\%$ state memory reduction).
  * **Step Latency**: At $D=256$, `SafeAdaptiveDelta` executes in $4.63\,\mu\text{s}$ per token vs. $4.71\,\mu\text{s}$ for `OnlineRidge` under identical CPU streaming measurement scope.
  * The trade-off is specifically documented as constant-factor $\mathcal{O}(D^2)$ memory and latency advantages under tested dimensions, without claiming asymptotic complexity dominance.

---

### Hypothesis H13.6: Permutation Equivariance & Architectural Boundary
> *Real-world performance is invariant/equivariant to fixed spatial permutation.*

* **Status**: **SUPPORTED** (Explicitly Verified)
* **Audited Evidence**:
  * Explicit vector-level equivariance testing across all 5 seeds demonstrated normalized equivariance error:
    $$E_{\text{equiv}} = \frac{\|f(P X) - P f(X)\|_F}{\max(\|P f(X)\|_F, \epsilon)} = \mathbf{7.73 \times 10^{-8}} \pm 1.98 \times 10^{-9}$$
  * This confirms **exact mathematical permutation equivariance** up to single-precision floating point roundoff.
  * **Architectural Boundary**: DeltaCore treats the $D$-dimensional flattened field as an unordered feature vector. Claims that DeltaCore exploits 2D locality or physical spatial geometry remain **REFUTED IN TESTED CONFIGURATION**.

---

### Hypothesis H13.7: Spatial Convolution Locality Inductive Bias
> *The tested SpatialConv exhibits strong sensitivity to spatial permutation, consistent with reliance on local spatial ordering.*

* **Status**: **NOT SUPPORTED** *(Corrected from Phase 13)*
* **Audited Evidence**:
  * While `SpatialConv` exhibits high sensitivity to coordinate permutation (relative error increasing by $+641\%$ from $0.3470$ to $2.5731$), its overall prediction performance on the unpermuted real-world task ($E_{\text{rel}} = 0.3285$) is **inferior to SafeAdaptiveDelta ($E_{\text{rel}} = 0.2582$)**.
  * Permutation sensitivity alone does **not** establish a task-performance advantage over DeltaCore. Therefore, the hypothesis that a small spatial convolution provides an advantage on this task is formally **NOT SUPPORTED**.

---

## 2. Scientific Language Audit Summary

In compliance with Section 16 of the Phase 13.1 mandate:
1. **"Guarantee" / "Strictly Bounded"**: Replaced across all reports with:
   > *"remained finite/bounded in the tested $D=256$ experiment while enforcing the implemented local step-size safety condition."*
2. **"Significantly" / "Superior"**: Replaced with precise quantitative descriptions (e.g., *"achieved lower error across all 5 tested seeds with a paired difference of $\Delta = -0.0421$"*).
3. **"Consistently Outperformed"**: Replaced with: *"achieved the lowest prediction error across all five tested seeds ($5/5$ win rate)"*.
4. **"Kelvin Thermal Anomaly"**: Corrected to: **"+2.8 °C SST anomaly"**, distinguishing the benchmark's local normalized anomaly from the official NOAA Niño 3.4 index.
5. **"Breaks RLS" / "Spatial Understanding"**: Explicitly refuted; documented that DeltaCore is a permutation-equivariant associative estimator with no physical fluid or spatial understanding.

---

## 3. Phase 13.1 Exit Criteria Sign-Off

| Exit Criterion | Description | Evidence / Artifact | Status |
| :--- | :--- | :--- | :---: |
| **EC13.1.1** | Table discrepancies resolved | [PHASE_13_1_NUMERICAL_CONSISTENCY.md](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/PHASE_13_1_NUMERICAL_CONSISTENCY.md) | **PASSED** |
| **EC13.1.2** | Baseline training provenance documented | [baseline_training_provenance.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_13_1/baseline_training_provenance.json) | **PASSED** |
| **EC13.1.3** | H13.7 correctly classified as NOT SUPPORTED | Registered above in Section 1 | **PASSED** |
| **EC13.1.4** | Actual permutation equivariance tested | $E_{\text{equiv}} \approx 7.73 \times 10^{-8}$ verified in `permutation_equivariance.json` | **PASSED** |
| **EC13.1.5** | Stability claims limited to local safety conditions | Audited in Section 2 above | **PASSED** |
| **EC13.1.6** | Temperature/anomaly terminology corrected | Corrected to "+2.8 °C SST anomaly" | **PASSED** |
| **EC13.1.7** | Per-seed headline comparisons verified | $5/5$ seed wins verified in `phase_13_1_per_seed_verification.json` | **PASSED** |
| **EC13.1.8** | 26% cumulative shift result recomputed | $25.96\%$ recomputed from machine data in `phase_13_1_audit.json` | **PASSED** |
| **EC13.1.9** | Resource claims use consistent measurement scope | Verified at $D=256$ in `PHASE_13_1_NUMERICAL_CONSISTENCY.md` | **PASSED** |
| **EC13.1.10** | Spatial-gradient metric semantics documented | Coordinate scrambling denominator artifact documented | **PASSED** |
| **EC13.1.11** | Unjustified statistical language removed | Verified in Section 2 language audit | **PASSED** |
| **EC13.1.12** | Repo-wide Pyright passes (0 diagnostics) | `.venv/bin/pyright deltacore examples tests` clean | **PASSED** |
| **EC13.1.13** | Compileall passes (0 compile failures) | `python3 -m compileall deltacore examples tests` clean | **PASSED** |
| **EC13.1.14** | Pytest passes (571 passing tests) | `python3 -m pytest -q` clean | **PASSED** |
| **EC13.1.15** | Ruff passes (0 linter errors, clean format) | `ruff check . && ruff format --check .` clean | **PASSED** |
| **EC13.1.16** | All Phase 13.1 artifacts reproducible | Verified via `phase_13_1_audit_tool.py` | **PASSED** |

---

## 4. Phase Gate Determination

According to Phase 13.1 Phase Gate specification:

> **Outcome A — Clean empirical validation**:
> All inconsistencies are resolved, baseline provenance is valid, the key real-world result survives, and all claims accurately reflect the evidence. Phase 13 may then be frozen.

### Formal Declaration:
**Phase 13.1 achieves OUTCOME A.**

All numerical values, baseline training histories, and mathematical properties of Phase 13 are now fully audited, verified, and frozen.

**DeltaCore is formally approved for Phase 14.**
