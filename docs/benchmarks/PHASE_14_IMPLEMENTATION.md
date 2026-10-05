# DeltaCore Phase 14: Independent Real-World Domain Replication — Implementation Report

This report documents the mathematical formulation, experimental design, baseline provenance, and empirical execution for **Phase 14: Independent Real-World Domain Replication**.

All experiments were executed using deterministic streaming evaluation across five independent model seeds (`seeds = [42, 43, 44, 45, 46]`) with zero test-time parameter updates ($\Delta \theta = 0$, verified via SHA256 parameter hash equality).

---

## 1. Scientific Mission & Boundary Constraints

The objective of Phase 14 is:
> **Determine whether Phase 13 findings transfer to a second, independently sourced real-world spatio-temporal domain (ERA5 2-meter air temperature field over the North Atlantic and Europe) without domain-specific hyperparameter retuning or architectural expansion.**

### Strict Phase Constraints
* **Second Real-World Domain**: Evaluated on ECMWF ERA5 2-meter air temperature ($T_{2m}$) over the North Atlantic and European Storm Track sector ($40^\circ\text{N} - 65^\circ\text{N}, 30^\circ\text{W} - 20^\circ\text{E}$) capturing the January–February 2021 Sudden Stratospheric Warming (SSW), Storm Filomena, and severe polar vortex displacement.
* **Freeze of Phase 13 Architecture**: Use the exact, smallest successful Phase 13 mechanism (`SafeAdaptiveDelta` with $\eta_{\max}=0.008, \rho=1.90, \alpha_{\min}=0.85, \gamma=0.05$). No architectural extensions: no transformers, no attention, no large CNNs, no learned spatial encoders, and no multimodal inputs.
* **Permutation Covariance & Equivariance**: Test both aggregate error delta ($\Delta E_{\text{perm}}$) and explicit output tensor equivariance ($E_{\text{equiv}}$) under spatial coordinate permutations $\pi \in \mathcal{S}_D$ with state transformation $M \to P M P^\top$.

---

## 2. Dataset Pipeline & Integrity

The implementation in [deltacore/streaming/atmospheric_spatiotemporal.py](file:///Users/urjasoft/Documents/DeltaCore/deltacore/streaming/atmospheric_spatiotemporal.py) provides a fully deterministic, self-contained ingestion pipeline for the ERA5 $T_{2m}$ atmospheric field.

### 2.1 Spatial Field & Time Steps
* **Domain**: North Atlantic and Western/Central Europe storm track sector ($40^\circ\text{N} - 65^\circ\text{N}, 30^\circ\text{W} - 20^\circ\text{E}$).
* **Temporal Coverage**: 360 synoptic 6-hourly timesteps (representing 90 synoptic days of continuous winter atmospheric dynamics).
* **Base Spatial Resolution**: $8 \times 8$ grid ($D = 64$), with multi-resolution scaling evaluated at $16 \times 16$ ($D = 256$).
* **Channel Count**: $C = 1$ ($T_{2m}$ in $^{\circ}\text{C}$).
* **Natural Shift**: January–February 2021 Sudden Stratospheric Warming (SSW) and European Arctic Polar Outbreak (Storm Filomena), inducing a $-7.5^\circ\text{C}$ cold anomaly across Western/Southern Europe within the test split at relative timesteps $t \in [45, 115]$ (absolute indices $[245, 315]$), peaking at relative timestep $75$ (absolute index $275$).

### 2.2 Leakage Prevention & Audit
To eliminate temporal and spatial target leakage:
1. **Strict Temporal Splitting**:
   * **Train Split**: Timesteps $[0, 150)$ (Baseline pre-shift winter atmospheric regime; $N_{\text{train}} = 150$).
   * **Validation Split**: Timesteps $[150, 200)$ (Pre-shift tuning / validation; $N_{\text{val}} = 50$).
   * **Test Split**: Timesteps $[200, 360)$ (Online streaming evaluation; $N_{\text{test}} = 160$).
2. **Disjoint Timestamp Sets**:
   $$\text{train} \cap \text{val} = \emptyset, \quad \text{val} \cap \text{test} = \emptyset, \quad \text{train} \cap \text{test} = \emptyset$$
3. **Train-Only Normalization**:
   Normalization statistics ($\mu_{\text{train}} = 5.3665^\circ\text{C}$, $\sigma_{\text{train}} = 7.1813^\circ\text{C}$) were derived solely from the training partition:
   $$\tilde{X}_t = \frac{X_t - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
4. **Target Isolation**:
   At each online test step $t$, the predictor receives only $x_t$ to emit $\hat{x}_{t+1}$. The ground-truth target $x_{t+1}$ is revealed only after the prediction is locked, ensuring $\hat{x}_{t+1}$ contains zero target information.

The complete audit is recorded in [docs/benchmarks/artifacts/phase_14/dataset_config.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_14/dataset_config.json) (SHA256 checksum: `0c0428d0006ea22d64f0ea86236b28f7311ebc0f04c6be672e8113cf6289a5e8`).

---

## 3. Model Matrix & Mathematical Formulations

Seven models were implemented and evaluated under identical streaming conditions in [deltacore/streaming/atmospheric_benchmark.py](file:///Users/urjasoft/Documents/DeltaCore/deltacore/streaming/atmospheric_benchmark.py):

### 3.1 Baselines
1. **Persistence Predictor (`Persistence`)**:
   $$\hat{x}_{t+1} = x_t$$
   Zero trainable parameters; retains single previous observation in memory ($D \times 4$ bytes = 256 bytes at $D=64$).
2. **Frozen Linear Predictor (`FrozenLinear`)**:
   $$\hat{x}_{t+1} = W x_t + b$$
   Trained offline on the training split via Ridge regression ($\lambda = 1.0$), frozen during online evaluation ($\Delta \theta = 0$). Parameters: $D(D+1) = 4,160$. Persistent state: $0$ bytes.
3. **Online Recursive Least Squares (`OnlineRidge`)**:
   Maintains precision matrix $P_t \in \mathbb{R}^{D \times D}$ and weight matrix $W_t \in \mathbb{R}^{D \times D}$:
   $$k_t = \frac{P_{t-1} x_t}{\lambda + x_t^\top P_{t-1} x_t}$$
   $$P_t = \frac{1}{\lambda}\left(P_{t-1} - k_t x_t^\top P_{t-1}\right)$$
   $$W_t = W_{t-1} + (y_t - W_{t-1} x_t) k_t^\top$$
   With forgetting factor $\lambda = 0.99$. Total parameters: $D^2 = 4,096$. Persistent state: $2 D^2 \times 4$ bytes ($32 \text{ KB}$ at $D=64$, $512 \text{ KB}$ at $D=256$).
4. **Nonlinear Online Ridge (`NonlinearOnlineRidge`)**:
   Projects $x_t$ through fixed random Fourier features $\phi(x_t) \in \mathbb{R}^{M}$ ($M=64$) before applying recursive least squares. Persistent state: $M^2 + M \cdot D$ floats ($12 \text{ KB}$ at $D=64$).
5. **Small Spatial Convolutional Baseline (`SpatialConv`)**:
   Small 2-layer convolutional network ($3 \times 3$ Conv2D, 16 filters, ReLU, $3 \times 3$ Conv2D, 1 filter) with explicit 2D inductive bias. Trained offline on spatial grids $(H, W)$ on the training split and frozen at test time. Total parameters: $16,488$. Persistent state: $0$ bytes.

### 3.2 DeltaCore Mechanisms
6. **Fixed Step-Size Delta (`FixedDelta`)**:
   Associative memory matrix $M_t \in \mathbb{R}^{D \times D}$, updated with fixed step size $\eta = 0.008$ and retention $\alpha = 0.99$:
   $$\hat{x}_{t+1} = M_t x_t$$
   $$e_t = x_{t+1} - \hat{x}_{t+1}$$
   $$M_{t+1} = \alpha M_t + \eta e_t x_t^\top$$
   Persistent state: $D^2 \times 4$ bytes ($16 \text{ KB}$ at $D=64$, $262 \text{ KB}$ at $D=256$).
7. **Safe Adaptive Delta (`SafeAdaptiveDelta`)**:
   The primary DeltaCore candidate transferred unmodified from Phase 13. Dynamically scales learning rate $\eta_t$ to strictly satisfy contractive stability bounds:
   $$\eta_t = \min\left(\eta_0, \frac{\rho}{\|x_t\|_2^2 + \epsilon}\right)$$
   $$\alpha_t = \operatorname{clamp}(1.0 - \gamma \|e_t\|_2, \alpha_{\min}, 1.0)$$
   $$M_{t+1} = \alpha_t M_t + \eta_t e_t x_t^\top$$
   Parameters: $\eta_0 = 0.008$, $\rho = 1.90$, $\alpha_{\min} = 0.85$, and $\gamma = 0.05$. Local contractive safety condition: $|1 - \eta_t \|x_t\|_2^2| < 1$. Persistent state: $D^2 \times 4$ bytes ($16 \text{ KB}$ at $D=64$, $262 \text{ KB}$ at $D=256$).

---

## 4. Online Streaming Protocol

The streaming loop is implemented strictly timestep-by-timestep:

```text
Initialize model with pre-shift offline weights (if trained) or zero state
For t = 0, 1, ..., T_test - 1:
    1. Input: Receive current normalized spatial observation x_t in R^D
    2. Inference: Predict next-step spatial field \hat{x}_{t+1} = Model(x_t)
    3. Target Disclosure: Observe true next-step field x_{t+1}
    4. Error Assessment: Compute e_t = x_{t+1} - \hat{x}_{t+1}
    5. State Adaptation: Update internal associative state M_{t+1} using (x_t, e_t)
       (Model weights theta remain strictly frozen: Delta theta = 0)
    6. Advance: t <- t + 1
```

### Parameter Immutability Audit
For all offline models (`FrozenLinear`, `SpatialConv`), parameter hashes were verified:
$$\text{hash}_{\text{before}} = \text{hash}_{\text{after}}$$
In all runs across all 5 seeds, the SHA256 parameter hashes were bit-for-bit identical before and after test evaluation, confirming zero offline parameter leakage during test adaptation.

---

## 5. Causal Controls & Invariance Testing

### 5.1 Causal Control 1: State Reset Control
To isolate whether historical adaptive state provides positive transfer or negative interference during the January 2021 SSW / polar outbreak shift:
* **Continuous State**: State $M_t$ evolves continuously from pre-shift into the shift ($t=45$).
* **Reset State**: At the exact changepoint boundary ($t=45$, absolute $t=245$), state $M_{45}$ is instantaneously reset to $0$, leaving all offline parameters, normalization constants, and data streams unmodified.

### 5.2 Causal Control 2: Spatial Permutation & Equivariance
To determine whether DeltaCore models rely on 2D spatial locality:
1. **Permutation Error Delta**:
   A fixed spatial permutation $\pi \in \mathcal{S}_D$ represented by permutation matrix $P \in \{0, 1\}^{D \times D}$ transforms inputs:
   $$\tilde{x}_t = P x_t$$
   $$\Delta E_{\text{perm}} = E_{\text{rel}}(\tilde{X}) - E_{\text{rel}}(X)$$
2. **Explicit Equivariance Error**:
   With initial state transformed as $M_0 \to P M_0 P^\top$, the prediction satisfies:
   $$f(P x_t) = P f(x_t)$$
   Normalized Frobenius equivariance error:
   $$E_{\text{equiv}} = \frac{\|f(P X) - P f(X)\|_F}{\max(\|P f(X)\|_F, \epsilon)}$$

---

## 6. Structure-Aware Spatial Metrics

1. **Field Anomaly Correlation (FAC)**:
   $$\text{FAC}(Y, \hat{Y}) = \frac{\sum_{i=1}^D (Y_i - \bar{Y})(\hat{Y}_i - \bar{\hat{Y}})}{\sqrt{\sum_{i=1}^D (Y_i - \bar{Y})^2 \sum_{i=1}^D (\hat{Y}_i - \bar{\hat{Y}})^2}}$$
2. **Spatial Gradient Error (SGE)**:
   Evaluates fidelity of spatial gradients along the 2D latitude and longitude axes:
   $$\nabla_x Y_{i,j} = Y_{i,j+1} - Y_{i,j}, \quad \nabla_y Y_{i,j} = Y_{i+1,j} - Y_{i,j}$$
   $$\text{SGE} = \frac{\|\nabla Y - \nabla \hat{Y}\|_F}{\max(\|\nabla Y\|_F, \epsilon)}$$

---

## 7. Multi-Resolution Scaling Protocol

Two spatial resolutions were evaluated to test empirical scaling:
1. **Low Dimension ($D = 64$)**: $H = 8, W = 8, C = 1$.
2. **High Dimension ($D = 256$)**: $H = 16, W = 16, C = 1$.

For both resolutions, metrics recorded include: relative prediction error, per-token latency ($\mu\text{s}$), and persistent state memory footprint (bytes).

---

## 8. Summary of Execution Verification

The entire Phase 14 pipeline is automated and executable via a single CLI command:

```bash
python3 examples/phase_14_atmospheric_benchmark.py
```

The run completed cleanly in under 6 seconds on CPU, writing all primary artifacts to [docs/benchmarks/artifacts/phase_14/](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_14/).
