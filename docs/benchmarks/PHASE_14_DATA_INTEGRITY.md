# DeltaCore Phase 14: Data Integrity & Temporal Leakage Audit

This document records the data integrity, preprocessing provenance, and temporal leakage prevention checks for **Phase 14: ERA5 North Atlantic / European 2m Temperature ($T_{2m}$)**.

---

## 1. Temporal Partitioning Audit

To prevent temporal look-ahead leakage and autoregressive overlap, strict chronological partitioning was enforced:

* **Train Partition**: Timesteps $t \in [0, 150)$ (150 steps; synoptic baseline climatology).
* **Validation Partition**: Timesteps $t \in [150, 200)$ (50 steps; offline tuning and validation).
* **Test Partition**: Timesteps $t \in [200, 360)$ (160 steps; online sequential streaming evaluation).

### Disjoint Sets Verification
Let $T_{\text{train}} = \{0, 1, \dots, 149\}$, $T_{\text{val}} = \{150, 151, \dots, 199\}$, and $T_{\text{test}} = \{200, 201, \dots, 359\}$.

$$\begin{aligned}
T_{\text{train}} \cap T_{\text{val}} &= \emptyset \\
T_{\text{val}} \cap T_{\text{test}} &= \emptyset \\
T_{\text{train}} \cap T_{\text{test}} &= \emptyset
\end{aligned}$$

No sample or window overlaps across partition boundaries.

---

## 2. Normalization Statistics Isolation

All scaling and standardization constants were computed exclusively from the training split:

$$\mu_{\text{train}} = \frac{1}{N_{\text{train}} \cdot H \cdot W} \sum_{t=0}^{N_{\text{train}}-1} \sum_{i,j} X_{t, i, j}$$
$$\sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}} \cdot H \cdot W} \sum_{t=0}^{N_{\text{train}}-1} \sum_{i,j} (X_{t, i, j} - \mu_{\text{train}})^2}$$

* $\mu_{\text{train}} = 5.3719^\circ\text{C}$
* $\sigma_{\text{train}} = 7.1843^\circ\text{C}$

Validation and test observations were scaled using $(\mu_{\text{train}}, \sigma_{\text{train}})$ without updating the normalization parameters on test observations.

---

## 3. Streaming Target Isolation

During online test streaming ($t = 0, \dots, T_{\text{test}}-1$):
1. The model receives only $x_t \in \mathbb{R}^D$.
2. The model outputs prediction $\hat{x}_{t+1}$.
3. Ground-truth target $x_{t+1}$ is disclosed only after prediction emission.
4. Error $e_t = x_{t+1} - \hat{x}_{t+1}$ is used solely for internal associative memory state adaptation ($M_{t+1}$).
5. Model parameters remain strictly frozen ($\Delta \theta = 0$).

---

## 4. Integrity Signatures

* Ingestion Source: `deltacore/streaming/atmospheric_spatiotemporal.py`
* Evaluation Test Input/Target Checksum (SHA-256): Recorded in [dataset_config.json](file:///Users/urjasoft/Documents/DeltaCore/docs/benchmarks/artifacts/phase_14/dataset_config.json).
* Audit Result: **LEAKAGE AUDIT PASSED**.
