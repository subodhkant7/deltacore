# Dataset Integrity & Leakage Prevention Audit Report

**Date of Audit**: 2026-10-05  
**Auditor**: DeltaCore Scientific Integrity Enforcement Engine  
**Dataset**: NOAA Optimum Interpolation Sea Surface Temperature (OISST v2.1) Equatorial Pacific  
**Artifact**: `docs/benchmarks/artifacts/phase_13/dataset_config.json`  
**Integrity Status**: PASSED (Zero Temporal Leakage, Zero Normalization Contamination)

---

## 1. Provenance & Cryptographic Verification

| Property | Value |
| :--- | :--- |
| **Dataset Name** | `NOAA_OISST_v2_1_Pacific` |
| **Version** | `2.1.2015_2016` |
| **Primary Source** | NOAA Physical Sciences Laboratory / NCEI |
| **Source URL** | `https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html` |
| **Download / Verification Date** | `2026-10-05` |
| **SHA-256 Digest** | `02656c32c65c9b369da7f7a264a754160a28f8045f2066fa64d06a9d701db787` |
| **License** | Public Domain / US Government Open Data |

---

## 2. Temporal Partitioning Audit

All samples are partitioned strictly by time. Random sampling across time is explicitly forbidden.

```text
Split Name       Timestep Range     Pair Count (h=1)    Calendar Days
----------------------------------------------------------------------
Training         t in [0, 150)            149             Days 1-150
Validation       t in [150, 200)           49             Days 151-200
Test (Streaming) t in [200, 360)          159             Days 201-360
----------------------------------------------------------------------
Total                                     357             360 Days
```

### Disjointness Proof
1. $\text{TrainTimestamps} \cap \text{ValTimestamps} = \{0 \dots 149\} \cap \{150 \dots 199\} = \emptyset$ (Disjoint: **True**)
2. $\text{ValTimestamps} \cap \text{TestTimestamps} = \{150 \dots 199\} \cap \{200 \dots 359\} = \emptyset$ (Disjoint: **True**)
3. $\text{TrainTimestamps} \cap \text{TestTimestamps} = \{0 \dots 149\} \cap \{200 \dots 359\} = \emptyset$ (Disjoint: **True**)

---

## 3. Normalization Contamination Audit

To prevent future lookahead bias:
* **Training Mean**: $\mu_{\text{train}} = 26.2307^\circ\text{C}$
* **Training Std**: $\sigma_{\text{train}} = 1.9421^\circ\text{C}$
* **Validation / Test Treatment**: All evaluation inputs $x_t$ and targets $y_t$ are scaled using $(\mu_{\text{train}}, \sigma_{\text{train}})$.
* **Test Set Climatological Mean**: $\mu_{\text{test}} = 26.4712^\circ\text{C}$ (reflecting the El Niño warm phase).
* **Leakage Verification**: The test mean is $+0.24\text{ K}$ above $\mu_{\text{train}}$. The model must dynamically adapt to this shift online rather than benefiting from a pre-computed zero-mean centering across the test period.

---

## 4. Streaming Target Availability Audit

During sequential test-time streaming ($t = 200 \dots 359$):
1. Input $x_t = X_t^{\text{norm}}$ is passed to `predict_step(x_t)` to produce $\hat{y}_t$.
2. Target $y_t = X_{t+1}^{\text{norm}}$ is **withheld** until after $\hat{y}_t$ is output.
3. Prediction error $e_t = y_t - \hat{y}_t$ is recorded.
4. Internal adaptive state $M_t$ is updated via `adapt_step(x_t, y_t)`.
5. Parameter vectors $\theta$ remain strictly immutable ($\Delta \theta = 0$). Parameter hashes $\text{SHA256}(\theta_{\text{before}}) == \text{SHA256}(\theta_{\text{after}})$ are verified.

---

## 5. Missing Data Treatment

* A realistic land mask isolates oceanic grid points ($M_{\text{ocean}} = 1$).
* Where controlled dropout is enabled (5%), missing cells are imputed using **causal spatial neighborhood interpolation** of the current frame $t$ only. No future frames $t' > t$ are accessed during imputation.

---

## 6. Audit Conclusion

The dataset ingestion pipeline adheres to all scientific constraints of Section 1, 2, 3, and 27. The benchmark is free of data leakage and ready for baseline and DeltaCore model evaluation.
