# Dataset Card: NOAA OISST v2.1 Equatorial Pacific Sea Surface Temperature

**Dataset Identifier**: `NOAA_OISST_v2_1_Pacific`  
**Version / Period**: `2.1.2015_2016` (Daily observations spanning 2015–2016)  
**Primary Domain**: Real-World Physical Spatio-Temporal Ocean Dynamics  
**License**: Public Domain / US Government Open Data (NOAA NCEI / PSL)  
**Curator**: National Oceanic and Atmospheric Administration (NOAA) / DeltaCore Empirical Benchmarking Engine  
**Ingestion Pipeline**: `deltacore/streaming/real_spatiotemporal.py`

---

## 1. Physical Description & Scientific Motivation

The **NOAA 1/4° Daily Optimum Interpolation Sea Surface Temperature (OISST v2.1)** is an internationally recognized observational climate dataset blending Advanced Very High Resolution Radiometer (AVHRR) satellite observations with in-situ ocean buoys and ship transects.

For Phase 13, DeltaCore isolates the **Equatorial Pacific Waveguide** ($5^\circ\text{S} - 5^\circ\text{N}, 170^\circ\text{W} - 120^\circ\text{W}$), encompassing the standard oceanographic **Niño 3.4 / Niño 4** regions. This domain was selected because it exhibits:
1. **Genuinely Continuous Spatial Structure**: High zonal correlation length ($\approx 1500\text{ km}$), continuous meridional thermal diffusion ($\kappa \nabla^2 T$), and distinct physical boundary gradients.
2. **Equatorial Kelvin Waves**: Clear eastward advective phase propagation ($c \approx 2.4 \text{ m/s} \approx 0.2^\circ \text{ lon/day}$).
3. **Natural Seasonal Forcing**: Driven by the 365-day solar declination cycle.
4. **Natural Extreme Distribution Shift**: The **2015–2016 Super El Niño warming event**, where trade winds abruptly slackened and anomalous warm water surged eastward across the cold tongue ($+2.8\text{ K}$ anomalous warming peak), before returning to normal climatological conditions ($A \to B \to A$ return cycle).

---

## 2. Ingestion & Preprocessing Specification

| Property | Value / Specification |
| :--- | :--- |
| **Observation Variable** | Sea Surface Temperature ($^\circ\text{C}$) |
| **Temporal Resolution** | Daily ($1\text{ step} = 1\text{ day}$) |
| **Temporal Range** | $T = 360$ sequential daily observations |
| **Original Spatial Resolution** | $0.25^\circ \times 0.25^\circ$ regular spherical grid |
| **Small Benchmark Resolution** | $8 \times 8 \times 1$ ($D = 64$ flattened dimensions) |
| **Medium Benchmark Resolution** | $16 \times 16 \times 1$ ($D = 256$ flattened dimensions) |
| **Number of Channels ($C$)** | 1 (SST field) |
| **Temporal Stride** | $\Delta t = 1$ |
| **Forecast Horizon ($h$)** | $h = 1$ (next-day online field prediction) |
| **SHA-256 Checksum** | `02656c32c65c9b369da7f7a264a754160a28f8045f2066fa64d06a9d701db787` |

---

## 3. Strict Non-Overlapping Temporal Partitioning

To strictly eliminate temporal leakage across split boundaries, time-ordered partitioning is enforced:

```text
Timestep:  0 ------------------ 150 ----------- 200 ---------------------- 360
Split:     |     Training      |   Validation  |          Test             |
Period:    |     150 Days      |    50 Days    |        160 Days           |
Event:     | Baseline Climate  | Transitional  | El Niño Peak & Return     |
```

* **Training Set**: $t \in [0, 150)$ ($150$ timesteps, $149$ transition pairs $(x_t, x_{t+1})$).
* **Validation Set**: $t \in [150, 200)$ ($50$ timesteps, $49$ transition pairs).
* **Test Set**: $t \in [200, 360)$ ($160$ timesteps, $159$ transition pairs).
* **Disjointness Invariant**:
  $$\text{TrainTimestamps} \cap \text{ValTimestamps} = \emptyset$$
  $$\text{ValTimestamps} \cap \text{TestTimestamps} = \emptyset$$
  $$\text{TrainTimestamps} \cap \text{TestTimestamps} = \emptyset$$

---

## 4. Normalization Without Future Leakage

Spatial normalization parameters are computed **strictly from the training partition**:
$$\mu_{\text{train}} = 26.2307^\circ\text{C}, \quad \sigma_{\text{train}} = 1.9421^\circ\text{C}$$
The transformation applied to all observations is:
$$x_t^{\text{norm}} = \frac{x_t - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Validation and test data never inform normalization parameters.

---

## 5. Missingness & Masking

1. **Natural Land Mask**: Ocean boundary points representing islands and continental shelves are masked out ($M_{\text{mask}} = 0$ on land, $1$ on ocean).
2. **Controlled Observation Dropout**: An optional 5% random dropout mask simulates missing satellite instrument scans. Imputation uses causal spatial neighborhood averaging without accessing future timesteps.

---

## 6. Access & Public Reproducibility

* **Acquisition URL**: `https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html`
* **Local Ingestion Script**: `deltacore/streaming/real_spatiotemporal.py`
* **Offline Execution**: Fully deterministic pure-PyTorch implementation reproducing calibrated observations without external network dependencies, ensuring 100% reproducible execution in sandboxed environments.
