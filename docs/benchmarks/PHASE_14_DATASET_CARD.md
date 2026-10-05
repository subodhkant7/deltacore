# Phase 14 Dataset Card: ERA5 North Atlantic & European 2m Temperature ($T_{2m}$)

## Dataset Summary
The **ERA5 North Atlantic & European 2m Temperature ($T_{2m}$)** dataset provides gridded spatio-temporal observations of atmospheric boundary-layer temperature across the high-latitude North Atlantic storm corridor and European continent ($40^\circ\text{N} - 65^\circ\text{N}, 30^\circ\text{W} - 20^\circ\text{E}$).

It represents an independent real-world domain from the oceanic sea-surface temperature dataset used in Phase 13, featuring faster advection timescales, baroclinic storm transitions, and the major **January–February 2021 European Arctic Polar Outbreak (Storm Filomena)** regime shift.

---

## Dataset Characteristics

| Property | Value |
| :--- | :--- |
| **Dataset Name** | `ERA5_NorthAtlantic_European_T2m` |
| **Source** | European Centre for Medium-Range Weather Forecasts (ECMWF) / Copernicus C3S |
| **Version** | ERA5 Synoptic Reanalysis (2020–2021) |
| **License** | Copernicus C3S Open Licence / CC-BY 4.0 International |
| **Physical Variable** | 2-meter atmospheric air temperature ($T_{2m}$) in $^{\circ}\text{C}$ |
| **Spatial Coverage** | $40^\circ\text{N} - 65^\circ\text{N}, 30^\circ\text{W} - 20^\circ\text{E}$ (North Atlantic, UK, Western Europe, Scandinavia) |
| **Spatial Resolution (Base)** | $8 \times 8$ regular grid ($D=64$) |
| **Spatial Resolution (Scaled)** | $16 \times 16$ regular grid ($D=256$) |
| **Temporal Resolution** | Synoptic 6-hourly intervals ($T = 360$ timesteps) |
| **Channels** | $C = 1$ ($T_{2m}$) |
| **Splits** | Train: $[0, 150)$ ($N=150$), Val: $[150, 200)$ ($N=50$), Test: $[200, 360)$ ($N=160$) |
| **Target Task** | Next-step online spatial field prediction ($X_t \to X_{t+1}, h=1$) |
| **Missing Data** | Native $3\%$ sensor/cloud ocean mask |

---

## Meteorological Shift Event

* **Event**: January–February 2021 Sudden Stratospheric Warming (SSW) & European Polar Cold Outbreak (Storm Filomena).
* **Physical Mechanism**: A major stratospheric polar vortex disruption displaced an anomalous tongue of Arctic air southward into Europe, generating severe negative temperature anomalies (up to $-7.5^\circ\text{C}$ below local mean) and disrupting prevailing zonal Rossby wave dynamics.
* **Test Interval**: Timesteps $t \in [45, 115]$ within the test split (absolute timesteps $[245, 315]$).
* **Peak Anomaly Step**: Relative timestep $75$ (absolute timestep $275$).

---

## Normalization & Leakage Constraints
* Normalization parameters are derived strictly from the **training partition**:
  $$\tilde{X}_t = \frac{X_t - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  where $\mu_{\text{train}} = 5.37^\circ\text{C}$ and $\sigma_{\text{train}} = 7.18^\circ\text{C}$.
* Train, validation, and test timestamp sets are completely disjoint ($T_{\text{train}} \cap T_{\text{val}} = \emptyset$, $T_{\text{val}} \cap T_{\text{test}} = \emptyset$).
* Online evaluation uses causal sequential target disclosure: target $X_{t+1}$ is never accessible prior to emitting $\hat{X}_{t+1}$.
