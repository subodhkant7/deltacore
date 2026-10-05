# DeltaCore Phase 14: Second Real-World Domain Selection

This document establishes the objective selection criteria, candidate evaluations, and formal selection rationale for the **second real-world spatio-temporal domain** in DeltaCore Phase 14.

In accordance with Section 2 of the Phase 14 mandate, this selection is conducted **prior to running any model evaluations on the benchmark**.

---

## 1. Objective Selection Criteria

To determine whether the findings of Phase 13 (which evaluated sea-surface temperature in the equatorial Pacific) transfer across physical domains, the second dataset must satisfy the following seven scientific criteria:

1. **Independent Physical Mechanism**: The physical processes governing the field must be fundamentally different from oceanic Kelvin wave thermocline advection (e.g. fast atmospheric synoptic advection, baroclinic instability, Rossby wave propagation).
2. **True Spatio-Temporal Topology**: Must consist of continuous, gridded spatial observations repeated regularly over time.
3. **Naturally Occurring Distribution Shift**: Must contain at least one well-documented, externally validated meteorological shift (e.g., sudden stratospheric warming, synoptic polar outbreak, or winter storm transition) rather than stationary synthetic noise.
4. **Reproducible & Deterministic Ingestion**: Data generation and preprocessing must be self-contained, fully deterministic, and runnable from a single CLI command with zero machine-specific path dependencies.
5. **Permissible Licensing**: Must be based on publicly available, open-access meteorological data (e.g., Copernicus ECMWF Open Data / CC-BY 4.0 or NOAA open access).
6. **Manageable Computational Footprint**: Must support evaluation on standard CPU workstations ($D \in \{64, 256\}$, sequence length $T \sim 360$) in under 10 seconds per run to allow full multi-seed statistical testing.
7. **Strict Separation of Train, Validation, and Test**: Temporal duration must be sufficient to permit disjoint chronological splitting without cross-boundary target contamination.

---

## 2. Candidate Datasets Considered

| Candidate Dataset | Source Domain | Spatial Continuous? | Temporal Cadence | Natural Shift Present? | Computational Size | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ERA5 North Atlantic / European $T_{2m}$** | Atmospheric Reanalysis | Yes (Regular 2D Grid) | 6-hourly / synoptic | Yes (Jan 2021 Sudden Stratospheric Warming & Storm Filomena) | Compact ($D=64 / 256$, $T=360$) | **SELECTED** |
| **Global High-Res ERA5 (0.25°)** | Global Atmosphere | Yes | Hourly | Yes | Enormous (>50 GB) | **REJECTED** (Prohibitive compute) |
| **NOAA GHCN-Daily Weather Stations** | Ground Stations | No (Sparse/Irregular) | Daily | Yes | Variable | **REJECTED** (No continuous 2D spatial grid) |
| **IMERG Satellite Radar Precipitation** | Precipitation | Yes | 30-min | Yes | Moderate | **REJECTED** (High sparsity, >85% zero inflation) |
| **NCEP/NCAR 500 hPa Geopotential Height** | Mid-Troposphere | Yes | 6-hourly | Yes | Moderate | **Viable alternative**, but $T_{2m}$ exhibits richer local thermal gradients |

---

## 3. Formal Selection & Dataset Profile

### Selected Domain: **ERA5 North Atlantic & European Storm Track 2-Meter Temperature ($T_{2m}$)**

* **Dataset Name**: `ERA5_NorthAtlantic_European_T2m`
* **Source Organization**: European Centre for Medium-Range Weather Forecasts (ECMWF) / Copernicus Climate Change Service (C3S)
* **Dataset Version**: ERA5 Synoptic Reanalysis (2020–2021)
* **License**: Copernicus C3S Open Licence / CC-BY 4.0 International
* **Geographical Bounding Box**: $40^\circ\text{N} - 65^\circ\text{N}, 30^\circ\text{W} - 20^\circ\text{E}$ (covering the North Atlantic storm corridor, British Isles, Western Europe, and the Nordic Sea)
* **Variable**: 2-meter air temperature ($T_{2m}$ in $^{\circ}\text{C}$)
* **Temporal Coverage**: 360 synoptic timesteps (6-hourly intervals spanning late 2020 to mid 2021)
* **Primary Benchmark Dimension**: $H=8, W=8, C=1 \implies D=64$
* **Multi-Resolution Scaling Dimension**: $H=16, W=16, C=1 \implies D=256$
* **Identified Meteorological Shift**: **January–February 2021 Sudden Stratospheric Warming (SSW) and European Arctic Polar Outbreak (Storm Filomena)**:
  * In January 2021, a major disruption of the stratospheric polar vortex triggered a massive southward displacement of Arctic cold air across Europe, causing extreme negative temperature anomalies ($-8^\circ\text{C}$ to $-14^\circ\text{C}$) and rapid baroclinic wave restructuring.
  * Located within the test split at relative timesteps $t \in [45, 115]$ (absolute timesteps $[245, 315]$), peaking at relative timestep $75$ (absolute timestep $275$).

---

## 4. Key Differences from Phase 13 OISST

| Characteristic | Phase 13 (NOAA OISST v2.1) | Phase 14 (ERA5 Atmospheric $T_{2m}$) | Physical Implication |
| :--- | :--- | :--- | :--- |
| **Physical Fluid Medium** | Ocean (upper mixed layer) | Atmosphere (boundary layer) | Atmospheric dynamics are $\approx 10\times$ faster than oceanic advection. |
| **Dominant Dynamics** | Slow equatorial Kelvin wave advection | Fast synoptic Rossby wave / baroclinic cyclones | Higher temporal variance, sharper spatial fronts. |
| **Thermal Inertia** | High oceanic thermal capacity | Low atmospheric heat capacity | Faster recovery and rapid transient shifts. |
| **Shift Anomaly Direction** | Positive thermal surge ($+2.8^\circ\text{C}$ El Niño) | Negative cold outbreak ($-6.5^\circ\text{C}$ Polar Outbreak) | Inverts anomaly sign and testing adaptation to rapid chilling. |
| **Spatial Gradient Energy** | Broad, diffuse equatorial gradients | Sharp frontal boundaries along storm tracks | Higher high-frequency spatial variation. |

This dataset provides a genuine, independent physical test of whether DeltaCore's adaptive state mechanisms transfer beyond oceanic sea-surface temperature.
