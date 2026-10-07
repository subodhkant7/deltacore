# DeltaCore Open-Source Release & Research Freeze Closure Report

---

## 1. Final Status

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          DELTACORE STATUS                              │
├────────────────────────────────────────────────────────────────────────┤
│ Core Research Implementation          FROZEN                           │
│ Telemetry Novelty Detection           NOT RECOMMENDED                  │
│ RecoveryOS Integration                PERMANENTLY REJECTED & FROZEN    │
│ Scientific Results                    VERIFIED / DOCUMENTED            │
│ Negative Findings                     PRESERVED                        │
│ Open-Source Archive (v0.2.0)          RELEASE-READY                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Canonical Decisive Benchmark Results

The following table constitutes the single authoritative numerical record for the decisive 20-seed non-stationary matched-marginal benchmark (`experiments/higher_order_regime_shift.py` v3.0.0, feature dimension $D = 128$, categorical marginals $\text{TVD} \le 0.05$):

| Metric | Value | Inferential Scope & Resampling Details |
| :--- | :---: | :--- |
| **Online Covariance Mean AUROC** | **0.5976 $\pm$ 0.0667** | Standard regularized 2nd-order Mahalanobis tracker |
| **DeltaCore Gated Mean AUROC** | **0.5711 $\pm$ 0.0801** | Adaptive associative matrix ($M_t \in \mathbb{R}^{D \times D}$) with score gating |
| **Mean Paired Difference ($\bar{d}$)** | **-0.0265** | $d_i = \text{AUROC}_{\text{DeltaCore},i} - \text{AUROC}_{\text{Covariance},i}$ across 20 evaluation seeds |
| **95% Paired Bootstrap CI** | **[-0.0478, -0.0072]** | $B = 10{,}000$ paired seed resamples, seed 42, percentile method (strictly excludes 0) |
| **DeltaCore Seed Wins** | **4 / 20** | DeltaCore won on 4 streams, lost on 16 streams |
| **Exact Two-Sided Binomial Sign Test** | **$p = 0.0118$** | $H_0: P(\text{DeltaCore wins}) = 0.5$, exact binomial calculation: $2 \times \sum_{j=0}^4 \binom{20}{j} 0.5^{20}$ |
| **Exact Paired Sign-Flip Randomization Test** | **$p = 0.0188$** | Exhaustive enumeration across all $2^{20} = 1{,}048{,}576$ sign configurations under exchangeability null ($19{,}728 / 1{,}048{,}576$) |
| **Paired Effect Size (Cohen's $d$)** | **-0.58** | Medium negative effect size for observed benchmark differences |
| **Online Centroid Gated Mean AUROC** | **0.5330 $\pm$ 0.0813** | First-order baseline; DeltaCore paired difference $+0.0382$ ($p = 0.0004$) |
| **Online PCA Gated Mean AUROC** | **0.5641 $\pm$ 0.0903** | Low-rank projection baseline; DeltaCore paired difference $+0.0070$ ($p = 0.8238$, CI $[-0.0074, 0.0209]$) |

---

## 3. Disambiguation of Statistical Methods

The three inferential procedures answer fundamentally distinct questions and must never be conflated:

### Inferential Question 1 — Mean Effect Uncertainty
- **Method**: Paired percentile bootstrap confidence interval ($B = 10{,}000$, seed 42).
- **Question**: *What sampling uncertainty surrounds the estimated mean paired performance difference?*
- **Canonical Result**: $95\%\text{ paired percentile bootstrap CI} = [\mathbf{-0.0478}, \mathbf{-0.0072}]$.
- **Interpretation**: The bootstrap interval for the mean paired difference excludes zero and is entirely negative for this benchmark.
- **Mandatory Caveat**: This percentile bootstrap interval is a resampling-based uncertainty estimate. It is not an assumption-free or mathematically exact confidence interval.

### Inferential Question 2 — Randomization / Magnitude
- **Method**: Exact paired sign-flip permutation/randomization test.
- **Question**: *Is the observed mean paired difference statistic ($\bar{d} = -0.0265$) inconsistent with a symmetric sign-exchangeability null?*
- **Canonical Result**: $2^{20} = 1{,}048{,}576$ exact sign assignments; $19{,}728$ assignments satisfy $|\bar{d}_{\text{perm}}| \ge |\bar{d}_{\text{observed}}|$; $p = 19{,}728 / 1{,}048{,}576 = 0.0188140869 \dots \approx \mathbf{0.0188}$.
- **Interpretation**: The $p$-value is exact with respect to exhaustive enumeration of the sign assignments under the specified exchangeability-of-signs null.
- **Mandatory Caveat**: "Exact" refers to exhaustive enumeration under the stated null; it does not mean assumption-free inference. Exact enumeration removes Monte Carlo approximation error, but the inferential validity of the test remains conditional on the null and exchangeability assumptions specified by the test.

### Inferential Question 3 — Direction / Majority
- **Method**: Exact two-sided binomial sign test.
- **Question**: *Does DeltaCore win approximately half of the paired streams?*
- **Canonical Result**: DeltaCore wins: $4 / 20$, DeltaCore loses: $16 / 20$, $p = \mathbf{0.0118}$.
- **Interpretation**: Under the two-sided 50/50 sign null, the observed directional imbalance is statistically significant for this benchmark.

Explicitly:
- **Sign test**: direction / majority across streams.
- **Sign-flip test**: magnitude / paired effect under sign exchangeability.
- **Bootstrap**: uncertainty interval for the mean difference.

---

## 4. Scope of Inference

> **The evidence is benchmark-specific. It establishes that, on the predefined 20-seed non-stationary matched-marginal benchmark with $D=128$ and the specified feature representation, DeltaCore Gated did not demonstrate an advantage over regularized Online Covariance, while the observed paired difference favored Online Covariance.**
>
> **The benchmark does not establish universal superiority of Online Covariance, universal inferiority of DeltaCore, or performance ordering under telemetry distributions, dimensions, drift processes, workloads, or deployment conditions not represented by the benchmark.**
>
> **The result therefore supports terminating this specific DeltaCore research direction for RecoveryOS, rather than claiming that adaptive telemetry detection as a broader problem has been solved.**

---

## 5. Practical vs. Statistical Significance

The statistical metrics must be interpreted alongside engineering costs rather than generalized into universal claims:
- **Observed Benchmark Effect**: Mean paired AUROC difference was $-0.0265$ (Cohen's $d = -0.58$). These quantify the observed benchmark effect and should not be converted into a universal production-performance claim.
- **Engineering Complexity & Latency**:
  - DeltaCore requires an unconstrained square associative matrix ($M_t \in \mathbb{R}^{D \times D}$) consuming $4 D^2$ bytes ($64\text{ KB}$ at $D = 128$) and median scoring latency of $25.9\text{ µs}$ ($8.6\times$ higher than first-order centroids at $3.0\text{ µs}$).
  - Online Covariance also consumes $O(D^2)$ state ($64\text{ KB}$) with comparable matrix operations.
- **Engineering Tradeoff**: Within the evaluated design and benchmark, DeltaCore incurred substantially greater state/latency cost than first-order methods without demonstrating a compensating performance advantage over regularized online covariance.

---

## 6. Canonical Scientific Conclusion

> **"Under the predefined 20-seed non-stationary matched-marginal benchmark, regularized Online Covariance significantly outperformed DeltaCore Gated under both the exact paired sign test and exact paired sign-flip randomization test."**
>
> **"DeltaCore Gated did not demonstrate a performance advantage over regularized Online Covariance on the decisive benchmark."**

DeltaCore successfully demonstrated deterministic controller semantics, local Lyapunov contraction, and exact state persistence, but did not establish a performance advantage over classical regularized second-order covariance estimation on the targeted benchmark.

---

## 7. Reproducibility

The complete benchmark and regression suite can be reproduced with:

```bash
# 1. Environment installation
pip install -e .

# 2. Run minimal adaptation & persistence example
python examples/basic_adaptation.py

# 3. Full regression suite (689 tests)
pytest -q

# 4. Static type checking & formatting
pyright
ruff check .
ruff format --check .

# 5. Bytecode compilation
python3 -m compileall deltacore examples tests experiments

# 6. Decisive 20-seed benchmark reproduction
python experiments/higher_order_regime_shift.py
```

---

## 8. Repository Integrity

| Check | Tool / Command | Result |
| :--- | :--- | :--- |
| **Unit Tests** | `pytest -q` | **689 passed in 30.26s** |
| **Static Types** | `pyright` | **0 errors, 0 warnings, 0 informations** |
| **Linter** | `ruff check .` | **All checks passed!** |
| **Formatter** | `ruff format --check .` | **198 files already formatted** |
| **Compilation** | `python3 -m compileall` | **Clean, 0 errors** |
| **Smoke Example** | `python examples/basic_adaptation.py` | **Passed cleanly** (checksum / persistence verified) |
| **Version Alignment** | `pyproject.toml`, `deltacore/__init__.py` | **0.2.0** |
| **Git Status** | `git status` | **Clean working tree** |

---

## 9. Closure Rule & RecoveryOS Decision

> **The RecoveryOS integration was rejected because DeltaCore did not demonstrate a sufficient empirical advantage over a simpler regularized online covariance baseline on the decisive benchmark, while incurring greater state complexity and latency. DeltaCore is therefore removed as an active RecoveryOS research dependency.**
>
> **Any future work on adaptive telemetry detection must be treated as a new independent research effort with a new hypothesis, benchmark design, and preregistered evaluation.**
