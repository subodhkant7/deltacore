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

| Procedure | Question | Null / Framework | Result |
| :--- | :--- | :--- | :--- |
| **Paired percentile bootstrap** | Mean-effect uncertainty | Resampling framework ($B = 10{,}000$, seed 42) | 95% CI `[-0.0478, -0.0072]` |
| **Exact paired sign-flip test** | Paired-effect magnitude | Exchangeability-of-signs null ($2^{20} = 1{,}048{,}576$ assignments) | `p = 0.0188` ($19{,}728$ extreme) |
| **Exact two-sided sign test** | Direction / majority | 50/50 sign null ($H_0: p = 0.5$) | `p = 0.0118` (4 wins / 16 losses) |

Exhaustive enumeration removes Monte Carlo approximation error from the sign-flip calculation; it does not make the inferential procedure assumption-free. The percentile bootstrap interval is a resampling-based uncertainty estimate and is not treated as an exact confidence interval.

### Inferential Question 1 — Mean Effect Uncertainty
- **Method**: Paired percentile bootstrap confidence interval ($B = 10{,}000$, seed 42).
- **Question**: *What sampling uncertainty surrounds the estimated mean paired performance difference?*
- **Canonical Result**: $95\%\text{ paired percentile bootstrap CI} = [\mathbf{-0.0478}, \mathbf{-0.0072}]$.
- **Interpretation**: The bootstrap interval for the mean paired difference excludes zero and is entirely negative for this benchmark.
- **Mandatory Caveat**: The percentile bootstrap interval is a resampling-based uncertainty estimate and is not treated as an exact confidence interval.

### Inferential Question 2 — Randomization / Magnitude
- **Method**: Exact paired sign-flip permutation/randomization test.
- **Question**: *Is the observed mean paired difference statistic ($\bar{d} = -0.0265$) inconsistent with a symmetric sign-exchangeability null?*
- **Canonical Result**: $2^{20} = 1{,}048{,}576$ exact sign assignments; $19{,}728$ assignments satisfy $|\bar{d}_{\text{perm}}| \ge |\bar{d}_{\text{observed}}|$; $p = 19{,}728 / 1{,}048{,}576 = 0.0188140869 \dots \approx \mathbf{0.0188}$.
- **Interpretation**: The $p$-value is exact with respect to exhaustive enumeration of the sign assignments under the specified exchangeability-of-signs null.
- **Mandatory Caveat**: "Exact" refers to exhaustive enumeration under the stated null; it does not mean assumption-free inference. Exhaustive enumeration removes Monte Carlo approximation error from the sign-flip calculation; it does not make the inferential procedure assumption-free.

### Inferential Question 3 — Direction / Majority
- **Method**: Exact two-sided binomial sign test.
- **Question**: *Does DeltaCore win approximately half of the paired streams?*
- **Canonical Result**: DeltaCore wins: $4 / 20$, DeltaCore loses: $16 / 20$, $p = \mathbf{0.0118}$.
- **Interpretation**: Under the two-sided 50/50 sign null, the observed directional imbalance is statistically significant for this benchmark.

---

## 4. Multiplicity and Inferential Roles

**Multiplicity and inferential roles.** The sign test and paired sign-flip test are separately defined inferential procedures addressing different questions. The exact two-sided sign test evaluates directional imbalance under a 50/50 sign null, while the exact paired sign-flip randomization test evaluates the magnitude of the paired mean difference under the stated exchangeability-of-signs null. They are not independent replications of a single hypothesis test. Their p-values are therefore reported separately for their respective null hypotheses and are not interpreted as a combined family-wise-error-controlled omnibus result.

The paired percentile bootstrap serves a different purpose: it provides a resampling-based uncertainty interval for the observed mean paired difference. It is not treated as a third hypothesis test.

---

## 5. Scope of Inference

> **The evidence is benchmark-specific. It establishes that, on the specified 20-seed non-stationary matched-marginal benchmark with D=128 and the stated feature representation, DeltaCore Gated did not demonstrate an advantage over regularized Online Covariance, while the observed paired difference favored Online Covariance. The benchmark does not establish universal superiority of Online Covariance, universal inferiority of DeltaCore, or performance ordering under telemetry distributions, dimensions, drift processes, workloads, or deployment conditions not represented by the benchmark. The result supports terminating this specific DeltaCore research direction for RecoveryOS; it does not establish that adaptive telemetry detection as a broader problem has been solved.**

---

## 6. Practical vs. Statistical Significance

The statistical metrics must be interpreted alongside engineering costs rather than generalized into universal claims:
- **Observed Benchmark Effect**:
  - Mean paired AUROC difference: $\bar{d} = -0.0265$
  - Paired Cohen's $d = -0.58$
  - Exact two-sided sign test: $p = 0.0118$ (directional null)
  - Exact paired sign-flip test: $p = 0.0188$ (magnitude / randomization null)
  - 95% paired percentile bootstrap CI: $[-0.0478, -0.0072]$
  These metrics quantify the observed benchmark effect and should not be converted into a universal production-performance claim.
- **Engineering Complexity & Latency**:
  - DeltaCore requires an unconstrained square associative matrix ($M_t \in \mathbb{R}^{D \times D}$) consuming $4 D^2$ bytes ($64\text{ KB}$ at $D = 128$) and median scoring latency of $25.9\text{ µs}$ ($8.6\times$ higher latency relative to first-order centroids at $3.0\text{ µs}$).
  - Online Covariance also consumes $O(D^2)$ state ($64\text{ KB}$) with comparable matrix operations.
- **Engineering Tradeoff**: Within the evaluated design and benchmark, DeltaCore incurred greater state and latency cost without demonstrating a compensating performance advantage over regularized online covariance.

---

## 7. Canonical Scientific Conclusion

> **"Under the 20-seed non-stationary matched-marginal benchmark, the observed paired difference favored regularized Online Covariance. The exact two-sided sign test rejected its 50/50 directional null at p = 0.0118, and the exact paired sign-flip randomization test rejected its stated exchangeability-of-signs null at p = 0.0188."**
>
> **"These tests address different inferential questions and are reported separately; the two p-values are not treated as independent confirmations of a single omnibus hypothesis."**
>
> **"DeltaCore Gated did not demonstrate a performance advantage over regularized Online Covariance on the decisive benchmark."**

DeltaCore successfully demonstrated deterministic controller semantics, local Lyapunov contraction, and exact state persistence, but did not establish a performance advantage over classical regularized second-order covariance estimation on the targeted benchmark.

---

## 8. Decisive Benchmark Identity

```text
Benchmark implementation:
experiments/higher_order_regime_shift.py

Benchmark version:
v3.0.0

Streams:
20 independent paired streams

Seeds:
0 through 19

Feature dimension:
D = 128

Feature representation:
unary + pairs + triples + lag

Marginal-invariance constraint:
Categorical TVD <= 0.05 vs Phase C
```

**Git Provenance**:
- Benchmark implementation introduced: commit `deda3eaaef1ebdeb3171198bce2f50070e38f1df` (*"experiments: complete higher-order baseline and evaluation-correction gate"*).
- Exact statistical validation & research freeze: commit `0ac487bf5d85be17267bd97133a5bef55c2e4767` (*"release: close research freeze with exact statistical validation"*).

---

## 9. Historical Metric Isolation

The values 0.9859 and 0.9594 are Phase 16/17 balanced classification accuracy metrics and are not part of the telemetry regime-detection AUROC record.

---

## 10. Reproducibility

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

## 11. Repository Integrity

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

## 12. Closure Rule & RecoveryOS Decision

> **The RecoveryOS integration was rejected because DeltaCore did not demonstrate a sufficient empirical advantage over a simpler regularized online covariance baseline on the decisive benchmark, while incurring greater state complexity and latency. DeltaCore is therefore removed as an active RecoveryOS research dependency.**
>
> **Any future work on adaptive telemetry detection must be treated as a new independent research effort with a new hypothesis, benchmark design, and preregistered evaluation.**
