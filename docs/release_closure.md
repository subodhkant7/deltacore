# DeltaCore Open-Source Release & Research Freeze Closure Report

---

## 1. Final Status

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          DELTACORE STATUS                              │
├────────────────────────────────────────────────────────────────────────┤
│ Core Research Implementation          FROZEN                           │
│ Telemetry Novelty Detection           NOT RECOMMENDED                  │
│ RecoveryOS Integration                REJECTED                         │
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
| **Exact Paired Sign-Flip Permutation Test** | **$p = 0.0188$** | Evaluated all $2^{20} = 1{,}048{,}576$ sign configurations under exchangeability null ($19{,}728 / 1{,}048{,}576$) |
| **Paired Effect Size (Cohen's $d$)** | **-0.58** | Medium negative effect size favoring Online Covariance |
| **Online Centroid Gated Mean AUROC** | **0.5330 $\pm$ 0.0813** | First-order baseline; DeltaCore paired difference $+0.0382$ ($p = 0.0004$) |
| **Online PCA Gated Mean AUROC** | **0.5641 $\pm$ 0.0903** | Low-rank projection baseline; DeltaCore paired difference $+0.0070$ ($p = 0.8238$, CI $[-0.0074, 0.0209]$) |

---

## 3. Disambiguation of Statistical Methods

The three inferential procedures answer fundamentally distinct questions and must never be conflated:

1. **Exact Two-Sided Binomial Sign Test** ($p = 0.0118$):
   - **Question**: *Does DeltaCore win on approximately half of the independent evaluation streams?*
   - **Data**: Only the *direction* (sign) of each paired difference ($d_i > 0$ vs. $d_i < 0$).
   - **Null**: $H_0: p = 0.5$. With $k = 4$ wins out of $n = 20$, the exact two-sided binomial $p$-value is $0.011818 \dots$, rejecting the null of equal win frequency.
2. **Exact Paired Sign-Flip Permutation Test** ($p = 0.0188$):
   - **Question**: *Is the observed mean paired difference statistic ($\bar{d} = -0.0265$) consistent with symmetric exchangeability of labels under the null?*
   - **Data**: The *magnitudes* of paired differences, conditioning on observed $|d_i|$.
   - **Null**: Under $H_0: \mathbb{E}[d_i] = 0$, each pair's sign is randomly $\pm 1$. Evaluated by exact enumeration of all $2^{20} = 1{,}048{,}576$ configurations without Monte Carlo approximation ($19{,}728$ assignments have $|\bar{d}| \ge 0.0265$).
3. **Paired Bootstrap Confidence Interval** ($[-0.0478, -0.0072]$):
   - **Question**: *What is the sampling uncertainty surrounding the mean paired difference estimate?*
   - **Data**: Paired stream units $(d_i)$ resampled with replacement ($B = 10{,}000$, seed 42).
   - **Interpretation**: The 95% percentile interval strictly excludes zero, indicating that the true mean difference under this benchmark distribution is negative.

---

## 4. Scientific Conclusion

> **Under non-stationary distribution drift with matched marginals ($\text{TVD} \le 0.05$), DeltaCore's auto-associative controller did not demonstrate a sufficient discriminative or operational advantage over regularized online covariance estimation to justify its additional algorithmic complexity, $O(D^2)$ state memory, and higher latency for telemetry regime detection.**

DeltaCore succeeded in establishing deterministic controller semantics, local Lyapunov contraction, and exact state persistence, but failed to establish operational superiority over classical second-order statistics.

---

## 5. Reproducibility

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

## 6. Repository Integrity

| Check | Tool / Command | Result |
| :--- | :--- | :--- |
| **Unit Tests** | `pytest -q` | **689 passed in ~30s** |
| **Static Types** | `pyright` | **0 errors, 0 warnings, 0 informations** |
| **Linter** | `ruff check .` | **All checks passed!** |
| **Formatter** | `ruff format --check .` | **198 files already formatted** |
| **Compilation** | `python3 -m compileall` | **Clean, 0 errors** |
| **Smoke Example** | `python examples/basic_adaptation.py` | **Passed cleanly** (checksum / persistence verified) |
| **Version Alignment** | `pyproject.toml`, `deltacore/__init__.py` | **0.2.0** |
| **Git Status** | `git status` | **Clean working tree** |

---

## 7. Closure Rule

> **DeltaCore is no longer an active RecoveryOS research dependency. Any future work on adaptive telemetry detection must be treated as a new independent research effort with a new hypothesis, benchmark design, and preregistered evaluation.**
