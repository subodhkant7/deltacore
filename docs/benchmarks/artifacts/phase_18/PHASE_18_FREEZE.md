# DeltaCore Phase 18 Final Freeze Record

This document cryptographically freezes all **Phase 18: Unseen Classification Regime Transfer & Falsification** artifacts, baselines, code implementations, test suites, and environment state.

---

## 1. Environment & Pre-Cleanup Commit

- **Pre-Cleanup Base Git Commit**: `bdd0efa279e901e26542bc4eb5b2dcca0bc8f86f`
- **Current Branch**: `main`
- **Git Remote**: `origin (https://github.com/subodhkant7/deltacore.git)`
- **Python Version**: `3.13.0 (v3.13.0:60403a5409f, Oct  7 2024, 00:37:40)`
- **PyTorch Version**: `2.6.0`
- **NumPy Version**: `2.5.3`
- **Package Version**: `0.1.0` (from `pyproject.toml`)
- **Pyright Status**: `0 errors, 0 warnings, 0 informations`
- **Ruff Linter Status**: `Clean (0 lint violations)`
- **Ruff Formatter Status**: `Clean (186 files formatted)`
- **Compileall Status**: `0 errors`
- **Test Suite Status**: `656 passed in 33.2s` (100% passing rate across all test suites)

---

## 2. Phase 18 Frozen Configuration Digest

- **Configuration File**: `docs/benchmarks/artifacts/phase_18/phase_18_config.json`
- **SHA-256 Digest**: `97da4ab2ea79a5c214b7a11f5cdb733f46b51ed91ceab150cf80730e848b9e5b`

### Frozen Tuple
$$
\eta_0 = 0.015, \quad \rho = 1.50, \quad \alpha_{\min} = 0.95, \quad \gamma = 0.10, \quad \epsilon = 10^{-6}
$$
- `seeds = [42, 43, 44, 45, 46]`
- `scaling_dims = [32, 64, 128, 256]`
- `steps_per_regime = 120`

---

## 3. Phase 18 Implementation File Hashes (SHA-256)

| Component / File | File Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **Phase 18 Models** | `deltacore/benchmarks/phase_18/models.py` | `5ebcdf1a95f82a79ce46b8f1c43746708cc7af4c5bee02a441c3b5d4450abd07` |
| **Multi-Family Generators** | `deltacore/benchmarks/phase_18/generators.py` | `498328ad23819e2b261feb3a775bf852a8c58e73ed5473d3c075b7d0aea3b3dd` |
| **Causal Streaming Protocol** | `deltacore/benchmarks/phase_18/protocol.py` | `272c4a0c3b51ec78c78bef3880e6acd9fb1176b11424f212ba2d11db516d5a3c` |
| **Controls & Shortcut Audits** | `deltacore/benchmarks/phase_18/controls.py` | `9c3266596eb6ae792178871b85d60e63b00ff13ad05c3f546ba0ae3738ad8875` |
| **Benchmark Runner & Serialization** | `deltacore/benchmarks/phase_18/run.py` | `e4a310f80ed9f7818db66092c5890e67fe89ed802e7d140c2642a1eb41a8a039` |
| **Hypothesis & Gate Analysis** | `deltacore/benchmarks/phase_18/analysis.py` | `d504e97aa9567b19a5666e8b7f6442b535dbba4670fb391ec62669f433ed7bf4` |
| **Phase 18 Observatory Plots** | `deltacore/observatory/phase_18_plots.py` | `44a659cef09a83249fbeaa3d31bb2a2cd2d0c9054c059ae5940558cde31033e9` |
| **Phase 18 Unit Tests** | `tests/test_phase_18_transfer.py` | `7a7e9d69d1c669a084e2862384b20a1482072d89eb581932b4fa56272d63ec23` |


---

## 4. Phase 18 Serialized Artifact Hashes (SHA-256)

| Artifact | File Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **Benchmark Configuration** | `docs/benchmarks/artifacts/phase_18/phase_18_config.json` | `97da4ab2ea79a5c214b7a11f5cdb733f46b51ed91ceab150cf80730e848b9e5b` |
| **Aggregate Results** | `docs/benchmarks/artifacts/phase_18/phase_18_results.json` | `b00f1b56182de415cbbfbc94d472355c8ca6ae8d8deba4c0dfa73a903cfd5a5e` |
| **Per-Seed Detailed Results** | `docs/benchmarks/artifacts/phase_18/phase_18_per_seed.json` | `a292fe9327dbf1e84019c8682ef774428f4119d98c89dcf7216a57f4ff1cad66` |
| **Shift Severity Results** | `docs/benchmarks/artifacts/phase_18/phase_18_shift_results.json` | `4645e98fc58689c8fa7c2e603c720d58cc52e155259d2fbbd5d26b2683f133ba` |
| **Retention & Mismatch Results** | `docs/benchmarks/artifacts/phase_18/phase_18_retention.json` | `24a39332ece0856b5dc6f44bf9d0217731eeec80835c452dd44b5a71ea452e7e` |
| **Dimensional Scaling Results** | `docs/benchmarks/artifacts/phase_18/phase_18_scaling.json` | `616ee96667f4f0cb4690e372ac54b21860cea1ec11ef86711eaeec8f5c713452` |
| **Cryptographic Parameter Hashes** | `docs/benchmarks/artifacts/phase_18/phase_18_hashes.json` | `6dccf64ed6c30640594d336dcf6f459d5633f0f0e70489901e6e07988bf9a36a` |
| **Shortcut & Negative Controls** | `docs/benchmarks/artifacts/phase_18/phase_18_controls.json` | `4c050c585674ff13b63626f4e5a2609c45b3933015ed9e0654ba2b00944a43f8` |
| **Failure Modes & Boundaries** | `docs/benchmarks/artifacts/phase_18/phase_18_failures.json` | `de209d9fa30fd497499624c739a9f2a00bb68a24a9afa7c7c7c94a17b9fc0825` |
| **Task Metadata** | `docs/benchmarks/artifacts/phase_18/phase_18_task_metadata.json` | `344bdb0ee2af96cb179714ccdf56b400618aca6824b0ea4c7733de2862be7b0e` |
| **Consistency Audit** | `docs/benchmarks/artifacts/phase_18/PHASE_18_CONSISTENCY_AUDIT.md` | `0b20e18b9030dd9a38cc356b109a83f46c6bc216189109cfedb439d830226fe2` |

---

## 5. Phase 18 Observatory Figures Hashes (SHA-256)

| Figure | Filename | SHA-256 Digest |
| :--- | :--- | :--- |
| **Plot 1** | `plot_18_task_transfer_overview.png` | `30995c892ff60e9a53f6b0f0621d5fb3524578a06b22ed54f29a295e38496fd3` |
| **Plot 2** | `plot_18_boundary_rotation.png` | `531c031a608d47ac525b5db0e2020554dc2b1cf0356b4cb087d0aecb1dda1f84` |
| **Plot 3** | `plot_18_boundary_translation.png` | `48398c7e5c9dad9192f2e65ffe57e57036714caba01441eb7a6a18dd20945d75` |
| **Plot 4** | `plot_18_nonlinear_boundary.png` | `f617c9452b2f9daf0d41dc9e1721c59a6d906a1e893c2260d9e7f0ddd873b672` |
| **Plot 5** | `plot_18_abrupt_vs_gradual.png` | `83c99db69159d2de29ab252a72aa5c464e564027a6c0be632732cf40c20e360a` |
| **Plot 6** | `plot_18_persistent_vs_reset.png` | `26d96a41761c1b0a13eb0eb67e6e2cc9a9f4a1b002e88d8366a8c909588a9b02` |
| **Plot 7** | `plot_18_state_on_off.png` | `5491f0913f1be4364c52aee4014f50e38a5471da87c42f3b8ff89dbc31688ed1` |
| **Plot 8** | `plot_18_recovery_curves.png` | `888b63861185cb825277bd3dce19e4166066a314119f96abb37094834e6e72e9` |
| **Plot 9** | `plot_18_cumulative_excess_loss.png` | `76f63fa8a14b08205a1c83381f9d1a3ebfabb0ac01ddcdfda4183cfffabe2cb1` |
| **Plot 10** | `plot_18_state_norm.png` | `c3bc0ccef30a0f6fa91ed0f0c0b8645aefe5a13c06b1535b839a653a64958f32` |
| **Plot 11** | `plot_18_adaptation_energy.png` | `ed5ff1ab955e5fcfd940ec6fdd2b43e2d2545649187a03160cb0d6b617908ab5` |
| **Plot 12** | `plot_18_safety_margin.png` | `a6b7e633849a57e3c960ddc55433714f2aea9b9ed00bbc7076c12739698d35c1` |
| **Plot 13** | `plot_18_scaling_accuracy.png` | `c0b7e17f3810b4ef7c29c60aa92959f91e1b6fbd7a3b6eee10db4c6dc11f17e8` |
| **Plot 14** | `plot_18_scaling_latency.png` | `24f3380ad6380d0f54e42f9414c03c77774bb224a0f651dcc5a2e45b06182f3a` |
| **Plot 15** | `plot_18_scaling_memory.png` | `7ad4c156c60eb7325cc189382f9166fa172d0c55d8c409c33fafc490561fdcd8` |
| **Plot 16** | `plot_18_label_shuffle.png` | `ad055cbcbd336cdb7a52a5a00effeba33c377856d483840917a87a61d9de27fb` |
| **Plot 17** | `plot_18_feature_permutation.png` | `0c987e060f2c2d956130e9b991534fa39719f3f42ee24d3aa98c09475bea85c6` |
| **Plot 18** | `plot_18_failure_cases.png` | `941ecacefcfb5e78b1c437e96207c427dc73c930ab894e4453617d53a9d927a5` |

---

## 6. Final Commit SHA

- **Pre-Cleanup Base Commit**: `bdd0efa279e901e26542bc4eb5b2dcca0bc8f86f`
- **Final Release Commit**: `374733bf99d4030955e81356d360cd45e58d648f`
