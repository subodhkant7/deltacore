# Phase 8: Paper Alignment & Provenance Specification

This document establishes the formal alignment, classification, and mathematical provenance between published literature—specifically **VisionHOPE** (Siran Peng, Tianshuo Zhang, Tianyu Fu, Weisong Zhao, Haoyuan Zhang, Jiankuo Zhao, Minghui Wu, Ping Jiang, Xiangyu Zhu, Chenxu Zhao, Zhen Lei, September 2026; [arXiv:2609.33325](https://arxiv.org/abs/2609.33325)) and **Nested Learning**—and the **DeltaCore Phase 8** five-memory reference core.

---

## 1. Provenance Classification Schema

Every mathematical equation, architectural concept, and state invariant in DeltaCore Phase 8 is explicitly cataloged under one or more of four provenance sources:

* **[A] DeltaCore Original Formulation**: Mathematical derivations, closed-form contraction bounds, dataclass schemas, and functional contracts formulated originally within DeltaCore.
* **[B] VisionHOPE Paper**: Theoretical concepts, architectural roles, and claims published in Peng et al. (September 2026).
* **[C] Nested Learning Source**: Foundational formulations of multi-memory associative systems and deep optimizers as associative memories.
* **[D] Engineering Implementation Choice**: Design decisions chosen for clean PyTorch CPU reference execution, autograd differentiability, and testability.

---

## 2. Paper Alignment & Classification Matrix

In accordance with Phase 8.1 audit rules, correspondence is classified using strictly defined labels:
`EXACT`, `CONCEPTUALLY ALIGNED`, `GENERALIZED`, `ENGINEERING DIFFERENCE`, and `NOT IMPLEMENTED`. Blanket "Yes" indicators are prohibited.

| Mechanism | VisionHOPE (Peng et al., 2026) | DeltaCore Phase 8 Reference Core | Classification | Provenance |
| :--- | :--- | :--- | :--- | :--- |
| **Five-Memory Co-Evolution** | Five coupled memories ($M^m, M^k, M^v, m^\eta, m^\alpha$) co-evolve where stored content and learning dynamics adapt together. | Five coupled states ($M_{\text{content}}, M_{\text{key}}, M_{\text{val}}, M_\eta, M_{\text{ret}}$) co-evolve via `FiveMemoryState` and `FiveMemorySystem`. | `CONCEPTUALLY ALIGNED` | [B, C] |
| **State Dimension Parameterization** | Homogeneous dimension $d$: $M^m, M^k, M^v \in \mathbb{R}^{d \times d}$ and $m^\eta, m^\alpha \in \mathbb{R}^{1 \times d}$. | Decoupled dimensions: $M_{\text{content}} \in \mathbb{R}^{V \times K}$, $M_{\text{key}} \in \mathbb{R}^{K \times D_{\text{in}}}$, $M_{\text{val}} \in \mathbb{R}^{V \times D_{\text{in}}}$, $M_\eta, M_{\text{ret}} \in \mathbb{R}^{D_{\text{meta}} \times D_{\text{feat}}}$. | `GENERALIZED` | [A, D] |
| **Key & Value Prior-State Timing** | Generated from previous-step state: $k_t = M_{t-1}^k x_t$, $v_t = M_{t-1}^v x_t$. | Generated from prior state passed to `step()`: $k_t = M_{\text{key}} x_t$, $v_t = M_{\text{val}} x_t$. | `EXACT` | [B] |
| **Key & Value Generation Projections** | Linear maps from token $x_t \in \mathbb{R}^d$ to key/value space $\mathbb{R}^d$. | Linear maps from token $x_t \in \mathbb{R}^{D_{\text{in}}}$ to key space $\mathbb{R}^K$ and value space $\mathbb{R}^V$. | `GENERALIZED` | [A, B] |
| **Key Normalization** | Stabilized $L_2$ normalization: $k_t = k_t^{\mathrm{raw}} / \sqrt{\|k_t^{\mathrm{raw}}\|_2^2 + 10^{-6}}$ ensuring $\|k_t\| \le 1$. | Raw linear key output without normalization in base `FiveMemorySystem`. | `ENGINEERING DIFFERENCE` | [A, D] |
| **Learning-Rate & Retention Prior-State Timing** | Generated from previous-step state: $\eta_t = \phi_\eta(m_{t-1}^\eta x_t)$, $\alpha_t = \phi_\alpha(m_{t-1}^\alpha x_t)$. | Generated from prior state: $u_\eta = M_\eta z_t$, $u_{\text{ret}} = M_{\text{ret}} z_t$. | `EXACT` | [B] |
| **Learning-Rate Generation Function ($\phi_\eta$)** | Softplus activation: $\eta_t = \gamma_\eta \operatorname{softplus}(s_t^\eta)$ with scale $\gamma_\eta = 0.025$. | Scaled sigmoid squashing: $\eta_t^{\mathrm{raw}} = \eta_{\max} \sigma(\dots)$. | `ENGINEERING DIFFERENCE` | [A, D] |
| **Retention Generation Function ($\phi_\alpha$)** | Shifted sigmoid: $\alpha_t = \sigma(s_t^\alpha + \log \frac{\alpha_{\mathrm{init}}}{1 - \alpha_{\mathrm{init}}})$ with $\alpha_{\mathrm{init}} = 0.9$. | Scaled sigmoid: $\lambda_t^{\mathrm{raw}} = \lambda_{\min} + (1 - \lambda_{\min}) \sigma(\dots)$ with $\lambda_t^{\mathrm{safe}} \in [\lambda_{\min}, 1]$. | `CONCEPTUALLY ALIGNED` | [A, B] |
| **Self-Generated Targets ($\widehat{v}_t^\square$)** | Each memory generates its own regression target: $\widehat{v}_t^\square = M_{t-1}^\square v_t$ for all $\square \in \{m, k, v, \eta, \alpha\}$. | Content reads key ($\hat{v}_t = M_{\text{content}} k_t$); auxiliary memories adapt via error gradient or error shock feedback. | `ENGINEERING DIFFERENCE` | [A, D] |
| **Learning Signals ($g_t^\square$)** | Unified discrepancy signal: $g_t^\square = M_{t-1}^\square(k_t - v_t) = M_{t-1}^\square \delta_t$. | Specialized error signals: $e_t$ (content), $M_{\text{content}}^\top e_t$ (key), $y_t - v_t$ (value), tanh error feedback (LR, ret). | `ENGINEERING DIFFERENCE` | [A, D] |
| **Content Memory Update Rule** | Retained DGD: $M_t^m = M_{t-1}^m (\alpha_t I - \eta_t k_t k_t^\top) - \eta_t g_t^m k_t^\top$. | Affine delta rule: $M_{t+1}^c = \lambda_t M_t^c (\dots) + \eta_t v_t k_t^\top$. | `CONCEPTUALLY ALIGNED` | [A, B, C] |
| **Auxiliary Memory Update Rules** | All five memories share identical SR-DGD recurrence: $M_t^\square = M_{t-1}^\square (\alpha_t I - \eta_t k_t k_t^\top - \eta_t \delta_t k_t^\top)$. | Modular updates with independent learning rates and retention decays per memory state. | `ENGINEERING DIFFERENCE` | [A, D] |
| **Stability Controller** | Two-stage soft injection cap + spectral clamp: $r_t = \sqrt{\|\delta_t\|^2 + \epsilon^2} + \epsilon$, $\bar\eta_t = \eta_t^{\mathrm{inj}}(1 - e^{-\eta_t/\eta_t^{\mathrm{inj}}})$, $\eta_t^{\mathrm{spec}} = \frac{2\alpha_t}{\|k_t\|^2}$, $\widetilde\eta_t = \min(\bar\eta_t, \eta_t^{\mathrm{spec}})$. | Phase 4 rank-1 contraction clamp: $\eta_t^{\mathrm{safe}} = \min(\eta_t^{\mathrm{raw}}, \frac{\beta}{\|k_t\|^2 + \epsilon})$ with fixed $\beta \in (0, 2)$. | `ENGINEERING DIFFERENCE` | [A, D] |
| **Operator Norm Bounds** | Guaranteed token-wise bounds: $\|A_t\|_2 \le \alpha_t$ and $\|B_t\|_2 < 1 - \alpha_t \implies \|T_t\|_2 < 1$. | Local rank-1 contraction on homogeneous operator $A_t$ when $\lambda_t = 1.0$; when $\lambda_t < 1$, $\|A_t\|_2$ can exceed $1$. | `ENGINEERING DIFFERENCE` | [A, B] |
| **Shared Right-Multiplicative Representation** | $M_t^\square = M_b^\square G_t$ with shared gain $G_t = G_{t-1} A_t + B_t$. | Not shared across memories; states update independently. | `NOT IMPLEMENTED` | [A] |
| **Parallel Chunking** | 2D row/column chunked associative scans. | Sequential unrolling in reference implementation. | `NOT IMPLEMENTED` (Deferred) | [A] |
| **2D Spatial Scans** | Four directional scans ($\rightarrow, \leftarrow, \downarrow, \uparrow$). | 1D sequential token sequences $[B, T, D]$. | `NOT IMPLEMENTED` (Deferred to Phase 9) | [A] |

---

## 3. Explicit Epistemic Boundaries

In accordance with AGENTS.md Directives 2, 7, 8, 9, and 10:
1. **Generalized Parameterization, Not Exact Reproduction**: DeltaCore Phase 8 is a generalized mathematical reference framework for self-modifying neural state. It is not an exact reproduction of VisionHOPE's specific vision backbone.
2. **Distinct Update Algebra**: VisionHOPE uses a shared right-multiplicative transition $M_t^\square = M_{t-1}^\square T_t$ driven by the key-value discrepancy $\delta_t = k_t - v_t$. DeltaCore uses decoupled associative mappings, back-projected gradients for key memory, and error-shock feedback for meta-controllers.
3. **Distinct Stability Mechanisms**: DeltaCore enforces local rank-1 contraction bounds ($\eta_t \le \beta / \|k_t\|^2$) derived from Phase 4; VisionHOPE enforces a two-stage soft injection cap and spectral clamp ensuring $\|T_t\|_2 < 1$.
4. **No Unproven Theoretical Claims**: No joint Lyapunov stability proof is asserted for DeltaCore's five-memory recurrence.
