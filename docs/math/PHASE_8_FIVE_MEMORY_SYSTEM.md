# Phase 8: Five-Memory Self-Modifying System Specification

This document provides the formal mathematical specification for the five coupled memory states comprising DeltaCore's Phase 8 reference core:
$$M_{\text{content}}, \quad M_{\text{key}}, \quad M_{\text{val}}, \quad M_{\eta}, \quad M_{\text{ret}}$$

---

## 1. Dimensional Architecture & Notation

Let $B$ denote batch size, $T$ sequence length, and $D_{\text{in}}$ token feature dimension.
The system does not assume all five memory states have identical dimensions:

| Memory State | Tensor Notation | Shape (Unbatched) | Shape (Batched) | Semantic Role |
| :--- | :--- | :--- | :--- | :--- |
| **Content Memory** | $M_{\text{content}}$ | $[V, K]$ | $[B, V, K]$ | Primary associative storage mapping key representations to target value predictions. |
| **Key Generation Memory** | $M_{\text{key}}$ | $[K, D_{\text{in}}]$ | $[B, K, D_{\text{in}}]$ | Input-to-key encoder generating memory addressing representations. |
| **Value Generation Memory** | $M_{\text{val}}$ | $[V, D_{\text{in}}]$ | $[B, V, D_{\text{in}}]$ | Input-to-value encoder generating target feature representations. |
| **Learning-Rate Memory** | $M_{\eta}$ | $[D_{\text{lr}}, D_{\text{feat}}]$ | $[B, D_{\text{lr}}, D_{\text{feat}}]$ | Self-referential state generating adaptive step-size scalars $\eta_t$. |
| **Retention Memory** | $M_{\text{ret}}$ | $[D_{\text{ret}}, D_{\text{feat}}]$ | $[B, D_{\text{ret}}, D_{\text{feat}}]$ | Self-referential state generating forgetting/retention factors $\lambda_t$. |

Here:
- $D_{\text{in}}$: Input token dimension.
- $K$: Key representation dimension.
- $V$: Value representation dimension.
- $D_{\text{feat}}$: Feedback/interaction feature dimension driving meta-controllers (default: $D_{\text{in}}$ or concatenated feature size).
- $D_{\text{lr}}$: Hidden dimension for learning rate projection (default: $1$ or $D_k$).
- $D_{\text{ret}}$: Hidden dimension for retention projection (default: $1$ or $D_k$).

---

## 2. State-by-State Specification

### 2.1. Content Memory ($M_{\text{content}}$)
* **Shape**: $[V, K]$ (unbatched) or $[B, V, K]$ (batched).
* **Meaning**: Stores learned associations between key queries and value targets.
* **Read Operation**:
  $$\hat{v}_t = M_{\text{content}, t} k_t \in \mathbb{R}^V$$
* **Update Target**: $M_{\text{content}, t+1} \in \mathbb{R}^{V \times K}$.
* **Update Rule**:
  $$M_{\text{content}, t+1} = \lambda_t^{\text{safe}} M_{\text{content}, t} + \eta_t^{\text{safe}} e_t k_t^\top$$
  where $e_t = v_t - \hat{v}_t$ is the prediction error vector.
* **Consumer**: Downstream output projection and error evaluation.
* **Directly Contributes to Output**: **Yes** ($\hat{v}_t$ is the primary output prediction).
* **Affects Future Updates**: **Yes** (determines prediction error $e_t$, which drives updates across all five states).

### 2.2. Key Generation Memory ($M_{\text{key}}$)
* **Shape**: $[K, D_{\text{in}}]$ (unbatched) or $[B, K, D_{\text{in}}]$ (batched).
* **Meaning**: Transforms raw input tokens into structured key vectors for associative content lookup.
* **Read Operation**:
  $$k_t = M_{\text{key}, t} x_t \in \mathbb{R}^K$$
* **Update Target**: $M_{\text{key}, t+1} \in \mathbb{R}^{K \times D_{\text{in}}}$.
* **Update Rule**:
  $$M_{\text{key}, t+1} = \lambda_{\text{key}} M_{\text{key}, t} + \eta_{\text{key}} (M_{\text{content}, t}^\top e_t) x_t^\top$$
  The gradient of content prediction loss with respect to key $k_t$ is $-M_{\text{content}}^\top e_t$. Key memory adapts in the error-reducing direction.
* **Consumer**: Content memory read ($\hat{v}_t = M_{\text{content}} k_t$) and content memory outer product update ($e_t k_t^\top$).
* **Directly Contributes to Output**: **No** (acts as intermediate addressing mechanism).
* **Affects Future Updates**: **Yes** (modulates future keys $k_{t+1}$ and key energy $\|k_t\|^2$).

### 2.3. Value Generation Memory ($M_{\text{val}}$)
* **Shape**: $[V, D_{\text{in}}]$ (unbatched) or $[B, V, D_{\text{in}}]$ (batched).
* **Meaning**: Transforms raw input tokens into target value representations for associative content binding.
* **Read Operation**:
  $$v_t = M_{\text{val}, t} x_t \in \mathbb{R}^V$$
  *(When external supervised targets $y_t$ are provided during sequence association, $y_t$ replaces or guides $v_t$)*.
* **Update Target**: $M_{\text{val}, t+1} \in \mathbb{R}^{V \times D_{\text{in}}}$.
* **Update Rule**:
  $$M_{\text{val}, t+1} = \lambda_{\text{val}} M_{\text{val}, t} + \eta_{\text{val}} e_{v, t} x_t^\top$$
  where $e_{v, t} = y_t - v_t$ if target is present, or self-supervised token displacement.
* **Consumer**: Prediction error calculation $e_t = v_t - \hat{v}_t$.
* **Directly Contributes to Output**: **No** (serves as target reference).
* **Affects Future Updates**: **Yes** (directly sets error signal $e_t$).

### 2.4. Learning-Rate Memory ($M_{\eta}$)
* **Shape**: $[D_{\text{lr}}, D_{\text{feat}}]$ (unbatched) or $[B, D_{\text{lr}}, D_{\text{feat}}]$ (batched).
* **Meaning**: An evolving self-referential controller memory that dynamically generates the scalar learning rate $\eta_t$.
* **Read Operation**:
  $$u_{\eta, t} = M_{\eta, t} z_t \in \mathbb{R}^{D_{\text{lr}}}$$
  $$\eta_t^{\text{raw}} = \eta_{\max} \cdot \sigma\left(\frac{1}{\sqrt{D_{\text{lr}}}} \sum_{i=1}^{D_{\text{lr}}} u_{\eta, t}[i] + b_\eta\right)$$
* **Update Target**: $M_{\eta, t+1} \in \mathbb{R}^{D_{\text{lr}} \times D_{\text{feat}}}$.
* **Update Rule**:
  $$M_{\eta, t+1} = \lambda_\eta M_{\eta, t} + \rho_\eta e_{\eta, t} z_t^\top$$
  where $e_{\eta, t} = \tanh\left(\frac{\|e_t\|_2 - \tau_\eta}{\tau_\eta + \epsilon}\right) \mathbf{1}_{D_{\text{lr}}}$ drives acceleration under high error and decrescendo under convergence.
* **Consumer**: Stability controller and content memory update rule.
* **Directly Contributes to Output**: **No**.
* **Affects Future Updates**: **Yes** (directly modulates effective step size $\eta_{t+1}^{\text{safe}}$ for future content updates).

### 2.5. Retention Memory ($M_{\text{ret}}$)
* **Shape**: $[D_{\text{ret}}, D_{\text{feat}}]$ (unbatched) or $[B, D_{\text{ret}}, D_{\text{feat}}]$ (batched).
* **Meaning**: An evolving self-referential controller memory that dynamically generates the memory retention decay factor $\lambda_t$.
* **Read Operation**:
  $$u_{\text{ret}, t} = M_{\text{ret}, t} z_t \in \mathbb{R}^{D_{\text{ret}}}$$
  $$\lambda_t^{\text{raw}} = \lambda_{\min} + (1 - \lambda_{\min}) \sigma\left(\frac{1}{\sqrt{D_{\text{ret}}}} \sum_{i=1}^{D_{\text{ret}}} u_{\text{ret}, t}[i] + b_{\text{ret}}\right)$$
* **Update Target**: $M_{\text{ret}, t+1} \in \mathbb{R}^{D_{\text{ret}} \times D_{\text{feat}}}$.
* **Update Rule**:
  $$M_{\text{ret}, t+1} = \lambda_{\text{ret}} M_{\text{ret}, t} + \rho_{\text{ret}} e_{\text{ret}, t} z_t^\top$$
  where $e_{\text{ret}, t} = -\tanh\left(\frac{\|e_t\|_2 - \tau_{\text{ret}}}{\tau_{\text{ret}} + \epsilon}\right) \mathbf{1}_{D_{\text{ret}}}$ reduces retention during unexpected shocks (flushing obsolete history) while restoring near-unity retention during stable tracking.
* **Consumer**: Content memory transition equation.
* **Directly Contributes to Output**: **No**.
* **Affects Future Updates**: **Yes** (controls memory persistence $\lambda_{t+1}^{\text{safe}}$ for all future states).

---

## 3. Co-Evolutionary Transition Graph

```
                   Input Token x_t
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
     Key Memory      Value Memory   Content Memory
    M_key, t          M_val, t      M_content, t
         │                │                │
         ▼                ▼                │
     Key k_t          Value v_t            │
         │                │                │
         ├────────────────┼────────► Read: v_hat_t = M k_t
         │                │                │
         │                ▼                ▼
         │             Error e_t = v_t - v_hat_t
         │                │
         │         ┌──────┴──────┐
         ▼         ▼             ▼
   Query z_t ──► M_eta, t     M_ret, t
                   │             │
                   ▼             ▼
                eta_raw       lambda_raw
                   │             │
                   ▼             ▼
               Stability     Spectral
              Controller       Clamp
                   │             │
                   ▼             ▼
                eta_safe     lambda_safe
                   │             │
                   └──────┬──────┘
                          │
                          ▼
            Co-Evolving Memory Updates:
              Delta M_content
              Delta M_key
              Delta M_val
              Delta M_eta
              Delta M_ret
                          │
                          ▼
              Five Next States (t+1)
```
