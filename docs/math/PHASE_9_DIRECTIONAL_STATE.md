# Phase 9: Directional State Independence

## 1. Mathematical Formulation

Let $Z \in \mathbb{R}^{B \times C \times H \times W}$ denote a 2D spatial feature map of batch size $B$, channels $C$, height $H$, and width $W$.
We consider the four canonical spatial routes:
$$\mathcal{R} = \{\text{RIGHT}, \text{LEFT}, \text{DOWN}, \text{UP}\}$$

For each route $r \in \mathcal{R}$, there exists an exact serialization operator:
$$\mathcal{P}_r: \mathbb{R}^{B \times C \times H \times W} \to \mathbb{R}^{B \times N \times C}, \quad N = HW$$
and an exact spatial restoration operator:
$$\mathcal{P}_r^{-1}: \mathbb{R}^{B \times N \times C} \to \mathbb{R}^{B \times C \times H \times W}$$
such that $\mathcal{P}_r^{-1}(\mathcal{P}_r(Z)) = Z$ identically for any rectangular dimension pair $(H, W)$.

## 2. Directional State Isolation Architecture

In DeltaCore's Phase 8 five-memory reference learner, the adaptive state at token step $t$ is encapsulated by the tuple:
$$S_t = \left(M_t^k, M_t^v, m_t^\eta, m_t^\alpha, W_t\right)$$
where:
- $M_t^k \in \mathbb{R}^{B \times D \times D_k}$: Key generation memory
- $M_t^v \in \mathbb{R}^{B \times D \times D_v}$: Value generation memory
- $m_t^\eta \in \mathbb{R}^{B \times D}$: Learning rate modulation vector
- $m_t^\alpha \in \mathbb{R}^{B \times D}$: Retention modulation vector
- $W_t \in \mathbb{R}^{B \times D_k \times D_v}$: Associative working memory

### The Directional Independence Axiom
Under no circumstances may the evolving memory state of route $r$ contaminate, influence, or couple with the memory state of route $r' \neq r$ during sequence traversal.
Each directional trajectory evolves exclusively within its own isolated state manifold:

```
                          Feature Map Z ∈ R^{B×C×H×W}
                                      │
       ┌──────────────────┬───────────┴───────────┬──────────────────┐
       ▼                  ▼                       ▼                  ▼
  P_RIGHT(Z)          P_LEFT(Z)               P_DOWN(Z)           P_UP(Z)
       │                  │                       │                  │
       ▼                  ▼                       ▼                  ▼
FiveMemory_RIGHT    FiveMemory_LEFT         FiveMemory_DOWN     FiveMemory_UP
State: S_{t}^{R}    State: S_{t}^{L}        State: S_{t}^{D}    State: S_{t}^{U}
       │                  │                       │                  │
       ▼                  ▼                       ▼                  ▼
     Y_{seq}^R          Y_{seq}^L               Y_{seq}^D          Y_{seq}^U
       │                  │                       │                  │
       ▼                  ▼                       ▼                  ▼
 P_RIGHT^{-1}(·)    P_LEFT^{-1}(·)          P_DOWN^{-1}(·)      P_UP^{-1}(·)
       │                  │                       │                  │
       └──────────────────┼───────────────────────┴──────────────────┘
                          ▼
                  Spatial Fusion: Y = Σ_r λ_r ⊙ Y^r
```

Formally:
$$\frac{\partial S_t^r}{\partial S_{t'}^{r'}} = 0, \quad \forall r \neq r', \quad \forall t, t' \in \{0, \dots, N\}$$

## 3. Directional State Trajectories

For route $r \in \{\text{RIGHT}, \text{LEFT}, \text{DOWN}, \text{UP}\}$, let $x_{t}^r = [\mathcal{P}_r(Z)]_{:, t, :}$. The recurrence is:
$$\begin{aligned}
k_t^r &= M_{t}^{k, r} x_t^r \\
v_t^r &= M_{t}^{v, r} x_t^r \\
\eta_t^r &= \phi_\eta(m_{t}^{\eta, r} x_t^r) \\
\alpha_t^r &= \phi_\alpha(m_{t}^{\alpha, r} x_t^r) \\
\hat{v}_t^r &= W_{t}^r k_t^r \\
e_t^r &= v_t^r - \hat{v}_t^r \\
W_{t+1}^r &= \alpha_t^r W_t^r + \eta_t^r e_t^r (k_t^r)^\top \\
S_{t+1}^r &= \mathcal{U}^r(S_t^r, x_t^r)
\end{aligned}$$

Each direction $r$ maintains independent initial states $S_0^r$, independent projections, and independent update parameters. This directly mirrors the formulation in VisionHOPE, where each directional branch instantiates an independent Self-Referential Nested Learner (SRNL).

## 4. Preservation of Mathematical Properties
1. **No Shared Global Buffers**: In accordance with DeltaCore Rule 5 (Avoid Hidden Global State), directional execution instantiates distinct state containers with zero shared mutable references.
2. **Determinism**: For fixed inputs $Z$ and fixed initialization parameters, each directional output $\mathcal{P}_r^{-1}(Y_{\text{seq}}^r)$ is strictly deterministic and independently reproducible.
3. **Observatory Decomposability**: Because states are partitioned by direction, the Adaptive State Observatory can record and plot per-direction energy, gradient norms, and retention trajectories without information loss.
