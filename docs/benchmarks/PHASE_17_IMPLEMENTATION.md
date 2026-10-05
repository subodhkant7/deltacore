# Phase 17 Implementation: Online Non-Stationary Classification

This document records the exact mathematical formulation, system architecture, offline/online separation protocols, and algorithmic updates implemented for **Phase 17: Online Non-Stationary Classification**.

---

## 1. Scientific Mission & Task Formulation

Phase 17 investigates the central scientific question:

$$
\boxed{
\text{Can adaptive associative state improve classification under changing data distributions without test-time parameter learning?}
}
$$

The benchmark is deliberately adversarial:
- **No vision backbones**, transformers, multi-head attention, large convolutional networks, or deep recurrent architectures.
- **Strict parameter immutability**: During online test streaming, all model parameters remain strictly frozen:
  $$\Delta\theta = 0$$
  This condition is verified via pre- and post-stream SHA-256 cryptographic hashes over parameter tensors.
- **Strict online sequencing (no label lookahead)**: For each time step $t$, the model receives input $x_t \in \mathbb{R}^D$ and must emit its prediction $\hat{y}_t$ using only the historical state $S_{t-1}$:
  $$\hat{y}_t = f(x_t, S_{t-1})$$
  Only after $\hat{y}_t$ and class probabilities $\hat{p}_t$ are recorded is the true integer label $y_t \in \{0, \dots, K-1\}$ revealed to the model's adaptation step:
  $$S_t = \text{Adapt}(S_{t-1}, x_t, y_t)$$

---

## 2. Mathematical Formulation & Architecture

### 2.1 DeltaCore Classification Architecture

The DeltaCore classification model consists of two cleanly separated modules:
1. **Offline Linear Head** $(\theta = \{W_{\mathrm{head}}, b_{\mathrm{head}}\})$:
   - Fixed weight matrix $W_{\mathrm{head}} \in \mathbb{R}^{K \times D}$ and bias $b_{\mathrm{head}} \in \mathbb{R}^K$ fit exclusively on stationary training data from Regime A.
   - Strictly frozen during online deployment ($\Delta\theta = 0$).
2. **Online Associative State** $(S_t = M_t \in \mathbb{R}^{D \times D})$:
   - Initialized to zero: $M_0 = \mathbf{0}_{D \times D}$.
   - Dynamically adapts feature representations via associative outer-product updates.

### 2.2 Prediction Step (Prior to Label Revelation)

Given $x_t \in \mathbb{R}^D$:
1. **Associative Feature Projection**:
   $$z_t = x_t + M_t x_t = (I + M_t) x_t$$
2. **Logit Evaluation & Softmax**:
   $$s_t = W_{\mathrm{head}} z_t + b_{\mathrm{head}} \in \mathbb{R}^K$$
   $$\hat{p}_{t, k} = \frac{\exp(s_{t, k})}{\sum_{j=1}^K \exp(s_{t, j})}$$
3. **Hard Class Decision**:
   $$\hat{y}_t = \arg\max_{k \in \{0, \dots, K-1\}} s_{t, k}$$

### 2.3 Online Adaptive State Update (After Label Revelation)

Upon revelation of $y_t \in \{0, \dots, K-1\}$:
1. **One-Hot Target & Logit Error**:
   $$e_{\mathrm{logit}, t} = y_{\mathrm{one\_hot}, t} - \hat{p}_t \in \mathbb{R}^K$$
2. **Feature-Space Backprojected Error**:
   $$e_{x, t} = W_{\mathrm{head}}^\top e_{\mathrm{logit}, t} \in \mathbb{R}^D$$
3. **Candidate Learning Rate with Error Normalization**:
   $$\eta_{\mathrm{cand}, t} = \frac{\eta_0}{1 + \gamma \|e_{x, t}\|_2}$$
4. **Contraction Bound & Effective Step Size**:
   $$\eta_t = \min\left(\eta_{\mathrm{cand}, t}, \frac{\rho}{\|x_t\|_2^2 + \epsilon}\right)$$
   where $\rho = 1.50$ enforces the spectral non-expansion condition $\eta_t \|x_t\|_2^2 \le \rho$.
5. **Safe Adaptive Retention**:
   $$\alpha_t = \max\left(\alpha_{\min}, 1 - \eta_t \|x_t\|_2^2\right)$$
   where $\alpha_{\min} = 0.95$.
6. **Associative Delta State Transition**:
   $$M_{t+1} = \alpha_t M_t + \eta_t e_{x, t} x_t^\top$$

---

## 3. Evaluated Model Matrix & Capacity Matching

The model matrix spans 10 distinct configurations across 4 paradigm families, matched approximately at $\sim 200$, $\sim 600$, and $\sim 2,500$ parameters ($D=32, K=6$):

| Paradigm | Model Class | Trainable Params $\theta$ | Test $\Delta\theta$ | Persistent State | Memory (Bytes) | Mean Latency ($\mu s$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical** | [OnlineLogisticRegression](file:///Users/urjasoft/Documents/DeltaCore/deltacore/classification/models.py#L167) | 198 | Online SGD | $W_t, b_t$ | 792 | 4.51 |
| **Classical** | [OnlineRidgeClassifier](file:///Users/urjasoft/Documents/DeltaCore/deltacore/classification/models.py#L211) | 192 | RLS Update | $P_t, W_t$ | 4,864 | 4.04 |
| **Classical** | [OnlineMulticlassLinear](file:///Users/urjasoft/Documents/DeltaCore/deltacore/classification/models.py#L267) | 192 | Online Step | $W_t, b_t$ | 768 | 3.27 |
| **Frozen Neural** | [FrozenLinearClassifier](file:///Users/urjasoft/Documents/DeltaCore/deltacore/classification/models.py#L90) | 198 | $0$ (Frozen) | None | 0 | 6.42 |
| **Frozen Neural** | [SmallMLPClassifier](file:///Users/urjasoft/Documents/DeltaCore/deltacore/classification/models.py#L125) | 630 | $0$ (Frozen) | None | 0 | 15.50 |
| **Recurrent** | [GRUClassifier](file:///Users/urjasoft/Documents/DeltaCore/deltacore/classification/models.py#L297) | 2,502 | $0$ (Frozen) | $h_t \in \mathbb{R}^{16}$ | 64 | 23.55 |
| **Recurrent** | [LSTMClassifier](file:///Users/urjasoft/Documents/DeltaCore/deltacore/classification/models.py#L340) | 3,302 | $0$ (Frozen) | $(h_t, c_t) \in \mathbb{R}^{32}$ | 128 | 23.76 |
| **DeltaCore** | [FixedDeltaClassifier](file:///Users/urjasoft/Documents/DeltaCore/deltacore/classification/models.py#L487) | 198 | $0$ (Frozen) | $M_t \in \mathbb{R}^{D \times D}$ | 4,096 | 7.06 |
| **DeltaCore** | [SafeAdaptiveDeltaClassifier](file:///Users/urjasoft/Documents/DeltaCore/deltacore/classification/models.py#L382) | 198 | $0$ (Frozen) | $M_t \in \mathbb{R}^{D \times D}$ | 4,096 | 6.43 |
| **Causal Control** | [StateOffAblationClassifier](file:///Users/urjasoft/Documents/DeltaCore/deltacore/classification/models.py#L552) | 198 | $0$ (Frozen) | None ($M_t \equiv 0$) | 0 | 4.45 |

---

## 4. Non-Stationary Shift Stream Generation & Shortcut Audit

### 4.1 Multi-Regime Sequence ($A \to B \to C \to A$)

The benchmark constructs non-stationary streaming sequences across 4 contiguous regimes (120 steps per regime, 480 steps total):
1. **Regime A (Directional Means)**:
   Class prototypes $P_A \in \mathbb{R}^{K \times D}$ are constructed from dense Rademacher binary codes $\pm 1 / \sqrt{D}$ normalized to $\|P_A^{(k)}\|_2 = \sqrt{D}$. Noise is isotropic Gaussian $\epsilon_t \sim \mathcal{N}(0, \sigma^2 I)$.
2. **Regime B (Covariance Structure Shift)**:
   Features undergo coordinate-covariance deformation via symmetric positive-definite transform $C_{\mathrm{cov}} = Q \Lambda Q^\top$ with condition number $\kappa \in \{2.5, 5.0, 10.0\}$, distorting the feature covariance ellipsoid while preserving class labels.
3. **Regime C (Decision-Boundary Rotation)**:
   The feature-to-label mapping rotates:
   $$P_C = R_{\mathrm{sub}} P_A$$
   where $R_{\mathrm{sub}}$ rotates pairs of class prototype directions in the prototype subspace by angle $\theta \in \{\pi/4, \pi/2.5, \pi/2\}$. Samples generated from $P_C$ are misaligned with offline head $W_{\mathrm{head}}$, requiring online adaptation.
4. **Regime A (Return)**:
   Exact return to Regime A distribution, testing representation forgetting and stale-state dynamics.

### 4.2 Shortcut Audit Results

Before model evaluation, a formal shortcut audit verified that the data stream cannot be classified trivially:
- **Matched Class Energy**: $\|x_t\|_2 = \sqrt{D} \approx 5.657$ for all classes ($\text{Max/Min Ratio} = 1.00000008$).
- **Matched Variances**: Class marginal variances matched ($\text{Max/Min Ratio} = 1.112 < 1.30$).
- **No 1D Feature Leak**: Maximum 1D nearest-centroid classification accuracy across any single feature coordinate is **40.8%**, strictly below the 60.0% leak threshold (theoretical chance is $16.7\%$).
- **Passed Shortcut Audit**: `passed_shortcut_audit: True`.
