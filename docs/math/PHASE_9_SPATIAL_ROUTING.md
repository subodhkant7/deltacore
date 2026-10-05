# Phase 9: 2D Spatial Routing & Directional Traversal Algebra

This document specifies the mathematical foundation of 2D spatial routing, coordinate serialization, and exact spatial restoration in DeltaCore Phase 9.

---

## 1. Feature Map Definition & Spatial Topologies

Let $Z \in \mathbb{R}^{B \times C \times H \times W}$ denote a generic continuous feature map where:
* $B \in \mathbb{N}_{\ge 1}$ is the batch size,
* $C \in \mathbb{N}_{\ge 1}$ is the channel dimension (token feature dimension $D_{\text{in}}$),
* $H \in \mathbb{N}_{\ge 1}$ is the spatial height (number of rows),
* $W \in \mathbb{N}_{\ge 1}$ is the spatial width (number of columns),
* $N = H \cdot W$ is the total number of spatial tokens per item.

DeltaCore does not assume square grids; all operations must support arbitrary rectangular geometries ($H \neq W$), including degenerate 1D strips ($1 \times W$ or $H \times 1$) and unit grids ($1 \times 1$).

---

## 2. Directional Traversal Routes ($\mathcal{D}$)

Following spatial scanning literature and VisionHOPE (Peng et al., 2026), four canonical directional routes $\mathcal{D} = \{\text{RIGHT}, \text{LEFT}, \text{DOWN}, \text{UP}\}$ are defined:

$$\mathcal{P}_r: \mathbb{R}^{B \times C \times H \times W} \longrightarrow \mathbb{R}^{B \times N \times C}$$
$$\mathcal{P}_r^{-1}: \mathbb{R}^{B \times N \times C} \longrightarrow \mathbb{R}^{B \times C \times H \times W}$$

Every route $\mathcal{P}_r$ bijectively indexes the 2D spatial coordinates $(h, w) \in \{0, \dots, H-1\} \times \{0, \dots, W-1\}$ into a 1D sequential index $n \in \{0, \dots, N-1\}$.

### 2.1. Route RIGHT ($\rightarrow$): Standard Row-Major Traversal
* **Coordinate Progression**: Sweeps left-to-right along row 0, then row 1, ..., through row $H-1$.
  $$(0, 0) \to (0, 1) \to \dots \to (0, W-1) \to (1, 0) \to \dots \to (H-1, W-1)$$
* **Index Mapping**:
  $$n(h, w) = h \cdot W + w$$
* **Tensor Implementation**:
  $$\mathcal{P}_{\text{RIGHT}}(Z) = \operatorname{reshape}\left(\operatorname{permute}(Z, (0, 2, 3, 1)), (B, N, C)\right)$$
* **Inverse Restoration**:
  $$\mathcal{P}_{\text{RIGHT}}^{-1}(S, H, W) = \operatorname{permute}\left(\operatorname{reshape}(S, (B, H, W, C)), (0, 3, 1, 2)\right)$$

### 2.2. Route LEFT ($\leftarrow$): Reverse Row-Major Traversal
* **Coordinate Progression**: Traverses the row-major sequence in exact reverse order (bottom-right to top-left).
  $$(H-1, W-1) \to (H-1, W-2) \to \dots \to (H-1, 0) \to (H-2, W-1) \to \dots \to (0, 0)$$
* **Index Mapping**:
  $$n(h, w) = (H - 1 - h) \cdot W + (W - 1 - w)$$
* **Tensor Implementation**:
  $$\mathcal{P}_{\text{LEFT}}(Z) = \operatorname{flip}\left(\mathcal{P}_{\text{RIGHT}}(Z), \operatorname{dim}=1\right)$$
* **Inverse Restoration**:
  $$\mathcal{P}_{\text{LEFT}}^{-1}(S, H, W) = \mathcal{P}_{\text{RIGHT}}^{-1}\left(\operatorname{flip}(S, \operatorname{dim}=1), H, W\right)$$

### 2.3. Route DOWN ($\downarrow$): Standard Column-Major Traversal
* **Coordinate Progression**: Sweeps top-to-bottom along column 0, then column 1, ..., through column $W-1$.
  $$(0, 0) \to (1, 0) \to \dots \to (H-1, 0) \to (0, 1) \to \dots \to (H-1, W-1)$$
* **Index Mapping**:
  $$n(h, w) = w \cdot H + h$$
* **Tensor Implementation**:
  $$\mathcal{P}_{\text{DOWN}}(Z) = \operatorname{reshape}\left(\operatorname{permute}(Z, (0, 3, 2, 1)), (B, N, C)\right)$$
* **Inverse Restoration**:
  $$\mathcal{P}_{\text{DOWN}}^{-1}(S, H, W) = \operatorname{permute}\left(\operatorname{reshape}(S, (B, W, H, C)), (0, 3, 2, 1)\right)$$

### 2.4. Route UP ($\uparrow$): Reverse Column-Major Traversal
* **Coordinate Progression**: Traverses the column-major sequence in exact reverse order (bottom-right column-wise to top-left).
  $$(H-1, W-1) \to (H-2, W-1) \to \dots \to (0, W-1) \to (H-1, W-2) \to \dots \to (0, 0)$$
* **Index Mapping**:
  $$n(h, w) = (W - 1 - w) \cdot H + (H - 1 - h)$$
* **Tensor Implementation**:
  $$\mathcal{P}_{\text{UP}}(Z) = \operatorname{flip}\left(\mathcal{P}_{\text{DOWN}}(Z), \operatorname{dim}=1\right)$$
* **Inverse Restoration**:
  $$\mathcal{P}_{\text{UP}}^{-1}(S, H, W) = \mathcal{P}_{\text{DOWN}}^{-1}\left(\operatorname{flip}(S, \operatorname{dim}=1), H, W\right)$$

---

## 3. Exact Round-Trip Inversion Identity

For every route $r \in \mathcal{D}$ and for any feature map $Z \in \mathbb{R}^{B \times C \times H \times W}$ of arbitrary rectangular shape and data type:

$$\mathcal{P}_r^{-1}\left(\mathcal{P}_r(Z), H, W\right) \equiv Z$$

This is a mathematical identity, not an approximation. It preserves:
1. Batch index $b \in \{0, \dots, B-1\}$,
2. Channel index $c \in \{0, \dots, C-1\}$,
3. Spatial coordinate $(h, w)$,
4. Floating-point precision (bitwise identity in FP32, FP64, BFloat16),
5. Computational graph autograd gradients ($\nabla_Z \mathcal{L} = \mathcal{P}_r^{-1}\left(\nabla_{\mathcal{P}_r(Z)} \mathcal{L}\right)$).

---

## 4. Chunk Alignment Geometry

When operating under boundary-refresh chunk execution (Section 2.3 of VisionHOPE):
* **Row-Aligned Routes (RIGHT, LEFT)**:
  Setting chunk size $C_{\text{chunk}} = W$ partitions the $N = H \cdot W$ token sequence into exactly $H$ chunks. Each chunk contains all tokens of exactly one horizontal row of width $W$.
* **Column-Aligned Routes (DOWN, UP)**:
  Setting chunk size $C_{\text{chunk}} = H$ partitions the $N = H \cdot W$ token sequence into exactly $W$ chunks. Each chunk contains all tokens of exactly one vertical column of height $H$.

DeltaCore supports both:
* **Paper-Aligned Mode**: Automatically assigns $C_\rightarrow = W, C_\leftarrow = W, C_\downarrow = H, C_\uparrow = H$.
* **Generic Mode**: Permits arbitrary user-specified chunk sizes $C_{\text{chunk}} \in [1, N]$.
