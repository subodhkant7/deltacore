# ==============================================================================
# DeltaCore: deltacore/memory/five_memory.py
# State container for the five coupled memory matrices.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FiveMemoryState:
    r"""Immutable state container for the five coupled memories in DeltaCore Phase 8.

    Mathematical Representation:
        - content:       M_{content} \in \mathbb{R}^{V \times K}      (or [B, V, K])
        - key:           M_{key}     \in \mathbb{R}^{K \times D_{in}} (or [B, K, D_{in}])
        - value:         M_{val}     \in \mathbb{R}^{V \times D_{in}} (or [B, V, D_{in}])
        - learning_rate: M_{\eta}    \in \mathbb{R}^{D_{lr} \times D_{feat}} (or [B, D_{lr}, D_{feat}])
        - retention:     M_{ret}     \in \mathbb{R}^{D_{ret} \times D_{feat}} (or [B, D_{ret}, D_{feat}])

    Invariants & Validation:
        1. All attributes must be floating-point PyTorch tensors.
        2. All attributes must reside on the same compute device and share identical dtype.
        3. All attributes must be either all unbatched (2D) or all batched (3D) with identical batch size.
        4. Shape compatibility:
           - content.shape[-1] == key.shape[-2] == K
           - content.shape[-2] == value.shape[-2] == V
           - key.shape[-1] == value.shape[-1] == D_in
           - learning_rate.shape[-1] == retention.shape[-1] == D_feat
        5. Container is frozen to prevent accidental in-place attribute mutation.
    """

    content: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    learning_rate: torch.Tensor
    retention: torch.Tensor

    def __post_init__(self) -> None:
        tensors = {
            "content": self.content,
            "key": self.key,
            "value": self.value,
            "learning_rate": self.learning_rate,
            "retention": self.retention,
        }

        # 1. Type and floating-point checks
        for name, t in tensors.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"FiveMemoryState '{name}' must be a torch.Tensor, got {type(t).__name__}."
                )
            if not torch.is_floating_point(t):
                raise TypeError(
                    f"FiveMemoryState '{name}' must have a floating-point dtype, got {t.dtype}."
                )

        # 2. Dtype and device consistency
        ref_dtype = self.content.dtype
        ref_device = self.content.device
        for name, t in tensors.items():
            if t.dtype != ref_dtype:
                raise ValueError(
                    f"FiveMemoryState dtype mismatch: 'content' is {ref_dtype} but '{name}' is {t.dtype}."
                )
            if t.device != ref_device:
                raise ValueError(
                    f"FiveMemoryState device mismatch: 'content' is {ref_device} but '{name}' is {t.device}."
                )

        # 3. Dimensionality (2D or 3D)
        ref_ndim = self.content.ndim
        if ref_ndim not in (2, 3):
            raise ValueError(
                f"FiveMemoryState tensors must be 2D [rows, cols] or 3D [B, rows, cols], got ndim={ref_ndim} for 'content'."
            )
        for name, t in tensors.items():
            if t.ndim != ref_ndim:
                raise ValueError(
                    f"FiveMemoryState dimensionality mismatch: 'content' has ndim={ref_ndim} but '{name}' has ndim={t.ndim}."
                )

        # 4. Batch size matching if 3D
        if ref_ndim == 3:
            b_size = self.content.shape[0]
            for name, t in tensors.items():
                if t.shape[0] != b_size:
                    raise ValueError(
                        f"FiveMemoryState batch size mismatch: 'content' has batch_size={b_size} but '{name}' has batch_size={t.shape[0]}."
                    )

        # 5. Semantic dimension compatibility
        v_c, k_c = self.content.shape[-2], self.content.shape[-1]
        k_k, din_k = self.key.shape[-2], self.key.shape[-1]
        v_v, din_v = self.value.shape[-2], self.value.shape[-1]
        _, dfeat_lr = self.learning_rate.shape[-2], self.learning_rate.shape[-1]
        _, dfeat_ret = self.retention.shape[-2], self.retention.shape[-1]

        if k_c != k_k:
            raise ValueError(
                f"Key dimension mismatch: 'content' expects K={k_c} but 'key' provides K={k_k}."
            )
        if v_c != v_v:
            raise ValueError(
                f"Value dimension mismatch: 'content' expects V={v_c} but 'value' provides V={v_v}."
            )
        if din_k != din_v:
            raise ValueError(
                f"Input dimension mismatch: 'key' expects D_in={din_k} but 'value' expects D_in={din_v}."
            )
        if dfeat_lr != dfeat_ret:
            raise ValueError(
                f"Feature dimension mismatch: 'learning_rate' expects D_feat={dfeat_lr} but 'retention' expects D_feat={dfeat_ret}."
            )

    @property
    def is_batched(self) -> bool:
        """True if states have a leading batch dimension [B, ...]."""
        return self.content.ndim == 3

    @property
    def batch_size(self) -> int | None:
        """Batch size B if batched, else None."""
        return self.content.shape[0] if self.is_batched else None

    @property
    def v_dim(self) -> int:
        """Value dimension V."""
        return self.content.shape[-2]

    @property
    def k_dim(self) -> int:
        """Key dimension K."""
        return self.content.shape[-1]

    @property
    def in_dim(self) -> int:
        """Input token dimension D_in."""
        return self.key.shape[-1]

    @property
    def feat_dim(self) -> int:
        """Meta-controller feature dimension D_feat."""
        return self.learning_rate.shape[-1]

    @property
    def lr_dim(self) -> int:
        """Learning-rate memory projection dimension D_lr."""
        return self.learning_rate.shape[-2]

    @property
    def ret_dim(self) -> int:
        """Retention memory projection dimension D_ret."""
        return self.retention.shape[-2]

    @property
    def dtype(self) -> torch.dtype:
        """Floating point precision dtype."""
        return self.content.dtype

    @property
    def device(self) -> torch.device:
        """Compute device."""
        return self.content.device

    def clone(self) -> FiveMemoryState:
        """Create a deep copy of the state with cloned tensors."""
        return FiveMemoryState(
            content=self.content.clone(),
            key=self.key.clone(),
            value=self.value.clone(),
            learning_rate=self.learning_rate.clone(),
            retention=self.retention.clone(),
        )

    def detach(self) -> FiveMemoryState:
        """Return a new state detached from the computational autograd graph."""
        return FiveMemoryState(
            content=self.content.detach(),
            key=self.key.detach(),
            value=self.value.detach(),
            learning_rate=self.learning_rate.detach(),
            retention=self.retention.detach(),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> FiveMemoryState:
        """Cast and/or move all five state tensors."""
        return FiveMemoryState(
            content=self.content.to(device=device, dtype=dtype),
            key=self.key.to(device=device, dtype=dtype),
            value=self.value.to(device=device, dtype=dtype),
            learning_rate=self.learning_rate.to(device=device, dtype=dtype),
            retention=self.retention.to(device=device, dtype=dtype),
        )

    @classmethod
    def zeros(
        cls,
        v_dim: int,
        k_dim: int,
        in_dim: int,
        feat_dim: int | None = None,
        lr_dim: int = 1,
        ret_dim: int = 1,
        batch_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> FiveMemoryState:
        """Factory method to initialize all five memory states to zeros.

        Args:
            v_dim: Value target dimension V.
            k_dim: Key query dimension K.
            in_dim: Input token dimension D_in.
            feat_dim: Feature dimension D_feat for meta-controllers (defaults to in_dim).
            lr_dim: Projection dimension D_lr for learning-rate memory (default: 1).
            ret_dim: Projection dimension D_ret for retention memory (default: 1).
            batch_size: Optional batch size B for batched 3D tensors.
            dtype: Tensor floating point precision.
            device: Compute device.
        """
        f_dim = in_dim if feat_dim is None else feat_dim

        def make(rows: int, cols: int) -> torch.Tensor:
            if batch_size is not None:
                return torch.zeros(batch_size, rows, cols, dtype=dtype, device=device)
            return torch.zeros(rows, cols, dtype=dtype, device=device)

        return cls(
            content=make(v_dim, k_dim),
            key=make(k_dim, in_dim),
            value=make(v_dim, in_dim),
            learning_rate=make(lr_dim, f_dim),
            retention=make(ret_dim, f_dim),
        )

    @classmethod
    def initialize(
        cls,
        v_dim: int,
        k_dim: int,
        in_dim: int,
        feat_dim: int | None = None,
        lr_dim: int = 1,
        ret_dim: int = 1,
        batch_size: int | None = None,
        init_scale: float = 0.01,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> FiveMemoryState:
        """Initialize five memory states with small normal noise for encoders and zeros for content."""
        f_dim = in_dim if feat_dim is None else feat_dim

        def make_noise(rows: int, cols: int) -> torch.Tensor:
            shape = (batch_size, rows, cols) if batch_size is not None else (rows, cols)
            noise = torch.randn(shape, generator=generator, dtype=dtype, device=device)
            return noise * init_scale

        def make_zeros(rows: int, cols: int) -> torch.Tensor:
            shape = (batch_size, rows, cols) if batch_size is not None else (rows, cols)
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            content=make_zeros(v_dim, k_dim),
            key=make_noise(k_dim, in_dim),
            value=make_noise(v_dim, in_dim),
            learning_rate=make_zeros(lr_dim, f_dim),
            retention=make_zeros(ret_dim, f_dim),
        )
