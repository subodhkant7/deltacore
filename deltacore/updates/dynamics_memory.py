r"""Dynamics memory container for self-referential control state.

Represents an internal memory state $C_t \in \mathbb{R}^{R \times D_c}$ that generates
control signals driving the primary content memory's learning dynamics.
"""

import torch


class DynamicsMemory:
    r"""Explicit state container for the learning dynamics controller memory.

    Mathematical definition:
        $$C_t \in \mathbb{R}^{R \times D_c}$$
        where $R$ is the control response dimension (default $R=1$ for scalar step size)
        and $D_c$ is the control feature dimension.

    Provides readout projection:
        $$r_t = C_t z_t \in \mathbb{R}^R$$
    and supports cloning, resetting, precision casting, and shape validation.
    """

    def __init__(self, data: torch.Tensor) -> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"DynamicsMemory data must be a torch.Tensor, got {type(data).__name__}."
            )
        if not torch.is_floating_point(data):
            raise TypeError(
                f"DynamicsMemory requires floating point dtype, got {data.dtype}."
            )
        if data.ndim not in (2, 3):
            raise ValueError(
                f"DynamicsMemory requires 2D [R, Dc] or 3D [B, R, Dc] tensor, "
                f"got ndim={data.ndim} with shape {list(data.shape)}."
            )
        if any(dim <= 0 for dim in data.shape):
            raise ValueError(
                f"DynamicsMemory dimensions must be strictly positive, got {list(data.shape)}."
            )

        self._data: torch.Tensor = data

    @classmethod
    def zeros(
        cls,
        r_dim: int = 1,
        d_c: int = 3,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        batch_size: int | None = None,
    ) -> "DynamicsMemory":
        """Initialize DynamicsMemory state with all zeros."""
        if r_dim <= 0 or d_c <= 0:
            raise ValueError(
                f"Dimensions must be positive, got r_dim={r_dim}, d_c={d_c}."
            )
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError(f"batch_size must be positive, got {batch_size}.")
            shape = (batch_size, r_dim, d_c)
        else:
            shape = (r_dim, d_c)

        return cls(torch.zeros(shape, dtype=dtype, device=device))

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "DynamicsMemory":
        """Wrap an existing tensor into a DynamicsMemory container."""
        return cls(tensor)

    @property
    def data(self) -> torch.Tensor:
        """Underlying dynamics memory tensor."""
        return self._data

    @property
    def r_dim(self) -> int:
        """Control response dimension $R$ (rows of $C$)."""
        return self._data.shape[-2]

    @property
    def d_c(self) -> int:
        """Control feature dimension $D_c$ (columns of $C$)."""
        return self._data.shape[-1]

    @property
    def shape(self) -> torch.Size:
        """Shape of the dynamics memory tensor."""
        return self._data.shape

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the dynamics memory."""
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        """Device hosting the dynamics memory."""
        return self._data.device

    def read(self, features: torch.Tensor) -> torch.Tensor:
        r"""Compute control response $r = C z$.

        Args:
            features: Feature vector $z \in \mathbb{R}^{D_c}$ or batched `[B, D_c]`.

        Returns:
            Projected control signal $r \in \mathbb{R}^R$ or `[B, R]`.
        """
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"features must be a torch.Tensor, got {type(features).__name__}."
            )
        if features.dtype != self.dtype:
            raise TypeError(
                f"Dtype mismatch: memory is {self.dtype}, features is {features.dtype}."
            )
        if features.device != self.device:
            raise ValueError(
                f"Device mismatch: memory on {self.device}, features on {features.device}."
            )

        if self._data.ndim == 2:
            if features.ndim == 1:
                if features.shape[0] != self.d_c:
                    raise ValueError(
                        f"Feature dimension mismatch: expected {self.d_c}, got {features.shape[0]}."
                    )
                # [R, Dc] @ [Dc] -> [R]
                return self._data @ features
            elif features.ndim == 2:
                # [B, Dc] @ [Dc, R] -> [B, R]
                return features @ self._data.T
            else:
                raise ValueError(
                    f"Unsupported feature ndim={features.ndim} for 2D dynamics memory."
                )

        elif self._data.ndim == 3:
            # Batched: [B, R, Dc] @ [B, Dc, 1] -> [B, R, 1] -> [B, R]
            if features.ndim != 2:
                raise ValueError(
                    f"Expected 2D [B, Dc] features for 3D dynamics memory, got {features.ndim}."
                )
            return torch.bmm(self._data, features.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(f"Invalid memory ndim={self._data.ndim}.")

    def clone(self) -> "DynamicsMemory":
        """Create an independent copy of this dynamics memory state."""
        return DynamicsMemory(self._data.clone())

    def reset(self) -> "DynamicsMemory":
        """Reset the dynamics memory state in-place to all zeros."""
        self._data.zero_()
        return self

    def to(self, *args, **kwargs) -> "DynamicsMemory":
        """Cast dynamics memory to another dtype or device."""
        return DynamicsMemory(self._data.to(*args, **kwargs))

    def __repr__(self) -> str:
        batch_prefix = f"batch={self._data.shape[0]}, " if self._data.ndim == 3 else ""
        return (
            f"DynamicsMemory({batch_prefix}r_dim={self.r_dim}, d_c={self.d_c}, "
            f"dtype={self.dtype}, device={self.device})"
        )
