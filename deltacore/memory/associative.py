"""Explicit associative memory state abstraction.

Represents an internal associative memory state matrix:
    M in R^{V x K}
"""

import torch

from deltacore.memory.read import read


class AssociativeMemory:
    r"""Explicit state container for a linear associative memory matrix.

    Mathematical definition:
        $$M \in \mathbb{R}^{V \times K}$$
        where $V$ is the value (target) dimension and $K$ is the key (query) dimension.

    This abstraction encapsulates memory state storage, validation, cloning,
    and retrieval delegation. Following the DeltaCore Project Constitution (Principle B),
    the memory container does NOT compute update rules; state transitions are
    exclusively computed by external update rules (e.g. HebbianRule, DeltaRule).

    Args:
        data: The underlying floating-point memory tensor of shape `[V, K]` or `[B, V, K]`.

    Raises:
        TypeError: If data is not a torch.Tensor or not a floating-point type.
        ValueError: If data is not 2D or 3D, or contains non-positive dimensions.

    Example:
        >>> mem = AssociativeMemory.zeros(v_dim=4, k_dim=3)
        >>> mem.shape
        torch.Size([4, 3])
        >>> k = torch.ones(3)
        >>> mem.read(k)
        tensor([0., 0., 0., 0.])
    """

    def __init__(self, data: torch.Tensor) -> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"AssociativeMemory data must be a torch.Tensor, got {type(data).__name__}."
            )
        if not torch.is_floating_point(data):
            raise TypeError(
                f"AssociativeMemory requires floating point dtype, got {data.dtype}."
            )
        if data.ndim not in (2, 3):
            raise ValueError(
                f"AssociativeMemory requires 2D [V, K] or 3D [B, V, K] tensor, "
                f"got ndim={data.ndim} with shape {list(data.shape)}."
            )
        if any(dim <= 0 for dim in data.shape):
            raise ValueError(
                f"AssociativeMemory dimensions must be strictly positive, got shape {list(data.shape)}."
            )

        self._data: torch.Tensor = data

    @classmethod
    def zeros(
        cls,
        v_dim: int,
        k_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        batch_size: int | None = None,
    ) -> "AssociativeMemory":
        """Initialize an associative memory state filled with zeros.

        Args:
            v_dim: Value (target) dimension $V > 0$.
            k_dim: Key (query) dimension $K > 0$.
            dtype: Floating point precision (e.g. torch.float32, torch.float64).
            device: Compute device (e.g. 'cpu', 'cuda').
            batch_size: Optional batch size $B$ for batched 3D memory `[B, V, K]`.

        Returns:
            An AssociativeMemory instance initialized to zeros.
        """
        if v_dim <= 0 or k_dim <= 0:
            raise ValueError(
                f"Dimensions must be positive, got v_dim={v_dim}, k_dim={k_dim}."
            )
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError(
                    f"batch_size must be positive, got batch_size={batch_size}."
                )
            shape = (batch_size, v_dim, k_dim)
        else:
            shape = (v_dim, k_dim)

        data = torch.zeros(shape, dtype=dtype, device=device)
        return cls(data)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "AssociativeMemory":
        """Create an AssociativeMemory instance wrapping an existing tensor.

        Args:
            tensor: 2D `[V, K]` or 3D `[B, V, K]` floating point tensor.

        Returns:
            AssociativeMemory instance.
        """
        return cls(tensor)

    @property
    def data(self) -> torch.Tensor:
        """Return the underlying memory tensor."""
        return self._data

    @property
    def v_dim(self) -> int:
        """Value dimension $V$ (rows of $M$)."""
        return self._data.shape[-2]

    @property
    def k_dim(self) -> int:
        """Key dimension $K$ (columns of $M$)."""
        return self._data.shape[-1]

    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying memory tensor."""
        return self._data.shape

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the memory tensor."""
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        """Device hosting the memory tensor."""
        return self._data.device

    def read(self, key: torch.Tensor) -> torch.Tensor:
        r"""Retrieve an associated value vector using key $k$.

        $$\hat{v} = M k$$

        Args:
            key: Query vector `[K]` or batched queries `[B, K]`.

        Returns:
            Retrieved value tensor `[V]` or `[B, V]`.
        """
        return read(self._data, key)

    def clone(self) -> "AssociativeMemory":
        """Create an independent copy of this memory state.

        Returns:
            A new AssociativeMemory containing cloned underlying data.
        """
        return AssociativeMemory(self._data.clone())

    def reset(self) -> "AssociativeMemory":
        """Reset the memory contents in-place to all zeros.

        Returns:
            Self (mutated in-place to zeros).
        """
        self._data.zero_()
        return self

    def to(self, *args, **kwargs) -> "AssociativeMemory":
        """Cast memory data to another dtype or device.

        Returns:
            A new AssociativeMemory instance with converted data.
        """
        new_data = self._data.to(*args, **kwargs)
        return AssociativeMemory(new_data)

    def __repr__(self) -> str:
        batch_prefix = f"batch={self._data.shape[0]}, " if self._data.ndim == 3 else ""
        return (
            f"AssociativeMemory({batch_prefix}v_dim={self.v_dim}, k_dim={self.k_dim}, "
            f"dtype={self.dtype}, device={self.device})"
        )
