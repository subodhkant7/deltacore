"""Linear associative memory read operation.

Computes the linear projection of a query/key vector through the associative memory:
    v_hat = M @ k
"""

import torch


def read(memory: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    r"""Retrieve an associated value vector from memory using a key.

    Mathematical definition:
        $$\hat{v} = M k$$

    Args:
        memory: The associative memory state tensor.
            Must be 2D of shape `[V, K]` or 3D of shape `[B, V, K]`.
        key: The query/key vector or batch of key vectors.
            Must be 1D of shape `[K]` when memory is 2D `[V, K]`,
            or 2D of shape `[B, K]` when memory is 2D or 3D.

    Returns:
        The retrieved value tensor $\hat{v}$.
        Shape is `[V]` if key is `[K]`, or `[B, V]` if key is `[B, K]`.

    Raises:
        TypeError: If memory or key is not a torch.Tensor, or if dtypes do not match.
        ValueError: If tensor dimensions are invalid, shapes are mismatched, or devices differ.

    Example:
        >>> M = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> k = torch.tensor([3.0, 5.0])
        >>> read(M, k)
        tensor([3., 5.])
    """
    if not isinstance(memory, torch.Tensor):
        raise TypeError(f"memory must be a torch.Tensor, got {type(memory).__name__}")
    if not isinstance(key, torch.Tensor):
        raise TypeError(f"key must be a torch.Tensor, got {type(key).__name__}")

    if memory.dtype != key.dtype:
        raise TypeError(
            f"Dtype mismatch: memory has dtype {memory.dtype}, but key has dtype {key.dtype}."
        )

    if memory.device != key.device:
        raise ValueError(
            f"Device mismatch: memory is on {memory.device}, but key is on {key.device}."
        )

    # 1. Simple 2D Memory: [V, K]
    if memory.ndim == 2:
        v_dim, k_dim = memory.shape
        if key.ndim == 1:
            if key.shape[0] != k_dim:
                raise ValueError(
                    f"Key dimension mismatch: memory requires key size {k_dim}, "
                    f"but received key of size {key.shape[0]}."
                )
            # Standard matrix-vector multiplication: [V, K] @ [K] -> [V]
            return memory @ key

        elif key.ndim == 2:
            batch_size, key_len = key.shape
            if key_len != k_dim:
                raise ValueError(
                    f"Key dimension mismatch: memory requires key size {k_dim}, "
                    f"but received key of shape [{batch_size}, {key_len}]."
                )
            # Batched keys against shared 2D memory: [B, K] @ [K, V] -> [B, V]
            return key @ memory.T

        else:
            raise ValueError(
                f"Invalid key shape for 2D memory: expected 1D [K] or 2D [B, K], "
                f"got ndim={key.ndim} with shape {list(key.shape)}."
            )

    # 2. Batched 3D Memory: [B, V, K]
    elif memory.ndim == 3:
        b_mem, v_dim, k_dim = memory.shape
        if key.ndim == 2:
            b_key, key_len = key.shape
            if b_mem != b_key:
                raise ValueError(
                    f"Batch dimension mismatch: memory batch size is {b_mem}, "
                    f"but key batch size is {b_key}."
                )
            if key_len != k_dim:
                raise ValueError(
                    f"Key dimension mismatch: memory requires key size {k_dim}, "
                    f"but received key size {key_len}."
                )
            # Batched matmul: [B, V, K] @ [B, K, 1] -> [B, V, 1] -> [B, V]
            return torch.bmm(memory, key.unsqueeze(-1)).squeeze(-1)

        else:
            raise ValueError(
                f"Invalid key shape for 3D memory: expected 2D [B, K], "
                f"got ndim={key.ndim} with shape {list(key.shape)}."
            )

    else:
        raise ValueError(
            f"Invalid memory dimensions: expected 2D [V, K] or 3D [B, V, K], "
            f"got ndim={memory.ndim} with shape {list(memory.shape)}."
        )
