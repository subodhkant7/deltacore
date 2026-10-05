# ==============================================================================
# DeltaCore: deltacore/scans/affine.py
# Affine scan operator abstraction and associative composition monoid.
# ==============================================================================

from dataclasses import dataclass
from typing import Self

import torch


@dataclass(frozen=True)
class AffineScanMetadata:
    r"""Preserves diagnostic stability metadata across associative composition."""

    min_stability_margin: float | None = None
    max_normalized_step: float | None = None
    clipped_count: int = 0


class AffineScanOperator:
    r"""Affine operator representing linear state transformation: $F(M) = M A + B$.

    Mathematical formulation:
        Given memory state $M \in \mathbb{R}^{V \times K}$,
        transition multiplier $A \in \mathbb{R}^{K \times K}$,
        and affine translation $B \in \mathbb{R}^{V \times K}$:
            $$F(M) = M A + B$$

    Composition law:
        Applying $F_1 = (A_1, B_1)$ followed by $F_2 = (A_2, B_2)$:
            $$F_2(F_1(M)) = (M A_1 + B_1) A_2 + B_2 = M (A_1 A_2) + (B_1 A_2 + B_2)$$
        Therefore, composition $(A_1, B_1) \otimes (A_2, B_2)$ is defined as:
            $$(A_1, B_1) \otimes (A_2, B_2) = (A_1 A_2, \; B_1 A_2 + B_2)$$

    Supports both unbatched ($A \in [K, K], B \in [V, K]$) and batched ($A \in [B, K, K], B \in [B, V, K]$) tensors.
    """

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        metadata: AffineScanMetadata | None = None,
    ) -> None:
        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
            raise TypeError("A and B must be torch.Tensor instances.")
        if A.dtype != B.dtype:
            raise TypeError(f"Dtype mismatch: A is {A.dtype}, B is {B.dtype}.")
        if A.device != B.device:
            raise ValueError(
                f"Device mismatch: A is on {A.device}, B is on {B.device}."
            )
        if A.ndim not in (2, 3) or B.ndim not in (2, 3):
            raise ValueError(
                f"A and B must be 2D or 3D tensors, got A.ndim={A.ndim}, B.ndim={B.ndim}."
            )
        if A.ndim != B.ndim:
            raise ValueError(
                f"A and B must have identical ndim, got A.ndim={A.ndim}, B.ndim={B.ndim}."
            )

        # Dimension validation
        if A.ndim == 2:
            k1, k2 = A.shape
            if k1 != k2:
                raise ValueError(f"A must be square [K, K], got shape {list(A.shape)}.")
            v_dim, k_b = B.shape
            if k_b != k1:
                raise ValueError(
                    f"Dimension mismatch: A is [{k1}, {k2}], B is [{v_dim}, {k_b}]. Expected B.shape[-1] == {k1}."
                )
        else:
            b_a, k1, k2 = A.shape
            b_b, v_dim, k_b = B.shape
            if b_a != b_b:
                raise ValueError(f"Batch dimension mismatch: A has {b_a}, B has {b_b}.")
            if k1 != k2:
                raise ValueError(
                    f"A must be square [B, K, K], got shape {list(A.shape)}."
                )
            if k_b != k1:
                raise ValueError(
                    f"Dimension mismatch: A is [{b_a}, {k1}, {k2}], B is [{b_b}, {v_dim}, {k_b}]."
                )

        self._A: torch.Tensor = A
        self._B: torch.Tensor = B
        self._metadata: AffineScanMetadata = (
            metadata if metadata is not None else AffineScanMetadata()
        )

    @property
    def A(self) -> torch.Tensor:
        """Transition matrix $A$."""
        return self._A

    @property
    def B(self) -> torch.Tensor:
        """Affine drive matrix $B$."""
        return self._B

    @property
    def metadata(self) -> AffineScanMetadata:
        """Preserved stability diagnostics."""
        return self._metadata

    @property
    def k_dim(self) -> int:
        """Key dimension $K$."""
        return self._A.shape[-1]

    @property
    def v_dim(self) -> int:
        """Value dimension $V$."""
        return self._B.shape[-2]

    @property
    def is_batched(self) -> bool:
        """Whether operator is batched [B, K, K]."""
        return self._A.ndim == 3

    @property
    def dtype(self) -> torch.dtype:
        return self._A.dtype

    @property
    def device(self) -> torch.device:
        return self._A.device

    def apply(self, memory: torch.Tensor) -> torch.Tensor:
        r"""Apply the affine operator to a memory state: $M \mapsto M A + B$.

        Args:
            memory: Memory tensor of shape `[V, K]` or `[B, V, K]`.

        Returns:
            Updated memory tensor of identical shape, dtype, and device.
        """
        if not isinstance(memory, torch.Tensor):
            raise TypeError(
                f"memory must be a torch.Tensor, got {type(memory).__name__}."
            )
        if memory.dtype != self.dtype:
            raise TypeError(
                f"Dtype mismatch: memory is {memory.dtype}, operator is {self.dtype}."
            )
        if memory.device != self.device:
            raise ValueError(
                f"Device mismatch: memory is on {memory.device}, operator is on {self.device}."
            )

        if not self.is_batched:
            if memory.ndim != 2:
                raise ValueError(
                    f"Expected 2D memory [V, K] for unbatched operator, got ndim={memory.ndim}."
                )
            if memory.shape != (self.v_dim, self.k_dim):
                raise ValueError(
                    f"Memory shape mismatch: expected [{self.v_dim}, {self.k_dim}], got {list(memory.shape)}."
                )
            # [V, K] @ [K, K] + [V, K]
            return memory @ self._A + self._B
        else:
            if memory.ndim != 3:
                raise ValueError(
                    f"Expected 3D memory [B, V, K] for batched operator, got ndim={memory.ndim}."
                )
            b = self._A.shape[0]
            if memory.shape != (b, self.v_dim, self.k_dim):
                raise ValueError(
                    f"Memory shape mismatch: expected [{b}, {self.v_dim}, {self.k_dim}], got {list(memory.shape)}."
                )
            # [B, V, K] @ [B, K, K] + [B, V, K]
            return torch.bmm(memory, self._A) + self._B

    def compose(self, second: "AffineScanOperator") -> "AffineScanOperator":
        r"""Compose this operator ($F_1$) with a subsequent operator ($F_2$).

        Sequential order:
            $(F_1 \otimes F_2)(M) = F_2(F_1(M))$
            $A_{\text{composed}} = A_1 A_2$
            $B_{\text{composed}} = B_1 A_2 + B_2$

        Args:
            second: Operator $F_2$ to execute after this operator.

        Returns:
            Composed AffineScanOperator representing the composite transformation.
        """
        if not isinstance(second, AffineScanOperator):
            raise TypeError(
                f"second must be an AffineScanOperator, got {type(second).__name__}."
            )
        if self.dtype != second.dtype:
            raise TypeError(
                f"Dtype mismatch in compose: self is {self.dtype}, second is {second.dtype}."
            )
        if self.device != second.device:
            raise ValueError(
                f"Device mismatch in compose: self on {self.device}, second on {second.device}."
            )
        if self.is_batched != second.is_batched:
            raise ValueError(
                f"Batch mode mismatch: self is_batched={self.is_batched}, second is_batched={second.is_batched}."
            )
        if self.k_dim != second.k_dim or self.v_dim != second.v_dim:
            raise ValueError(
                f"Dimension mismatch in compose: self is ({self.v_dim}, {self.k_dim}), "
                f"second is ({second.v_dim}, {second.k_dim})."
            )

        if not self.is_batched:
            new_A = self._A @ second._A
            new_B = self._B @ second._A + second._B
        else:
            new_A = torch.bmm(self._A, second._A)
            new_B = torch.bmm(self._B, second._A) + second._B

        # Stability metadata composition
        m1 = self._metadata
        m2 = second._metadata
        min_margin = None
        if m1.min_stability_margin is not None and m2.min_stability_margin is not None:
            min_margin = min(m1.min_stability_margin, m2.min_stability_margin)
        elif m1.min_stability_margin is not None:
            min_margin = m1.min_stability_margin
        else:
            min_margin = m2.min_stability_margin

        max_norm_step = None
        if m1.max_normalized_step is not None and m2.max_normalized_step is not None:
            max_norm_step = max(m1.max_normalized_step, m2.max_normalized_step)
        elif m1.max_normalized_step is not None:
            max_norm_step = m1.max_normalized_step
        else:
            max_norm_step = m2.max_normalized_step

        new_meta = AffineScanMetadata(
            min_stability_margin=min_margin,
            max_normalized_step=max_norm_step,
            clipped_count=m1.clipped_count + m2.clipped_count,
        )

        return AffineScanOperator(new_A, new_B, metadata=new_meta)

    @classmethod
    def identity(
        cls,
        k_dim: int,
        v_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        batch_size: int | None = None,
    ) -> "AffineScanOperator":
        r"""Construct the affine identity operator: $I(M) = M I_K + 0 = M$.

        Args:
            k_dim: Key dimension $K$.
            v_dim: Value dimension $V$.
            dtype: Floating point dtype.
            device: Computing device.
            batch_size: Optional batch size for batched identity operator.

        Returns:
            AffineScanOperator with $A = I_K$ and $B = 0_{V \times K}$.
        """
        if k_dim <= 0 or v_dim <= 0:
            raise ValueError(
                f"Dimensions must be positive, got k_dim={k_dim}, v_dim={v_dim}."
            )
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError(f"batch_size must be positive, got {batch_size}.")
            eye = (
                torch.eye(k_dim, dtype=dtype, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            zeros = torch.zeros(batch_size, v_dim, k_dim, dtype=dtype, device=device)
        else:
            eye = torch.eye(k_dim, dtype=dtype, device=device)
            zeros = torch.zeros(v_dim, k_dim, dtype=dtype, device=device)

        return cls(eye, zeros)

    def to(self, *args, **kwargs) -> Self:
        """Cast operator to another dtype or device."""
        return self.__class__(
            self._A.to(*args, **kwargs),
            self._B.to(*args, **kwargs),
            metadata=self._metadata,
        )

    def __repr__(self) -> str:
        batch_prefix = f"batch={self._A.shape[0]}, " if self.is_batched else ""
        return (
            f"AffineScanOperator({batch_prefix}v_dim={self.v_dim}, k_dim={self.k_dim}, "
            f"dtype={self.dtype}, device={self.device})"
        )
