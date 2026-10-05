# ==============================================================================
# DeltaCore: deltacore/benchmarks/baselines/factory.py
# Factory for instantiating standardized benchmark baselines.
# ==============================================================================

from typing import Any

import torch

from deltacore.benchmarks.baselines.base import BaseBaseline
from deltacore.benchmarks.baselines.wrappers import (
    AdaptiveDeltaBaseline,
    FixedDeltaBaseline,
    FrozenBaseline,
    SafeSelfReferentialBaseline,
    SelfReferentialBaseline,
)


def get_baseline(
    name: str,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype | str = torch.float32,
    device: torch.device | str = "cpu",
    **kwargs: Any,
) -> BaseBaseline:
    """Instantiate a standardized baseline wrapper by name.

    Args:
        name: Baseline identifier ('frozen', 'fixed', 'adaptive',
            'self_referential', 'safe_self_referential').
        key_dim: Key vector dimension.
        value_dim: Value vector dimension.
        dtype: Numerical precision.
        device: Target execution device.
        **kwargs: Additional baseline-specific parameters.

    Returns:
        Instantiated BaseBaseline instance.

    Raises:
        ValueError: If baseline name is unrecognized.
    """
    if isinstance(dtype, str):
        torch_dtype: torch.dtype = getattr(torch, dtype)
    else:
        torch_dtype = dtype

    key = name.lower().replace("-", "_")
    if key in {"frozen", "baseline_a"}:
        return FrozenBaseline(
            key_dim=key_dim, value_dim=value_dim, dtype=torch_dtype, device=device
        )
    elif key in {"fixed", "fixed_delta", "fixeddelta", "baseline_b"}:
        step_size = kwargs.get("step_size", 0.1)
        return FixedDeltaBaseline(
            key_dim=key_dim,
            value_dim=value_dim,
            step_size=step_size,
            dtype=torch_dtype,
            device=device,
        )
    elif key in {"adaptive", "adaptive_delta", "adaptivedelta", "baseline_c"}:
        max_step_size = kwargs.get("max_step_size", 0.5)
        alpha = kwargs.get("alpha", 1.0)
        return AdaptiveDeltaBaseline(
            key_dim=key_dim,
            value_dim=value_dim,
            max_step_size=max_step_size,
            alpha=alpha,
            dtype=torch_dtype,
            device=device,
        )
    elif key in {"self_referential", "selfreferential", "self_ref", "baseline_d"}:
        eta_max = kwargs.get("eta_max", 0.5)
        rho_max = kwargs.get("rho_max", 0.2)
        dynamics_dim = kwargs.get("dynamics_dim", 3)
        return SelfReferentialBaseline(
            key_dim=key_dim,
            value_dim=value_dim,
            dynamics_dim=dynamics_dim,
            eta_max=eta_max,
            rho_max=rho_max,
            dtype=torch_dtype,
            device=device,
        )
    elif key in {
        "safe_self_referential",
        "safeselfreferential",
        "safe_self_ref",
        "baseline_e",
    }:
        eta_max = kwargs.get("eta_max", 0.5)
        rho_max = kwargs.get("rho_max", 0.2)
        dynamics_dim = kwargs.get("dynamics_dim", 3)
        beta = kwargs.get("beta", 1.0)
        beta_c = kwargs.get("beta_c", 1.0)
        epsilon = kwargs.get("epsilon", 1e-4)
        return SafeSelfReferentialBaseline(
            key_dim=key_dim,
            value_dim=value_dim,
            dynamics_dim=dynamics_dim,
            eta_max=eta_max,
            rho_max=rho_max,
            beta=beta,
            beta_c=beta_c,
            epsilon=epsilon,
            dtype=torch_dtype,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown baseline name '{name}'. Expected one of: "
            f"['frozen', 'fixed', 'adaptive', 'self_referential', 'safe_self_referential']."
        )
