r"""Stability controllers, norm bounds, and numerical regularization.

This module provides mathematical controllers that enforce local non-expansion
conditions for adaptive and self-referential memory updates:
    - SafeStepSizeController: Enforces \eta_t \|k_t\|^2 \le \beta < 2
    - SafeDynamicsRateController: Enforces \rho_t \|z_t\|^2 \le \beta_C < 2
    - UnconstrainedController: Identity passthrough with active telemetry
"""

from deltacore.stability.controllers import (
    BaseStabilityController,
    SafeDynamicsRateController,
    SafeStepSizeController,
    StabilityConstraintResult,
    UnconstrainedController,
)

__all__: list[str] = [
    "BaseStabilityController",
    "UnconstrainedController",
    "SafeStepSizeController",
    "SafeDynamicsRateController",
    "StabilityConstraintResult",
]
