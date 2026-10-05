"""Update rule primitives and mathematical transition operators.

This module hosts Hebbian outer-product, error-correcting delta rules,
adaptive step-size controllers, dynamics memory containers, and self-referential
adaptive systems.
"""

from deltacore.updates.adaptive_delta import (
    AdaptiveDeltaRule,
    AdaptiveDeltaStepResult,
)
from deltacore.updates.controllers import (
    ConstantStepSize,
    ErrorConditionedStepSize,
    InputConditionedStepSize,
    StateConditionedStepSize,
    StepSizeController,
)
from deltacore.updates.delta import DeltaRule, DeltaStepResult
from deltacore.updates.dynamics_memory import DynamicsMemory
from deltacore.updates.five_memory import (
    FiveMemoryConfig,
    FiveMemoryScanResult,
    FiveMemoryStepResult,
    FiveMemorySystem,
)
from deltacore.updates.hebbian import HebbianRule, HebbianStepResult
from deltacore.updates.self_referential import (
    ControlFeatureExtractor,
    CoupledState,
    ErrorProportionalTargetGenerator,
    SelfReferentialScanResult,
    SelfReferentialStepResult,
    SelfReferentialSystem,
    TargetGenerator,
)

__all__: list[str] = [
    "DeltaRule",
    "DeltaStepResult",
    "HebbianRule",
    "HebbianStepResult",
    "AdaptiveDeltaRule",
    "AdaptiveDeltaStepResult",
    "StepSizeController",
    "ConstantStepSize",
    "InputConditionedStepSize",
    "ErrorConditionedStepSize",
    "StateConditionedStepSize",
    "DynamicsMemory",
    "CoupledState",
    "SelfReferentialStepResult",
    "SelfReferentialScanResult",
    "TargetGenerator",
    "ErrorProportionalTargetGenerator",
    "ControlFeatureExtractor",
    "SelfReferentialSystem",
    "FiveMemoryConfig",
    "FiveMemoryStepResult",
    "FiveMemoryScanResult",
    "FiveMemorySystem",
]
