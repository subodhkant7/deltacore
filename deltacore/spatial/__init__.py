"""DeltaCore Spatial Routing Package.

Provides generic 2D spatial serialization, directional routing,
restoration, directional five-memory execution, and fusion operators
for multidimensional feature maps.
"""

from deltacore.spatial.coordinates import (
    compute_coordinate_statistics,
    coordinate_statistic_control,
    generate_2d_coordinates,
    inject_2d_coordinates,
    pixel_shuffle_control,
    translate_spatial_patterns,
)
from deltacore.spatial.directional import (
    DirectionalExecutionResult,
    DirectionalFiveMemory,
)
from deltacore.spatial.fusion import (
    EqualFusion,
    LearnedChannelFusion,
    SpatialFusion,
)
from deltacore.spatial.operator import (
    SpatialAdaptiveOperator,
    SpatialOperatorResult,
)
from deltacore.spatial.routes import (
    ALL_ROUTES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    RouteDirection,
    SpatialRoute,
    get_spatial_route,
)

__all__ = [
    "SpatialRoute",
    "RouteDirection",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UP",
    "ALL_ROUTES",
    "get_spatial_route",
    "DirectionalFiveMemory",
    "DirectionalExecutionResult",
    "SpatialFusion",
    "EqualFusion",
    "LearnedChannelFusion",
    "SpatialAdaptiveOperator",
    "SpatialOperatorResult",
    "generate_2d_coordinates",
    "inject_2d_coordinates",
    "translate_spatial_patterns",
    "pixel_shuffle_control",
    "compute_coordinate_statistics",
    "coordinate_statistic_control",
]
