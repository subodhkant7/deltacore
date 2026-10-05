# ==============================================================================
# DeltaCore: deltacore/observatory/events.py
# Automated trajectory event extraction without causal inference.
# ==============================================================================

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from deltacore.observatory.schema import StateTrajectory


@dataclass
class TrajectoryEvent:
    """Structured discrete event extracted from a neural-state trajectory.

    Non-causal specification:
        Events record discrete temporal occurrences (step, value, and context).
        They do not assert causal mechanisms between events.

    Attributes:
        event_type: Categorical identifier of the event.
        step: Sequence step index at which the event occurred.
        value: Numerical value associated with the event (or None if categorical).
        context: Descriptive contextual metadata (e.g. threshold, window, prior value).
    """

    event_type: str
    step: int
    value: float | None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to serializable dictionary."""
        return asdict(self)


def extract_events(
    trajectory: StateTrajectory,
    tau: float | None = None,
    window: int | None = None,
) -> list[TrajectoryEvent]:
    """Extract all standardized events from a StateTrajectory in chronological order.

    Detects:
        - shift_boundary: Declared sequence distribution shift point.
        - first_threshold_crossing: First step after shock where error <= tau * shock_error.
        - sustained_recovery: First step after shock where error remains <= tau * shock_error for window steps.
        - first_nonfinite_state: First occurrence of non-finite internal state or output.
        - maximum_update: Global peak of primary update norm ||Delta M_t||_F.
        - maximum_state_norm: Global peak of memory norm ||M_t||_F.
        - minimum_stability_margin: Global minimum of contraction margin S_t = 2 - eta_t * ||k_t||^2.
        - clipping_events: Steps where step size or dynamics rate was actively clipped.
        - controller_regime_changes: Steps where stability margin transitioned across S_t = 0.

    Args:
        trajectory: Analyzed StateTrajectory.
        tau: Recovery fractional threshold (defaults to metadata['tau'] or 0.5).
        window: Sustained recovery window length (defaults to metadata['window'] or 4).

    Returns:
        List of TrajectoryEvent objects sorted by step.
    """
    events: list[TrajectoryEvent] = []
    steps = trajectory.steps
    if not steps:
        return events

    meta = trajectory.metadata or {}
    t_shift = meta.get("shift_position")
    tau_val = tau if tau is not None else meta.get("tau", 0.5)
    win_val = window if window is not None else meta.get("window", 4)

    # 1. Shift boundary event
    if t_shift is not None and 0 <= t_shift < len(steps):
        s_step = steps[t_shift]
        events.append(
            TrajectoryEvent(
                event_type="shift_boundary",
                step=t_shift,
                value=s_step.error_norm,
                context={
                    "shift_type": meta.get("shift_type", "unknown"),
                    "pre_shift_error": steps[t_shift - 1].error_norm
                    if t_shift > 0
                    else None,
                },
            )
        )

    # 2. First non-finite state
    first_nonfin = trajectory.first_nonfinite_step
    if first_nonfin is not None and 0 <= first_nonfin < len(steps):
        events.append(
            TrajectoryEvent(
                event_type="first_nonfinite_state",
                step=first_nonfin,
                value=None,
                context={"prior_step": first_nonfin - 1 if first_nonfin > 0 else 0},
            )
        )

    # Filter to finite steps for numerical extrema calculations
    finite_steps = [
        s
        for s in steps
        if s.observed
        and (s.finite_state is True)
        and (first_nonfin is None or s.step < first_nonfin)
    ]

    # 3. Maximum update norm
    valid_upds = [
        (s.step, s.update_norm)
        for s in finite_steps
        if s.update_norm is not None and math.isfinite(s.update_norm)
    ]
    if valid_upds:
        max_upd_step, max_upd_val = max(valid_upds, key=lambda x: x[1])
        events.append(
            TrajectoryEvent(
                event_type="maximum_update",
                step=max_upd_step,
                value=max_upd_val,
                context={"total_steps": len(steps)},
            )
        )

    # 4. Maximum state norm
    valid_norms = [
        (s.step, s.memory_norm)
        for s in finite_steps
        if s.memory_norm is not None and math.isfinite(s.memory_norm)
    ]
    if valid_norms:
        max_norm_step, max_norm_val = max(valid_norms, key=lambda x: x[1])
        events.append(
            TrajectoryEvent(
                event_type="maximum_state_norm",
                step=max_norm_step,
                value=max_norm_val,
                context={"initial_norm": steps[0].memory_norm},
            )
        )

    # 5. Minimum stability margin
    valid_margins = [
        (s.step, s.stability_margin)
        for s in finite_steps
        if s.stability_margin is not None and math.isfinite(s.stability_margin)
    ]
    if valid_margins:
        min_marg_step, min_marg_val = min(valid_margins, key=lambda x: x[1])
        events.append(
            TrajectoryEvent(
                event_type="minimum_stability_margin",
                step=min_marg_step,
                value=min_marg_val,
                context={"contractive": min_marg_val >= 0.0},
            )
        )

    # 6. Clipping events
    for s in finite_steps:
        if s.clip_event:
            events.append(
                TrajectoryEvent(
                    event_type="clipping_event",
                    step=s.step,
                    value=s.step_size,
                    context={"normalized_step": s.normalized_step},
                )
            )

    # 7. Controller regime changes (crossing S_t = 0 boundary)
    for i in range(1, len(valid_margins)):
        prev_step, prev_m = valid_margins[i - 1]
        curr_step, curr_m = valid_margins[i]
        if (prev_m >= 0.0 and curr_m < 0.0) or (prev_m < 0.0 and curr_m >= 0.0):
            regime = "expansive" if curr_m < 0.0 else "contractive"
            events.append(
                TrajectoryEvent(
                    event_type="controller_regime_change",
                    step=curr_step,
                    value=curr_m,
                    context={"from_margin": prev_m, "to_regime": regime},
                )
            )

    # 8. First passage and sustained recovery after shift
    if t_shift is not None and t_shift < len(steps):
        shock_step = steps[t_shift]
        shock_err = shock_step.error_norm
        if shock_err is not None and math.isfinite(shock_err) and shock_err > 0:
            target_err = tau_val * shock_err

            # First passage crossing
            first_cross_step: int | None = None
            first_cross_val: float | None = None
            for s in finite_steps:
                if s.step >= t_shift and s.error_norm is not None:
                    if s.error_norm <= target_err:
                        first_cross_step = s.step
                        first_cross_val = s.error_norm
                        break

            if first_cross_step is not None:
                events.append(
                    TrajectoryEvent(
                        event_type="first_threshold_crossing",
                        step=first_cross_step,
                        value=first_cross_val,
                        context={
                            "latency_from_shift": first_cross_step - t_shift,
                            "threshold": target_err,
                            "shock_error": shock_err,
                        },
                    )
                )

            # Sustained recovery (W consecutive steps below target_err)
            sustained_step: int | None = None
            sustained_val: float | None = None
            for i in range(len(finite_steps)):
                s = finite_steps[i]
                if s.step >= t_shift:
                    # Check window
                    window_steps = finite_steps[i : i + win_val]
                    if len(window_steps) == win_val:
                        all_below = all(
                            ws.error_norm is not None and ws.error_norm <= target_err
                            for ws in window_steps
                        )
                        if all_below:
                            sustained_step = s.step
                            sustained_val = s.error_norm
                            break

            if sustained_step is not None:
                events.append(
                    TrajectoryEvent(
                        event_type="sustained_recovery",
                        step=sustained_step,
                        value=sustained_val,
                        context={
                            "latency_from_shift": sustained_step - t_shift,
                            "window_length": win_val,
                            "threshold": target_err,
                        },
                    )
                )

    # Sort all events chronologically by step index
    events.sort(key=lambda ev: ev.step)
    return events
