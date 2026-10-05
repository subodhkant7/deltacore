# ==============================================================================
# DeltaCore: deltacore/benchmarks/baselines/wrappers.py
# Standardized baseline wrappers for DeltaCore benchmark evaluation.
# ==============================================================================

import torch

from deltacore.benchmarks.baselines.base import (
    BaseBaseline,
    BaselineTrajectoryResult,
)
from deltacore.memory.associative import AssociativeMemory
from deltacore.stability.controllers import (
    SafeDynamicsRateController,
    SafeStepSizeController,
    UnconstrainedController,
)
from deltacore.updates.adaptive_delta import AdaptiveDeltaRule
from deltacore.updates.controllers import ErrorConditionedStepSize
from deltacore.updates.delta import DeltaRule
from deltacore.updates.dynamics_memory import DynamicsMemory
from deltacore.updates.self_referential import (
    CoupledState,
    SelfReferentialSystem,
)


def _check_finiteness(tensor: torch.Tensor) -> bool:
    """Check if tensor elements and Frobenius norm are strictly finite."""
    return bool(
        torch.isfinite(tensor).all().item()
        and torch.isfinite(torch.linalg.norm(tensor.float())).item()
    )


class FrozenBaseline(BaseBaseline):
    """Baseline A: Frozen associative memory with zero test-time adaptation."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            name="Frozen",
            key_dim=key_dim,
            value_dim=value_dim,
            dtype=dtype,
            device=device,
        )

    def reset(self, initial_memory: torch.Tensor | None = None) -> None:
        pass

    def run_sequence(
        self,
        keys: torch.Tensor,
        targets: torch.Tensor,
        initial_memory: torch.Tensor | None = None,
    ) -> BaselineTrajectoryResult:
        t_steps = keys.shape[0]
        if initial_memory is None:
            m = torch.zeros(
                (self.value_dim, self.key_dim),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            m = initial_memory.to(dtype=self.dtype, device=self.device).clone()

        preds = []
        errors = []
        m_norm = float(torch.linalg.norm(m).item())
        mem_norms = [m_norm] * t_steps
        upd_norms = [0.0] * t_steps
        step_sizes = [0.0] * t_steps
        stab_margins = [2.0] * t_steps
        norm_steps = [0.0] * t_steps

        all_finite = True
        first_nonfinite = None

        for t in range(t_steps):
            k = keys[t]
            v = targets[t]
            pred = m @ k
            err = v - pred
            preds.append(pred)
            errors.append(err)

            if not (_check_finiteness(pred) and _check_finiteness(err)):
                all_finite = False
                if first_nonfinite is None:
                    first_nonfinite = t

        preds_t = torch.stack(preds, dim=0)
        errors_t = torch.stack(errors, dim=0)
        final_err = float(torch.linalg.norm(errors_t[-1]).item())

        return BaselineTrajectoryResult(
            predictions=preds_t,
            errors=errors_t,
            final_memory=m,
            memory_norms=mem_norms,
            update_norms=upd_norms,
            step_sizes=step_sizes,
            stability_margins=stab_margins,
            normalized_steps=norm_steps,
            clip_count=0,
            all_states_finite=all_finite,
            terminal_state_finite=_check_finiteness(m),
            first_nonfinite_step=first_nonfinite,
            max_state_norm=m_norm,
            max_update_norm=0.0,
            final_error_norm=final_err,
        )


class FixedDeltaBaseline(BaseBaseline):
    """Baseline B: Fixed Delta-rule associative memory (Phase 1)."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        step_size: float = 0.1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            name="FixedDelta",
            key_dim=key_dim,
            value_dim=value_dim,
            dtype=dtype,
            device=device,
        )
        self.step_size = step_size
        self.rule = DeltaRule(step_size=step_size)

    def reset(self, initial_memory: torch.Tensor | None = None) -> None:
        pass

    def run_sequence(
        self,
        keys: torch.Tensor,
        targets: torch.Tensor,
        initial_memory: torch.Tensor | None = None,
    ) -> BaselineTrajectoryResult:
        t_steps = keys.shape[0]
        if initial_memory is None:
            m = torch.zeros(
                (self.value_dim, self.key_dim),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            m = initial_memory.to(dtype=self.dtype, device=self.device).clone()

        preds = []
        errors = []
        mem_norms = []
        upd_norms = []
        step_sizes = []
        stab_margins = []
        norm_steps = []

        all_finite = True
        first_nonfinite = None

        for t in range(t_steps):
            k = keys[t]
            v = targets[t]

            pred = m @ k
            err = v - pred
            preds.append(pred)
            errors.append(err)

            delta = self.step_size * torch.outer(err, k)
            m = m + delta

            m_finite = _check_finiteness(m)
            p_finite = _check_finiteness(pred)
            if not (m_finite and p_finite):
                all_finite = False
                if first_nonfinite is None:
                    first_nonfinite = t

            k_norm_sq = float((k @ k).item())
            gamma = self.step_size * k_norm_sq
            margin = 2.0 - gamma

            upd_norms.append(float(torch.linalg.norm(delta).item()))
            mem_norms.append(float(torch.linalg.norm(m).item()))
            step_sizes.append(self.step_size)
            norm_steps.append(gamma)
            stab_margins.append(margin)

        preds_t = torch.stack(preds, dim=0)
        errors_t = torch.stack(errors, dim=0)
        final_err = float(torch.linalg.norm(errors_t[-1]).item())

        return BaselineTrajectoryResult(
            predictions=preds_t,
            errors=errors_t,
            final_memory=m,
            memory_norms=mem_norms,
            update_norms=upd_norms,
            step_sizes=step_sizes,
            stability_margins=stab_margins,
            normalized_steps=norm_steps,
            clip_count=0,
            all_states_finite=all_finite,
            terminal_state_finite=_check_finiteness(m),
            first_nonfinite_step=first_nonfinite,
            max_state_norm=max(mem_norms) if mem_norms else 0.0,
            max_update_norm=max(upd_norms) if upd_norms else 0.0,
            final_error_norm=final_err,
        )


class AdaptiveDeltaBaseline(BaseBaseline):
    """Baseline C: Adaptive Delta-rule with dynamic step size (Phase 2)."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        max_step_size: float = 0.5,
        alpha: float = 1.0,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            name="AdaptiveDelta",
            key_dim=key_dim,
            value_dim=value_dim,
            dtype=dtype,
            device=device,
        )
        self.controller = ErrorConditionedStepSize(eta_max=max_step_size, scale=alpha)
        self.rule = AdaptiveDeltaRule(controller=self.controller)

    def reset(self, initial_memory: torch.Tensor | None = None) -> None:
        pass

    def run_sequence(
        self,
        keys: torch.Tensor,
        targets: torch.Tensor,
        initial_memory: torch.Tensor | None = None,
    ) -> BaselineTrajectoryResult:
        t_steps = keys.shape[0]
        if initial_memory is None:
            m = torch.zeros(
                (self.value_dim, self.key_dim),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            m = initial_memory.to(dtype=self.dtype, device=self.device).clone()

        preds = []
        errors = []
        mem_norms = []
        upd_norms = []
        step_sizes = []
        stab_margins = []
        norm_steps = []

        all_finite = True
        first_nonfinite = None

        for t in range(t_steps):
            k = keys[t]
            v = targets[t]

            step_res = self.rule.step(m, k, v)
            pred = step_res.prediction
            err = step_res.error
            delta = step_res.update
            eta_val = float(step_res.step_size.item())
            m = step_res.new_memory
            if isinstance(m, AssociativeMemory):
                m = m.data

            preds.append(pred)
            errors.append(err)

            m_finite = _check_finiteness(m)
            p_finite = _check_finiteness(pred)
            if not (m_finite and p_finite):
                all_finite = False
                if first_nonfinite is None:
                    first_nonfinite = t

            k_norm_sq = float((k @ k).item())
            gamma = eta_val * k_norm_sq
            margin = 2.0 - gamma

            upd_norms.append(float(torch.linalg.norm(delta).item()))
            mem_norms.append(float(torch.linalg.norm(m).item()))
            step_sizes.append(eta_val)
            norm_steps.append(gamma)
            stab_margins.append(margin)

        preds_t = torch.stack(preds, dim=0)
        errors_t = torch.stack(errors, dim=0)
        final_err = float(torch.linalg.norm(errors_t[-1]).item())

        return BaselineTrajectoryResult(
            predictions=preds_t,
            errors=errors_t,
            final_memory=m,
            memory_norms=mem_norms,
            update_norms=upd_norms,
            step_sizes=step_sizes,
            stability_margins=stab_margins,
            normalized_steps=norm_steps,
            clip_count=0,
            all_states_finite=all_finite,
            terminal_state_finite=_check_finiteness(m),
            first_nonfinite_step=first_nonfinite,
            max_state_norm=max(mem_norms) if mem_norms else 0.0,
            max_update_norm=max(upd_norms) if upd_norms else 0.0,
            final_error_norm=final_err,
        )


class SelfReferentialBaseline(BaseBaseline):
    """Baseline D: Unconstrained coupled self-referential memory (Phase 3)."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        dynamics_dim: int = 3,
        eta_max: float = 0.5,
        rho_max: float = 0.2,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            name="SelfReferential",
            key_dim=key_dim,
            value_dim=value_dim,
            dtype=dtype,
            device=device,
        )
        self.dynamics_dim = dynamics_dim
        self.system = SelfReferentialSystem(
            eta_max=eta_max,
            rho=rho_max,
            content_stability_controller=UnconstrainedController(),
            dynamics_stability_controller=UnconstrainedController(),
        )

    def reset(self, initial_memory: torch.Tensor | None = None) -> None:
        pass

    def run_sequence(
        self,
        keys: torch.Tensor,
        targets: torch.Tensor,
        initial_memory: torch.Tensor | None = None,
    ) -> BaselineTrajectoryResult:
        keys_dev = keys.to(device=self.device, dtype=self.dtype)
        targets_dev = targets.to(device=self.device, dtype=self.dtype)

        if initial_memory is None:
            m_init = AssociativeMemory.zeros(
                self.value_dim,
                self.key_dim,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            m_init = AssociativeMemory(
                initial_memory.to(device=self.device, dtype=self.dtype).clone()
            )

        c_init = DynamicsMemory.zeros(
            1, self.dynamics_dim, dtype=self.dtype, device=self.device
        )
        init_state = CoupledState(content_memory=m_init, dynamics_memory=c_init)

        scan_res = self.system.scan(keys_dev, targets_dev, initial_state=init_state)

        preds = scan_res.predictions
        errors = targets_dev - preds

        mem_states = scan_res.memory_states or []
        dyn_states = scan_res.dynamics_states or []
        mem_norms = [float(torch.linalg.norm(m).item()) for m in mem_states]
        upd_norms = [float(u.item()) for u in scan_res.content_update_norms] or [
            0.0
        ] * len(keys_dev)
        step_sizes = [float(s.item()) for s in scan_res.step_sizes] or [0.0] * len(
            keys_dev
        )

        if scan_res.normalized_steps is not None and len(scan_res.normalized_steps) > 0:
            norm_steps = [float(n.item()) for n in scan_res.normalized_steps]
        else:
            norm_steps = [0.0] * len(keys_dev)

        if (
            scan_res.stability_margins is not None
            and len(scan_res.stability_margins) > 0
        ):
            stab_margins = [float(m.item()) for m in scan_res.stability_margins]
        else:
            stab_margins = [2.0 - ns for ns in norm_steps]

        all_finite = True
        first_nonfinite = None
        for idx in range(len(mem_states)):
            m_fin = _check_finiteness(mem_states[idx])
            c_fin = (
                _check_finiteness(dyn_states[idx]) if idx < len(dyn_states) else True
            )
            p_fin = _check_finiteness(preds[idx]) if idx < len(preds) else True
            if not (m_fin and c_fin and p_fin):
                all_finite = False
                if first_nonfinite is None:
                    first_nonfinite = idx

        final_m = (
            mem_states[-1] if len(mem_states) > 0 else init_state.content_memory.data
        )
        final_err = (
            float(torch.linalg.norm(errors[-1]).item()) if len(errors) > 0 else 0.0
        )

        return BaselineTrajectoryResult(
            predictions=preds,
            errors=errors,
            final_memory=final_m,
            memory_norms=mem_norms,
            update_norms=upd_norms,
            step_sizes=step_sizes,
            stability_margins=stab_margins,
            normalized_steps=norm_steps,
            clip_count=0,
            all_states_finite=all_finite,
            terminal_state_finite=_check_finiteness(final_m),
            first_nonfinite_step=first_nonfinite,
            max_state_norm=max(mem_norms) if mem_norms else 0.0,
            max_update_norm=max(upd_norms) if upd_norms else 0.0,
            final_error_norm=final_err,
        )


class SafeSelfReferentialBaseline(BaseBaseline):
    """Baseline E: Stability-controlled self-referential memory (Phase 4)."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        dynamics_dim: int = 3,
        eta_max: float = 0.5,
        rho_max: float = 0.2,
        beta: float = 1.0,
        beta_c: float = 1.0,
        epsilon: float = 1e-4,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            name="SafeSelfReferential",
            key_dim=key_dim,
            value_dim=value_dim,
            dtype=dtype,
            device=device,
        )
        self.dynamics_dim = dynamics_dim
        self.system = SelfReferentialSystem(
            eta_max=eta_max,
            rho=rho_max,
            content_stability_controller=SafeStepSizeController(beta=beta, eps=epsilon),
            dynamics_stability_controller=SafeDynamicsRateController(
                beta_c=beta_c, eps=epsilon
            ),
        )

    def reset(self, initial_memory: torch.Tensor | None = None) -> None:
        pass

    def run_sequence(
        self,
        keys: torch.Tensor,
        targets: torch.Tensor,
        initial_memory: torch.Tensor | None = None,
    ) -> BaselineTrajectoryResult:
        keys_dev = keys.to(device=self.device, dtype=self.dtype)
        targets_dev = targets.to(device=self.device, dtype=self.dtype)

        if initial_memory is None:
            m_init = AssociativeMemory.zeros(
                self.value_dim,
                self.key_dim,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            m_init = AssociativeMemory(
                initial_memory.to(device=self.device, dtype=self.dtype).clone()
            )

        c_init = DynamicsMemory.zeros(
            1, self.dynamics_dim, dtype=self.dtype, device=self.device
        )
        init_state = CoupledState(content_memory=m_init, dynamics_memory=c_init)

        scan_res = self.system.scan(keys_dev, targets_dev, initial_state=init_state)

        preds = scan_res.predictions
        errors = targets_dev - preds

        mem_states = scan_res.memory_states or []
        dyn_states = scan_res.dynamics_states or []
        mem_norms = [float(torch.linalg.norm(m).item()) for m in mem_states]
        upd_norms = [float(u.item()) for u in scan_res.content_update_norms] or [
            0.0
        ] * len(keys_dev)
        step_sizes = [float(s.item()) for s in scan_res.step_sizes] or [0.0] * len(
            keys_dev
        )

        if scan_res.normalized_steps is not None and len(scan_res.normalized_steps) > 0:
            norm_steps = [float(n.item()) for n in scan_res.normalized_steps]
        else:
            norm_steps = [0.0] * len(keys_dev)

        if (
            scan_res.stability_margins is not None
            and len(scan_res.stability_margins) > 0
        ):
            stab_margins = [float(m.item()) for m in scan_res.stability_margins]
        else:
            stab_margins = [2.0 - ns for ns in norm_steps]

        clips = 0
        if scan_res.step_size_clips is not None:
            clips += int(scan_res.step_size_clips.sum().item())
        if scan_res.dynamics_rate_clips is not None:
            clips += int(scan_res.dynamics_rate_clips.sum().item())

        all_finite = True
        first_nonfinite = None
        for idx in range(len(mem_states)):
            m_fin = _check_finiteness(mem_states[idx])
            c_fin = (
                _check_finiteness(dyn_states[idx]) if idx < len(dyn_states) else True
            )
            p_fin = _check_finiteness(preds[idx]) if idx < len(preds) else True
            if not (m_fin and c_fin and p_fin):
                all_finite = False
                if first_nonfinite is None:
                    first_nonfinite = idx

        final_m = (
            mem_states[-1] if len(mem_states) > 0 else init_state.content_memory.data
        )
        final_err = (
            float(torch.linalg.norm(errors[-1]).item()) if len(errors) > 0 else 0.0
        )

        return BaselineTrajectoryResult(
            predictions=preds,
            errors=errors,
            final_memory=final_m,
            memory_norms=mem_norms,
            update_norms=upd_norms,
            step_sizes=step_sizes,
            stability_margins=stab_margins,
            normalized_steps=norm_steps,
            clip_count=clips,
            all_states_finite=all_finite,
            terminal_state_finite=_check_finiteness(final_m),
            first_nonfinite_step=first_nonfinite,
            max_state_norm=max(mem_norms) if mem_norms else 0.0,
            max_update_norm=max(upd_norms) if upd_norms else 0.0,
            final_error_norm=final_err,
        )
