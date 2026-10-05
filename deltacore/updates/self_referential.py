r"""Minimal self-referential adaptive memory system.

Implements two coupled memories:
    1. Content Memory M_t in R^{V x K}: Stores associative input-output bindings.
    2. Dynamics Memory C_t in R^{1 x Dc}: Modulates learning rates \eta_t online.

Both memories co-evolve using delta-rule update mechanics:
    \Delta M_t = \eta_t (v_t - M_t k_t) k_t^\top
    \Delta C_t = \rho_t (c_t^{target} - C_t z_t) z_t^\top
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn

from deltacore.memory.associative import AssociativeMemory
from deltacore.memory.read import read
from deltacore.stability.controllers import (
    BaseStabilityController,
    UnconstrainedController,
)
from deltacore.updates.dynamics_memory import DynamicsMemory


@dataclass(frozen=True)
class CoupledState:
    r"""Coupled state container holding content and dynamics memories."""

    content_memory: AssociativeMemory
    dynamics_memory: DynamicsMemory

    def clone(self) -> "CoupledState":
        """Create an independent copy of both memories."""
        return CoupledState(
            content_memory=self.content_memory.clone(),
            dynamics_memory=self.dynamics_memory.clone(),
        )


@dataclass(frozen=True)
class SelfReferentialStepResult:
    r"""Structured diagnostic output for a coupled self-referential update step."""

    prediction: torch.Tensor
    error: torch.Tensor
    control_features: torch.Tensor
    step_size: torch.Tensor
    content_update: torch.Tensor
    dynamics_prediction: torch.Tensor
    dynamics_error: torch.Tensor
    dynamics_update: torch.Tensor
    new_content_memory: AssociativeMemory
    new_dynamics_memory: DynamicsMemory
    new_state: CoupledState
    raw_step_size: torch.Tensor | None = None
    safe_step_size: torch.Tensor | None = None
    step_size_clipped: torch.Tensor | None = None
    raw_dynamics_rate: torch.Tensor | None = None
    safe_dynamics_rate: torch.Tensor | None = None
    dynamics_rate_clipped: torch.Tensor | None = None
    normalized_step: torch.Tensor | None = None
    stability_margin: torch.Tensor | None = None
    normalized_dynamics_rate: torch.Tensor | None = None
    dynamics_stability_margin: torch.Tensor | None = None


@dataclass(frozen=True)
class SelfReferentialScanResult:
    r"""Diagnostic output of a sequential scan over a self-referential system."""

    predictions: torch.Tensor
    final_state: CoupledState
    step_sizes: torch.Tensor
    errors: torch.Tensor
    content_update_norms: torch.Tensor
    dynamics_update_norms: torch.Tensor
    raw_step_sizes: torch.Tensor | None = None
    step_size_clips: torch.Tensor | None = None
    dynamics_rate_clips: torch.Tensor | None = None
    normalized_steps: torch.Tensor | None = None
    normalized_dynamics_rates: torch.Tensor | None = None
    stability_margins: torch.Tensor | None = None
    dynamics_stability_margins: torch.Tensor | None = None
    memory_states: list[torch.Tensor] | None = None
    dynamics_states: list[torch.Tensor] | None = None

    def __iter__(self) -> Iterator[torch.Tensor | CoupledState]:
        """Allow tuple unpacking."""
        yield self.predictions
        yield self.final_state
        yield self.step_sizes


class TargetGenerator(ABC, nn.Module):
    r"""Abstract base class for dynamics memory target generators."""

    @abstractmethod
    def forward(
        self,
        error: torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        content_memory: AssociativeMemory | torch.Tensor,
        dynamics_memory: DynamicsMemory | torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the control target $c_t^{\text{target}}$.

        Args:
            error: Prediction residual $e_t = v_t - \hat{v}_t$.
            key: Input key vector $k_t$.
            target: Target value vector $v_t$.
            content_memory: Current content memory.
            dynamics_memory: Current dynamics memory.

        Returns:
            Target response tensor $c_t^{\text{target}} \in \mathbb{R}^1$ or `[B, 1]`.
        """
        pass


class ErrorProportionalTargetGenerator(TargetGenerator):
    r"""Generates dynamics targets proportional to prediction error magnitude.

    Mathematical formulation:
        $$c_t^{\text{target}} = \alpha \cdot \|e_t\|_2 + \beta$$

    Parameters:
        alpha ($\alpha$): Scaling factor.
        beta ($\beta$): Bias / baseline target.
        eps: Epsilon preventing gradient singularity at zero error.
    """

    def __init__(
        self, alpha: float = 1.0, beta: float = 0.0, eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.eps = float(eps)

    def forward(
        self,
        error: torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        content_memory: AssociativeMemory | torch.Tensor,
        dynamics_memory: DynamicsMemory | torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.alpha.to(dtype=error.dtype, device=error.device)
        beta = self.beta.to(dtype=error.dtype, device=error.device)

        if error.ndim == 1:
            err_norm = torch.sqrt(torch.sum(error**2) + self.eps)
            # Shape [1]
            return (alpha * err_norm + beta).unsqueeze(0)
        elif error.ndim == 2:
            # Batched: [B, 1]
            err_norm = torch.sqrt(torch.sum(error**2, dim=-1, keepdim=True) + self.eps)
            return alpha * err_norm + beta
        else:
            raise ValueError(f"Unsupported error ndim={error.ndim}.")


class ControlFeatureExtractor(nn.Module):
    r"""Extracts control features $z_t \in \mathbb{R}^{D_c}$ from observable state.

    Standard features ($D_c = 3$):
        $$z_t = \begin{bmatrix} \|e_t\|_2 \\ \|M_t\|_F \\ 1.0 \end{bmatrix}$$
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    @property
    def d_c(self) -> int:
        """Control feature dimension ($D_c = 3$)."""
        return 3

    def forward(
        self,
        error: torch.Tensor,
        content_memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Extract control feature vector z_t."""
        m_data = (
            content_memory.data
            if isinstance(content_memory, AssociativeMemory)
            else content_memory
        )

        if error.ndim == 1:
            err_norm = torch.sqrt(torch.sum(error**2) + self.eps)
            mem_norm = torch.sqrt(torch.sum(m_data**2) + self.eps)
            bias = torch.tensor(1.0, dtype=error.dtype, device=error.device)
            return torch.stack([err_norm, mem_norm, bias])  # Shape [3]

        elif error.ndim == 2:
            # Batched: [B, 3]
            batch_size = error.shape[0]
            err_norm = torch.sqrt(
                torch.sum(error**2, dim=-1, keepdim=True) + self.eps
            )  # [B, 1]
            if m_data.ndim == 3:
                mem_norm = torch.sqrt(
                    torch.sum(m_data**2, dim=(-2, -1), keepdim=True) + self.eps
                )  # [B, 1]
            else:
                mem_norm = torch.sqrt(torch.sum(m_data**2) + self.eps).expand(
                    batch_size, 1
                )

            bias = torch.ones((batch_size, 1), dtype=error.dtype, device=error.device)
            return torch.cat([err_norm, mem_norm, bias], dim=-1)  # [B, 3]

        else:
            raise ValueError(f"Unsupported error ndim={error.ndim}.")


class SelfReferentialSystem(nn.Module):
    r"""Coupled self-referential adaptive memory system.

    Coordinates joint transitions of ContentMemory M_t and DynamicsMemory C_t.
    """

    def __init__(
        self,
        eta_max: float = 1.0,
        rho: float = 0.1,
        target_generator: TargetGenerator | None = None,
        feature_extractor: ControlFeatureExtractor | None = None,
        content_stability_controller: BaseStabilityController | None = None,
        dynamics_stability_controller: BaseStabilityController | None = None,
    ) -> None:
        r"""Initialize SelfReferentialSystem.

        Args:
            eta_max: Strict upper bound on content step size (\eta_t <= eta_max).
            rho: Learning rate for dynamics memory updates (\rho >= 0).
            target_generator: Component generating control targets c_t^{target}.
            feature_extractor: Component extracting control features z_t.
            content_stability_controller: Stability controller for content step size \eta_t.
            dynamics_stability_controller: Stability controller for dynamics rate \rho_t.
        """
        super().__init__()
        if eta_max <= 0.0:
            raise ValueError(f"eta_max must be positive, got {eta_max}.")
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")

        self.eta_max = float(eta_max)
        self.rho = float(rho)
        self.target_generator = (
            target_generator
            if target_generator is not None
            else ErrorProportionalTargetGenerator()
        )
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else ControlFeatureExtractor()
        )
        self.content_stability_controller = (
            content_stability_controller
            if content_stability_controller is not None
            else UnconstrainedController()
        )
        self.dynamics_stability_controller = (
            dynamics_stability_controller
            if dynamics_stability_controller is not None
            else UnconstrainedController()
        )

    def step(
        self,
        key: torch.Tensor,
        target: torch.Tensor,
        state: CoupledState,
        frozen_dynamics: bool = False,
    ) -> SelfReferentialStepResult:
        r"""Execute a single coupled self-referential update step.

        Args:
            key: Query vector `[K]` or batched `[B, K]`.
            target: Target vector `[V]` or batched `[B, V]`.
            state: CoupledState holding content_memory and dynamics_memory.
            frozen_dynamics: If True, C_t generates \eta_t but does not update (C_{t+1} = C_t).

        Returns:
            SelfReferentialStepResult containing all intermediate quantities, safety metrics, and new state.
        """
        if not isinstance(state, CoupledState):
            raise TypeError(f"state must be CoupledState, got {type(state).__name__}.")

        m_mem = state.content_memory
        c_mem = state.dynamics_memory

        m_data = m_mem.data
        c_data = c_mem.data

        # 1. Content Prediction & Residual Error
        pred = read(m_data, key)
        error = target - pred

        # 2. Control Features z_t
        z_t = self.feature_extractor(error, m_data, key)

        # 3. Dynamics Readout r_t = C_t @ z_t
        r_t = c_mem.read(z_t)

        # 4. Raw Step Size from Dynamics Readout: \eta_t^{raw} = \eta_{max} * \sigma(r_t)
        eta_raw = self.eta_max * torch.sigmoid(r_t)
        if eta_raw.ndim == 1 and eta_raw.shape[0] == 1:
            eta_raw_scalar = eta_raw[0]
        else:
            eta_raw_scalar = eta_raw.squeeze(-1)

        # 4.5. Stability Controller on Content Step Size
        step_stab_res = self.content_stability_controller.safe_step(eta_raw_scalar, key)
        eta_safe = step_stab_res.safe_value

        # 5. Content Memory Update: \Delta M_t = \eta_t^{safe} * error @ key^T
        if m_data.ndim == 2:
            content_outer = torch.outer(error, key)
            delta_m = eta_safe * content_outer
            new_m_data = m_data + delta_m
        elif m_data.ndim == 3:
            content_outer = torch.bmm(error.unsqueeze(-1), key.unsqueeze(-2))
            scale_m = eta_safe.view(-1, 1, 1) if eta_safe.ndim == 1 else eta_safe
            delta_m = scale_m * content_outer
            new_m_data = m_data + delta_m
        else:
            raise ValueError(f"Invalid content memory ndim={m_data.ndim}.")

        new_content_mem = AssociativeMemory(new_m_data)

        # 6. Dynamics Memory Target & Residual Error
        c_target = self.target_generator(error, key, target, m_mem, c_mem)
        dynamics_error = c_target - r_t  # [1] or [B, 1]

        # 6.5. Stability Controller on Dynamics Learning Rate
        rate_stab_res = self.dynamics_stability_controller.safe_rate(self.rho, z_t)
        rho_safe = rate_stab_res.safe_value

        # 7. Dynamics Memory Update: \Delta C_t = \rho^{safe} * dynamics_error @ z_t^T
        if frozen_dynamics:
            delta_c = torch.zeros_like(c_data)
            new_c_data = c_data.clone()
        else:
            if c_data.ndim == 2:
                # dynamics_error is [1], z_t is [Dc] -> outer product is [1, Dc]
                dynamics_outer = torch.outer(dynamics_error.view(-1), z_t)
                delta_c = rho_safe * dynamics_outer
                new_c_data = c_data + delta_c
            elif c_data.ndim == 3:
                # Batched: [B, 1, 1] @ [B, 1, Dc] -> [B, 1, Dc]
                dynamics_outer = torch.bmm(
                    dynamics_error.unsqueeze(-1), z_t.unsqueeze(-2)
                )
                scale_c = rho_safe.view(-1, 1, 1) if rho_safe.ndim == 1 else rho_safe
                delta_c = scale_c * dynamics_outer
                new_c_data = c_data + delta_c
            else:
                raise ValueError(f"Invalid dynamics memory ndim={c_data.ndim}.")

        new_dynamics_mem = DynamicsMemory(new_c_data)
        new_state = CoupledState(
            content_memory=new_content_mem,
            dynamics_memory=new_dynamics_mem,
        )

        return SelfReferentialStepResult(
            prediction=pred,
            error=error,
            control_features=z_t,
            step_size=eta_safe,
            content_update=delta_m,
            dynamics_prediction=r_t,
            dynamics_error=dynamics_error,
            dynamics_update=delta_c,
            new_content_memory=new_content_mem,
            new_dynamics_memory=new_dynamics_mem,
            new_state=new_state,
            raw_step_size=eta_raw_scalar,
            safe_step_size=eta_safe,
            step_size_clipped=step_stab_res.clipped,
            raw_dynamics_rate=rate_stab_res.raw_value,
            safe_dynamics_rate=rho_safe,
            dynamics_rate_clipped=rate_stab_res.clipped,
            normalized_step=step_stab_res.normalized_value,
            stability_margin=step_stab_res.stability_margin,
            normalized_dynamics_rate=rate_stab_res.normalized_value,
            dynamics_stability_margin=rate_stab_res.stability_margin,
        )

    def scan(
        self,
        keys: torch.Tensor,
        targets: torch.Tensor,
        initial_state: CoupledState | None = None,
        frozen_dynamics: bool = False,
    ) -> SelfReferentialScanResult:
        r"""Execute sequential unrolling of the self-referential memory system.

        Args:
            keys: Key sequence `[B, T, K]` or `[T, K]`.
            targets: Target sequence `[B, T, V]` or `[T, V]`.
            initial_state: Optional CoupledState. If None, initialized to zeros.
            frozen_dynamics: If True, dynamics memory state remains frozen.

        Returns:
            SelfReferentialScanResult with predictions, step sizes, errors, update norms, and stability telemetry.
        """
        is_unbatched = keys.ndim == 2

        if is_unbatched:
            seq_len, k_dim = keys.shape
            _, v_dim = targets.shape

            if initial_state is None:
                c_mem = AssociativeMemory.zeros(
                    v_dim=v_dim,
                    k_dim=k_dim,
                    dtype=keys.dtype,
                    device=keys.device,
                )
                d_mem = DynamicsMemory.zeros(
                    r_dim=1,
                    d_c=self.feature_extractor.d_c,
                    dtype=keys.dtype,
                    device=keys.device,
                )
                current_state = CoupledState(
                    content_memory=c_mem, dynamics_memory=d_mem
                )
            else:
                current_state = initial_state.clone()

            preds_list = []
            etas_list = []
            errs_list = []
            c_norms_list = []
            d_norms_list = []
            raw_etas_list = []
            step_clips_list = []
            rate_clips_list = []
            norm_steps_list = []
            norm_rates_list = []
            margins_list = []
            dyn_margins_list = []
            mem_states_list = [current_state.content_memory.data.clone()]
            dyn_states_list = [current_state.dynamics_memory.data.clone()]

            for t in range(seq_len):
                kt = keys[t]
                vt = targets[t]

                res = self.step(
                    kt, vt, state=current_state, frozen_dynamics=frozen_dynamics
                )

                preds_list.append(res.prediction)
                errs_list.append(res.error)
                etas_list.append(res.step_size)
                c_norms_list.append(torch.linalg.norm(res.content_update))
                d_norms_list.append(torch.linalg.norm(res.dynamics_update))
                if res.raw_step_size is not None:
                    raw_etas_list.append(res.raw_step_size)
                if res.step_size_clipped is not None:
                    step_clips_list.append(res.step_size_clipped)
                if res.dynamics_rate_clipped is not None:
                    rate_clips_list.append(res.dynamics_rate_clipped)
                if res.normalized_step is not None:
                    norm_steps_list.append(res.normalized_step)
                if res.normalized_dynamics_rate is not None:
                    norm_rates_list.append(res.normalized_dynamics_rate)
                if res.stability_margin is not None:
                    margins_list.append(res.stability_margin)
                if res.dynamics_stability_margin is not None:
                    dyn_margins_list.append(res.dynamics_stability_margin)

                current_state = res.new_state
                mem_states_list.append(current_state.content_memory.data.clone())
                dyn_states_list.append(current_state.dynamics_memory.data.clone())

            if seq_len > 0:
                stacked_preds = torch.stack(preds_list, dim=0)
                stacked_etas = torch.stack(etas_list, dim=0)
                stacked_errs = torch.stack(errs_list, dim=0)
                stacked_cnorms = torch.stack(c_norms_list, dim=0)
                stacked_dnorms = torch.stack(d_norms_list, dim=0)
                stacked_raw_etas = (
                    torch.stack(raw_etas_list, dim=0) if raw_etas_list else None
                )
                stacked_step_clips = (
                    torch.stack(step_clips_list, dim=0) if step_clips_list else None
                )
                stacked_rate_clips = (
                    torch.stack(rate_clips_list, dim=0) if rate_clips_list else None
                )
                stacked_norm_steps = (
                    torch.stack(norm_steps_list, dim=0) if norm_steps_list else None
                )
                stacked_norm_rates = (
                    torch.stack(norm_rates_list, dim=0) if norm_rates_list else None
                )
                stacked_margins = (
                    torch.stack(margins_list, dim=0) if margins_list else None
                )
                stacked_dyn_margins = (
                    torch.stack(dyn_margins_list, dim=0) if dyn_margins_list else None
                )
            else:
                stacked_preds = torch.empty(
                    (0, v_dim), dtype=keys.dtype, device=keys.device
                )
                stacked_etas = torch.empty((0,), dtype=keys.dtype, device=keys.device)
                stacked_errs = torch.empty(
                    (0, v_dim), dtype=keys.dtype, device=keys.device
                )
                stacked_cnorms = torch.empty((0,), dtype=keys.dtype, device=keys.device)
                stacked_dnorms = torch.empty((0,), dtype=keys.dtype, device=keys.device)
                stacked_raw_etas = None
                stacked_step_clips = None
                stacked_rate_clips = None
                stacked_norm_steps = None
                stacked_norm_rates = None
                stacked_margins = None
                stacked_dyn_margins = None

            return SelfReferentialScanResult(
                predictions=stacked_preds,
                final_state=current_state,
                step_sizes=stacked_etas,
                errors=stacked_errs,
                content_update_norms=stacked_cnorms,
                dynamics_update_norms=stacked_dnorms,
                raw_step_sizes=stacked_raw_etas,
                step_size_clips=stacked_step_clips,
                dynamics_rate_clips=stacked_rate_clips,
                normalized_steps=stacked_norm_steps,
                normalized_dynamics_rates=stacked_norm_rates,
                stability_margins=stacked_margins,
                dynamics_stability_margins=stacked_dyn_margins,
                memory_states=mem_states_list,
                dynamics_states=dyn_states_list,
            )

        # Batched case: keys is [B, T, K], targets is [B, T, V]
        batch_size, seq_len, k_dim = keys.shape
        _, _, v_dim = targets.shape

        if initial_state is None:
            c_mem = AssociativeMemory.zeros(
                v_dim=v_dim,
                k_dim=k_dim,
                dtype=keys.dtype,
                device=keys.device,
                batch_size=batch_size,
            )
            d_mem = DynamicsMemory.zeros(
                r_dim=1,
                d_c=self.feature_extractor.d_c,
                dtype=keys.dtype,
                device=keys.device,
                batch_size=batch_size,
            )
            current_state = CoupledState(content_memory=c_mem, dynamics_memory=d_mem)
        else:
            m_data = initial_state.content_memory.data
            c_data = initial_state.dynamics_memory.data
            if m_data.ndim == 2:
                m_batched = m_data.unsqueeze(0).expand(batch_size, -1, -1).clone()
            else:
                m_batched = m_data.clone()

            if c_data.ndim == 2:
                c_batched = c_data.unsqueeze(0).expand(batch_size, -1, -1).clone()
            else:
                c_batched = c_data.clone()

            current_state = CoupledState(
                content_memory=AssociativeMemory(m_batched),
                dynamics_memory=DynamicsMemory(c_batched),
            )

        preds_list = []
        etas_list = []
        errs_list = []
        c_norms_list = []
        d_norms_list = []
        raw_etas_list = []
        step_clips_list = []
        rate_clips_list = []
        norm_steps_list = []
        norm_rates_list = []

        for t in range(seq_len):
            kt = keys[:, t, :]
            vt = targets[:, t, :]

            res = self.step(
                kt, vt, state=current_state, frozen_dynamics=frozen_dynamics
            )

            preds_list.append(res.prediction)
            errs_list.append(res.error)
            etas_list.append(res.step_size)
            c_norms_list.append(torch.linalg.norm(res.content_update, dim=(-2, -1)))
            d_norms_list.append(torch.linalg.norm(res.dynamics_update, dim=(-2, -1)))
            if res.raw_step_size is not None:
                raw_etas_list.append(res.raw_step_size)
            if res.step_size_clipped is not None:
                step_clips_list.append(res.step_size_clipped)
            if res.dynamics_rate_clipped is not None:
                rate_clips_list.append(res.dynamics_rate_clipped)
            if res.normalized_step is not None:
                norm_steps_list.append(res.normalized_step)
            if res.normalized_dynamics_rate is not None:
                norm_rates_list.append(res.normalized_dynamics_rate)

            current_state = res.new_state

        if seq_len > 0:
            stacked_preds = torch.stack(preds_list, dim=1)
            stacked_etas = torch.stack(etas_list, dim=1)
            stacked_errs = torch.stack(errs_list, dim=1)
            stacked_cnorms = torch.stack(c_norms_list, dim=1)
            stacked_dnorms = torch.stack(d_norms_list, dim=1)
            stacked_raw_etas = (
                torch.stack(raw_etas_list, dim=1) if raw_etas_list else None
            )
            stacked_step_clips = (
                torch.stack(step_clips_list, dim=1) if step_clips_list else None
            )
            stacked_rate_clips = (
                torch.stack(rate_clips_list, dim=1) if rate_clips_list else None
            )
            stacked_norm_steps = (
                torch.stack(norm_steps_list, dim=1) if norm_steps_list else None
            )
            stacked_norm_rates = (
                torch.stack(norm_rates_list, dim=1) if norm_rates_list else None
            )
        else:
            stacked_preds = torch.empty(
                (batch_size, 0, v_dim), dtype=keys.dtype, device=keys.device
            )
            stacked_etas = torch.empty(
                (batch_size, 0), dtype=keys.dtype, device=keys.device
            )
            stacked_errs = torch.empty(
                (batch_size, 0, v_dim), dtype=keys.dtype, device=keys.device
            )
            stacked_cnorms = torch.empty(
                (batch_size, 0), dtype=keys.dtype, device=keys.device
            )
            stacked_dnorms = torch.empty(
                (batch_size, 0), dtype=keys.dtype, device=keys.device
            )
            stacked_raw_etas = None
            stacked_step_clips = None
            stacked_rate_clips = None
            stacked_norm_steps = None
            stacked_norm_rates = None

        return SelfReferentialScanResult(
            predictions=stacked_preds,
            final_state=current_state,
            step_sizes=stacked_etas,
            errors=stacked_errs,
            content_update_norms=stacked_cnorms,
            dynamics_update_norms=stacked_dnorms,
            raw_step_sizes=stacked_raw_etas,
            step_size_clips=stacked_step_clips,
            dynamics_rate_clips=stacked_rate_clips,
            normalized_steps=stacked_norm_steps,
            normalized_dynamics_rates=stacked_norm_rates,
            stability_margins=None,
            dynamics_stability_margins=None,
            memory_states=None,
            dynamics_states=None,
        )
