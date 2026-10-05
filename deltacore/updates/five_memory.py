# ==============================================================================
# DeltaCore: deltacore/updates/five_memory.py
# Five-memory self-modifying reference learner.
# ==============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from deltacore.memory.five_memory import FiveMemoryState


@dataclass(frozen=True)
class FiveMemoryConfig:
    r"""Configuration hyperparameters for the FiveMemorySystem.

    Attributes:
        v_dim: Target value dimension V.
        k_dim: Query key dimension K.
        in_dim: Input token dimension D_in.
        feat_dim: Feature dimension D_feat for meta-controllers (default: in_dim).
        lr_dim: Hidden projection dimension for learning-rate memory (default: 1).
        ret_dim: Hidden projection dimension for retention memory (default: 1).
        eta_max: Maximum raw learning rate scalar under soft injection cap.
        eta_bias: Bias offset for learning rate squashing.
        ret_min: Minimum retention factor.
        ret_bias: Bias offset for retention squashing.
        eta_key: Adaptation rate for key generation memory.
        lambda_key: Retention decay for key generation memory.
        eta_val: Adaptation rate for value generation memory.
        lambda_val: Retention decay for value generation memory.
        rho_eta: Update rate for learning-rate controller memory.
        lambda_eta: Retention decay for learning-rate controller memory.
        tau_eta: Error reference threshold for learning rate acceleration.
        rho_ret: Update rate for retention controller memory.
        lambda_ret: Retention decay for retention controller memory.
        tau_ret: Error reference threshold for retention shock decrescendo.
        beta: Phase 4 contraction bound (beta in (0, 2), default: 1.9).
        epsilon: Numerical denominator stabilizer.
        apply_stability_control: If True, clamps eta_t <= beta / (||k_t||^2 + eps).
    """

    v_dim: int
    k_dim: int
    in_dim: int
    feat_dim: int | None = None
    lr_dim: int = 1
    ret_dim: int = 1
    eta_max: float = 1.0
    eta_bias: float = 0.0
    ret_min: float = 0.0
    ret_bias: float = 2.0
    eta_key: float = 0.01
    lambda_key: float = 1.0
    eta_val: float = 0.01
    lambda_val: float = 1.0
    rho_eta: float = 0.05
    lambda_eta: float = 0.99
    tau_eta: float = 0.5
    rho_ret: float = 0.05
    lambda_ret: float = 0.99
    tau_ret: float = 0.5
    beta: float = 1.9
    epsilon: float = 1e-6
    apply_stability_control: bool = True

    def get_feat_dim(self) -> int:
        return self.in_dim if self.feat_dim is None else self.feat_dim


@dataclass(frozen=True)
class FiveMemoryStepResult:
    r"""Diagnostic telemetry and transition results for a single five-memory step.

    Exposes all intermediate representations, signals, and update tensors for full observability.
    """

    input: torch.Tensor
    key_representation: torch.Tensor
    value_representation: torch.Tensor
    content_prediction: torch.Tensor
    prediction_error: torch.Tensor
    raw_learning_rate: torch.Tensor
    safe_learning_rate: torch.Tensor
    raw_retention: torch.Tensor
    safe_retention: torch.Tensor
    content_update: torch.Tensor
    key_memory_update: torch.Tensor
    value_memory_update: torch.Tensor
    learning_rate_update: torch.Tensor
    retention_update: torch.Tensor
    new_state: FiveMemoryState
    normalized_step: torch.Tensor | None = None
    stability_margin: torch.Tensor | None = None
    clip_event: torch.Tensor | None = None


@dataclass(frozen=True)
class FiveMemoryScanResult:
    r"""Sequential trajectory output for FiveMemorySystem.scan()."""

    predictions: torch.Tensor
    final_state: FiveMemoryState
    step_results: list[FiveMemoryStepResult]
    errors: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    raw_learning_rates: torch.Tensor
    safe_learning_rates: torch.Tensor
    raw_retentions: torch.Tensor
    safe_retentions: torch.Tensor
    content_memory_norms: torch.Tensor
    key_memory_norms: torch.Tensor
    value_memory_norms: torch.Tensor
    learning_rate_memory_norms: torch.Tensor
    retention_memory_norms: torch.Tensor
    content_update_norms: torch.Tensor
    key_update_norms: torch.Tensor
    value_update_norms: torch.Tensor
    learning_rate_update_norms: torch.Tensor
    retention_update_norms: torch.Tensor
    stability_margins: torch.Tensor
    normalized_steps: torch.Tensor
    clip_events: torch.Tensor


class FiveMemorySystem(nn.Module):
    r"""Reference implementation of the five-memory self-modifying learning system.

    Implements co-evolution of:
        M_content, M_key, M_val, M_eta, M_ret

    Supports both unbatched [D_in] and batched [B, D_in] step evaluations,
    and sequence scans over [B, T, D_in] or [T, D_in].
    """

    def __init__(
        self,
        config: FiveMemoryConfig,
        trainable_dynamics: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.trainable_dynamics = trainable_dynamics

        if self.trainable_dynamics:
            self.eta_key = nn.Parameter(
                torch.full((config.k_dim, 1), float(config.eta_key))
            )
            self.lambda_key = nn.Parameter(
                torch.full((config.k_dim, 1), float(config.lambda_key))
            )
            self.eta_val = nn.Parameter(
                torch.full((config.v_dim, 1), float(config.eta_val))
            )
            self.lambda_val = nn.Parameter(
                torch.full((config.v_dim, 1), float(config.lambda_val))
            )
            self.rho_eta = nn.Parameter(
                torch.full((config.lr_dim, 1), float(config.rho_eta))
            )
            self.lambda_eta = nn.Parameter(
                torch.full((config.lr_dim, 1), float(config.lambda_eta))
            )
            self.rho_ret = nn.Parameter(
                torch.full((config.ret_dim, 1), float(config.rho_ret))
            )
            self.lambda_ret = nn.Parameter(
                torch.full((config.ret_dim, 1), float(config.lambda_ret))
            )
            self.tau_eta = nn.Parameter(torch.tensor(float(config.tau_eta)))
            self.tau_ret = nn.Parameter(torch.tensor(float(config.tau_ret)))
            self.eta_bias = nn.Parameter(torch.tensor(float(config.eta_bias)))
            self.ret_bias = nn.Parameter(torch.tensor(float(config.ret_bias)))

    def step(
        self,
        x: torch.Tensor,
        state: FiveMemoryState,
        target: torch.Tensor | None = None,
    ) -> FiveMemoryStepResult:
        r"""Compute a single forward read, error evaluation, and coupled five-memory state transition.

        Args:
            x: Input token representation [D_in] or batched [B, D_in].
            state: Current FiveMemoryState.
            target: Optional target value vector [V] or [B, V]. If None, target is generated by M_val.

        Returns:
            FiveMemoryStepResult containing next state and comprehensive diagnostic tensors.
        """
        is_batched = state.is_batched
        if is_batched:
            if (
                x.ndim != 2
                or x.shape[0] != state.batch_size
                or x.shape[-1] != state.in_dim
            ):
                raise ValueError(
                    f"Batched input must have shape [{state.batch_size}, {state.in_dim}], got {list(x.shape)}."
                )
        else:
            if x.ndim != 1 or x.shape[-1] != state.in_dim:
                raise ValueError(
                    f"Unbatched input must have shape [{state.in_dim}], got {list(x.shape)}."
                )

        cfg = self.config

        # Resolve dynamics quantities (trainable vs static configuration)
        if self.trainable_dynamics:
            eta_bias = self.eta_bias.to(dtype=x.dtype, device=x.device)
            ret_bias = self.ret_bias.to(dtype=x.dtype, device=x.device)
            lam_k = self.lambda_key.to(dtype=state.key.dtype, device=state.key.device)
            eta_k = self.eta_key.to(dtype=state.key.dtype, device=state.key.device)
            lam_v = self.lambda_val.to(
                dtype=state.value.dtype, device=state.value.device
            )
            eta_v = self.eta_val.to(dtype=state.value.dtype, device=state.value.device)
            lam_e = self.lambda_eta.to(
                dtype=state.learning_rate.dtype, device=state.learning_rate.device
            )
            rho_e = self.rho_eta.to(
                dtype=state.learning_rate.dtype, device=state.learning_rate.device
            )
            tau_e = self.tau_eta.to(dtype=x.dtype, device=x.device)
            lam_r = self.lambda_ret.to(
                dtype=state.retention.dtype, device=state.retention.device
            )
            rho_r = self.rho_ret.to(
                dtype=state.retention.dtype, device=state.retention.device
            )
            tau_r = self.tau_ret.to(dtype=x.dtype, device=x.device)
        else:
            eta_bias = cfg.eta_bias
            ret_bias = cfg.ret_bias
            lam_k = cfg.lambda_key
            eta_k = cfg.eta_key
            lam_v = cfg.lambda_val
            eta_v = cfg.eta_val
            lam_e = cfg.lambda_eta
            rho_e = cfg.rho_eta
            tau_e = cfg.tau_eta
            lam_r = cfg.lambda_ret
            rho_r = cfg.rho_ret
            tau_r = cfg.tau_ret

        # 1. Key Generation: k_t = M_key * x_t
        if is_batched:
            k_t = torch.bmm(state.key, x.unsqueeze(-1)).squeeze(-1)  # [B, K]
        else:
            k_t = torch.matmul(state.key, x)  # [K]

        # 2. Value Generation: v_t = M_val * x_t
        if is_batched:
            v_gen = torch.bmm(state.value, x.unsqueeze(-1)).squeeze(-1)  # [B, V]
        else:
            v_gen = torch.matmul(state.value, x)  # [V]

        # Effective target for content memory error: external target or generated value
        v_eff = target if target is not None else v_gen

        # 3. Content Memory Read: v_hat_t = M_content * k_t
        if is_batched:
            v_hat = torch.bmm(state.content, k_t.unsqueeze(-1)).squeeze(-1)  # [B, V]
        else:
            v_hat = torch.matmul(state.content, k_t)  # [V]

        # Prediction Error: e_t = v_eff - v_hat
        e_t = v_eff - v_hat

        # 4. Meta-controller query features z_t
        # Default: input token representation x
        z_t = x

        # 5. Learning-Rate Read & Stability Contraction:
        # u_eta = M_eta * z_t
        if is_batched:
            u_eta = torch.bmm(state.learning_rate, z_t.unsqueeze(-1)).squeeze(
                -1
            )  # [B, D_lr]
            sum_eta = torch.sum(u_eta, dim=-1, keepdim=True) / math.sqrt(
                max(1, state.lr_dim)
            )
            raw_eta = cfg.eta_max * torch.sigmoid(sum_eta + eta_bias)  # [B, 1]
            k_sq_norm = torch.sum(k_t * k_t, dim=-1, keepdim=True)  # [B, 1]
        else:
            u_eta = torch.matmul(state.learning_rate, z_t)  # [D_lr]
            sum_eta = torch.sum(u_eta) / math.sqrt(max(1, state.lr_dim))
            raw_eta = cfg.eta_max * torch.sigmoid(sum_eta + eta_bias)  # scalar
            k_sq_norm = torch.sum(k_t * k_t)  # scalar

        # Safe Learning Rate via Phase 4 Contraction Controller
        if cfg.apply_stability_control:
            eta_bound = cfg.beta / (k_sq_norm + cfg.epsilon)
            safe_eta = torch.minimum(raw_eta, eta_bound)
            clip_event = raw_eta > eta_bound
        else:
            safe_eta = raw_eta
            clip_event = torch.zeros_like(raw_eta, dtype=torch.bool)

        norm_step = safe_eta * k_sq_norm
        stab_margin = 2.0 - norm_step

        # 6. Retention Read & Spectral Clamp:
        # u_ret = M_ret * z_t
        if is_batched:
            u_ret = torch.bmm(state.retention, z_t.unsqueeze(-1)).squeeze(
                -1
            )  # [B, D_ret]
            sum_ret = torch.sum(u_ret, dim=-1, keepdim=True) / math.sqrt(
                max(1, state.ret_dim)
            )
            raw_ret = cfg.ret_min + (1.0 - cfg.ret_min) * torch.sigmoid(
                sum_ret + ret_bias
            )  # [B, 1]
        else:
            u_ret = torch.matmul(state.retention, z_t)  # [D_ret]
            sum_ret = torch.sum(u_ret) / math.sqrt(max(1, state.ret_dim))
            raw_ret = cfg.ret_min + (1.0 - cfg.ret_min) * torch.sigmoid(
                sum_ret + ret_bias
            )  # scalar

        safe_ret = torch.clamp(raw_ret, 0.0, 1.0)

        # 7. State Transitions for the Five Memories:

        # 7.1 Content Memory Update
        if is_batched:
            outer_ek = torch.bmm(e_t.unsqueeze(-1), k_t.unsqueeze(-2))  # [B, V, K]
            ret_c = safe_ret.unsqueeze(-1)  # [B, 1, 1]
            eta_c = safe_eta.unsqueeze(-1)  # [B, 1, 1]
            content_upd = (ret_c - 1.0) * state.content + eta_c * outer_ek
            new_content = state.content + content_upd
        else:
            outer_ek = torch.outer(e_t, k_t)  # [V, K]
            content_upd = (safe_ret - 1.0) * state.content + safe_eta * outer_ek
            new_content = state.content + content_upd

        # 7.2 Key Generation Memory Update
        # Gradient of content error w.r.t. key is - M_content^T * e_t
        if is_batched:
            g_k = torch.bmm(state.content.transpose(-1, -2), e_t.unsqueeze(-1)).squeeze(
                -1
            )  # [B, K]
            outer_gx = torch.bmm(g_k.unsqueeze(-1), x.unsqueeze(-2))  # [B, K, D_in]
            key_upd = (lam_k - 1.0) * state.key + eta_k * outer_gx
            new_key = state.key + key_upd
        else:
            g_k = torch.matmul(state.content.t(), e_t)  # [K]
            outer_gx = torch.outer(g_k, x)  # [K, D_in]
            key_upd = (lam_k - 1.0) * state.key + eta_k * outer_gx
            new_key = state.key + key_upd

        # 7.3 Value Generation Memory Update
        # Driven by external target discrepancy e_v = target - v_gen if provided, else self-supervised
        if target is not None:
            e_val = target - v_gen
        else:
            e_val = torch.zeros_like(v_gen)

        if is_batched:
            outer_vx = torch.bmm(e_val.unsqueeze(-1), x.unsqueeze(-2))  # [B, V, D_in]
            val_upd = (lam_v - 1.0) * state.value + eta_v * outer_vx
            new_val = state.value + val_upd
        else:
            outer_vx = torch.outer(e_val, x)  # [V, D_in]
            val_upd = (lam_v - 1.0) * state.value + eta_v * outer_vx
            new_val = state.value + val_upd

        # 7.4 Learning-Rate Controller Memory Update
        # Driven by error feedback tanh((||e_t|| - tau_eta) / (tau_eta + eps))
        if is_batched:
            e_norm = torch.linalg.norm(e_t, dim=-1, keepdim=True)  # [B, 1]
            e_eta_sc = torch.tanh((e_norm - tau_e) / (tau_e + cfg.epsilon))  # [B, 1]
            e_eta = e_eta_sc.expand(-1, state.lr_dim)  # [B, D_lr]
            outer_ez = torch.bmm(
                e_eta.unsqueeze(-1), z_t.unsqueeze(-2)
            )  # [B, D_lr, D_feat]
            lr_upd = (lam_e - 1.0) * state.learning_rate + rho_e * outer_ez
            new_lr = state.learning_rate + lr_upd
        else:
            e_norm = torch.linalg.norm(e_t)
            e_eta_sc = torch.tanh((e_norm - tau_e) / (tau_e + cfg.epsilon))
            e_eta = e_eta_sc.expand(state.lr_dim)
            outer_ez = torch.outer(e_eta, z_t)
            lr_upd = (lam_e - 1.0) * state.learning_rate + rho_e * outer_ez
            new_lr = state.learning_rate + lr_upd

        # 7.5 Retention Controller Memory Update
        # Shock-reduction drive: decreases retention under unexpected high errors
        if is_batched:
            e_ret_sc = -torch.tanh((e_norm - tau_r) / (tau_r + cfg.epsilon))  # [B, 1]
            e_ret = e_ret_sc.expand(-1, state.ret_dim)  # [B, D_ret]
            outer_rz = torch.bmm(
                e_ret.unsqueeze(-1), z_t.unsqueeze(-2)
            )  # [B, D_ret, D_feat]
            ret_upd = (lam_r - 1.0) * state.retention + rho_r * outer_rz
            new_ret = state.retention + ret_upd
        else:
            e_ret_sc = -torch.tanh((e_norm - tau_r) / (tau_r + cfg.epsilon))
            e_ret = e_ret_sc.expand(state.ret_dim)
            outer_rz = torch.outer(e_ret, z_t)
            ret_upd = (lam_r - 1.0) * state.retention + rho_r * outer_rz
            new_ret = state.retention + ret_upd

        new_state = FiveMemoryState(
            content=new_content,
            key=new_key,
            value=new_val,
            learning_rate=new_lr,
            retention=new_ret,
        )

        return FiveMemoryStepResult(
            input=x,
            key_representation=k_t,
            value_representation=v_gen,
            content_prediction=v_hat,
            prediction_error=e_t,
            raw_learning_rate=raw_eta,
            safe_learning_rate=safe_eta,
            raw_retention=raw_ret,
            safe_retention=safe_ret,
            content_update=content_upd,
            key_memory_update=key_upd,
            value_memory_update=val_upd,
            learning_rate_update=lr_upd,
            retention_update=ret_upd,
            new_state=new_state,
            normalized_step=norm_step,
            stability_margin=stab_margin,
            clip_event=clip_event,
        )

    def scan(
        self,
        inputs: torch.Tensor,
        initial_state: FiveMemoryState,
        targets: torch.Tensor | None = None,
    ) -> FiveMemoryScanResult:
        r"""Execute reference sequential unrolling over input sequence without mutating input state.

        Args:
            inputs: Token sequence tensor [B, T, D_in] (batched) or [T, D_in] (unbatched).
            initial_state: Starting FiveMemoryState at t=0.
            targets: Optional targets [B, T, V] or [T, V].

        Returns:
            FiveMemoryScanResult containing predictions, final state, and trajectory histories.
        """
        is_batched = inputs.ndim == 3
        if is_batched:
            b_size, t_steps, d_in = inputs.shape
            if not initial_state.is_batched or initial_state.batch_size != b_size:
                raise ValueError(
                    f"Inputs are batched [B={b_size}], but initial_state has batch_size={initial_state.batch_size}."
                )
        else:
            if inputs.ndim != 2:
                raise ValueError(
                    f"Inputs must have shape [B, T, D_in] or [T, D_in], got {list(inputs.shape)}."
                )
            t_steps, d_in = inputs.shape
            if initial_state.is_batched:
                raise ValueError(
                    "Inputs are unbatched [T, D_in], but initial_state is batched."
                )

        current_state = initial_state
        step_results: list[FiveMemoryStepResult] = []

        preds_list: list[torch.Tensor] = []
        errors_list: list[torch.Tensor] = []
        keys_list: list[torch.Tensor] = []
        vals_list: list[torch.Tensor] = []
        raw_etas_list: list[torch.Tensor] = []
        safe_etas_list: list[torch.Tensor] = []
        raw_rets_list: list[torch.Tensor] = []
        safe_rets_list: list[torch.Tensor] = []

        norm_c_mems = [torch.linalg.norm(current_state.content.float(), dim=(-2, -1))]
        norm_k_mems = [torch.linalg.norm(current_state.key.float(), dim=(-2, -1))]
        norm_v_mems = [torch.linalg.norm(current_state.value.float(), dim=(-2, -1))]
        norm_lr_mems = [
            torch.linalg.norm(current_state.learning_rate.float(), dim=(-2, -1))
        ]
        norm_ret_mems = [
            torch.linalg.norm(current_state.retention.float(), dim=(-2, -1))
        ]

        norm_c_upds: list[torch.Tensor] = []
        norm_k_upds: list[torch.Tensor] = []
        norm_v_upds: list[torch.Tensor] = []
        norm_lr_upds: list[torch.Tensor] = []
        norm_ret_upds: list[torch.Tensor] = []

        margins_list: list[torch.Tensor] = []
        norm_steps_list: list[torch.Tensor] = []
        clips_list: list[torch.Tensor] = []

        for t in range(t_steps):
            x_t = inputs[:, t] if is_batched else inputs[t]
            y_t = None
            if targets is not None:
                y_t = targets[:, t] if is_batched else targets[t]

            res = self.step(x=x_t, state=current_state, target=y_t)
            step_results.append(res)
            current_state = res.new_state

            preds_list.append(res.content_prediction)
            errors_list.append(res.prediction_error)
            keys_list.append(res.key_representation)
            vals_list.append(res.value_representation)
            raw_etas_list.append(res.raw_learning_rate)
            safe_etas_list.append(res.safe_learning_rate)
            raw_rets_list.append(res.raw_retention)
            safe_rets_list.append(res.safe_retention)

            norm_c_mems.append(
                torch.linalg.norm(current_state.content.float(), dim=(-2, -1))
            )
            norm_k_mems.append(
                torch.linalg.norm(current_state.key.float(), dim=(-2, -1))
            )
            norm_v_mems.append(
                torch.linalg.norm(current_state.value.float(), dim=(-2, -1))
            )
            norm_lr_mems.append(
                torch.linalg.norm(current_state.learning_rate.float(), dim=(-2, -1))
            )
            norm_ret_mems.append(
                torch.linalg.norm(current_state.retention.float(), dim=(-2, -1))
            )

            norm_c_upds.append(
                torch.linalg.norm(res.content_update.float(), dim=(-2, -1))
            )
            norm_k_upds.append(
                torch.linalg.norm(res.key_memory_update.float(), dim=(-2, -1))
            )
            norm_v_upds.append(
                torch.linalg.norm(res.value_memory_update.float(), dim=(-2, -1))
            )
            norm_lr_upds.append(
                torch.linalg.norm(res.learning_rate_update.float(), dim=(-2, -1))
            )
            norm_ret_upds.append(
                torch.linalg.norm(res.retention_update.float(), dim=(-2, -1))
            )

            if res.stability_margin is not None:
                margins_list.append(res.stability_margin)
            if res.normalized_step is not None:
                norm_steps_list.append(res.normalized_step)
            if res.clip_event is not None:
                clips_list.append(res.clip_event)

        dim_cat = 1 if is_batched else 0

        return FiveMemoryScanResult(
            predictions=torch.stack(preds_list, dim=dim_cat),
            final_state=current_state,
            step_results=step_results,
            errors=torch.stack(errors_list, dim=dim_cat),
            keys=torch.stack(keys_list, dim=dim_cat),
            values=torch.stack(vals_list, dim=dim_cat),
            raw_learning_rates=torch.stack(raw_etas_list, dim=dim_cat),
            safe_learning_rates=torch.stack(safe_etas_list, dim=dim_cat),
            raw_retentions=torch.stack(raw_rets_list, dim=dim_cat),
            safe_retentions=torch.stack(safe_rets_list, dim=dim_cat),
            content_memory_norms=torch.stack(norm_c_mems, dim=dim_cat),
            key_memory_norms=torch.stack(norm_k_mems, dim=dim_cat),
            value_memory_norms=torch.stack(norm_v_mems, dim=dim_cat),
            learning_rate_memory_norms=torch.stack(norm_lr_mems, dim=dim_cat),
            retention_memory_norms=torch.stack(norm_ret_mems, dim=dim_cat),
            content_update_norms=torch.stack(norm_c_upds, dim=dim_cat),
            key_update_norms=torch.stack(norm_k_upds, dim=dim_cat),
            value_update_norms=torch.stack(norm_v_upds, dim=dim_cat),
            learning_rate_update_norms=torch.stack(norm_lr_upds, dim=dim_cat),
            retention_update_norms=torch.stack(norm_ret_upds, dim=dim_cat),
            stability_margins=torch.stack(margins_list, dim=dim_cat)
            if margins_list
            else torch.empty(0),
            normalized_steps=torch.stack(norm_steps_list, dim=dim_cat)
            if norm_steps_list
            else torch.empty(0),
            clip_events=torch.stack(clips_list, dim=dim_cat)
            if clips_list
            else torch.empty(0),
        )
