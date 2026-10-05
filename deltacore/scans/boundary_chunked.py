"""DeltaCore Boundary-Refresh Chunk Execution.

Implements chunked execution with boundary-state refresh semantics:
    boundary_state (S_b)
          │
          ▼
    generate token-dependent quantities (k_t, v_t, η_t, α_t) using boundary_state
          │
          ▼
    within-chunk sequential memory updates (M_{t+1}^c, M_{t+1}^k, ...)
          │
          ▼
    chunk final state S_{b+1}
          │
          ▼
    new boundary_state

CRITICAL DISTINCTION:
This execution model is deliberately distinct from Phase 5 exact associative
affine chunking (ExactAffineChunkScan).
- In ExactAffineChunkScan: token transition operators (A_t, B_t) are already known,
  and associative chunk composition is algebraically exact for the mathematical
  recurrence M_{t+1} = M_t A_t + B_t across all chunk sizes C >= 1.
- In BoundaryRefreshChunkScan: token-dependent quantities (k_t, v_t, η_t, α_t)
  are generated using the boundary memory state S_b, which is held frozen across
  the chunk of length C.
  * For C = 1: BoundaryRefreshChunkScan reproduces the fully token-indexed reference
    FiveMemorySystem recurrence exactly.
  * For C > 1: This represents an algorithmic chunking formulation where boundary
    staleness produces genuine trajectory differences. Larger C is NOT mathematically
    equivalent to C = 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from deltacore.memory.five_memory import FiveMemoryState
from deltacore.updates.five_memory import FiveMemoryConfig, FiveMemorySystem


@dataclass(frozen=True)
class BoundaryChunkResult:
    r"""Diagnostic telemetry and trajectory output for BoundaryRefreshChunkScan.

    Attributes:
        predictions: Output sequence [B, T, V] or [T, V].
        final_state: Final FiveMemoryState at t=T.
        boundary_states: List of boundary states captured at the start of each chunk.
        chunk_boundaries: List of token indices where boundaries occurred (e.g. [0, C, 2C, ...]).
        errors: Prediction errors [B, T, V] or [T, V].
        keys: Generated key representations [B, T, K] or [T, K].
        values: Generated value representations [B, T, V] or [T, V].
        safe_learning_rates: Contraction-stabilized learning rates [B, T, 1] or [T, 1].
        safe_retentions: Clamped retention factors [B, T, 1] or [T, 1].
        content_memory_norms: Frobenius norm of content memory over sequence.
        trajectory: Full sequence of FiveMemoryState instances if recorded, else None.
    """

    predictions: torch.Tensor
    final_state: FiveMemoryState
    boundary_states: list[FiveMemoryState]
    chunk_boundaries: list[int]
    errors: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    safe_learning_rates: torch.Tensor
    safe_retentions: torch.Tensor
    content_memory_norms: torch.Tensor
    trajectory: list[FiveMemoryState] | None = None


class BoundaryRefreshChunkScan(nn.Module):
    r"""Boundary-Refresh Chunk Execution Operator for the Five-Memory Reference Learner.

    Within each chunk [t_start, t_end):
    1. Boundary state S_b is held fixed to generate token-dependent quantities:
           k_t = M_b^k x_t
           v_t = M_b^v x_t
           η_t = \phi_\eta(m_b^\eta x_t)
           α_t = \phi_\alpha(m_b^\alpha x_t)
    2. Working memory updates are accumulated sequentially in token order.
    3. The accumulated chunk final state becomes the next chunk boundary state S_{b+1}.

    Args:
        system: FiveMemorySystem instance providing configuration and projection parameters.
        default_chunk_size: Default chunk length C (must be >= 1).
    """

    def __init__(
        self,
        system: FiveMemorySystem,
        default_chunk_size: int = 1,
    ) -> None:
        super().__init__()
        if default_chunk_size < 1:
            raise ValueError(
                f"default_chunk_size must be >= 1, got {default_chunk_size}"
            )
        self.system = system
        self.default_chunk_size = default_chunk_size

    @property
    def config(self) -> FiveMemoryConfig:
        return self.system.config

    def scan(
        self,
        inputs: torch.Tensor,
        initial_state: FiveMemoryState,
        targets: torch.Tensor | None = None,
        chunk_size: int | None = None,
        record_trajectory: bool = False,
    ) -> BoundaryChunkResult:
        r"""Execute boundary-refresh chunked recurrence over input sequence.

        Args:
            inputs: Token sequence tensor [B, T, D_in] (batched) or [T, D_in] (unbatched).
            initial_state: Starting FiveMemoryState at t=0.
            targets: Optional targets [B, T, V] or [T, V].
            chunk_size: Contiguous chunk size C (default: self.default_chunk_size).
            record_trajectory: If True, saves all intermediate FiveMemoryStates.

        Returns:
            BoundaryChunkResult with complete trajectory telemetry.
        """
        c_size = self.default_chunk_size if chunk_size is None else chunk_size
        if c_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {c_size}")

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

        cfg = self.config
        current_state = initial_state
        boundary_states: list[FiveMemoryState] = []
        chunk_boundaries: list[int] = []
        trajectory: list[FiveMemoryState] | None = [] if record_trajectory else None

        preds_list: list[torch.Tensor] = []
        errors_list: list[torch.Tensor] = []
        keys_list: list[torch.Tensor] = []
        vals_list: list[torch.Tensor] = []
        safe_etas_list: list[torch.Tensor] = []
        safe_rets_list: list[torch.Tensor] = []
        content_norms_list: list[torch.Tensor] = []

        # Resolve dynamics quantities (trainable vs static configuration)
        if getattr(self.system, "trainable_dynamics", False):
            eta_bias = self.system.eta_bias.to(dtype=inputs.dtype, device=inputs.device)
            ret_bias = self.system.ret_bias.to(dtype=inputs.dtype, device=inputs.device)
            lam_k = self.system.lambda_key.to(
                dtype=initial_state.key.dtype, device=initial_state.key.device
            )
            eta_k = self.system.eta_key.to(
                dtype=initial_state.key.dtype, device=initial_state.key.device
            )
            lam_v = self.system.lambda_val.to(
                dtype=initial_state.value.dtype, device=initial_state.value.device
            )
            eta_v = self.system.eta_val.to(
                dtype=initial_state.value.dtype, device=initial_state.value.device
            )
            lam_e = self.system.lambda_eta.to(
                dtype=initial_state.learning_rate.dtype,
                device=initial_state.learning_rate.device,
            )
            rho_e = self.system.rho_eta.to(
                dtype=initial_state.learning_rate.dtype,
                device=initial_state.learning_rate.device,
            )
            tau_e = self.system.tau_eta.to(dtype=inputs.dtype, device=inputs.device)
            lam_r = self.system.lambda_ret.to(
                dtype=initial_state.retention.dtype,
                device=initial_state.retention.device,
            )
            rho_r = self.system.rho_ret.to(
                dtype=initial_state.retention.dtype,
                device=initial_state.retention.device,
            )
            tau_r = self.system.tau_ret.to(dtype=inputs.dtype, device=inputs.device)
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

        # Iterate over chunks
        for chunk_start in range(0, t_steps, c_size):
            chunk_end = min(chunk_start + c_size, t_steps)
            chunk_boundaries.append(chunk_start)

            # Hold boundary state fixed for generating token-dependent representations
            boundary_state = current_state
            boundary_states.append(boundary_state)

            # Process tokens within the chunk sequentially
            for t in range(chunk_start, chunk_end):
                if is_batched:
                    x_t = inputs[:, t, :]
                    target_t = targets[:, t, :] if targets is not None else None
                else:
                    x_t = inputs[t, :]
                    target_t = targets[t, :] if targets is not None else None

                # 1. Key generation using boundary state M_b^k
                if is_batched:
                    k_t = torch.bmm(boundary_state.key, x_t.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, K]
                else:
                    k_t = torch.matmul(boundary_state.key, x_t)  # [K]

                # 2. Value generation using boundary state M_b^v
                if is_batched:
                    v_gen = torch.bmm(boundary_state.value, x_t.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, V]
                else:
                    v_gen = torch.matmul(boundary_state.value, x_t)  # [V]

                v_eff = target_t if target_t is not None else v_gen

                # 3. Content memory read using CURRENT working memory M_t^c
                if is_batched:
                    v_hat = torch.bmm(current_state.content, k_t.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, V]
                else:
                    v_hat = torch.matmul(current_state.content, k_t)  # [V]

                e_t = v_eff - v_hat

                # 4. Learning-rate controller read using boundary state M_b^eta
                z_t = x_t
                if is_batched:
                    u_eta = torch.bmm(
                        boundary_state.learning_rate, z_t.unsqueeze(-1)
                    ).squeeze(-1)
                    sum_eta = torch.sum(u_eta, dim=-1, keepdim=True) / math.sqrt(
                        max(1, boundary_state.lr_dim)
                    )
                    raw_eta = cfg.eta_max * torch.sigmoid(sum_eta + eta_bias)
                    k_sq_norm = torch.sum(k_t * k_t, dim=-1, keepdim=True)
                else:
                    u_eta = torch.matmul(boundary_state.learning_rate, z_t)
                    sum_eta = torch.sum(u_eta) / math.sqrt(
                        max(1, boundary_state.lr_dim)
                    )
                    raw_eta = cfg.eta_max * torch.sigmoid(sum_eta + eta_bias)
                    k_sq_norm = torch.sum(k_t * k_t)

                if cfg.apply_stability_control:
                    eta_bound = cfg.beta / (k_sq_norm + cfg.epsilon)
                    safe_eta = torch.minimum(raw_eta, eta_bound)
                else:
                    safe_eta = raw_eta

                # 5. Retention controller read using boundary state M_b^ret
                if is_batched:
                    u_ret = torch.bmm(
                        boundary_state.retention, z_t.unsqueeze(-1)
                    ).squeeze(-1)
                    sum_ret = torch.sum(u_ret, dim=-1, keepdim=True) / math.sqrt(
                        max(1, boundary_state.ret_dim)
                    )
                    raw_ret = cfg.ret_min + (1.0 - cfg.ret_min) * torch.sigmoid(
                        sum_ret + ret_bias
                    )
                else:
                    u_ret = torch.matmul(boundary_state.retention, z_t)
                    sum_ret = torch.sum(u_ret) / math.sqrt(
                        max(1, boundary_state.ret_dim)
                    )
                    raw_ret = cfg.ret_min + (1.0 - cfg.ret_min) * torch.sigmoid(
                        sum_ret + ret_bias
                    )

                safe_ret = torch.clamp(raw_ret, 0.0, 1.0)

                # 6. Sequential Working Memory Updates:
                # 6.1 Content Memory Update
                if is_batched:
                    outer_ek = torch.bmm(
                        e_t.unsqueeze(-1), k_t.unsqueeze(-2)
                    )  # [B, V, K]
                    ret_c = safe_ret.unsqueeze(-1)
                    eta_c = safe_eta.unsqueeze(-1)
                    content_upd = (
                        ret_c - 1.0
                    ) * current_state.content + eta_c * outer_ek
                    new_content = current_state.content + content_upd
                else:
                    outer_ek = torch.outer(e_t, k_t)
                    content_upd = (
                        safe_ret - 1.0
                    ) * current_state.content + safe_eta * outer_ek
                    new_content = current_state.content + content_upd

                # 6.2 Key Generation Memory Update
                if is_batched:
                    g_k = torch.bmm(
                        current_state.content.transpose(-1, -2), e_t.unsqueeze(-1)
                    ).squeeze(-1)
                    outer_gx = torch.bmm(g_k.unsqueeze(-1), x_t.unsqueeze(-2))
                    key_upd = (lam_k - 1.0) * current_state.key + eta_k * outer_gx
                    new_key = current_state.key + key_upd
                else:
                    g_k = torch.matmul(current_state.content.t(), e_t)
                    outer_gx = torch.outer(g_k, x_t)
                    key_upd = (lam_k - 1.0) * current_state.key + eta_k * outer_gx
                    new_key = current_state.key + key_upd

                # 6.3 Value Generation Memory Update
                if target_t is not None:
                    e_val = target_t - v_gen
                else:
                    e_val = torch.zeros_like(v_gen)

                if is_batched:
                    outer_vx = torch.bmm(e_val.unsqueeze(-1), x_t.unsqueeze(-2))
                    val_upd = (lam_v - 1.0) * current_state.value + eta_v * outer_vx
                    new_val = current_state.value + val_upd
                else:
                    outer_vx = torch.outer(e_val, x_t)
                    val_upd = (lam_v - 1.0) * current_state.value + eta_v * outer_vx
                    new_val = current_state.value + val_upd

                # 6.4 Learning-Rate Controller Memory Update
                if is_batched:
                    e_norm = torch.linalg.norm(e_t, dim=-1, keepdim=True)
                    e_eta_sc = torch.tanh((e_norm - tau_e) / (tau_e + cfg.epsilon))
                    e_eta = e_eta_sc.expand(-1, current_state.lr_dim)
                    outer_ez = torch.bmm(e_eta.unsqueeze(-1), z_t.unsqueeze(-2))
                    lr_upd = (
                        lam_e - 1.0
                    ) * current_state.learning_rate + rho_e * outer_ez
                    new_lr = current_state.learning_rate + lr_upd
                else:
                    e_norm = torch.linalg.norm(e_t)
                    e_eta_sc = torch.tanh((e_norm - tau_e) / (tau_e + cfg.epsilon))
                    e_eta = e_eta_sc.expand(current_state.lr_dim)
                    outer_ez = torch.outer(e_eta, z_t)
                    lr_upd = (
                        lam_e - 1.0
                    ) * current_state.learning_rate + rho_e * outer_ez
                    new_lr = current_state.learning_rate + lr_upd

                # 6.5 Retention Controller Memory Update
                if is_batched:
                    e_ret_sc = -torch.tanh((e_norm - tau_r) / (tau_r + cfg.epsilon))
                    e_ret = e_ret_sc.expand(-1, current_state.ret_dim)
                    outer_rz = torch.bmm(e_ret.unsqueeze(-1), z_t.unsqueeze(-2))
                    ret_upd = (lam_r - 1.0) * current_state.retention + rho_r * outer_rz
                    new_ret = current_state.retention + ret_upd
                else:
                    e_ret_sc = -torch.tanh((e_norm - tau_r) / (tau_r + cfg.epsilon))
                    e_ret = e_ret_sc.expand(current_state.ret_dim)
                    outer_rz = torch.outer(e_ret, z_t)
                    ret_upd = (lam_r - 1.0) * current_state.retention + rho_r * outer_rz
                    new_ret = current_state.retention + ret_upd

                current_state = FiveMemoryState(
                    content=new_content,
                    key=new_key,
                    value=new_val,
                    learning_rate=new_lr,
                    retention=new_ret,
                )

                if trajectory is not None:
                    trajectory.append(current_state)

                preds_list.append(v_hat)
                errors_list.append(e_t)
                keys_list.append(k_t)
                vals_list.append(v_gen)
                safe_etas_list.append(safe_eta)
                safe_rets_list.append(safe_ret)

                c_norm = torch.linalg.norm(
                    current_state.content, dim=(-2, -1) if is_batched else (-2, -1)
                )
                content_norms_list.append(c_norm)

        # Stack outputs along time dimension
        dim_stack = 1 if is_batched else 0
        predictions = torch.stack(preds_list, dim=dim_stack)
        errors = torch.stack(errors_list, dim=dim_stack)
        keys = torch.stack(keys_list, dim=dim_stack)
        values = torch.stack(vals_list, dim=dim_stack)
        safe_etas = torch.stack(safe_etas_list, dim=dim_stack)
        safe_rets = torch.stack(safe_rets_list, dim=dim_stack)
        content_norms = torch.stack(content_norms_list, dim=dim_stack)

        return BoundaryChunkResult(
            predictions=predictions,
            final_state=current_state,
            boundary_states=boundary_states,
            chunk_boundaries=chunk_boundaries,
            errors=errors,
            keys=keys,
            values=values,
            safe_learning_rates=safe_etas,
            safe_retentions=safe_rets,
            content_memory_norms=content_norms,
            trajectory=trajectory,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: FiveMemoryState,
        targets: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> BoundaryChunkResult:
        """Alias forward to scan for standard PyTorch module execution."""
        return self.scan(inputs, initial_state, targets=targets, chunk_size=chunk_size)
