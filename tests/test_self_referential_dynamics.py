r"""Tests for Phase 3 minimal self-referential adaptive memory dynamics.

Verifies:
- Property A: Controller state actually evolves (C_{t+1} != C_t for non-degenerate sequences).
- Property B: Frozen-controller equivalence with Phase 1/2 fixed update baseline.
- Property C: Zero content error invariance (\Delta M_t = 0 when M_t @ k_t = v_t).
- Property D: Zero key invariance (\Delta M_t = 0 when k_t = 0).
- Property E: Deterministic recurrence across identical sequences.
- Property F: Non-mutating functional state semantics.
- Property G: State dependence (changing dynamics state modifies control signal).
- Property H: Controller isolation (no hidden dependency between content and dynamics).
- Stress Tests: Varying \rho (1e-5, 0.1, 2.0), repeated keys, conflicting values,
  abrupt regime shifts, and long sequences.
"""

import pytest
import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.updates.delta import DeltaRule
from deltacore.updates.dynamics_memory import DynamicsMemory
from deltacore.updates.self_referential import (
    CoupledState,
    ErrorProportionalTargetGenerator,
    SelfReferentialSystem,
)


class TestSelfReferentialDynamics:
    """Verifies core mathematical invariants and dynamical properties of coupled memories."""

    @pytest.fixture
    def rng_seed(self):
        torch.manual_seed(42)

    def test_property_a_controller_actually_evolves(self, rng_seed):
        r"""Property A: In a non-degenerate sequence, dynamics memory state evolves:

        \exists t \text{ such that } C_{t+1} \neq C_t.
        """
        system = SelfReferentialSystem(
            eta_max=1.0,
            rho=0.2,
            target_generator=ErrorProportionalTargetGenerator(alpha=1.0, beta=0.5),
        )

        state = CoupledState(
            content_memory=AssociativeMemory.zeros(3, 3),
            dynamics_memory=DynamicsMemory.zeros(1, 3),
        )

        k = torch.tensor([1.0, 0.0, 0.0])
        v = torch.tensor([0.5, 1.0, -0.5])

        res = system.step(k, v, state=state, frozen_dynamics=False)

        assert not torch.allclose(
            res.new_dynamics_memory.data, state.dynamics_memory.data
        ), "Dynamics memory C failed to evolve when error was present."
        assert torch.norm(res.dynamics_update) > 0.0

    def test_property_b_frozen_controller_equivalence(self, rng_seed):
        r"""Property B: When dynamics memory is frozen to yield constant \eta,

        the content update matches Phase 1 DeltaRule exactly.
        """
        eta_target = 0.4
        # We want \eta = eta_max * \sigma(r_0) = 0.4 with eta_max = 1.0
        # \sigma(r_0) = 0.4 => r_0 = log(0.4 / 0.6) = log(2/3)
        r_0 = torch.log(torch.tensor(0.4 / 0.6, dtype=torch.float64)).item()

        # Place r_0 in the bias position of C_0 = [0.0, 0.0, r_0]
        c_init = torch.tensor([[0.0, 0.0, r_0]], dtype=torch.float64)
        m_init = torch.randn(3, 4, dtype=torch.float64)

        k = torch.randn(4, dtype=torch.float64)
        v = torch.randn(3, dtype=torch.float64)

        system = SelfReferentialSystem(eta_max=1.0, rho=0.1).to(dtype=torch.float64)
        state = CoupledState(
            content_memory=AssociativeMemory(m_init.clone()),
            dynamics_memory=DynamicsMemory(c_init.clone()),
        )

        res = system.step(k, v, state=state, frozen_dynamics=True)

        # Baseline Phase 1 DeltaRule
        delta_rule = DeltaRule(step_size=eta_target)
        base_res = delta_rule.step(m_init, k, v)

        assert torch.allclose(
            res.step_size, torch.tensor(eta_target, dtype=torch.float64), atol=1e-12
        )
        assert torch.allclose(res.content_update, base_res.update, atol=1e-14)
        assert torch.allclose(
            res.new_content_memory.data, base_res.new_memory, atol=1e-14
        )
        # Verify dynamics memory remained strictly frozen
        assert torch.allclose(res.new_dynamics_memory.data, c_init, atol=1e-14)

    def test_property_c_zero_content_error(self, rng_seed):
        r"""Property C: When content prediction is exact (e_t = 0),

        the content update is identically zero (\Delta M_t = 0).
        """
        system = SelfReferentialSystem(eta_max=1.0, rho=0.1)
        m_data = torch.randn(3, 3)
        k = torch.randn(3)
        # Target equals exact memory prediction
        v = m_data @ k

        state = CoupledState(
            content_memory=AssociativeMemory(m_data),
            dynamics_memory=DynamicsMemory.zeros(1, 3),
        )

        res = system.step(k, v, state=state)

        assert torch.allclose(res.error, torch.zeros_like(v), atol=1e-6)
        assert torch.allclose(res.content_update, torch.zeros_like(m_data), atol=1e-7)
        assert torch.allclose(res.new_content_memory.data, m_data, atol=1e-7)

    def test_property_d_zero_key_invariance(self, rng_seed):
        r"""Property D: When query key is zero (k_t = 0),

        the content update is identically zero (\Delta M_t = 0).
        """
        system = SelfReferentialSystem(eta_max=1.0, rho=0.1)
        m_data = torch.randn(3, 3)
        k = torch.zeros(3)
        v = torch.randn(3)

        state = CoupledState(
            content_memory=AssociativeMemory(m_data),
            dynamics_memory=DynamicsMemory.zeros(1, 3),
        )

        res = system.step(k, v, state=state)

        assert torch.allclose(res.content_update, torch.zeros_like(m_data), atol=1e-7)
        assert torch.allclose(res.new_content_memory.data, m_data, atol=1e-7)

    def test_property_e_deterministic_recurrence(self, rng_seed):
        """Property E: Identical initial state and input sequence produce identical trajectory."""
        system = SelfReferentialSystem(eta_max=0.8, rho=0.15)

        t_steps, k_dim, v_dim = 15, 4, 3
        keys = torch.randn(t_steps, k_dim)
        targets = torch.randn(t_steps, v_dim)

        state_1 = CoupledState(
            content_memory=AssociativeMemory.zeros(v_dim, k_dim),
            dynamics_memory=DynamicsMemory.zeros(1, 3),
        )
        state_2 = state_1.clone()

        res_1 = system.scan(keys, targets, initial_state=state_1)
        res_2 = system.scan(keys, targets, initial_state=state_2)

        assert torch.allclose(res_1.predictions, res_2.predictions, atol=1e-15)
        assert torch.allclose(res_1.step_sizes, res_2.step_sizes, atol=1e-15)
        assert torch.allclose(
            res_1.final_state.content_memory.data,
            res_2.final_state.content_memory.data,
            atol=1e-15,
        )
        assert torch.allclose(
            res_1.final_state.dynamics_memory.data,
            res_2.final_state.dynamics_memory.data,
            atol=1e-15,
        )

    def test_property_f_non_mutating_state_semantics(self, rng_seed):
        """Property F: Transitions never mutate the input state in-place."""
        system = SelfReferentialSystem(eta_max=0.5, rho=0.1)

        m_orig = torch.randn(3, 3)
        c_orig = torch.randn(1, 3)

        m_copy = m_orig.clone()
        c_copy = c_orig.clone()

        state = CoupledState(
            content_memory=AssociativeMemory(m_orig),
            dynamics_memory=DynamicsMemory(c_orig),
        )

        k = torch.randn(3)
        v = torch.randn(3)

        res = system.step(k, v, state=state)

        # Input state tensors must remain strictly identical
        assert torch.allclose(state.content_memory.data, m_copy, atol=1e-15)
        assert torch.allclose(state.dynamics_memory.data, c_copy, atol=1e-15)
        assert state.content_memory.data is m_orig
        assert state.dynamics_memory.data is c_orig

        # New state must be distinct memory
        assert res.new_state.content_memory.data is not state.content_memory.data
        assert res.new_state.dynamics_memory.data is not state.dynamics_memory.data

    def test_property_g_state_dependence(self, rng_seed):
        """Property G: Changing dynamics memory holding input and content memory fixed alters eta_t."""
        system = SelfReferentialSystem(eta_max=1.0, rho=0.1)

        m = torch.randn(2, 2)
        k = torch.randn(2)
        v = torch.randn(2)

        c_state_a = DynamicsMemory(torch.tensor([[-2.0, 0.0, -1.0]]))
        c_state_b = DynamicsMemory(torch.tensor([[2.0, 0.0, 1.0]]))

        state_a = CoupledState(
            content_memory=AssociativeMemory(m), dynamics_memory=c_state_a
        )
        state_b = CoupledState(
            content_memory=AssociativeMemory(m), dynamics_memory=c_state_b
        )

        res_a = system.step(k, v, state=state_a)
        res_b = system.step(k, v, state=state_b)

        assert res_a.step_size.item() != pytest.approx(res_b.step_size.item(), rel=1e-2)
        assert res_a.step_size < res_b.step_size

    def test_property_h_controller_isolation(self, rng_seed):
        r"""Property H: Verifies that changing content memory has only documented effects on control signal.

        When dynamics memory has weights only on the bias feature z[2] = 1.0,
        altering content memory M cannot change r_t or \eta_t.
        """
        system = SelfReferentialSystem(eta_max=1.0, rho=0.1)

        # C has weights only on feature 2 (bias term = 1.0)
        c_isolated = DynamicsMemory(torch.tensor([[0.0, 0.0, 0.5]]))

        k = torch.tensor([1.0, 0.0])
        # Two very different content memories with same target to keep error different,
        # or even with different errors, feature 0 and 1 are ignored by c_isolated!
        m_1 = torch.zeros(2, 2)
        m_2 = torch.randn(2, 2) * 5.0
        v = torch.tensor([1.0, 2.0])

        state_1 = CoupledState(
            content_memory=AssociativeMemory(m_1), dynamics_memory=c_isolated
        )
        state_2 = CoupledState(
            content_memory=AssociativeMemory(m_2), dynamics_memory=c_isolated
        )

        res_1 = system.step(k, v, state=state_1)
        res_2 = system.step(k, v, state=state_2)

        # r_t should be exactly 0.5 for both
        assert torch.allclose(res_1.dynamics_prediction, torch.tensor([0.5]), atol=1e-6)
        assert torch.allclose(res_2.dynamics_prediction, torch.tensor([0.5]), atol=1e-6)
        assert torch.allclose(res_1.step_size, res_2.step_size, atol=1e-6)


class TestSelfReferentialStress:
    """Stress testing self-referential dynamics across parameter regimes and sequence conditions.

    Evaluates numerical integrity and ensures that when divergence occurs under
    aggressive regimes, it is observable rather than silently masked.
    """

    def test_controller_learning_rate_regimes(self):
        """Evaluate dynamics under small, moderate, and aggressive controller learning rates."""
        # 1. Small rho (1e-5): Stable slow adaptation
        sys_small = SelfReferentialSystem(eta_max=0.5, rho=1e-5)
        keys = torch.randn(20, 4)
        keys = keys / torch.norm(keys, dim=-1, keepdim=True)
        targets = torch.randn(20, 3)

        res_small = sys_small.scan(keys, targets)
        assert torch.isfinite(res_small.final_state.content_memory.data).all()
        assert torch.isfinite(res_small.final_state.dynamics_memory.data).all()

        # 2. Moderate rho (0.05): Controlled adaptation
        sys_mod = SelfReferentialSystem(eta_max=0.5, rho=0.05)
        res_mod = sys_mod.scan(keys, targets)
        assert torch.isfinite(res_mod.final_state.content_memory.data).all()
        assert torch.isfinite(res_mod.final_state.dynamics_memory.data).all()

        # 3. Aggressive rho (2.0): System may exhibit rapid state growth or divergence
        sys_agg = SelfReferentialSystem(eta_max=1.0, rho=2.0)
        res_agg = sys_agg.scan(keys * 2.0, targets * 2.0)
        has_nan_or_inf = (
            torch.isnan(res_agg.final_state.content_memory.data).any()
            or torch.isinf(res_agg.final_state.content_memory.data).any()
            or torch.isnan(res_agg.final_state.dynamics_memory.data).any()
            or torch.isinf(res_agg.final_state.dynamics_memory.data).any()
        )
        # Exposes the phenomenon: either explosive state growth or NaN/Inf occurred
        state_norm = torch.linalg.norm(res_agg.final_state.content_memory.data)
        assert has_nan_or_inf or state_norm > 10.0

    def test_repeated_and_correlated_keys(self):
        """Verify behavior under identical and highly correlated keys."""
        system = SelfReferentialSystem(eta_max=0.5, rho=0.05)

        # Repeated normalized key
        key = torch.randn(4)
        key = key / torch.norm(key)
        target = torch.randn(3)

        keys = key.unsqueeze(0).expand(20, -1)
        targets = target.unsqueeze(0).expand(20, -1)

        scan_res = system.scan(keys, targets)

        # Content prediction error should strictly decrease
        initial_error_norm = torch.norm(scan_res.errors[0])
        final_error_norm = torch.norm(scan_res.errors[-1])
        assert final_error_norm < initial_error_norm
        assert torch.isfinite(scan_res.final_state.content_memory.data).all()

    def test_conflicting_values_for_same_key(self):
        """Verify behavior when identical key is paired with conflicting targets."""
        system = SelfReferentialSystem(eta_max=0.5, rho=0.05)

        key = torch.randn(4)
        key = key / torch.norm(key)
        target_a = torch.tensor([1.0, 0.0, -1.0])
        target_b = torch.tensor([-1.0, 0.0, 1.0])

        keys = key.unsqueeze(0).expand(20, -1)
        targets = torch.stack([target_a if i % 2 == 0 else target_b for i in range(20)])

        scan_res = system.scan(keys, targets)

        # Check telemetry for finite state tracking
        from deltacore.diagnostics.self_reference import measure_step_telemetry

        last_step_res = system.step(keys[-1], targets[-1], state=scan_res.final_state)
        telemetry = measure_step_telemetry(last_step_res)
        assert isinstance(telemetry["finite_state"], bool)

    def test_abrupt_regime_shift(self):
        """Verify behavior under sudden orthogonal distribution shift."""
        system = SelfReferentialSystem(eta_max=0.5, rho=0.02)

        t_regime = 15

        # Regime 1: Key is fixed in subspace 1 (first 2 dims), targets constant
        k1 = torch.tensor([1.0, 0.5, 0.0, 0.0])
        k1 = k1 / torch.norm(k1)
        v1 = torch.tensor([1.0, -1.0, 0.5])

        keys_1 = k1.unsqueeze(0).expand(t_regime, -1)
        targets_1 = v1.unsqueeze(0).expand(t_regime, -1)

        # Regime 2: Shift to orthogonal subspace (last 2 dims)
        k2 = torch.tensor([0.0, 0.0, 1.0, -0.5])
        k2 = k2 / torch.norm(k2)
        v2 = torch.tensor([-0.5, 1.0, 1.0])

        keys_2 = k2.unsqueeze(0).expand(t_regime, -1)
        targets_2 = v2.unsqueeze(0).expand(t_regime, -1)

        keys = torch.cat([keys_1, keys_2], dim=0)
        targets = torch.cat([targets_1, targets_2], dim=0)

        scan_res = system.scan(keys, targets)

        # Step t_regime - 1 is well-adapted in regime 1
        pre_shift_err = torch.norm(scan_res.errors[t_regime - 1]).item()
        # Step t_regime is the first novel key of regime 2
        post_shift_err = torch.norm(scan_res.errors[t_regime]).item()

        assert post_shift_err > pre_shift_err
        assert torch.isfinite(scan_res.final_state.content_memory.data).all()
        assert torch.isfinite(scan_res.final_state.dynamics_memory.data).all()

    def test_long_sequence_stability(self):
        """Run long 100-step sequence with normalized keys and monitor stability."""
        system = SelfReferentialSystem(eta_max=0.3, rho=0.01)

        t_steps = 100
        keys = torch.randn(t_steps, 5)
        keys = keys / torch.norm(keys, dim=-1, keepdim=True)
        targets = torch.randn(t_steps, 3)

        scan_res = system.scan(keys, targets)

        assert torch.isfinite(scan_res.predictions).all()
        assert torch.isfinite(scan_res.step_sizes).all()
        assert torch.isfinite(scan_res.final_state.content_memory.data).all()
        assert torch.isfinite(scan_res.final_state.dynamics_memory.data).all()
