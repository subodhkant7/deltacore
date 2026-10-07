#!/usr/bin/env python3
"""DeltaCore Minimal Smoke Example: Basic Adaptation & State Recovery.

Demonstrates the canonical public API for DeltaCore:
    1. Initialize AdaptiveController
    2. Encode structured telemetry events with DeterministicFeatureHasher
    3. Execute single-step inference and online adaptation (`step()`)
    4. Inspect pre-update diagnostics (reconstruction_residual, stability_margin)
    5. Evaluate score-before-update gating (`score()` without state mutation)
    6. Serialize adaptive state to disk (`save_state()`)
    7. Restore into a fresh controller instance (`load_state()`)
    8. Verify deterministic trajectory continuation from restored state

Usage:
    python examples/basic_adaptation.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch  # noqa: E402

from deltacore import (  # noqa: E402
    AdaptiveController,
    ControllerConfig,
    DeterministicFeatureHasher,
)


def main() -> None:
    print("=" * 72)
    print("DeltaCore Smoke Example: End-to-End Online Adaptation & Persistence")
    print("=" * 72)

    # 1. Initialize Feature Hasher and Controller
    feature_dim = 64
    hasher = DeterministicFeatureHasher(dim=feature_dim, normalize=True)
    config = ControllerConfig(
        dim=feature_dim,
        eta0=0.03,
        rho=1.50,
        gamma=0.10,
        epsilon=1e-6,
    )
    controller = AdaptiveController(config)
    print(
        f"[1] Initialized AdaptiveController (dim={feature_dim}, eta0={config.eta0}, rho={config.rho})"
    )

    # 2. Define sequence of structured telemetry events
    nominal_event = {
        "service": "checkout",
        "operation": "process_payment",
        "provider": "stripe",
        "http_status": 200,
        "transport": "http2",
        "outcome": "success",
    }
    anomalous_event = {
        "service": "checkout",
        "operation": "process_payment",
        "provider": "stripe",
        "http_status": 504,
        "transport": "gateway_timeout",
        "outcome": "failure",
    }

    # 3. Stream nominal events and observe adaptive residual contraction
    print("\n[2] Streaming nominal events (auto-associative mode, v_t = x_t):")
    x_nom = hasher.encode(nominal_event)

    for t in range(1, 6):
        res = controller.step(x_nom, adapt=True)
        print(
            f"    Step {t:02d} | Pre-update Residual: {res.reconstruction_residual:.4f} | "
            f"Step Size: {res.step_size:.4f} | Margin: {res.stability_margin:.4f} | "
            f"State Norm: {res.state_norm:.4f}"
        )

    # 4. Demonstrate Score-Before-Update Gating
    print(
        "\n[3] Evaluating anomalous event via non-mutating score() (Score-Before-Update):"
    )
    x_anom = hasher.encode(anomalous_event)

    score_res = controller.score(x_anom)
    pre_score_norm = controller.get_state().norm().item()
    print(
        f"    Anomalous Pre-update Residual: {score_res.reconstruction_residual:.4f} "
        f"(elevated relative to nominal baseline)"
    )
    print(
        f"    Adapted: {score_res.adapted} | Update Norm: {score_res.update_norm:.4f}"
    )
    print(
        f"    State norm before score: {pre_score_norm:.4f} | after score: {controller.get_state().norm().item():.4f}"
    )
    assert controller.get_state().norm().item() == pre_score_norm, (
        "score() must NOT mutate state!"
    )

    # 5. State Serialization & Recovery
    print("\n[4] State Persistence & Recovery Check:")
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "deltacore_checkpoint.json"
        controller.save_state(state_file)
        print(
            f"    Saved state to: {state_file.name} (payload size: {state_file.stat().st_size} bytes)"
        )

        # Create fresh controller instance and load state
        restored_controller = AdaptiveController(config)
        restored_controller.load_state(state_file)
        print("    Restored state into a fresh AdaptiveController instance.")

        # Verify state equivalence
        orig_state = controller.get_state()
        rest_state = restored_controller.get_state()
        diff = float(torch.linalg.norm(orig_state - rest_state).item())
        print(f"    Frobenius difference between original and restored: {diff:.1e}")
        assert diff == 0.0, "Restored state must match bit-for-bit!"

        # 6. Stream continuation test
        res_orig = controller.step(x_nom, adapt=True)
        res_rest = restored_controller.step(x_nom, adapt=True)
        assert torch.allclose(res_orig.prediction, res_rest.prediction), (
            "Predictions must match exactly!"
        )
        print(
            f"    Continued step residual match: orig={res_orig.reconstruction_residual:.4f}, rest={res_rest.reconstruction_residual:.4f}"
        )

    print("\n" + "=" * 72)
    print("SUCCESS: DeltaCore smoke example passed cleanly.")
    print("=" * 72)


if __name__ == "__main__":
    main()
