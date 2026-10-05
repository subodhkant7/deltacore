# ==============================================================================
# DeltaCore: tests/observatory/test_plots_smoke.py
# Smoke tests for static publication-quality plots (Plots A through I).
# ==============================================================================

from pathlib import Path

from deltacore.observatory.plots import (
    generate_all_plots,
    plot_comparative_recovery_curves,
    plot_error_vs_time,
    plot_failure_step_distribution,
    plot_key_correlation_vs_error,
    plot_memory_norm_vs_time,
    plot_recovery_latency_vs_energy,
    plot_stability_margin_vs_time,
    plot_step_size_vs_time,
    plot_update_norm_vs_time,
)
from deltacore.observatory.schema import StateTrajectory, TrajectoryStep


def _make_dummy_trajectory(
    model: str = "TestModel", failure_step: int | None = None
) -> StateTrajectory:
    steps = []
    for t in range(10):
        if failure_step is not None and t >= failure_step:
            steps.append(
                TrajectoryStep(
                    step=t,
                    error_norm=None,
                    step_size=None,
                    memory_norm=None,
                    finite_state=False,
                )
            )
        else:
            steps.append(
                TrajectoryStep(
                    step=t,
                    error_norm=1.0 / (t + 1),
                    step_size=0.1,
                    step_size_change=0.0,
                    memory_norm=1.0 + 0.1 * t,
                    dynamics_memory_norm=0.5,
                    update_norm=0.2 / (t + 1),
                    dynamics_update_norm=0.05,
                    normalized_step=0.2,
                    stability_margin=1.8,
                    finite_state=True,
                )
            )
    return StateTrajectory(
        model=model,
        task="distribution_shift",
        seed=42,
        steps=steps,
        all_states_finite=failure_step is None,
        first_nonfinite_step=failure_step,
        metadata={"shift_position": 4, "tau": 0.5, "window": 2},
    )


def test_core_plots_smoke(tmp_path: Path):
    """Verify that Plots A through E render and save without error."""
    traj = _make_dummy_trajectory("HealthyModel")

    p_a = plot_error_vs_time(traj, output_path=tmp_path / "plot_a.png")
    assert Path(p_a).is_file() and Path(p_a).stat().st_size > 0

    p_b = plot_step_size_vs_time(traj, output_path=tmp_path / "plot_b.png")
    assert Path(p_b).is_file() and Path(p_b).stat().st_size > 0

    p_c = plot_memory_norm_vs_time(traj, output_path=tmp_path / "plot_c.png")
    assert Path(p_c).is_file() and Path(p_c).stat().st_size > 0

    p_d = plot_update_norm_vs_time(traj, output_path=tmp_path / "plot_d.png")
    assert Path(p_d).is_file() and Path(p_d).stat().st_size > 0

    p_e = plot_stability_margin_vs_time(traj, output_path=tmp_path / "plot_e.png")
    assert Path(p_e).is_file() and Path(p_e).stat().st_size > 0


def test_failure_plot_smoke(tmp_path: Path):
    """Verify that failed trajectories visibly render failure annotations."""
    traj_fail = _make_dummy_trajectory("FailedModel", failure_step=6)
    p_fail = plot_error_vs_time(
        traj_fail, output_path=tmp_path / "plot_a_fail.png", annotate=True
    )
    assert Path(p_fail).is_file() and Path(p_fail).stat().st_size > 0


def test_comparative_plots_smoke(tmp_path: Path):
    """Verify Plots F through I render and save without error."""
    traj1 = _make_dummy_trajectory("Model1")
    traj2 = _make_dummy_trajectory("Model2", failure_step=7)

    # Plot F: Comparative recovery
    p_f = plot_comparative_recovery_curves(
        [traj1, traj2], output_path=tmp_path / "plot_f.png"
    )
    assert Path(p_f).is_file() and Path(p_f).stat().st_size > 0

    # Plot G: Key correlation vs error
    p_g = plot_key_correlation_vs_error(
        [0.0, 0.5, 0.9], [0.001, 0.02, 0.35], output_path=tmp_path / "plot_g.png"
    )
    assert Path(p_g).is_file() and Path(p_g).stat().st_size > 0

    # Plot H: Latency vs Energy
    p_h = plot_recovery_latency_vs_energy(
        ["M1", "M2"], [5, None], [12.0, 25.0], output_path=tmp_path / "plot_h.png"
    )
    assert Path(p_h).is_file() and Path(p_h).stat().st_size > 0

    # Plot I: Failure step distribution
    p_i = plot_failure_step_distribution(
        {"M1": [None, None], "M2": [12, 14]}, output_path=tmp_path / "plot_i.png"
    )
    assert Path(p_i).is_file() and Path(p_i).stat().st_size > 0


def test_generate_all_plots(tmp_path: Path):
    traj = _make_dummy_trajectory("AllPlotsModel")
    out_dir = tmp_path / "all_plots"
    paths = generate_all_plots(traj, output_dir=out_dir)
    assert len(paths) == 5
    for p in paths.values():
        assert p.is_file() and p.stat().st_size > 0


def test_spatial_plots_smoke(tmp_path: Path):
    """Verify Plots J through N render and save without error."""
    import torch

    from deltacore.observatory.plots import (
        plot_chunk_size_difference,
        plot_directional_learning_rate_map,
        plot_directional_memory_activity_map,
        plot_directional_output_magnitude,
        plot_fused_spatial_output,
    )

    dummy_maps = {
        "RIGHT": torch.randn(1, 3, 4, 4),
        "LEFT": torch.randn(1, 3, 4, 4),
        "DOWN": torch.randn(1, 3, 4, 4),
        "UP": torch.randn(1, 3, 4, 4),
    }

    # Plot J: Directional output magnitude
    pj = plot_directional_output_magnitude(
        dummy_maps, output_path=tmp_path / "plot_j.png"
    )
    assert Path(pj).is_file() and Path(pj).stat().st_size > 0

    # Plot K: Directional learning-rate map
    pk = plot_directional_learning_rate_map(
        dummy_maps, output_path=tmp_path / "plot_k.png"
    )
    assert Path(pk).is_file() and Path(pk).stat().st_size > 0

    # Plot L: Directional memory-activity map
    pl = plot_directional_memory_activity_map(
        dummy_maps, output_path=tmp_path / "plot_l.png"
    )
    assert Path(pl).is_file() and Path(pl).stat().st_size > 0

    # Plot M: Fused spatial output
    pm = plot_fused_spatial_output(
        torch.randn(1, 3, 4, 4), output_path=tmp_path / "plot_m.png"
    )
    assert Path(pm).is_file() and Path(pm).stat().st_size > 0

    # Plot N: Chunk size difference
    pn = plot_chunk_size_difference(
        {1: 0.0, 2: 0.05, 4: 0.12, 8: 0.28}, output_path=tmp_path / "plot_n.png"
    )
    assert Path(pn).is_file() and Path(pn).stat().st_size > 0


def test_learning_plots_smoke(tmp_path: Path):
    """Verify Plots O through T render and save without error."""
    from deltacore.observatory.plots import (
        plot_absolute_vs_relative_error,
        plot_accuracy_vs_directions,
        plot_metric_vs_directional_route,
        plot_runtime_vs_chunk_size,
        plot_train_val_loss,
        plot_validation_error_vs_chunk_size,
    )

    # Plot O
    po = plot_train_val_loss(
        [0.8, 0.5, 0.3, 0.2],
        [0.85, 0.55, 0.35, 0.25],
        output_path=tmp_path / "plot_o.png",
    )
    assert Path(po).is_file() and Path(po).stat().st_size > 0

    # Plot P
    pp = plot_accuracy_vs_directions(
        {"1-Dir": 65.0, "2-Dir": 78.0, "4-Dir": 84.0},
        output_path=tmp_path / "plot_p.png",
    )
    assert Path(pp).is_file() and Path(pp).stat().st_size > 0

    # Plot Q
    pq = plot_validation_error_vs_chunk_size(
        {1: 0.15, 2: 0.18, 4: 0.22, 8: 0.30}, output_path=tmp_path / "plot_q.png"
    )
    assert Path(pq).is_file() and Path(pq).stat().st_size > 0

    # Plot R
    pr = plot_runtime_vs_chunk_size(
        {1: 5.2, 2: 5.5, 4: 5.1, 8: 5.0}, output_path=tmp_path / "plot_r.png"
    )
    assert Path(pr).is_file() and Path(pr).stat().st_size > 0

    # Plot S
    ps = plot_metric_vs_directional_route(
        {"RIGHT": 0.25, "LEFT": 0.24, "DOWN": 0.22, "UP": 0.23, "All": 0.18},
        output_path=tmp_path / "plot_s.png",
    )
    assert Path(ps).is_file() and Path(ps).stat().st_size > 0

    # Plot T
    pt = plot_absolute_vs_relative_error(
        ["Zero", "Linear", "DeltaCore-1D", "DeltaCore-4D"],
        [28.0, 14.5, 11.2, 8.4],
        [1.0, 0.52, 0.40, 0.30],
        output_path=tmp_path / "plot_t.png",
    )
    assert Path(pt).is_file() and Path(pt).stat().st_size > 0
