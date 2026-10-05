# ==============================================================================
# DeltaCore: tests/observatory/test_replay_and_cli.py
# Unit tests for deterministic replay and CLI execution.
# ==============================================================================

from pathlib import Path

from deltacore.benchmarks.baselines.factory import get_baseline
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.tasks import get_task
from deltacore.observatory.cli import main
from deltacore.observatory.replay import execute_replay
from deltacore.observatory.report import generate_observatory_report


def test_replay_exact_match(tmp_path: Path):
    """Verify that deterministic replay of a synthetic task produces exact match."""
    cfg = BenchmarkConfig(
        experiment_name="stationary_recall",
        seed=123,
        controller="fixed",
        key_dim=16,
        value_dim=16,
        sequence_length=20,
    )
    task = get_task("stationary_recall")
    model = get_baseline("fixed", key_dim=16, value_dim=16)

    # 1. Run task and save RunResult
    res = task.run(model, cfg)
    res_path = tmp_path / "result.json"
    res.save(res_path)

    # 2. Replay
    replay_out = execute_replay(res_path)
    assert replay_out.status == "exact match"
    assert replay_out.max_absolute_diff == 0.0
    assert replay_out.regenerated_trajectory is not None
    assert len(replay_out.regenerated_trajectory) == 20


def test_replay_detects_metric_tampering(tmp_path: Path):
    """Verify that if a stored metric is modified, replay reports 'different'."""
    cfg = BenchmarkConfig(
        experiment_name="stationary_recall",
        seed=123,
        controller="fixed",
        key_dim=16,
        value_dim=16,
        sequence_length=20,
    )
    task = get_task("stationary_recall")
    model = get_baseline("fixed", key_dim=16, value_dim=16)

    res = task.run(model, cfg)
    # Tamper with a metric
    res.metrics["final_error"] = 999.99
    res_path = tmp_path / "tampered.json"
    res.save(res_path)

    replay_out = execute_replay(res_path)
    assert replay_out.status == "different"
    assert replay_out.max_absolute_diff > 1.0


def test_cli_inspect_and_plot_and_replay(tmp_path: Path):
    """Smoke test CLI subcommands on a real result."""
    cfg = BenchmarkConfig(
        experiment_name="distribution_shift",
        seed=42,
        controller="fixed",
        key_dim=16,
        value_dim=16,
        sequence_length=20,
    )
    task = get_task("distribution_shift")
    model = get_baseline("fixed", key_dim=16, value_dim=16)

    res = task.run(model, cfg)
    res_path = tmp_path / "dist_shift.json"
    res.save(res_path)

    # 1. Test CLI inspect
    report_file = tmp_path / "report.md"
    exit_code = main(["inspect", str(res_path), "--report", str(report_file)])
    assert exit_code == 0
    assert report_file.is_file() and report_file.stat().st_size > 0

    # 2. Test CLI plot
    plots_dir = tmp_path / "plots"
    exit_code = main(["plot", str(res_path), "--output", str(plots_dir)])
    assert exit_code == 0
    assert (plots_dir / "plot_a_error_vs_time.png").is_file()

    # 3. Test CLI replay
    exit_code = main(["replay", str(res_path)])
    assert exit_code == 0


def test_report_generation(tmp_path: Path):
    """Verify that Markdown reports compile with valid structure and no NaN crashes."""
    cfg = BenchmarkConfig(
        experiment_name="key_interference",
        seed=77,
        controller="fixed",
        key_dim=16,
        value_dim=16,
        sequence_length=15,
    )
    task = get_task("key_interference")
    model = get_baseline("fixed", key_dim=16, value_dim=16)

    res = task.run(model, cfg)
    replay_res = execute_replay(res)
    assert replay_res.regenerated_trajectory is not None

    rep_path = tmp_path / "obs_report.md"
    content = generate_observatory_report(
        replay_res.regenerated_trajectory,
        replay_result=replay_res,
        output_path=rep_path,
    )
    assert "DeltaCore State Observatory Report" in content
    assert "Trajectory Summary & Fingerprint" in content
    assert "Epistemic Guardrails & Scientific Limitations" in content
    assert rep_path.is_file()
