# ==============================================================================
# DeltaCore: tests/benchmarks/test_cli.py
# Unit tests for the benchmark command-line interface.
# ==============================================================================

from pathlib import Path

import pytest

from deltacore.benchmarks.cli import build_parser, main


def test_cli_parser_list() -> None:
    """Verify CLI list command builds without error."""
    parser = build_parser()
    args = parser.parse_args(["list"])
    assert args.command == "list"


def test_cli_parser_run_defaults() -> None:
    """Verify CLI run command parses task and default options."""
    parser = build_parser()
    args = parser.parse_args(["run", "distribution_shift"])
    assert args.command == "run"
    assert args.task == "distribution_shift"
    assert args.seq_len == 128
    assert args.dtype == "float32"
    assert args.device == "cpu"
    assert args.seeds == [0, 1, 2, 3, 4]


def test_cli_parser_custom_args() -> None:
    """Verify custom CLI options parse properly."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "stationary_recall",
            "--models",
            "fixed",
            "safe_self_referential",
            "--seeds",
            "7",
            "8",
            "--seq-len",
            "64",
            "--key-dim",
            "32",
            "--val-dim",
            "32",
            "--dtype",
            "float64",
            "--device",
            "cpu",
            "--output",
            "custom_results",
        ]
    )
    assert args.task == "stationary_recall"
    assert args.models == ["fixed", "safe_self_referential"]
    assert args.seeds == [7, 8]
    assert args.seq_len == 64
    assert args.key_dim == 32
    assert args.val_dim == 32
    assert args.dtype == "float64"
    assert args.output == "custom_results"


def test_cli_invalid_task_rejected() -> None:
    """Invalid task name must trigger an argument error."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "non_existent_task"])


def test_cli_main_execution(tmp_path: Path) -> None:
    """Test full execution of main with run command into a temporary directory."""
    exit_code = main(
        [
            "run",
            "distribution_shift",
            "--models",
            "fixed",
            "--seeds",
            "0",
            "--seq-len",
            "16",
            "--output",
            str(tmp_path / "cli_results"),
            "--quiet",
        ]
    )
    assert exit_code == 0
    # Check that artifact directory was generated
    assert (tmp_path / "cli_results" / "distribution_shift").exists()
