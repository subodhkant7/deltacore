"""Regression tests for DeltaCore packaging, dependencies, and environment integrity.

Verifies:
1. PyTorch is declared as a runtime dependency in pyproject.toml.
2. PyTorch is available and >= 2.2.0 at runtime.
3. Core DeltaCore modules (including deltacore.updates.delta) import cleanly.
4. DeltaRule operator instantiates and computes update steps.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import tomllib
import torch

from deltacore.updates.delta import DeltaRule


def test_torch_declared_in_pyproject_dependencies() -> None:
    """Verify that PyTorch is explicitly declared in pyproject.toml dependencies."""
    project_root = Path(__file__).resolve().parent.parent
    pyproject_path = project_root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml must exist in project root"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project_data = data.get("project", {})
    dependencies = project_data.get("dependencies", [])

    torch_deps = [d for d in dependencies if d.startswith("torch")]
    assert len(torch_deps) > 0, (
        "torch must be declared as a required runtime dependency in [project].dependencies"
    )
    assert "torch>=2.2.0" in torch_deps[0], (
        f"Expected torch>=2.2.0, found {torch_deps[0]}"
    )


def test_torch_version_constraint() -> None:
    """Verify that runtime PyTorch version satisfies project requirement (>= 2.2.0)."""
    version_str = torch.__version__.split("+")[0]
    major, minor = (int(x) for x in version_str.split(".")[:2])
    assert (major > 2) or (major == 2 and minor >= 2), (
        f"PyTorch version must be >= 2.2.0, got {torch.__version__}"
    )


def test_core_modules_importability() -> None:
    """Verify clean importability of key DeltaCore architectural modules."""
    core_modules = [
        "deltacore",
        "deltacore.updates.delta",
        "deltacore.updates.adaptive_delta",
        "deltacore.updates.self_referential",
        "deltacore.memory.associative",
        "deltacore.memory.five_memory",
        "deltacore.scans.sequential",
        "deltacore.scans.boundary_chunked",
        "deltacore.scans.adaptive",
        "deltacore.spatial.operator",
        "deltacore.streaming.models",
        "deltacore.streaming.spatiotemporal_regimes",
        "deltacore.observatory.analysis",
    ]
    for mod_name in core_modules:
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "__file__"), (
            f"Module {mod_name} must have a valid __file__ attribute"
        )


def test_delta_rule_functional_execution() -> None:
    """Verify DeltaRule (from deltacore.updates.delta) instantiates and computes forward steps."""
    k = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float32)
    v = torch.tensor([0.2, 0.8], dtype=torch.float32)
    M = torch.zeros(2, 3, dtype=torch.float32)

    rule = DeltaRule(step_size=0.1)
    res = rule.step(M, k, v)
    assert res.prediction.shape == (2,)
    assert res.error.shape == (2,)
    assert res.update.shape == (2, 3)
    assert torch.is_tensor(res.new_memory)
    assert res.new_memory.shape == (2, 3)
