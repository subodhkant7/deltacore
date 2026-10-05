"""Infrastructure tests for DeltaCore repository foundation.

These tests verify package importability, namespace structure, versioning,
Python compatibility, and strict dependency boundaries as established in Phase 0.
"""

import importlib
import sys
from types import ModuleType

import pytest


def test_package_import():
    """Verify that the root deltacore package can be cleanly imported."""
    import deltacore

    assert isinstance(deltacore, ModuleType)
    assert hasattr(deltacore, "__file__")


def test_package_version_exists():
    """Verify that deltacore declares a valid non-empty string version."""
    import deltacore

    assert hasattr(deltacore, "__version__")
    assert isinstance(deltacore.__version__, str)
    assert len(deltacore.__version__.strip()) > 0


def test_test_discovery_works():
    """Verify that pytest discovery runs and assertions behave as expected."""
    assert True


@pytest.mark.parametrize(
    "submodule_name",
    [
        "deltacore.memory",
        "deltacore.updates",
        "deltacore.stability",
        "deltacore.scans",
        "deltacore.diagnostics",
        "deltacore.benchmarks",
    ],
)
def test_basic_submodule_imports(submodule_name: str):
    """Verify that all Phase 0 architectural submodules can be imported cleanly."""
    mod = importlib.import_module(submodule_name)
    assert isinstance(mod, ModuleType)
    assert hasattr(mod, "__file__")


def test_python_version_requirements():
    """Verify that the active Python environment meets the minimum version requirement."""
    # DeltaCore requires Python >= 3.10
    assert sys.version_info >= (3, 10), (
        f"DeltaCore requires Python >= 3.10, currently running on "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )


def test_no_unexpected_runtime_dependencies():
    """Verify that importing deltacore does not pull unexpected heavy dependencies.

    Only standard library, torch, and numpy are permitted as foundational
    dependencies. Tested in a clean isolated subprocess to prevent test-ordering pollution.
    """
    import subprocess

    forbidden_modules = [
        "scipy",
        "sklearn",
        "transformers",
        "accelerate",
        "deepspeed",
        "triton",
        "wandb",
        "tensorboard",
        "matplotlib",
    ]

    check_code = f"""
import deltacore
import sys
forbidden = {forbidden_modules}
for mod in forbidden:
    if mod in sys.modules:
        print(f"FAILED: {{mod}} loaded in sys.modules")
        sys.exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", check_code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Importing deltacore leaked forbidden dependency:\n{result.stdout}\n{result.stderr}"
    )
