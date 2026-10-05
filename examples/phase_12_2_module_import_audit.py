"""Comprehensive import audit across all DeltaCore modules for Phase 12.2.

Iterates over every .py file in deltacore/, dynamically loads it via importlib,
and records the outcome in docs/benchmarks/artifacts/phase_12_2/module_import_audit.json.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


def audit_module_imports() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    deltacore_dir = project_root / "deltacore"

    py_files = sorted(deltacore_dir.rglob("*.py"))
    results = []

    for file_path in py_files:
        rel_path = file_path.relative_to(project_root)
        # Convert path to module name: e.g. deltacore/updates/delta.py -> deltacore.updates.delta
        parts = list(rel_path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join(parts)

        entry: dict[str, Any] = {
            "module": module_name,
            "file_path": str(rel_path),
            "status": "IMPORT_OK",
            "exception_type": None,
            "exception_message": None,
        }

        try:
            mod = importlib.import_module(module_name)
            # Verify module has __file__
            if not hasattr(mod, "__file__"):
                entry["status"] = "IMPORT_ERROR"
                entry["exception_type"] = "MissingFileAttribute"
                entry["exception_message"] = "Imported object has no __file__"
        except ModuleNotFoundError as e:
            missing_pkg = getattr(e, "name", str(e))
            if missing_pkg in ("torch", "numpy", "matplotlib"):
                entry["status"] = "ENVIRONMENT_ERROR"
            else:
                entry["status"] = "IMPORT_ERROR"
            entry["exception_type"] = type(e).__name__
            entry["exception_message"] = str(e)
        except ImportError as e:
            entry["status"] = "IMPORT_ERROR"
            entry["exception_type"] = type(e).__name__
            entry["exception_message"] = str(e)
        except Exception as e:
            entry["status"] = "IMPORT_ERROR"
            entry["exception_type"] = type(e).__name__
            entry["exception_message"] = f"{type(e).__name__}: {e}"

        results.append(entry)

    total_modules = len(results)
    passed_modules = sum(1 for r in results if r["status"] == "IMPORT_OK")
    failed_modules = sum(1 for r in results if r["status"] != "IMPORT_OK")

    summary = {
        "audit_phase": "12.2",
        "total_modules_audited": total_modules,
        "import_ok_count": passed_modules,
        "import_error_count": failed_modules,
        "success_rate": passed_modules / total_modules if total_modules > 0 else 0.0,
        "verdict": "ALL_MODULES_IMPORT_OK"
        if failed_modules == 0
        else "IMPORT_FAILURES_DETECTED",
        "modules": results,
    }

    return summary


def main() -> None:
    output_dir = Path("docs/benchmarks/artifacts/phase_12_2")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = audit_module_imports()
    output_path = output_dir / "module_import_audit.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Audited {summary['total_modules_audited']} modules.")
    print(
        f"Import OK: {summary['import_ok_count']} / {summary['total_modules_audited']}"
    )
    print(f"Verdict: {summary['verdict']}")
    print(f"Artifact saved to {output_path}")


if __name__ == "__main__":
    main()
