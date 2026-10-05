# ==============================================================================
# DeltaCore: deltacore/benchmarks/__main__.py
# Module entrypoint for running benchmarks via `python3 -m deltacore.benchmarks`.
# ==============================================================================

import sys

from deltacore.benchmarks.cli import main

if __name__ == "__main__":
    sys.exit(main())
