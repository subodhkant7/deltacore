# ==============================================================================
# DeltaCore: deltacore/observatory/__main__.py
# CLI execution entrypoint: python3 -m deltacore.observatory
# ==============================================================================

import sys

from deltacore.observatory.cli import main

if __name__ == "__main__":
    sys.exit(main())
