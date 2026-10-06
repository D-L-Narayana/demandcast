"""``python -m demandcast ...`` entry point (same as the ``demandcast`` console script)."""

from .cli import main

raise SystemExit(main())
