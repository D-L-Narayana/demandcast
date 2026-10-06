"""Regenerate ``examples/mini`` from the synthetic generator.

The example dataset is ``SimConfig(n_stores=2, n_products=3, start=2024-01-01, days=120,
seed=3)`` exported with ``demandcast.export.export_dataset``; a test asserts the committed files
match, so re-run this script whenever the generator changes::

    python examples/make_mini.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    sys.path.insert(0, str(HERE.parent))  # import the checkout this script lives in
    from demandcast import db
    from demandcast.export import export_dataset
    from demandcast.simulate import SimConfig, generate

    conn = db.connect(":memory:")
    db.init_schema(conn)
    generate(conn, SimConfig(2, 3, date(2024, 1, 1), 120, 3))
    counts = export_dataset(conn, HERE / "mini")
    for table, n in counts.items():
        print(f"{table}.csv: {n} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
