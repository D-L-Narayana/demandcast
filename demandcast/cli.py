"""Command-line entry point.

demandcast init        --db demandcast.db [--stores 10 --products 40 --days 730]
demandcast run         --db demandcast.db [--horizon 28 --service-level 0.95]
demandcast query NAME  --db demandcast.db [--run-id N] [--limit 20]
demandcast dashboard   --db demandcast.db --out dashboard/index.html
demandcast stats       --db demandcast.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from . import __version__, dashboard, db, pipeline
from .simulate import SimConfig, generate


def _print_rows(rows: list[dict], limit: int) -> None:
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(_cell(r[c])) for r in rows[:limit])) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows[:limit]:
        print("  ".join(_cell(r[c]).ljust(widths[c]) for c in cols))
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows")


def _cell(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="demandcast", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", action="version", version=f"demandcast {__version__}")
    p.add_argument("--db", default="demandcast.db", help="SQLite database path")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create schema and generate a synthetic dataset")
    s.add_argument("--stores", type=int, default=10)
    s.add_argument("--products", type=int, default=40)
    s.add_argument("--days", type=int, default=730)
    s.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    s.add_argument("--seed", type=int, default=42)

    r = sub.add_parser("run", help="backtest, forecast and generate replenishment orders")
    r.add_argument("--horizon", type=int, default=28)
    r.add_argument("--folds", type=int, default=4)
    r.add_argument("--service-level", type=float, default=0.95)
    r.add_argument("--review-period", type=int, default=7)

    q = sub.add_parser("query", help="execute a named analytics query")
    q.add_argument("name", nargs="?", help="query name (omit to list)")
    q.add_argument("--run-id", type=int)
    q.add_argument("--store-id", type=int)
    q.add_argument("--product-id", type=int)
    q.add_argument("--limit", type=int, default=20)
    q.add_argument("--json", action="store_true")

    d = sub.add_parser("dashboard", help="render the HTML dashboard")
    d.add_argument("--out", default="dashboard/index.html")
    d.add_argument("--run-id", type=int)

    sub.add_parser("stats", help="print row counts per table")

    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    conn = db.connect(a.db)

    if a.cmd == "init":
        db.init_schema(conn)
        if conn.execute("SELECT COUNT(*) FROM sales_daily").fetchone()[0]:
            print("database already contains data; delete it to regenerate", file=sys.stderr)
            return 1
        counts = generate(conn, SimConfig(a.stores, a.products, a.start, a.days, a.seed))
        print(json.dumps(counts, indent=2))
        return 0

    if a.cmd == "run":
        cfg = pipeline.RunConfig(
            horizon_days=a.horizon,
            n_folds=a.folds,
            service_level=a.service_level,
            review_period_days=a.review_period,
        )
        run_id = pipeline.run(conn, cfg)
        row = dict(conn.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone())
        print(json.dumps(row, indent=2, default=str))
        return 0

    if a.cmd == "query":
        if not a.name:
            print("\n".join(sorted(db.QUERIES)))
            return 0
        params = {}
        run_id = a.run_id or pipeline.latest_successful_run(conn)
        if run_id is not None:
            params["run_id"] = run_id
        if a.store_id is not None:
            params["store_id"] = a.store_id
        if a.product_id is not None:
            params["product_id"] = a.product_id
        needed = {k for k in ("run_id", "store_id", "product_id") if f":{k}" in db.QUERIES[a.name]}
        rows = db.run_query(conn, a.name, {k: params[k] for k in needed})
        if a.json:
            print(json.dumps(rows[: a.limit], indent=2, default=str))
        else:
            _print_rows(rows, a.limit)
        return 0

    if a.cmd == "dashboard":
        out = dashboard.render(conn, a.out, a.run_id)
        print(f"wrote {out}")
        return 0

    if a.cmd == "stats":
        print(json.dumps(db.table_counts(conn), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
