"""CSV / JSON exports: order book, forecasts, backtest metrics and the whole dataset.

* :func:`export_orders`    — rows of the ``replenishment_summary`` named query for a run, plus
  ``stockout_risk`` / ``priority`` / ``requested_qty`` when ``replenishment_orders`` has those
  columns (schema v2); same ordering as the query (order cost descending).
* :func:`export_forecasts` — one row per series x target day with the store code and SKU.
* :func:`export_metrics`   — every backtest candidate per series (``selected`` = 1 marks the
  winner).
* :func:`export_dataset`   — the six loadable tables as ``<table>.csv`` (header = table
  columns, ISO dates, UTF-8), exactly what :func:`demandcast.ingest.load_dataset` consumes.

``run_id=None`` means the latest successful run.  CSV files are written with the standard
library's ``csv`` module (NULL → empty field); JSON files hold a list of row objects.

Spreadsheet safety — the ``cells`` policy (shared with ``query --format csv``, see
:mod:`demandcast.csvsafe`)
    The three *reporting* exports are meant to be opened in a spreadsheet, and text that came
    in through CSV ingest (store codes, cities, SKUs, names, categories, reasons) may look like
    a formula: ``=1+1``, ``+1``, ``-1``, ``@x``, or the same behind a tab or an invisible
    character.  With the default ``cells="safe"`` every such text cell is written as ``'`` +
    text so a spreadsheet treats it as text (the apostrophe is the conventional marker; some
    applications display it).  Numbers, dates and empty cells are untouched, and database
    values are never modified.  ``cells="raw"`` writes every cell verbatim for machine
    consumers.  JSON output is never transformed, whatever ``cells`` says.

    :func:`export_dataset` is **raw machine data** for ``demandcast load`` — a lossless round
    trip of the six tables, not a report.  It has no ``cells`` option (the CLI rejects
    ``--cells`` for it) and should not be opened in a spreadsheet that evaluates formulas.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import csvsafe, db, pipeline
from .ingest import LOADABLE_TABLES

FORMATS = ("csv", "json")
ORDER_EXTRA_COLUMNS = ("stockout_risk", "priority", "requested_qty")

Rows = tuple[list[str], list[dict[str, Any]]]


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _check_format(fmt: str) -> str:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)} (got '{fmt}')")
    return fmt


def _resolve_run_id(conn: sqlite3.Connection, run_id: int | None) -> int:
    if run_id is None:
        run_id = pipeline.latest_successful_run(conn)
        if run_id is None:
            raise ValueError("no successful forecast run found — run `demandcast run` first")
    row = conn.execute("SELECT run_id FROM forecast_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"run {run_id} not found")
    return int(run_id)


def _query(conn: sqlite3.Connection, sql: str, params: Any = ()) -> Rows:
    """Run ``sql`` and return (column names, rows as dicts) independent of ``row_factory``."""
    cur = conn.execute(sql, params)
    columns = [d[0] for d in cur.description]
    rows = [dict(zip(columns, tuple(r), strict=True)) for r in cur.fetchall()]
    return columns, rows


def _orders(conn: sqlite3.Connection, run_id: int) -> Rows:
    columns, rows = _query(conn, db.QUERIES["replenishment_summary"], {"run_id": run_id})
    present = {r[1] for r in conn.execute("PRAGMA table_info(replenishment_orders)")}
    extra = [c for c in ORDER_EXTRA_COLUMNS if c in present and c not in columns]
    if extra:
        sql = (
            f"SELECT order_id, {', '.join(_q(c) for c in extra)} "
            "FROM replenishment_orders WHERE run_id = ?"
        )
        by_id = {r["order_id"]: r for r in _query(conn, sql, (run_id,))[1]}
        for row in rows:
            src = by_id.get(row.get("order_id"), {})
            for c in extra:
                row[c] = src.get(c)
        columns = columns + extra
    return columns, rows


def _forecasts(conn: sqlite3.Connection, run_id: int) -> Rows:
    return _query(
        conn,
        "SELECT f.*, st.store_code, p.sku FROM forecasts f "
        "JOIN stores st ON st.store_id = f.store_id "
        "JOIN products p ON p.product_id = f.product_id "
        "WHERE f.run_id = ? ORDER BY f.store_id, f.product_id, f.target_day",
        (run_id,),
    )


def _metrics(conn: sqlite3.Connection, run_id: int) -> Rows:
    return _query(
        conn,
        "SELECT m.*, st.store_code, p.sku FROM backtest_metrics m "
        "JOIN stores st ON st.store_id = m.store_id "
        "JOIN products p ON p.product_id = m.product_id "
        "WHERE m.run_id = ? ORDER BY m.store_id, m.product_id, m.selected DESC, m.model_name",
        (run_id,),
    )


_EXPORTERS: dict[str, Callable[[sqlite3.Connection, int], Rows]] = {
    "orders": _orders,
    "forecasts": _forecasts,
    "metrics": _metrics,
}


def _write(
    columns: list[str],
    rows: list[dict[str, Any]],
    out: str | Path,
    fmt: str,
    cells: str = "safe",
) -> Path:
    """Write ``rows`` to ``out`` as CSV (``cells`` policy applied) or JSON (always verbatim).

    The policy is validated before anything is created on disk; ``cells="safe"`` passes the
    rows through :func:`demandcast.csvsafe.safe_rows`, which only touches string cells.
    """
    csvsafe.check_policy(cells)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        out_rows = csvsafe.safe_rows(rows) if cells == "safe" else rows
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(out_rows)
    else:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, default=str)
            fh.write("\n")
    return path


def _export(
    kind: str,
    conn: sqlite3.Connection,
    run_id: int | None,
    out: str | Path,
    fmt: str,
    cells: str = "safe",
) -> Path:
    _check_format(fmt)
    csvsafe.check_policy(cells)
    rid = _resolve_run_id(conn, run_id)
    columns, rows = _EXPORTERS[kind](conn, rid)
    return _write(columns, rows, out, fmt, cells)


def export_orders(
    conn: sqlite3.Connection,
    run_id: int | None,
    out: str | Path,
    fmt: str = "csv",
    cells: str = "safe",
) -> Path:
    """Write the order book of ``run_id`` (default: latest successful run) to ``out``.

    ``cells``: ``"safe"`` (default) makes CSV text cells spreadsheet-safe, ``"raw"`` writes
    them verbatim; JSON output is never transformed.  See the module docstring.
    """
    return _export("orders", conn, run_id, out, fmt, cells)


def export_forecasts(
    conn: sqlite3.Connection,
    run_id: int | None,
    out: str | Path,
    fmt: str = "csv",
    cells: str = "safe",
) -> Path:
    """Write the point forecasts and intervals of ``run_id`` to ``out`` (``cells`` as above)."""
    return _export("forecasts", conn, run_id, out, fmt, cells)


def export_metrics(
    conn: sqlite3.Connection,
    run_id: int | None,
    out: str | Path,
    fmt: str = "csv",
    cells: str = "safe",
) -> Path:
    """Write the backtest metrics of every candidate model of ``run_id`` (``cells`` as above)."""
    return _export("metrics", conn, run_id, out, fmt, cells)


def export_dataset(conn: sqlite3.Connection, out_dir: str | Path) -> dict[str, int]:
    """Write ``<table>.csv`` for the six loadable tables into ``out_dir``; returns row counts.

    Header = the table's columns (``PRAGMA table_info`` order), rows ordered by primary key,
    NULL as an empty field, dates as stored (ISO ``YYYY-MM-DD``).  The files are **raw machine
    data** for :func:`demandcast.ingest.load_dataset` (every value verbatim, so the round trip
    is lossless); they are not spreadsheet-safe reports — use the reporting exports for that.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for table in LOADABLE_TABLES:
        info = conn.execute(f"PRAGMA table_info({_q(table)})").fetchall()
        columns = [r[1] for r in info]
        pk = [r[1] for r in sorted((r for r in info if r[5] > 0), key=lambda r: r[5])]
        col_list = ", ".join(_q(c) for c in columns)
        order = ", ".join(_q(c) for c in (pk or columns))
        cur = conn.execute(f"SELECT {col_list} FROM {_q(table)} ORDER BY {order}")
        n = 0
        with (directory / f"{table}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for r in cur:
                writer.writerow(tuple(r))
                n += 1
        counts[table] = n
    return counts


# ---- CLI plugin ----------------------------------------------------------------------------------

_CELLS_HELP = (
    "CSV cell policy for orders/forecasts/metrics (default: safe). 'safe' prefixes text cells "
    "that a spreadsheet would evaluate as a formula (first visible character =, +, -, @ or a "
    "leading tab/CR/LF/invisible character) with an apostrophe so they stay text; numbers, "
    "dates and JSON output are never changed. 'raw' writes every cell verbatim for machine "
    "consumers. Not accepted for 'dataset', which is always raw machine data for `load`."
)


def _cli_export(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    """``demandcast export {orders,forecasts,metrics,dataset} --out PATH [--cells safe|raw]``."""
    ensure_schema = getattr(db, "ensure_schema", None)  # schema v2 helper (present since 0.4.0)
    if callable(ensure_schema):
        ensure_schema(conn)
    cells_opt: str | None = getattr(args, "cells", None)
    if args.what == "dataset":
        if cells_opt is not None:
            raise ValueError(
                "dataset export is always raw machine data for `demandcast load` — "
                "--cells applies to orders, forecasts and metrics only"
            )
        if args.format == "json":
            raise ValueError("dataset export is CSV only (one <table>.csv per table)")
        counts = export_dataset(conn, args.out)
        summary = {"export": "dataset", "path": str(Path(args.out)), "rows": counts, "cells": "raw"}
        print(json.dumps(summary, indent=2))
        return 0
    fmt = args.format or ("json" if Path(args.out).suffix.lower() == ".json" else "csv")
    _check_format(fmt)
    cells = csvsafe.check_policy(cells_opt or "safe")
    run_id = _resolve_run_id(conn, args.run_id)
    columns, rows = _EXPORTERS[args.what](conn, run_id)
    path = _write(columns, rows, args.out, fmt, cells)
    payload: dict[str, Any] = {
        "export": args.what,
        "run_id": run_id,
        "format": fmt,
        "rows": len(rows),
        "path": str(path),
    }
    if fmt == "csv":
        payload["cells"] = cells
    print(json.dumps(payload, indent=2))
    return 0


def register_cli(sub: argparse._SubParsersAction) -> None:
    """Register ``export {orders,forecasts,metrics,dataset} --out PATH [--run-id N] [--format]
    [--cells safe|raw]``."""
    p = sub.add_parser(
        "export",
        help="export the order book, forecasts, backtest metrics or the whole dataset",
        description=(
            "Export run results as CSV/JSON (format defaults to the --out suffix) or the six "
            "core tables as <table>.csv files that `demandcast load --dir` can re-import. "
            "Reporting CSVs (orders, forecasts, metrics) are spreadsheet-safe by default "
            "(--cells safe); the dataset export is always raw machine data for `load` and does "
            "not accept --cells."
        ),
    )
    p.add_argument("what", choices=("orders", "forecasts", "metrics", "dataset"))
    p.add_argument(
        "--out", required=True, help="output file (orders/forecasts/metrics) or directory (dataset)"
    )
    p.add_argument("--run-id", type=int, default=None, help="default: latest successful run")
    p.add_argument("--format", choices=FORMATS, default=None)
    p.add_argument("--cells", choices=csvsafe.CELL_POLICIES, default=None, help=_CELLS_HELP)
    p.set_defaults(handler=_cli_export, creates_db=False)
