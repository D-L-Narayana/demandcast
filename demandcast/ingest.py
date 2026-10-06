"""Validated CSV ingest: bring your own stores / products / calendar / promotions / sales /
inventory snapshots (see ``docs/data-format.md`` for the column specification).

Every file is validated row by row *before* anything is written: column types (integer,
number, ISO date), range checks (non-negatives, enumerations, ``discount_pct`` in (0, 1)),
foreign-key existence (``store_id`` / ``product_id``), duplicate keys inside the file and
clashes on UNIQUE columns (``store_code``, ``sku``).  Invalid rows are rejected with a reason
(``"row N: <reason>"`` where N is the line number in the file, the header being line 1) and the
valid rows are loaded; ``strict=True`` turns any rejection into an :class:`IngestError`
(a ``ValueError`` carrying the full report) and loads nothing.

Modes
    ``insert``   existing keys are rejected ("key … already exists")
    ``upsert``   ``INSERT … ON CONFLICT(<primary key>) DO UPDATE SET col = excluded.col``
    ``replace``  delete the rows whose keys appear in the file, then insert the file's rows
                 (child rows survive because foreign keys are checked when the load commits)

Sales rows get their calendar rows auto-filled (:func:`ensure_calendar`, ``holiday_name`` NULL)
and a default ``revenue = round(units_sold * unit_price, 2)`` when the column is absent or
empty.  Every non-dry-run load appends a provenance row to ``data_loads`` (schema v2).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sqlite3
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import db
from .db import insert_many

log = logging.getLogger("demandcast.ingest")

LOADABLE_TABLES = (
    "stores",
    "products",
    "calendar",
    "promotions",
    "sales_daily",
    "inventory_snapshots",
)
MODES = ("insert", "upsert", "replace")
MAX_ERRORS = 20  # messages kept in LoadReport.errors (counts always cover every row)
PROGRESS_EVERY_ROWS = 50_000
MAX_CALENDAR_SPAN_DAYS = 36_600  # ~100 years; guards the auto-fill against typo years

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CALENDAR_COLUMNS = (
    "day",
    "day_of_week",
    "week_of_year",
    "month",
    "year",
    "is_weekend",
    "holiday_name",
)
_FK_ID_COLUMN = {"stores": "store_id", "products": "product_id"}
_SAVEPOINT = "demandcast_load"


# ---- report -------------------------------------------------------------------------------------


@dataclass
class LoadReport:
    """Outcome of one CSV load (also the JSON printed by ``demandcast load``)."""

    table: str
    inserted: int = 0
    updated: int = 0
    rejected: int = 0
    errors: list[str] = field(default_factory=list)  # first MAX_ERRORS messages "row N: reason"
    rows_read: int = 0
    mode: str = "upsert"
    source: str = ""
    dry_run: bool = False
    calendar_days_added: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class IngestError(ValueError):
    """Raised in strict mode after the whole file was validated; ``.report`` has the details."""

    def __init__(self, message: str, report: LoadReport) -> None:
        super().__init__(message)
        self.report = report


# ---- table specification ------------------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    fk: dict[str, set[int]]  # referenced table -> existing ids
    prices: dict[int, float]  # product_id -> unit_price (default revenue for sales rows)


@dataclass(frozen=True)
class _Column:
    name: str
    kind: str  # int | float | text | date | flag
    required: bool = True
    default: Any = None
    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None
    choices: tuple[str, ...] | None = None
    fk: str | None = None


@dataclass(frozen=True)
class _TableSpec:
    name: str
    columns: tuple[_Column, ...]
    key: tuple[str, ...]
    unique: tuple[str, ...] = ()
    row_check: Callable[[dict[str, Any], _Context], str | None] | None = None


def _calendar_fields(d: date) -> dict[str, Any]:
    return {
        "day_of_week": d.weekday(),
        "week_of_year": d.isocalendar()[1],
        "month": d.month,
        "year": d.year,
        "is_weekend": 1 if d.weekday() >= 5 else 0,
    }


def _check_product(row: dict[str, Any], _ctx: _Context) -> str | None:
    if row["unit_price"] < row["unit_cost"]:
        return f"unit_price must be >= unit_cost (got {row['unit_price']} < {row['unit_cost']})"
    return None


def _check_promotion(row: dict[str, Any], _ctx: _Context) -> str | None:
    if row["end_day"] < row["start_day"]:
        return f"end_day must be >= start_day (got {row['end_day']} < {row['start_day']})"
    return None


def _check_calendar(row: dict[str, Any], _ctx: _Context) -> str | None:
    derived = _calendar_fields(date.fromisoformat(row["day"]))
    for name in ("day_of_week", "month", "year", "is_weekend"):
        given = row.get(name)
        if given is not None and given != derived[name]:
            return f"{name} does not match day {row['day']} (expected {derived[name]}, got {given})"
    for name, value in derived.items():
        if row.get(name) is None:
            row[name] = value
    return None


def _fill_revenue(row: dict[str, Any], ctx: _Context) -> str | None:
    if row.get("revenue") is None:
        row["revenue"] = round(row["units_sold"] * ctx.prices[row["product_id"]], 2)
    return None


_SPECS: dict[str, _TableSpec] = {
    "stores": _TableSpec(
        "stores",
        (
            _Column("store_id", "int", ge=1),
            _Column("store_code", "text"),
            _Column("city", "text"),
            _Column("region", "text"),
            _Column("format", "text", choices=("flagship", "standard", "express")),
            _Column("opened_on", "date"),
        ),
        key=("store_id",),
        unique=("store_code",),
    ),
    "products": _TableSpec(
        "products",
        (
            _Column("product_id", "int", ge=1),
            _Column("sku", "text"),
            _Column("name", "text"),
            _Column("category", "text"),
            _Column("unit_cost", "float", gt=0),
            _Column("unit_price", "float", ge=0),
            _Column("case_pack", "int", required=False, default=1, ge=1),
            _Column("lead_time_days", "int", ge=0),
            _Column("shelf_life_days", "int", required=False, gt=0),
        ),
        key=("product_id",),
        unique=("sku",),
        row_check=_check_product,
    ),
    "calendar": _TableSpec(
        "calendar",
        (
            _Column("day", "date"),
            _Column("day_of_week", "int", required=False),
            _Column("week_of_year", "int", required=False, ge=1, le=53),
            _Column("month", "int", required=False),
            _Column("year", "int", required=False),
            _Column("is_weekend", "flag", required=False),
            _Column("holiday_name", "text", required=False),
        ),
        key=("day",),
        row_check=_check_calendar,
    ),
    "promotions": _TableSpec(
        "promotions",
        (
            _Column("promo_id", "int", ge=1),
            _Column("product_id", "int", fk="products"),
            _Column("store_id", "int", required=False, fk="stores"),
            _Column("start_day", "date"),
            _Column("end_day", "date"),
            _Column("discount_pct", "float", gt=0, lt=1),
        ),
        key=("promo_id",),
        row_check=_check_promotion,
    ),
    "sales_daily": _TableSpec(
        "sales_daily",
        (
            _Column("store_id", "int", fk="stores"),
            _Column("product_id", "int", fk="products"),
            _Column("day", "date"),
            _Column("units_sold", "int", ge=0),
            _Column("revenue", "float", required=False, ge=0),
            _Column("stockout_flag", "flag", required=False, default=0),
        ),
        key=("store_id", "product_id", "day"),
        row_check=_fill_revenue,
    ),
    "inventory_snapshots": _TableSpec(
        "inventory_snapshots",
        (
            _Column("store_id", "int", fk="stores"),
            _Column("product_id", "int", fk="products"),
            _Column("snapshot_day", "date"),
            _Column("on_hand", "int", ge=0),
            _Column("on_order", "int", required=False, default=0, ge=0),
        ),
        key=("store_id", "product_id", "snapshot_day"),
    ),
}


def _spec(table: str) -> _TableSpec:
    if table not in _SPECS:
        raise ValueError(f"'{table}' is not loadable; choose one of {', '.join(LOADABLE_TABLES)}")
    return _SPECS[table]


# ---- value parsing -------------------------------------------------------------------------------


def _parse_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if math.isfinite(f) and f.is_integer() else None


def _parse_float(s: str) -> float | None:
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def _parse_date(s: str) -> str | None:
    """Strict ``YYYY-MM-DD`` (same on every Python version); returns the ISO string."""
    if not _ISO_DATE.match(s):
        return None
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        return None


def _bounds_error(col: _Column, value: float) -> str | None:
    """'<col> must be > 0 and < 1 (got 1.0)' when ``value`` violates any configured bound."""
    violated = (
        (col.gt is not None and not value > col.gt)
        or (col.ge is not None and not value >= col.ge)
        or (col.lt is not None and not value < col.lt)
        or (col.le is not None and not value <= col.le)
    )
    if not violated:
        return None
    wanted = []
    if col.gt is not None:
        wanted.append(f"> {col.gt:g}")
    elif col.ge is not None:
        wanted.append(f">= {col.ge:g}")
    if col.lt is not None:
        wanted.append(f"< {col.lt:g}")
    elif col.le is not None:
        wanted.append(f"<= {col.le:g}")
    return f"{col.name} must be {' and '.join(wanted)} (got {value})"


def _parse_value(col: _Column, s: str, ctx: _Context) -> tuple[Any, str | None]:
    value: Any
    if col.kind == "int":
        value = _parse_int(s)
        if value is None:
            return None, f"{col.name} must be an integer (got '{s}')"
    elif col.kind == "float":
        value = _parse_float(s)
        if value is None:
            return None, f"{col.name} must be a number (got '{s}')"
    elif col.kind == "date":
        value = _parse_date(s)
        if value is None:
            return None, f"{col.name} must be an ISO date YYYY-MM-DD (got '{s}')"
    elif col.kind == "flag":
        value = _parse_int(s)
        if value not in (0, 1):
            return None, f"{col.name} must be 0 or 1 (got '{s}')"
    else:
        value = s
    if col.choices is not None and value not in col.choices:
        return None, f"{col.name} must be one of {', '.join(col.choices)} (got '{s}')"
    if col.kind in ("int", "float"):
        err = _bounds_error(col, value)
        if err:
            return None, err
    if col.fk is not None and value not in ctx.fk[col.fk]:
        return None, f"unknown {col.name} {value}"
    return value, None


def _validate_row(
    spec: _TableSpec, raw: dict[str, Any], ctx: _Context
) -> tuple[dict[str, Any] | None, str | None]:
    row: dict[str, Any] = {}
    for col in spec.columns:
        s = (raw.get(col.name) or "").strip()
        if s == "":
            if col.required:
                return None, f"{col.name} is required"
            row[col.name] = col.default
            continue
        value, err = _parse_value(col, s, ctx)
        if err:
            return None, err
        row[col.name] = value
    if spec.row_check is not None:
        err = spec.row_check(row, ctx)
        if err:
            return None, err
    return row, None


def _key_str(spec: _TableSpec, key: tuple[Any, ...]) -> str:
    return ", ".join(f"{k}={v}" for k, v in zip(spec.key, key, strict=True))


# ---- database helpers ----------------------------------------------------------------------------


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@contextmanager
def _savepoint(conn: sqlite3.Connection, name: str = _SAVEPOINT) -> Iterator[None]:
    """Atomic block that works both standalone (commits) and inside a caller's transaction."""
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    try:
        conn.execute(f"RELEASE {name}")
    except sqlite3.DatabaseError:  # e.g. a deferred FK violation surfacing at commit
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise


def _context(conn: sqlite3.Connection, spec: _TableSpec) -> _Context:
    fk: dict[str, set[int]] = {}
    for table in {c.fk for c in spec.columns if c.fk is not None}:
        col = _FK_ID_COLUMN[table]
        fk[table] = {r[0] for r in conn.execute(f"SELECT {_q(col)} FROM {_q(table)}")}
    prices: dict[int, float] = {}
    if spec.name == "sales_daily":
        prices = {
            r[0]: float(r[1]) for r in conn.execute("SELECT product_id, unit_price FROM products")
        }
    return _Context(fk=fk, prices=prices)


def _unique_maps(conn: sqlite3.Connection, spec: _TableSpec) -> dict[str, dict[Any, Any]]:
    """{unique column: {value: key value}} for the rows already in the table."""
    out: dict[str, dict[Any, Any]] = {}
    for col in spec.unique:
        sql = f"SELECT {_q(col)}, {_q(spec.key[0])} FROM {_q(spec.name)}"
        out[col] = {r[0]: r[1] for r in conn.execute(sql)}
    return out


def _existing_keys(
    conn: sqlite3.Connection, spec: _TableSpec, rows: list[dict[str, Any]]
) -> set[tuple[Any, ...]]:
    """Keys of ``rows`` that already exist in the table (one temp-table join, not N lookups)."""
    types = {r[1]: r[2] for r in conn.execute(f"PRAGMA table_info({_q(spec.name)})")}
    key_cols = ", ".join(_q(k) for k in spec.key)
    typed = ", ".join(f"{_q(k)} {types.get(k, '')}".strip() for k in spec.key)
    placeholders = ", ".join("?" for _ in spec.key)
    conn.execute("DROP TABLE IF EXISTS temp._load_keys")
    conn.execute(f"CREATE TEMP TABLE _load_keys ({typed}, PRIMARY KEY ({key_cols})) WITHOUT ROWID")
    conn.executemany(
        f"INSERT OR IGNORE INTO temp._load_keys ({key_cols}) VALUES ({placeholders})",
        (tuple(r[k] for k in spec.key) for r in rows),
    )
    cond = " AND ".join(f"t.{_q(k)} = k.{_q(k)}" for k in spec.key)
    select = ", ".join(f"t.{_q(k)}" for k in spec.key)
    existing = {
        tuple(r)
        for r in conn.execute(
            f"SELECT {select} FROM {_q(spec.name)} t "
            f"WHERE EXISTS (SELECT 1 FROM temp._load_keys k WHERE {cond})"
        )
    }
    conn.execute("DROP TABLE temp._load_keys")
    return existing


def _write_rows(
    conn: sqlite3.Connection,
    spec: _TableSpec,
    rows: list[dict[str, Any]],
    existing: set[tuple[Any, ...]],
    mode: str,
) -> None:
    cols = [c.name for c in spec.columns]
    col_list = ", ".join(_q(c) for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    insert_sql = f"INSERT INTO {_q(spec.name)} ({col_list}) VALUES ({placeholders})"
    tuples = [tuple(r[c] for c in cols) for r in rows]
    if mode == "upsert":
        key_cols = ", ".join(_q(k) for k in spec.key)
        updates = ", ".join(f"{_q(c)} = excluded.{_q(c)}" for c in cols if c not in spec.key)
        conn.executemany(f"{insert_sql} ON CONFLICT({key_cols}) DO UPDATE SET {updates}", tuples)
    elif mode == "replace":
        if existing:
            # children keep referencing the re-inserted parents: check FKs at commit instead
            conn.execute("PRAGMA defer_foreign_keys = ON")
            where = " AND ".join(f"{_q(k)} = ?" for k in spec.key)
            conn.executemany(f"DELETE FROM {_q(spec.name)} WHERE {where}", sorted(existing))
        conn.executemany(insert_sql, tuples)
    else:  # insert: rows with existing keys were already rejected
        conn.executemany(insert_sql, tuples)


def _record_load(conn: sqlite3.Connection, report: LoadReport) -> None:
    notes = f"rows_read={report.rows_read}; calendar_days_added={report.calendar_days_added}"
    if report.errors:
        notes += f"; first_error={report.errors[0]}"
    conn.execute(
        "INSERT INTO data_loads (loaded_at, source, table_name, mode, rows_inserted, "
        "rows_updated, rows_rejected, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            report.source,
            report.table,
            report.mode,
            report.inserted,
            report.updated,
            report.rejected,
            notes,
        ),
    )


# ---- public API ----------------------------------------------------------------------------------


def ensure_calendar(conn: sqlite3.Connection, start_day: str, end_day: str) -> int:
    """Insert the calendar rows missing in ``[start_day, end_day]``; returns how many were added.

    Derived fields: ``day_of_week`` (Monday = 0), ISO ``week_of_year``, ``month``, ``year``,
    ``is_weekend`` (Saturday/Sunday); ``holiday_name`` is NULL.  Existing rows are untouched.
    """
    start, end = _parse_date(start_day), _parse_date(end_day)
    if start is None or end is None:
        raise ValueError(
            f"start_day and end_day must be ISO dates YYYY-MM-DD (got {start_day!r}, {end_day!r})"
        )
    if start > end:
        raise ValueError(f"start_day {start} is after end_day {end}")
    first = date.fromisoformat(start)
    span = (date.fromisoformat(end) - first).days + 1
    if span > MAX_CALENDAR_SPAN_DAYS:
        raise ValueError(
            f"calendar range {start}..{end} spans {span} days (> {MAX_CALENDAR_SPAN_DAYS}); "
            "check the dates in your file"
        )
    present = {
        r[0]
        for r in conn.execute("SELECT day FROM calendar WHERE day BETWEEN ? AND ?", (start, end))
    }
    rows = []
    for i in range(span):
        d = first + timedelta(days=i)
        iso = d.isoformat()
        if iso in present:
            continue
        f = _calendar_fields(d)
        rows.append(
            (iso, f["day_of_week"], f["week_of_year"], f["month"], f["year"], f["is_weekend"], None)
        )
    if not rows:
        return 0
    with _savepoint(conn, _SAVEPOINT + "_calendar"):
        insert_many(conn, "calendar", list(_CALENDAR_COLUMNS), rows)
    return len(rows)


def load_csv(
    conn: sqlite3.Connection,
    table: str,
    path: str | Path,
    *,
    mode: str = "upsert",
    dry_run: bool = False,
    strict: bool = False,
) -> LoadReport:
    """Validate ``path`` (UTF-8 CSV with header) and load its rows into ``table``.

    Returns a :class:`LoadReport`; raises ``ValueError`` for file-level problems (unknown
    table/mode, missing required columns) and :class:`IngestError` in strict mode when any row
    is rejected.  ``dry_run`` validates and reports without writing anything.
    """
    spec = _spec(table)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)} (got '{mode}')")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: file not found")
    report = LoadReport(table, mode=mode, source=str(path), dry_run=dry_run)
    ctx = _context(conn, spec)
    unique_in_db = _unique_maps(conn, spec)

    valid: list[dict[str, Any]] = []
    lines: list[int] = []
    rejected: list[tuple[int, str]] = []
    seen_keys: dict[tuple[Any, ...], int] = {}
    seen_unique: dict[str, dict[Any, int]] = {u: {} for u in spec.unique}

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        if not header:
            raise ValueError(f"{path}: empty file (a header row is required)")
        names = [h.strip().lower() for h in header]
        reader.fieldnames = names
        required = [c.name for c in spec.columns if c.required and c.name not in names]
        if required:
            raise ValueError(f"{path}: missing required column(s): {', '.join(required)}")
        unknown = [n for n in names if n not in {c.name for c in spec.columns}]
        if unknown:
            log.warning("%s: ignoring unknown column(s) %s", path.name, ", ".join(unknown))
        for raw in reader:
            line = reader.line_num
            report.rows_read += 1
            if PROGRESS_EVERY_ROWS and report.rows_read % PROGRESS_EVERY_ROWS == 0:
                log.info("%s: %d rows read", table, report.rows_read)
            row, err = _validate_row(spec, raw, ctx)
            if row is None:
                rejected.append((line, err or "invalid row"))
                continue
            key = tuple(row[k] for k in spec.key)
            if key in seen_keys:
                rejected.append(
                    (
                        line,
                        f"duplicate key ({_key_str(spec, key)}) first seen at row {seen_keys[key]}",
                    )
                )
                continue
            clash = None
            for col in spec.unique:
                value = row[col]
                if value in seen_unique[col]:
                    clash = f"duplicate {col} '{value}' first seen at row {seen_unique[col][value]}"
                    break
                owner = unique_in_db[col].get(value)
                if owner is not None and owner != key[0]:
                    clash = f"{col} '{value}' is already used by {spec.key[0]} {owner}"
                    break
            if clash:
                rejected.append((line, clash))
                continue
            seen_keys[key] = line
            for col in spec.unique:
                seen_unique[col][row[col]] = line
            valid.append(row)
            lines.append(line)

    with _savepoint(conn):
        existing = _existing_keys(conn, spec, valid) if valid else set()
        if mode == "insert" and existing:
            kept: list[dict[str, Any]] = []
            for row, line in zip(valid, lines, strict=True):
                key = tuple(row[k] for k in spec.key)
                if key in existing:
                    hint = "use mode 'upsert' or 'replace'"
                    rejected.append((line, f"key ({_key_str(spec, key)}) already exists ({hint})"))
                else:
                    kept.append(row)
            valid, existing = kept, set()
        rejected.sort()
        report.rejected = len(rejected)
        report.errors = [f"row {line}: {msg}" for line, msg in rejected[:MAX_ERRORS]]
        report.updated = len(existing)
        report.inserted = len(valid) - len(existing)
        if strict and rejected:
            raise IngestError(
                f"{table}: {len(rejected)} of {report.rows_read} rows rejected (strict mode); "
                f"first: {report.errors[0]}",
                report,
            )
        if spec.name == "sales_daily" and valid:
            days = [r["day"] for r in valid]
            report.calendar_days_added = ensure_calendar(conn, min(days), max(days))
        if not dry_run:
            _write_rows(conn, spec, valid, existing, mode)
            _record_load(conn, report)
        else:
            conn.execute(f"ROLLBACK TO {_SAVEPOINT}")  # keep the counts, undo the calendar fill
    log.info(
        "%s%s: inserted=%d updated=%d rejected=%d (mode=%s, %d data rows in %s)",
        table,
        " [dry run]" if dry_run else "",
        report.inserted,
        report.updated,
        report.rejected,
        mode,
        report.rows_read,
        path.name,
    )
    return report


def load_dataset(conn: sqlite3.Connection, directory: str | Path, **kw: Any) -> list[LoadReport]:
    """Load every ``<table>.csv`` found in ``directory`` in foreign-key order.

    Keyword arguments are passed to :func:`load_csv`.  In strict mode the first failing table
    raises after the tables before it were committed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")
    reports = []
    for table in LOADABLE_TABLES:
        path = directory / f"{table}.csv"
        if path.is_file():
            reports.append(load_csv(conn, table, path, **kw))
    if not reports:
        raise ValueError(
            f"no loadable files in {directory} (expected {', '.join(LOADABLE_TABLES)} .csv)"
        )
    return reports


# ---- CLI plugin ----------------------------------------------------------------------------------

_FLAG_TO_TABLE = {
    "stores": "stores",
    "products": "products",
    "calendar": "calendar",
    "promotions": "promotions",
    "sales": "sales_daily",
    "inventory": "inventory_snapshots",
}


def _cli_load(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    """``demandcast load``: initialise the schema, load the given files, print a JSON report."""
    db.init_schema(conn)
    files: dict[str, Path] = {}
    if args.dir:
        directory = Path(args.dir)
        if not directory.is_dir():
            print(f"error: {directory} is not a directory", file=sys.stderr)
            return 1
        for table in LOADABLE_TABLES:
            path = directory / f"{table}.csv"
            if path.is_file():
                files[table] = path
    for flag, table in _FLAG_TO_TABLE.items():
        value = getattr(args, flag, None)
        if value:
            files[table] = Path(value)
    if not files:
        print("error: nothing to load — pass --dir DIR or at least one table file", file=sys.stderr)
        return 1
    reports: list[LoadReport] = []
    ok = True
    for table in LOADABLE_TABLES:
        if table not in files:
            continue
        try:
            reports.append(
                load_csv(
                    conn,
                    table,
                    files[table],
                    mode=args.mode,
                    dry_run=args.dry_run,
                    strict=args.strict,
                )
            )
        except IngestError as exc:
            reports.append(exc.report)
            ok = False
            break
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    payload = {
        "ok": ok,
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "strict": bool(args.strict),
        "loads": [r.to_dict() for r in reports],
    }
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


def register_cli(sub: argparse._SubParsersAction) -> None:
    """Register ``load [--dir DIR] [--<table> FILE]... [--mode M] [--dry-run] [--strict]``."""
    p = sub.add_parser(
        "load",
        help="load validated CSV files (stores, products, calendar, promotions, sales, inventory)",
        description=(
            "Validate and load CSV files. Rows that fail validation are reported and skipped "
            "(or, with --strict, make the whole file fail). Missing calendar days for loaded "
            "sales are filled automatically."
        ),
    )
    p.add_argument("--dir", help="directory containing <table>.csv files (loaded in FK order)")
    p.add_argument("--stores", metavar="FILE")
    p.add_argument("--products", metavar="FILE")
    p.add_argument("--calendar", metavar="FILE")
    p.add_argument("--promotions", metavar="FILE")
    p.add_argument("--sales", metavar="FILE", help="sales_daily rows")
    p.add_argument("--inventory", metavar="FILE", help="inventory_snapshots rows")
    p.add_argument("--mode", choices=MODES, default="upsert")
    p.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    p.add_argument("--strict", action="store_true", help="any rejected row fails the whole file")
    p.set_defaults(handler=_cli_load, creates_db=True)
