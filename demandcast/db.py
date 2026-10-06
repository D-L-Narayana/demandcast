"""Thin SQLite access layer.

* ``connect`` - connection factory with the project's PRAGMA defaults.
* ``init_schema`` / ``migrate`` / ``ensure_schema`` / ``schema_version`` - schema bootstrap and
  additive, idempotent migrations tracked through ``PRAGMA user_version`` (``SCHEMA_VERSION``).
* ``load_queries`` / ``QUERIES`` / ``query_params`` / ``run_query`` / ``explain_query`` - the
  ``-- name:`` analytics query catalogue kept in ``sql/analytics.sql``.
* ``insert_many`` / ``upsert_many`` / ``transaction`` / ``table_counts`` / ``integrity_check`` -
  bulk writes, nest-safe transactions and diagnostics.
"""

from __future__ import annotations

import itertools
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

SQL_DIR = Path(__file__).parent / "sql"
_QUERY_MARKER = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)

#: Schema version written to ``PRAGMA user_version`` by ``schema.sql`` and ``migrate()``.
SCHEMA_VERSION = 2

# --- schema v2 migration (contract C5) -------------------------------------------------------
# Fresh databases get these objects inline from sql/schema.sql; migrate() applies the very same
# additive changes to databases created by earlier releases (v0.3.0 had no user_version => 0).
# Every statement is guarded - PRAGMA table_info for columns, IF NOT EXISTS for tables and
# indexes - so running the migration again is a no-op.
_V2_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("forecast_runs", "config_json", "TEXT"),
    ("forecast_runs", "interval_level", "REAL"),
    ("forecast_runs", "engine_version", "TEXT"),
    ("forecasts", "promo_flag", "INTEGER NOT NULL DEFAULT 0"),
    ("replenishment_orders", "stockout_risk", "REAL"),
    ("replenishment_orders", "priority", "REAL"),
    ("replenishment_orders", "requested_qty", "INTEGER"),  # qty before budget allocation
)
_V2_TABLES: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS forecast_evaluations (
    run_id          INTEGER NOT NULL REFERENCES forecast_runs(run_id),
    store_id        INTEGER NOT NULL REFERENCES stores(store_id),
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    model_name      TEXT    NOT NULL,
    n_days          INTEGER NOT NULL CHECK (n_days > 0),
    mae             REAL    NOT NULL,
    wape            REAL,
    bias            REAL    NOT NULL,
    coverage        REAL    NOT NULL CHECK (coverage BETWEEN 0 AND 1),
    abs_error_sum   REAL    NOT NULL,
    actual_sum      REAL    NOT NULL,
    stockout_days   INTEGER NOT NULL DEFAULT 0,
    evaluated_at    TEXT    NOT NULL,
    PRIMARY KEY (run_id, store_id, product_id)
) WITHOUT ROWID""",
    """CREATE TABLE IF NOT EXISTS data_loads (
    load_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    loaded_at       TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    table_name      TEXT    NOT NULL,
    mode            TEXT    NOT NULL CHECK (mode IN ('insert', 'upsert', 'replace')),
    rows_inserted   INTEGER NOT NULL,
    rows_updated    INTEGER NOT NULL DEFAULT 0,
    rows_rejected   INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
)""",
)
# (table the index lives on, DDL) - the table must exist before the index can be created.
_V2_INDEXES: tuple[tuple[str, str], ...] = (
    (
        "forecasts",
        "CREATE INDEX IF NOT EXISTS idx_forecasts_series_day "
        "ON forecasts(store_id, product_id, target_day)",
    ),
    (
        "promotions",
        "CREATE INDEX IF NOT EXISTS idx_promotions_product "
        "ON promotions(product_id, start_day, end_day)",
    ),
)


# --- errors ----------------------------------------------------------------------------------


class SchemaMissingError(RuntimeError):
    """The database has no DemandCast tables (the core table ``sales_daily`` is absent)."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "database has no DemandCast schema (table sales_daily is missing); "
            "run `demandcast init` or `demandcast load` first"
        )


class SchemaVersionError(RuntimeError):
    """The database was written by a newer release than this code understands."""

    def __init__(self, found: int) -> None:
        self.found = found
        super().__init__(
            f"database schema version {found} is newer than this release supports "
            f"({SCHEMA_VERSION}); upgrade demandcast"
        )

    def __reduce__(self) -> tuple[Any, ...]:  # picklable despite the custom __init__ signature
        return type(self), (self.found,)


class UnknownQueryError(KeyError):
    """``name`` is not one of the named analytics queries (lists the available names)."""

    def __init__(self, name: str, available: Iterable[str] = ()) -> None:
        self.name = name
        self.available = tuple(sorted(available))
        super().__init__(f"Unknown query '{name}'. Available: {list(self.available)}")

    def __str__(self) -> str:  # KeyError would repr-quote the message
        return str(self.args[0])

    def __reduce__(self) -> tuple[Any, ...]:  # picklable despite the custom __init__ signature
        return type(self), (self.name, self.available)


class MissingParameterError(KeyError):
    """A named query references ``:params`` that were not supplied (see ``missing``)."""

    def __init__(self, name: str, missing: Iterable[str] = ()) -> None:
        self.name = name
        self.missing = tuple(sorted(missing))
        plural = "s" if len(self.missing) != 1 else ""
        listed = ", ".join(f":{m}" for m in self.missing)
        super().__init__(f"query '{name}' is missing required parameter{plural}: {listed}")

    def __str__(self) -> str:  # KeyError would repr-quote the message
        return str(self.args[0])

    def __reduce__(self) -> tuple[Any, ...]:  # picklable despite the custom __init__ signature
        return type(self), (self.name, self.missing)


# --- connection & schema ---------------------------------------------------------------------


def _quote(identifier: str) -> str:
    """Quote an SQL identifier with double quotes (embedded quotes are doubled)."""
    return '"' + identifier.replace('"', '""') + '"'


def connect(path: str | Path = ":memory:", timeout: float = 30.0) -> sqlite3.Connection:
    """Open a connection with sane defaults (FKs on, WAL for file DBs, dict-like rows).

    ``timeout`` is how long (seconds) a statement waits for a lock held by another connection;
    it is passed to ``sqlite3.connect`` and mirrored into ``PRAGMA busy_timeout``.
    """
    conn = sqlite3.connect(str(path), timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {max(0, round(timeout * 1000))}")
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    """Return ``PRAGMA user_version`` (0 for an empty database or a pre-0.4.0 schema)."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({_quote(table)})")}


def _check_not_newer(conn: sqlite3.Connection) -> None:
    found = schema_version(conn)
    if found > SCHEMA_VERSION:
        raise SchemaVersionError(found)


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the schema (``CREATE ... IF NOT EXISTS``) and then ``migrate()`` it.

    Idempotent: safe on an empty database, on a current one and on a database created by an
    earlier release (whose existing tables receive the new columns through the migration).
    """
    _check_not_newer(conn)
    conn.executescript((SQL_DIR / "schema.sql").read_text())
    conn.commit()
    migrate(conn)


def migrate(conn: sqlite3.Connection) -> int:
    """Bring an existing database up to ``SCHEMA_VERSION`` with additive DDL; return the version.

    Columns are added only when ``PRAGMA table_info`` does not list them, tables and indexes use
    ``IF NOT EXISTS`` and ``PRAGMA user_version`` is raised to ``SCHEMA_VERSION`` - so the call
    is idempotent and running it on a current database changes nothing. Existing rows are kept
    (new columns are NULL or take their DEFAULT). Everything runs in one transaction (a
    savepoint when the caller already has one open). Raises ``SchemaMissingError`` when the
    core tables are absent (use ``init_schema``) and ``SchemaVersionError`` when the file was
    written by a newer release.
    """
    _check_not_newer(conn)
    tables = _table_names(conn)
    if "sales_daily" not in tables:
        raise SchemaMissingError()
    with transaction(conn):
        present = {t: _column_names(conn, t) for t in {c[0] for c in _V2_COLUMNS} & tables}
        for table, column, declaration in _V2_COLUMNS:
            if table in present and column not in present[table]:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        for ddl in _V2_TABLES:
            conn.execute(ddl)
        for table, ddl in _V2_INDEXES:
            if table in tables:
                conn.execute(ddl)
        if schema_version(conn) < SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return SCHEMA_VERSION


def ensure_schema(conn: sqlite3.Connection) -> int:
    """Fail fast on an uninitialised database; otherwise migrate it and return the version."""
    if "sales_daily" not in _table_names(conn):
        raise SchemaMissingError()
    return migrate(conn)


# --- named queries ---------------------------------------------------------------------------


def load_queries(path: Path = SQL_DIR / "analytics.sql") -> dict[str, str]:
    """Parse a .sql file with `-- name: xyz` markers into {name: sql}; duplicate names raise."""
    text = path.read_text()
    parts = _QUERY_MARKER.split(text)
    # parts = [preamble, name1, body1, name2, body2, ...]
    queries: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        name, body = parts[i], parts[i + 1]
        if name in queries:
            raise ValueError(f"duplicate query name '{name}' in {path}")
        body = "\n".join(line for line in body.splitlines() if not line.strip().startswith("--"))
        queries[name] = body.strip()
    return queries


QUERIES = load_queries()

# Things that may legitimately contain a colon without being a parameter: string literals
# ('12:30', with '' as the escape), quoted identifiers, line comments and block comments.
_SQL_NOISE = re.compile(
    r"'(?:[^']|'')*'" r'|"(?:[^"]|"")*"' r"|--[^\n]*" r"|/\*.*?\*/",
    re.DOTALL,
)
# :name parameters; the look-behind skips `::` casts and anything glued to a word.
_NAMED_PARAM = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")


@cache
def _params_in(sql: str) -> frozenset[str]:
    return frozenset(_NAMED_PARAM.findall(_SQL_NOISE.sub(" ", sql)))


def _query_sql(name: str) -> str:
    try:
        return QUERIES[name]
    except KeyError:
        raise UnknownQueryError(name, QUERIES) from None


def query_params(name: str) -> frozenset[str]:
    """Return the ``:param`` names a named query references (string literals/comments ignored)."""
    return _params_in(_query_sql(name))


def run_query(
    conn: sqlite3.Connection, name: str, params: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Execute a named analytics query and return rows as plain dicts.

    Raises ``UnknownQueryError`` (a ``KeyError`` listing the available names) for an unknown
    ``name`` and ``MissingParameterError`` (also a ``KeyError``, naming the absent ``:params``)
    before anything is sent to SQLite. Extra keys in ``params`` are ignored.
    """
    sql = _query_sql(name)
    bound = dict(params or {})
    missing = _params_in(sql) - bound.keys()
    if missing:
        raise MissingParameterError(name, missing)
    cur = conn.execute(sql, bound)
    columns = [d[0] for d in cur.description or ()]
    return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def explain_query(
    conn: sqlite3.Connection, name: str, params: Mapping[str, Any] | None = None
) -> list[str]:
    """Return the ``EXPLAIN QUERY PLAN`` detail lines of a named query (for docs and tuning).

    Parameters that are not supplied are bound as NULL - the plan does not depend on values.
    """
    sql = _query_sql(name)
    bound: dict[str, Any] = dict.fromkeys(_params_in(sql))
    bound.update(params or {})
    rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}", bound).fetchall()
    return [str(tuple(r)[-1]) for r in rows]  # columns: id, parent, notused, detail


# --- bulk writes -----------------------------------------------------------------------------


def insert_many(
    conn: sqlite3.Connection, table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]
) -> int:
    """Bulk ``INSERT`` (identifiers double-quoted); return the number of rows inserted."""
    cols = ", ".join(_quote(c) for c in columns)
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO {_quote(table)} ({cols}) VALUES ({placeholders})"
    cur = conn.executemany(sql, rows)
    return max(cur.rowcount, 0)


def upsert_many(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    conflict_columns: Sequence[str],
) -> int:
    """Bulk ``INSERT ... ON CONFLICT(conflict_columns) DO UPDATE SET col = excluded.col``.

    Every column that is not part of the conflict key is overwritten from the incoming row;
    when all columns belong to the key the statement degrades to ``DO NOTHING``. Returns the
    number of rows inserted or updated (rows skipped by ``DO NOTHING`` are not counted).
    """
    cols = list(columns)
    keys = list(conflict_columns)
    if not keys:
        raise ValueError("conflict_columns must name at least one column")
    unknown = [k for k in keys if k not in cols]
    if unknown:
        raise ValueError(f"conflict_columns not present in columns: {unknown}")
    updates = [c for c in cols if c not in keys]
    action = (
        "DO UPDATE SET " + ", ".join(f"{_quote(c)} = excluded.{_quote(c)}" for c in updates)
        if updates
        else "DO NOTHING"
    )
    sql = (
        f"INSERT INTO {_quote(table)} ({', '.join(_quote(c) for c in cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)}) "
        f"ON CONFLICT({', '.join(_quote(k) for k in keys)}) {action}"
    )
    cur = conn.executemany(sql, rows)
    return max(cur.rowcount, 0)


_SAVEPOINT_IDS = itertools.count(1)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit transaction block: commit on success, rollback on any exception.

    At top level (no transaction open) this is ``BEGIN`` / ``COMMIT`` / ``ROLLBACK``. When the
    connection is already inside a transaction - an outer ``transaction()`` block or the
    implicit transaction the sqlite3 module opens before DML - a ``SAVEPOINT`` is used instead:
    ``RELEASE`` on success, ``ROLLBACK TO`` + ``RELEASE`` on error, so an inner failure undoes
    only the inner work and leaves the outer transaction open. Blocks nest to any depth.
    """
    if not conn.in_transaction:
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")  # inside the try: a failing COMMIT must still roll back
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        return
    savepoint = f"demandcast_sp_{next(_SAVEPOINT_IDS)}"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        yield conn
        conn.execute(f"RELEASE {savepoint}")
    except BaseException:
        if conn.in_transaction:
            conn.execute(f"ROLLBACK TO {savepoint}")
            conn.execute(f"RELEASE {savepoint}")
        raise


# --- diagnostics -----------------------------------------------------------------------------


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row count per user table (``sqlite_%`` internals excluded), in ``sqlite_master`` order."""
    names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    return {n: int(conn.execute(f"SELECT COUNT(*) FROM {_quote(n)}").fetchone()[0]) for n in names}


def integrity_check(conn: sqlite3.Connection) -> list[str]:
    """Return one line per ``PRAGMA foreign_key_check`` violation (empty list = all FKs resolve).

    Useful after bulk loads performed with ``PRAGMA foreign_keys = OFF``. Each line names the
    child table, the offending rowid (WITHOUT ROWID tables have none) and the parent table.
    """
    problems: list[str] = []
    for table, rowid, parent, fk_index in conn.execute("PRAGMA foreign_key_check"):
        where = f"rowid {rowid}" if rowid is not None else "a row"
        problems.append(
            f"{table}: {where} references a missing {parent} row (foreign key #{fk_index})"
        )
    return problems
