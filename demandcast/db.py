"""Thin SQLite access layer: connection factory, schema bootstrap, named-query loader."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SQL_DIR = Path(__file__).parent / "sql"
_QUERY_MARKER = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a connection with sane defaults (FKs on, WAL for file DBs, dict-like rows)."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript((SQL_DIR / "schema.sql").read_text())
    conn.commit()


def load_queries(path: Path = SQL_DIR / "analytics.sql") -> dict[str, str]:
    """Parse a .sql file with `-- name: xyz` markers into {name: sql}."""
    text = path.read_text()
    parts = _QUERY_MARKER.split(text)
    # parts = [preamble, name1, body1, name2, body2, ...]
    queries: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        name, body = parts[i], parts[i + 1]
        body = "\n".join(line for line in body.splitlines() if not line.strip().startswith("--"))
        queries[name] = body.strip()
    return queries


QUERIES = load_queries()


def run_query(
    conn: sqlite3.Connection, name: str, params: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Execute a named analytics query and return rows as plain dicts."""
    if name not in QUERIES:
        raise KeyError(f"Unknown query '{name}'. Available: {sorted(QUERIES)}")
    cur = conn.execute(QUERIES[name], dict(params or {}))
    return [dict(r) for r in cur.fetchall()]


def insert_many(
    conn: sqlite3.Connection, table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]
) -> int:
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    cur = conn.executemany(sql, rows)
    return cur.rowcount


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Explicit transaction block: commit on success, rollback on any exception."""
    try:
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = [
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    return {n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}
