"""Command-line entry point.

    demandcast init       [--stores N --products N --days N --start DATE --seed N]
                          [--no-data] [--snapshot-every N] [--future-promo-days N]
    demandcast run        [--horizon N --folds N --service-level P --review-period N --workers N]
                          [--cutoff DATE] [--models a,b] [--interval LEVEL]
                          [--interval-method empirical|normal] [--criterion mae|wape|mase]
                          [--no-promo] [--order-budget X] [--service-level-by-class A=0.98,...]
    demandcast runs       [--limit N] [--json]
    demandcast query NAME [--format table|csv|json] [--cells safe|raw] [--param k=v]...
                          [--limit N]  (0 = all)
    demandcast query --list
    demandcast dashboard  [--out dashboard/index.html] [--run-id N]
    demandcast stats

Global options: --db PATH (default demandcast.db), -v/--verbose, -q/--quiet, --version.
Optional commands (evaluate, load, export) are discovered from COMMAND_PLUGINS when their
modules are importable. Exit codes: 0 ok, 1 operational error (printed as "error: ..." on
stderr, never a traceback), 2 usage error. Only `init` (and plugins that opt in) may create
the database file; every other command refuses a missing --db path.

`query --format csv` is spreadsheet-safe by default (--cells safe): text cells that a
spreadsheet would evaluate as formulas (= + - @ first, or a leading tab/CR/LF) are written
with a leading ' marker; numbers, dates and empty cells are never changed and the database
is never modified. `--cells raw` writes every cell verbatim for machine consumers; table and
JSON output are unaffected either way.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import importlib
import json
import logging
import re
import sqlite3
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from . import __version__, dashboard, db, pipeline, replenish, simulate
from .csvsafe import CELL_POLICIES, safe_rows
from .models import MODEL_REGISTRY

log = logging.getLogger("demandcast.cli")

# Modules whose register_cli(sub) adds sub-commands; a module that cannot be imported is
# skipped silently (C10). Plugin parsers set handler=fn(conn, args) -> int and creates_db.
COMMAND_PLUGINS: tuple[str, ...] = ("demandcast.evaluate", "demandcast.ingest", "demandcast.export")

MISSING_DB_HINT = "run `demandcast init` or `demandcast load`"
ABC_CLASSES = ("A", "B", "C")
PROMO_PREFIX = "promo_"
Handler = Callable[[sqlite3.Connection, argparse.Namespace], int]

_PARAM_RE = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")  # :name placeholders in a SQL text
_RUN_COLUMNS = (
    "run_id",
    "status",
    "started_at",
    "finished_at",
    "cutoff_day",
    "horizon_days",
    "series_count",
    "notes",
)
_NO_LIMIT = 2**31 - 1


class UnsupportedFlagError(ValueError):
    """A flag was given whose target dataclass field does not exist in this build."""

    def __init__(self, flag: str) -> None:
        super().__init__(f"this build does not support {flag}")
        self.flag = flag


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_table(rows: list[dict[str, Any]], shown: list[dict[str, Any]]) -> None:
    """Fixed-width table of `shown` (a prefix of `rows`); works for any prefix length (D5)."""
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max([len(c), *(len(_cell(r[c])) for r in shown)]) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in shown:
        print("  ".join(_cell(r[c]).ljust(widths[c]) for c in cols))
    if len(rows) > len(shown):
        print(f"... {len(rows) - len(shown)} more rows")


def _print_csv(shown: list[dict[str, Any]]) -> None:
    if not shown:
        return
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(list(shown[0].keys()))
    for r in shown:
        writer.writerow(["" if v is None else v for v in r.values()])


def _emit_rows(rows: list[dict[str, Any]], fmt: str, limit: int, cells: str = "safe") -> None:
    """Print `rows` as table / csv / json; `limit` <= 0 means every row.

    `cells` is the CSV cell policy (see `demandcast.csvsafe`): "safe" (default) prefixes text
    cells that a spreadsheet would evaluate as formulas with ' so they stay text, "raw" writes
    every cell verbatim. It only affects `--format csv`; table and JSON output never change.
    """
    shown = rows if limit <= 0 else rows[:limit]
    if fmt == "json":
        print(json.dumps(shown, indent=2, default=str))
    elif fmt == "csv":
        _print_csv(safe_rows(shown) if cells == "safe" else shown)
    else:
        _print_table(rows, shown)


def _coerce(value: str) -> int | float | str:
    """Coerce a --param value: int, then float, then the string itself."""
    with contextlib.suppress(ValueError):
        return int(value)
    with contextlib.suppress(ValueError):
        return float(value)
    return value


def _required_params(name: str) -> set[str]:
    """The :params a named query needs (db.query_params when present, else a regex scan)."""
    getter = getattr(db, "query_params", None)
    if callable(getter):
        with contextlib.suppress(KeyError):
            return set(getter(name))
    return set(_PARAM_RE.findall(db.QUERIES[name]))


def _print_query_list(detailed: bool) -> None:
    names = sorted(db.QUERIES)
    if not detailed:
        print("\n".join(names))
        return
    width = max((len(n) for n in names), default=0)
    for n in names:
        params = " ".join(f":{p}" for p in sorted(_required_params(n))) or "-"
        print(f"{n.ljust(width)}  {params}")


def _field_defaults(cls: Any) -> dict[str, Any]:
    """Default values of the plain-default fields of a dataclass (used in --help texts)."""
    return {
        f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING
    }


def _dataclass_kwargs(cls: Any, given: Mapping[str, tuple[str, Any]]) -> dict[str, Any]:
    """Map CLI flags onto the fields that exist on the dataclass `cls`.

    `given` maps field name -> (flag, value). A value of None means the flag was not passed,
    so the dataclass default applies. A flag that *was* passed but has no matching field in
    this build raises UnsupportedFlagError instead of being silently ignored.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for field, (flag, value) in given.items():
        if value is None:
            continue
        if field not in names:
            raise UnsupportedFlagError(flag)
        kwargs[field] = value
    return kwargs


def _require_min(flag: str, value: float | None, minimum: float) -> None:
    if value is not None and value < minimum:
        raise ValueError(f"{flag} must be >= {minimum}, got {value}")


def _parse_models(spec: str | None) -> tuple[str, ...] | None:
    """Parse `--models a,b`; plain names must be registered, promo_<name> wraps a base model."""
    if spec is None:
        return None
    names = tuple(part.strip() for part in spec.split(",") if part.strip())
    if not names:
        raise ValueError("--models expects a comma-separated list of model names")
    unknown = [
        n
        for n in names
        if n not in MODEL_REGISTRY
        and not (n.startswith(PROMO_PREFIX) and n[len(PROMO_PREFIX) :] in MODEL_REGISTRY)
    ]
    if unknown:
        listed = ", ".join(repr(n) for n in unknown)
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"unknown model {listed} (available: {available}, or promo_<name>)")
    return names


def parse_service_levels(spec: str) -> dict[str, float]:
    """Parse ``"A=0.98,B=0.95"`` into ``{"A": 0.98, "B": 0.95}``.

    Delegates to ``replenish.parse_service_levels`` when the replenishment module provides
    it; otherwise applies the same rules locally: classes A/B/C, levels in [0.5, 1.0),
    ValueError on anything else.
    """
    shared = getattr(replenish, "parse_service_levels", None)
    if callable(shared):
        return dict(shared(spec))
    return _parse_service_levels_local(spec)


def _parse_service_levels_local(spec: str) -> dict[str, float]:
    if not spec.strip():
        raise ValueError("expected CLASS=LEVEL pairs such as A=0.98,B=0.95,C=0.90")
    levels: dict[str, float] = {}
    for item in spec.split(","):
        key, sep, raw = (part.strip() for part in item.partition("="))
        if not sep or key not in ABC_CLASSES:
            raise ValueError(
                f"bad entry {item.strip()!r}: expected one of A=, B=, C= followed by a level"
            )
        try:
            level = float(raw)
        except ValueError:
            raise ValueError(f"service level for class {key} is not a number: {raw!r}") from None
        if not 0.5 <= level < 1.0:
            raise ValueError(f"service level for class {key} must be in [0.5, 1.0), got {level}")
        levels[key] = level
    return levels


def _ordered(row: Mapping[str, Any], first: Sequence[str]) -> dict[str, Any]:
    out = {k: row[k] for k in first if k in row}
    out.update((k, v) for k, v in row.items() if k not in out)
    return out


def _list_runs(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    """Newest-first run rows via pipeline.list_runs when present, else inline SQL."""
    n = limit if limit > 0 else _NO_LIMIT
    lister = getattr(pipeline, "list_runs", None)
    if callable(lister):
        rows = [dict(r) for r in lister(conn, n)]
    else:
        cur = conn.execute(
            f"SELECT {', '.join(_RUN_COLUMNS)} FROM forecast_runs ORDER BY run_id DESC LIMIT ?",
            (n,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return [_ordered(r, _RUN_COLUMNS) for r in rows]


def _db_file_missing(path: str) -> bool:
    return path != ":memory:" and not Path(path).exists()


def _require_schema(conn: sqlite3.Connection, db_path: str) -> None:
    """Fail clearly (instead of a `no such table` traceback) when the schema is absent (D6).

    Uses db.ensure_schema (schema v2: raises SchemaMissingError, otherwise migrates) when the
    db layer provides it, else a plain existence check against the v0.3.0 db layer.
    """
    ensure = getattr(db, "ensure_schema", None)
    if callable(ensure):
        ensure(conn)
        return
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sales_daily'"
    ).fetchone()
    if row is None:
        error_type = getattr(db, "SchemaMissingError", RuntimeError)
        raise error_type(f"database {db_path} has no DemandCast schema ({MISSING_DB_HINT} first)")


def _wrapped_errors() -> tuple[type[BaseException], ...]:
    """Exception types reported as ``error: ...`` (exit 1) instead of a traceback.

    db.SchemaMissingError / db.MissingParameterError are looked up lazily so this module
    works against the v0.3.0 db layer as well as schema v2 (they subclass RuntimeError /
    KeyError, which are wrapped anyway; listing them documents the contract).
    """
    return (
        getattr(db, "SchemaMissingError", RuntimeError),
        getattr(db, "MissingParameterError", KeyError),
        RuntimeError,  # pipeline / dashboard / evaluate (NoActualsError)
        ValueError,  # bad flag values, parse errors
        KeyError,  # unknown query or model names
        sqlite3.Error,  # missing tables / columns, locked or unreadable database files
        OSError,  # unreadable input files, unwritable output paths
    )


def _describe(exc: BaseException) -> str:
    """Human-readable message for a wrapped exception (KeyError repr-quotes its argument)."""
    if isinstance(exc, KeyError) and exc.args:
        message = str(exc.args[0])
    else:
        message = str(exc) or exc.__class__.__name__
    if isinstance(exc, sqlite3.OperationalError) and "no such table" in message:
        message += f" ({MISSING_DB_HINT} first)"
    return message


def _logging_flags(argv: Sequence[str]) -> tuple[bool, bool]:
    """Pre-scan -v/--verbose and -q/--quiet so plugin discovery can already log."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("-v", "--verbose", action="store_true")
    pre.add_argument("-q", "--quiet", action="store_true")
    flags, _ = pre.parse_known_args(list(argv))
    return bool(flags.verbose), bool(flags.quiet)


def _configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("demandcast").setLevel(level)


# --------------------------------------------------------------------------------------
# command handlers: fn(conn, args) -> exit code
# --------------------------------------------------------------------------------------


def _cmd_init(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    for flag, value in (("--stores", args.stores), ("--products", args.products)):
        _require_min(flag, value, 1)
    _require_min("--days", args.days, 1)
    _require_min("--snapshot-every", args.snapshot_every, 1)
    _require_min("--future-promo-days", args.future_promo_days, 0)
    kwargs = _dataclass_kwargs(
        simulate.SimConfig,
        {
            "n_stores": ("--stores", args.stores),
            "n_products": ("--products", args.products),
            "start": ("--start", args.start),
            "days": ("--days", args.days),
            "seed": ("--seed", args.seed),
            "snapshot_every_days": ("--snapshot-every", args.snapshot_every),
            "future_promo_days": ("--future-promo-days", args.future_promo_days),
        },
    )
    db.init_schema(conn)
    if args.no_data:
        if kwargs:
            log.warning("--no-data given: dataset flags %s are ignored", sorted(kwargs))
        print(json.dumps(db.table_counts(conn), indent=2))
        return 0
    if conn.execute("SELECT COUNT(*) FROM sales_daily").fetchone()[0]:
        raise RuntimeError("database already contains data; delete it to regenerate")
    counts = simulate.generate(conn, simulate.SimConfig(**kwargs))
    print(json.dumps(counts, indent=2))
    return 0


def _cmd_run(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    _require_min("--horizon", args.horizon, 1)
    _require_min("--folds", args.folds, 1)
    _require_min("--review-period", args.review_period, 0)
    _require_min("--workers", args.workers, 0)
    _require_min("--order-budget", args.order_budget, 0.0)
    if args.service_level is not None and not 0.5 <= args.service_level < 1.0:
        raise ValueError(f"--service-level must be in [0.5, 1.0), got {args.service_level}")
    if args.interval is not None and not 0.0 < args.interval < 1.0:
        raise ValueError(f"--interval must be in (0, 1), got {args.interval}")
    models = _parse_models(args.models)
    overrides: dict[str, float] | None = None
    if args.service_level_by_class is not None:
        try:
            overrides = parse_service_levels(args.service_level_by_class)
        except ValueError as exc:
            raise ValueError(f"--service-level-by-class: {exc}") from exc
    kwargs = _dataclass_kwargs(
        pipeline.RunConfig,
        {
            "horizon_days": ("--horizon", args.horizon),
            "n_folds": ("--folds", args.folds),
            "service_level": ("--service-level", args.service_level),
            "review_period_days": ("--review-period", args.review_period),
            "workers": ("--workers", args.workers),
            "cutoff_day": ("--cutoff", args.cutoff),
            "models": ("--models", models),
            "interval_level": ("--interval", args.interval),
            "interval_method": ("--interval-method", args.interval_method),
            "criterion": ("--criterion", args.criterion),
            "promo_aware": ("--no-promo", False if args.no_promo else None),
            "order_budget": ("--order-budget", args.order_budget),
            "service_level_overrides": ("--service-level-by-class", overrides),
        },
    )
    run_id = pipeline.run(conn, pipeline.RunConfig(**kwargs))
    row = conn.execute("SELECT * FROM forecast_runs WHERE run_id = ?", (run_id,)).fetchone()
    print(json.dumps(dict(row) if row is not None else {"run_id": run_id}, indent=2, default=str))
    return 0


def _cmd_runs(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    rows = _list_runs(conn, args.limit)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        _print_table(rows, rows)
    return 0


def _cmd_query(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    name = args.name
    if name not in db.QUERIES:
        raise KeyError(f"unknown query '{name}' (available: {', '.join(sorted(db.QUERIES))})")
    params: dict[str, Any] = {}
    for item in args.param:
        key, sep, value = item.partition("=")
        key = key.strip().lstrip(":")
        if not sep or not key:
            raise ValueError(f"--param expects KEY=VALUE, got {item!r}")
        params[key] = _coerce(value.strip())
    for key in ("run_id", "store_id", "product_id"):  # the original dedicated flags
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    needed = _required_params(name)
    if "run_id" in needed and "run_id" not in params:
        latest = pipeline.latest_successful_run(conn)
        if latest is None:
            raise RuntimeError(
                f"query '{name}' needs :run_id but no successful run exists yet "
                "(run `demandcast run` first, or pass --run-id)"
            )
        params["run_id"] = latest
    missing = sorted(needed - params.keys())
    if missing:
        wanted = ", ".join(f":{m}" for m in missing)
        raise ValueError(f"query '{name}' requires {wanted}; pass --param {missing[0]}=VALUE")
    rows = db.run_query(conn, name, {k: v for k, v in params.items() if k in needed})
    fmt = args.format or ("json" if args.json else "table")
    _emit_rows(rows, fmt, args.limit, args.cells)
    return 0


def _cmd_dashboard(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    out = dashboard.render(conn, args.out, args.run_id)
    print(f"wrote {out}")
    return 0


def _cmd_stats(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    print(json.dumps(db.table_counts(conn), indent=2))
    return 0


# --------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------


def _add_init(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    d = _field_defaults(simulate.SimConfig)
    p = sub.add_parser("init", help="create the schema and (unless --no-data) a synthetic dataset")
    p.add_argument("--stores", type=int, help=f"number of stores (default {d.get('n_stores')})")
    p.add_argument(
        "--products", type=int, help=f"number of products (default {d.get('n_products')})"
    )
    p.add_argument("--days", type=int, help=f"days of history (default {d.get('days')})")
    p.add_argument(
        "--start",
        type=date.fromisoformat,
        metavar="DATE",
        help=f"first day of history (default {d.get('start')})",
    )
    p.add_argument("--seed", type=int, help=f"random seed (default {d.get('seed')})")
    p.add_argument(
        "--no-data", action="store_true", help="create the schema only (then `load` your CSVs)"
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        metavar="N",
        help="write an inventory snapshot every N days instead of only on the last day",
    )
    p.add_argument(
        "--future-promo-days",
        type=int,
        metavar="N",
        help="also schedule promotions up to N days after the last sales day",
    )
    p.set_defaults(handler=_cmd_init, creates_db=True)


def _add_run(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    d = _field_defaults(pipeline.RunConfig)
    p = sub.add_parser("run", help="backtest, forecast and generate replenishment orders")
    p.add_argument(
        "--horizon",
        type=int,
        metavar="DAYS",
        help=f"forecast horizon (default {d.get('horizon_days')})",
    )
    p.add_argument(
        "--folds",
        type=int,
        metavar="N",
        help=f"rolling-origin backtest folds (default {d.get('n_folds')})",
    )
    p.add_argument(
        "--service-level",
        type=float,
        metavar="P",
        help=f"cycle service level in [0.5, 1) (default {d.get('service_level')})",
    )
    p.add_argument(
        "--review-period",
        type=int,
        metavar="DAYS",
        help=f"order review period (default {d.get('review_period_days')})",
    )
    p.add_argument(
        "--workers", type=int, metavar="N", help="process pool size, 0 = all CPUs (default 0)"
    )
    p.add_argument(
        "--cutoff",
        type=date.fromisoformat,
        metavar="DATE",
        help="backdated run: use sales up to DATE only (enables `evaluate` later)",
    )
    p.add_argument(
        "--models", metavar="A,B", help="restrict the candidate models (comma-separated names)"
    )
    p.add_argument(
        "--interval",
        type=float,
        metavar="LEVEL",
        help=f"prediction-interval level in (0, 1) (default {d.get('interval_level', 0.8)})",
    )
    p.add_argument(
        "--interval-method",
        choices=("empirical", "normal"),
        help="how intervals are built (default empirical)",
    )
    p.add_argument(
        "--criterion", choices=("mae", "wape", "mase"), help="model-selection metric (default mae)"
    )
    p.add_argument(
        "--no-promo", action="store_true", help="ignore promotions (no promo-aware candidates)"
    )
    p.add_argument(
        "--order-budget",
        type=float,
        metavar="AMOUNT",
        help="cap the total order cost; lowest-priority orders are deferred",
    )
    p.add_argument(
        "--service-level-by-class",
        metavar="A=0.98,B=0.95,C=0.90",
        help="service level per ABC class (overrides --service-level for that class)",
    )
    p.set_defaults(handler=_cmd_run, creates_db=False)


def _add_runs(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("runs", help="list forecast runs, newest first")
    p.add_argument("--limit", type=int, default=20, help="rows to show, 0 = all (default 20)")
    p.add_argument("--json", action="store_true", help="JSON instead of a table")
    p.set_defaults(handler=_cmd_runs, creates_db=False)


def _add_query(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("query", help="execute a named analytics query (omit NAME to list them)")
    p.add_argument("name", nargs="?", metavar="NAME", help="query name")
    p.add_argument(
        "--list", action="store_true", help="list the queries with their required parameters"
    )
    p.add_argument("--run-id", type=int, help="run to report on (default: latest successful)")
    p.add_argument("--store-id", type=int)
    p.add_argument("--product-id", type=int)
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="bind :KEY (repeatable; values are coerced int -> float -> str)",
    )
    p.add_argument("--limit", type=int, default=20, help="rows to print, 0 = all (default 20)")
    p.add_argument(
        "--format", choices=("table", "csv", "json"), help="output format (default table)"
    )
    p.add_argument("--json", action="store_true", help="alias of --format json")
    p.add_argument(
        "--cells",
        choices=CELL_POLICIES,
        default="safe",
        help=(
            "affects --format csv only: 'safe' (default) writes text cells that a spreadsheet "
            "would evaluate as formulas (first visible character = + - @, or a leading "
            "tab/CR/LF) with a leading ' so they stay text; 'raw' writes every cell verbatim"
        ),
    )
    p.set_defaults(handler=_cmd_query, creates_db=False)


def _add_dashboard(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("dashboard", help="render the self-contained HTML dashboard")
    p.add_argument(
        "--out", default="dashboard/index.html", help="output file (default: %(default)s)"
    )
    p.add_argument("--run-id", type=int, help="run to render (default: latest successful)")
    p.set_defaults(handler=_cmd_dashboard, creates_db=False)


def _add_stats(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("stats", help="print row counts per table as JSON")
    p.add_argument("--json", action="store_true", help="accepted for symmetry (always JSON)")
    p.set_defaults(handler=_cmd_stats, creates_db=False)


def _load_plugins(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register optional commands from COMMAND_PLUGINS; unimportable modules are skipped."""
    for name in COMMAND_PLUGINS:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            log.debug("plugin %s not loaded: %s", name, exc)
            continue
        register = getattr(module, "register_cli", None)
        if not callable(register):
            log.debug("plugin %s has no register_cli(); skipped", name)
            continue
        try:
            register(sub)
        except Exception as exc:
            log.warning("plugin %s failed to register its commands: %s", name, exc)


def build_parser() -> argparse.ArgumentParser:
    """Build the full parser: built-in commands first, then plugin commands (C10)."""
    parser = argparse.ArgumentParser(
        prog="demandcast",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"demandcast {__version__}")
    parser.add_argument(
        "--db", default="demandcast.db", help="SQLite database path (default: %(default)s)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging on stderr")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only warnings/errors on stderr (results still print)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    _add_init(sub)
    _add_run(sub)
    _add_runs(sub)
    _add_query(sub)
    _add_dashboard(sub)
    _add_stats(sub)
    _load_plugins(sub)
    return parser


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, open the database and dispatch to the command handler.

    Returns the exit code: 0 on success, 1 for operational errors (reported as
    ``error: <message>`` on stderr, never a traceback), 2 for usage errors (argparse).
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    _configure_logging(*_logging_flags(args_list))
    parser = build_parser()
    args = parser.parse_args(args_list)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:  # a plugin registered a parser without a handler
        parser.error(f"command {args.cmd!r} has no handler")
    if args.cmd == "query" and (args.list or not args.name):
        _print_query_list(detailed=bool(args.list))  # needs no database at all
        return 0

    db_path = str(args.db)
    creates_db = bool(getattr(args, "creates_db", False))
    if not creates_db and _db_file_missing(db_path):
        print(f"error: database {db_path} does not exist ({MISSING_DB_HINT})", file=sys.stderr)
        return 1
    conn: sqlite3.Connection | None = None
    try:
        if creates_db and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = db.connect(db_path)
        if not creates_db and args.cmd != "stats":
            _require_schema(conn, db_path)
        return int(handler(conn, args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except _wrapped_errors() as exc:
        print(f"error: {_describe(exc)}", file=sys.stderr)
        log.debug("command %s failed", args.cmd, exc_info=True)
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
