"""CLI contract tests (C10): plugin hook, DB-file rule, error wrapping, formats and flags.

Every test drives ``cli.main([...])`` in-process against SQLite files under ``tmp_path``.
The shared dataset is deliberately tiny (2 stores x 4 products x 200 days, one run with
``--workers 1``) so the whole file stays well under its 5 s budget.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import io
import json
import logging
import runpy
import sqlite3
import subprocess
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pytest

import demandcast
from demandcast import cli, csvsafe, db, pipeline, replenish, simulate

INIT_ARGS = ["init", "--stores", "2", "--products", "4", "--days", "200"]
RUN_ARGS = ["run", "--horizon", "14", "--folds", "2", "--workers", "1"]
N_SERIES = 2 * 4
N_DAYS = 200


# --------------------------------------------------------------------------------------
# helpers & fixtures
# --------------------------------------------------------------------------------------


def invoke(capsys, argv: list[str]) -> tuple[int, str, str]:
    """Run ``cli.main`` and return ``(exit_code, stdout, stderr)``.

    argparse exits (usage errors, --help, --version) are turned into their exit code; any
    other exception escaping the CLI is a contract violation (D6) and fails the test.
    """
    try:
        code = cli.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception as exc:
        raise AssertionError(f"cli.main({argv}) let {exc!r} escape as a traceback") from exc
    out, err = capsys.readouterr()
    return code, out, err


def _capture(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _scalar(path: str, sql: str):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


def _has_field(cls, name: str) -> bool:
    return any(f.name == name for f in dataclasses.fields(cls))


@dataclass(frozen=True)
class CliDb:
    path: str
    init_out: str
    run_out: str
    run_id: int


@pytest.fixture(scope="module")
def cli_db(tmp_path_factory: pytest.TempPathFactory) -> CliDb:
    """2 x 4 x 200 dataset initialised and forecast once through the CLI (read-mostly)."""
    path = str(tmp_path_factory.mktemp("cli") / "demo.db")
    code, init_out = _capture(["--db", path, *INIT_ARGS])
    assert code == 0, init_out
    code, run_out = _capture(["--db", path, *RUN_ARGS])
    assert code == 0, run_out
    return CliDb(path, init_out, run_out, json.loads(run_out)["run_id"])


@pytest.fixture
def schema_only_db(tmp_path) -> str:
    """A database with the schema but no rows at all (what `init --no-data` produces)."""
    path = str(tmp_path / "schema.db")
    conn = db.connect(path)
    db.init_schema(conn)
    conn.close()
    return path


@pytest.fixture
def empty_file_db(tmp_path) -> str:
    """An SQLite file without a single DemandCast table."""
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()
    assert path.exists()
    return str(path)


def _fake_plugin(name: str, records: dict, *, creates_db: bool = False, outcome=0):
    """An in-memory plugin module exposing ``register_cli(sub)`` exactly as C10 describes."""
    module = types.ModuleType(name)

    def handler(conn, args):
        records["conn"] = conn
        records["args"] = args
        if isinstance(outcome, BaseException):
            raise outcome
        print(f"hello {args.who}")
        return outcome

    def register_cli(sub):
        p = sub.add_parser("hello", help="fake plugin command")
        p.add_argument("--who", default="world")
        p.set_defaults(handler=handler, creates_db=creates_db)

    module.register_cli = register_cli
    return module


# --------------------------------------------------------------------------------------
# plugin hook (C10)
# --------------------------------------------------------------------------------------


def test_plugin_default_names_match_contract():
    assert cli.COMMAND_PLUGINS == ("demandcast.evaluate", "demandcast.ingest", "demandcast.export")


def test_plugin_missing_module_is_skipped_and_fake_plugin_is_dispatched(
    cli_db, capsys, monkeypatch
):
    records: dict = {}
    mod = _fake_plugin("demandcast_fake_plugin_ok", records)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(
        cli, "COMMAND_PLUGINS", ("demandcast.plugin_that_does_not_exist", mod.__name__)
    )
    code, out, err = invoke(capsys, ["--db", cli_db.path, "hello", "--who", "ops"])
    assert code == 0, err
    assert out.strip() == "hello ops"
    assert isinstance(records["conn"], sqlite3.Connection)
    assert records["args"].who == "ops"
    # built-in commands keep working next to a missing plugin
    code, out, err = invoke(capsys, ["--db", cli_db.path, "stats"])
    assert code == 0, err
    assert json.loads(out)["sales_daily"] == N_SERIES * N_DAYS


@pytest.mark.parametrize(
    ("outcome", "expected_code", "needle"),
    [(3, 3, ""), (RuntimeError("boom"), 1, "error: boom"), (ValueError("bad"), 1, "error: bad")],
)
def test_plugin_handler_exit_code_and_errors_are_wrapped(
    cli_db, capsys, monkeypatch, outcome, expected_code, needle
):
    records: dict = {}
    mod = _fake_plugin("demandcast_fake_plugin_rc", records, outcome=outcome)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(cli, "COMMAND_PLUGINS", (mod.__name__,))
    code, _, err = invoke(capsys, ["--db", cli_db.path, "hello"])
    assert code == expected_code
    assert needle in err
    assert "Traceback" not in err


def test_plugin_keyboard_interrupt_is_reported_not_dumped(cli_db, capsys, monkeypatch):
    records: dict = {}
    mod = _fake_plugin("demandcast_fake_plugin_int", records, outcome=KeyboardInterrupt())
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(cli, "COMMAND_PLUGINS", (mod.__name__,))
    try:
        code = cli.main(["--db", cli_db.path, "hello"])
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped cli.main")
    except SystemExit as exc:
        code = exc.code
    assert code == 130
    assert "interrupted" in capsys.readouterr().err


def test_plugin_module_without_register_cli_is_ignored(cli_db, capsys, monkeypatch):
    mod = types.ModuleType("demandcast_fake_plugin_bare")
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(cli, "COMMAND_PLUGINS", (mod.__name__,))
    code, out, err = invoke(capsys, ["--db", cli_db.path, "stats"])
    assert code == 0, err
    assert "sales_daily" in json.loads(out)


def test_plugin_with_creates_db_may_create_the_file(tmp_path, capsys, monkeypatch):
    records: dict = {}
    mod = _fake_plugin("demandcast_fake_plugin_load", records, creates_db=True)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(cli, "COMMAND_PLUGINS", (mod.__name__,))
    target = tmp_path / "new.db"
    code, out, err = invoke(capsys, ["--db", str(target), "hello"])
    assert code == 0, err
    assert out.strip() == "hello world"
    assert target.exists()


def test_plugin_evaluate_is_registered_post_integration(cli_db, capsys):
    assert _has_field(pipeline.RunConfig, "cutoff_day")
    last_day = _scalar(cli_db.path, "SELECT MAX(day) FROM sales_daily")
    cutoff = (date.fromisoformat(last_day) - timedelta(days=14)).isoformat()
    code, _, err = invoke(capsys, ["--db", cli_db.path, *RUN_ARGS, "--cutoff", cutoff])
    assert code == 0, err
    code, out, err = invoke(capsys, ["--db", cli_db.path, "evaluate", "--json"])
    assert code == 0, err
    summary = json.loads(out)
    assert 0.0 <= summary["coverage"] <= 1.0


def test_plugin_load_and_export_round_trip_post_integration(cli_db, tmp_path, capsys):
    out_dir = tmp_path / "dataset"
    code, _, err = invoke(capsys, ["--db", cli_db.path, "export", "dataset", "--out", str(out_dir)])
    assert code == 0, err
    target = tmp_path / "loaded.db"
    code, _, err = invoke(capsys, ["--db", str(target), "load", "--dir", str(out_dir)])
    assert code == 0, err
    assert target.exists()
    code, out, err = invoke(capsys, ["--db", str(target), "stats"])
    assert code == 0, err
    assert json.loads(out)["sales_daily"] == N_SERIES * N_DAYS


# --------------------------------------------------------------------------------------
# DB-file rule: only init / creates_db commands may create the SQLite file
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--workers", "1"],
        ["query", "abc_classification"],
        ["runs"],
        ["stats"],
        ["dashboard", "--out", "{tmp}/index.html"],
    ],
)
def test_dbrule_missing_db_file_is_an_error_and_is_not_created(tmp_path, capsys, argv):
    missing = tmp_path / "missing.db"
    argv = [a.replace("{tmp}", str(tmp_path)) for a in argv]
    code, out, err = invoke(capsys, ["--db", str(missing), *argv])
    assert code == 1
    assert err.startswith(f"error: database {missing} does not exist")
    assert "demandcast init" in err
    assert "Traceback" not in err
    assert out == ""
    assert not missing.exists()
    assert not (tmp_path / "index.html").exists()


def test_dbrule_init_creates_the_file_and_memory_is_always_allowed(tmp_path, capsys):
    target = tmp_path / "fresh.db"
    code, out, err = invoke(
        capsys, ["--db", str(target), "init", "--stores", "1", "--products", "1", "--days", "40"]
    )
    assert code == 0, err
    assert target.exists()
    assert json.loads(out)["sales_daily"] == 40
    code, out, err = invoke(capsys, ["--db", ":memory:", "stats"])
    assert code == 0, err
    assert json.loads(out) == {}


def test_dbrule_init_creates_missing_parent_directories(tmp_path, capsys):
    target = tmp_path / "nested" / "dir" / "fresh.db"
    code, _, err = invoke(capsys, ["--db", str(target), "init", "--no-data"])
    assert code == 0, err
    assert target.exists()


# --------------------------------------------------------------------------------------
# error wrapping (D6) and `--limit 0` (D5)
# --------------------------------------------------------------------------------------


def test_errors_unknown_query_name_is_a_clean_exit_1(cli_db, capsys):
    code, out, err = invoke(capsys, ["--db", cli_db.path, "query", "does_not_exist"])
    assert code == 1
    assert err.startswith("error:")
    assert "does_not_exist" in err
    assert "abc_classification" in err  # tells the user what is available
    assert "Traceback" not in err
    assert out == ""


def test_errors_missing_query_params_are_reported(cli_db, capsys):
    code, _, err = invoke(capsys, ["--db", cli_db.path, "query", "series_history"])
    assert code == 1
    assert err.startswith("error:")
    assert ":store_id" in err
    assert ":product_id" in err
    assert "Traceback" not in err


def test_errors_run_id_query_without_a_successful_run(schema_only_db, capsys):
    code, _, err = invoke(
        capsys, ["--db", schema_only_db, "query", "forecast_accuracy_leaderboard"]
    )
    assert code == 1
    assert err.startswith("error:")
    assert "run" in err
    assert "Traceback" not in err


def test_errors_dashboard_without_a_run(schema_only_db, tmp_path, capsys):
    out_file = tmp_path / "d.html"
    code, _, err = invoke(capsys, ["--db", schema_only_db, "dashboard", "--out", str(out_file)])
    assert code == 1
    assert err.startswith("error:")
    assert "Traceback" not in err
    assert not out_file.exists()


def test_errors_run_on_empty_dataset(schema_only_db, capsys):
    code, _, err = invoke(capsys, ["--db", schema_only_db, "run", "--workers", "1"])
    assert code == 1
    assert err.startswith("error:")
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--workers", "1"],
        ["query", "abc_classification"],
        ["runs"],
        ["dashboard", "--out", "{tmp}/d.html"],
    ],
)
def test_errors_uninitialised_db_file_gives_a_hint(empty_file_db, tmp_path, capsys, argv):
    argv = [a.replace("{tmp}", str(tmp_path)) for a in argv]
    code, _, err = invoke(capsys, ["--db", empty_file_db, *argv])
    assert code == 1
    assert err.startswith("error:")
    assert "demandcast init" in err
    assert "Traceback" not in err


def test_errors_query_limit_zero_means_all_rows(cli_db, capsys):  # D5
    base = ["--db", cli_db.path, "query", "series_history", "--store-id", "1", "--product-id", "1"]
    code, out, err = invoke(capsys, base)
    assert code == 0, err
    lines = out.strip().splitlines()
    assert len(lines) == 2 + 20 + 1  # header, rule, 20 rows, "... N more rows"
    assert lines[-1] == f"... {N_DAYS - 20} more rows"
    code, out, err = invoke(capsys, [*base, "--limit", "0"])
    assert code == 0, err
    lines = out.strip().splitlines()
    assert len(lines) == 2 + N_DAYS
    assert "more rows" not in out
    code, out, err = invoke(capsys, [*base, "--limit", "0", "--format", "json"])
    assert code == 0, err
    assert len(json.loads(out)) == N_DAYS


def test_errors_schema_missing_error_from_the_db_layer_is_wrapped(
    cli_db, tmp_path, capsys, monkeypatch
):
    """Schema v2 (C5): db.ensure_schema guards every non-creating command except stats."""

    class SchemaMissingError(RuntimeError):
        pass

    calls: list[str] = []

    def fake_ensure_schema(conn):
        calls.append("ensure")
        raise SchemaMissingError("run `demandcast init` or `demandcast load` first")

    monkeypatch.setattr(db, "SchemaMissingError", SchemaMissingError, raising=False)
    monkeypatch.setattr(db, "ensure_schema", fake_ensure_schema, raising=False)
    code, _, err = invoke(capsys, ["--db", cli_db.path, "query", "abc_classification"])
    assert code == 1
    assert err.strip() == "error: run `demandcast init` or `demandcast load` first"
    assert calls == ["ensure"]
    code, _, err = invoke(capsys, ["--db", cli_db.path, "stats"])
    assert code == 0, err
    code, _, err = invoke(capsys, ["--db", str(tmp_path / "new.db"), "init", "--no-data"])
    assert code == 0, err
    assert calls == ["ensure"]  # stats and schema-creating commands skip the check


def test_errors_missing_parameter_error_from_the_db_layer_is_wrapped(cli_db, capsys, monkeypatch):
    class MissingParameterError(KeyError):
        pass

    def fake_run_query(conn, name, params=None):
        raise MissingParameterError("query 'abc_classification' is missing :run_id")

    monkeypatch.setattr(db, "MissingParameterError", MissingParameterError, raising=False)
    monkeypatch.setattr(db, "run_query", fake_run_query)
    code, _, err = invoke(capsys, ["--db", cli_db.path, "query", "abc_classification"])
    assert code == 1
    assert err.strip() == "error: query 'abc_classification' is missing :run_id"


# --------------------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _LegacySimConfig:  # the v0.3.0 generator config: no snapshot / future-promo fields
    n_stores: int = 10
    n_products: int = 40
    start: date = date(2024, 1, 1)
    days: int = 730
    seed: int = 42
    stockout_prob_scale: float = 1.0


@dataclass(frozen=True)
class _FullSimConfig(_LegacySimConfig):  # + the C8 fields
    snapshot_every_days: int | None = None
    future_promo_days: int = 0


def test_init_prints_counts_json(cli_db):
    counts = json.loads(cli_db.init_out)
    assert counts["stores"] == 2
    assert counts["products"] == 4
    assert counts["sales_daily"] == N_SERIES * N_DAYS
    assert counts["inventory_snapshots"] >= N_SERIES


def test_init_no_data_creates_schema_only(tmp_path, capsys):
    target = tmp_path / "schema.db"
    code, out, err = invoke(capsys, ["--db", str(target), "init", "--no-data"])
    assert code == 0, err
    counts = json.loads(out)
    assert counts["sales_daily"] == 0
    assert "forecast_runs" in counts
    assert target.exists()
    # a second init on the same (still empty) database is harmless
    code, _, err = invoke(capsys, ["--db", str(target), "init", "--no-data"])
    assert code == 0, err


def test_init_refuses_to_overwrite_existing_data(cli_db, capsys):
    code, _, err = invoke(capsys, ["--db", cli_db.path, *INIT_ARGS])
    assert code == 1
    assert err.startswith("error:")
    assert "already contains data" in err


@pytest.mark.parametrize("flag", [["--snapshot-every", "7"], ["--future-promo-days", "14"]])
def test_init_unsupported_flag_is_an_error_not_ignored(tmp_path, capsys, monkeypatch, flag):
    monkeypatch.setattr(simulate, "SimConfig", _LegacySimConfig)
    monkeypatch.setattr(
        simulate, "generate", lambda conn, cfg=None: pytest.fail("must not generate")
    )
    argv = [
        "--db",
        str(tmp_path / "x.db"),
        "init",
        "--stores",
        "1",
        "--products",
        "1",
        "--days",
        "40",
    ]
    code, _, err = invoke(capsys, [*argv, *flag])
    assert code == 1
    assert err.strip() == f"error: this build does not support {flag[0]}"


def test_init_new_flags_reach_simconfig_when_supported(tmp_path, capsys, monkeypatch):
    seen: dict = {}

    def fake_generate(conn, cfg=None):
        seen["cfg"] = cfg
        return {"stores": 2, "sales_daily": 0}

    monkeypatch.setattr(simulate, "SimConfig", _FullSimConfig)
    monkeypatch.setattr(simulate, "generate", fake_generate)
    argv = [
        "--db",
        str(tmp_path / "x.db"),
        "init",
        "--stores",
        "2",
        "--products",
        "4",
        "--days",
        "100",
    ]
    argv += [
        "--start",
        "2023-06-01",
        "--seed",
        "3",
        "--snapshot-every",
        "7",
        "--future-promo-days",
        "14",
    ]
    code, out, err = invoke(capsys, argv)
    assert code == 0, err
    cfg = seen["cfg"]
    assert isinstance(cfg, _FullSimConfig)
    assert (cfg.n_stores, cfg.n_products, cfg.days, cfg.seed) == (2, 4, 100, 3)
    assert cfg.start == date(2023, 6, 1)
    assert cfg.snapshot_every_days == 7
    assert cfg.future_promo_days == 14
    assert json.loads(out) == {"stores": 2, "sales_daily": 0}
    # flags that are not given are left to the dataclass defaults
    code, _, err = invoke(capsys, ["--db", str(tmp_path / "y.db"), "init"])
    assert code == 0, err
    assert seen["cfg"] == _FullSimConfig()


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _LegacyRunConfig:  # the v0.3.0 RunConfig
    horizon_days: int = 28
    n_folds: int = 4
    min_history_days: int = 84
    review_period_days: int = 7
    service_level: float = 0.95
    interval_z: float = 1.2816
    workers: int = 0


@dataclass(frozen=True)
class _FullRunConfig:  # the C6 RunConfig
    horizon_days: int = 28
    n_folds: int = 4
    min_history_days: int = 84
    review_period_days: int = 7
    service_level: float = 0.95
    interval_level: float = 0.8
    interval_method: str = "empirical"
    workers: int = 0
    cutoff_day: date | None = None
    models: tuple[str, ...] | None = None
    promo_aware: bool = True
    criterion: str = "mae"
    order_budget: float | None = None
    service_level_overrides: Mapping[str, float] | None = None


NEW_RUN_FLAGS = [
    ["--cutoff", "2024-06-01"],
    ["--models", "seasonal_naive"],
    ["--interval", "0.9"],
    ["--interval-method", "normal"],
    ["--criterion", "wape"],
    ["--no-promo"],
    ["--order-budget", "100"],
    ["--service-level-by-class", "A=0.98"],
]


def test_run_prints_the_forecast_runs_row_as_json(cli_db):
    row = json.loads(cli_db.run_out)
    assert row["status"] == "succeeded"
    assert row["series_count"] == N_SERIES
    assert row["horizon_days"] == 14
    assert isinstance(row["run_id"], int)
    for key in ("started_at", "finished_at", "cutoff_day", "notes"):
        assert key in row


@pytest.mark.parametrize("flag", NEW_RUN_FLAGS)
def test_run_unsupported_flag_is_an_error_not_ignored(cli_db, capsys, monkeypatch, flag):
    monkeypatch.setattr(pipeline, "RunConfig", _LegacyRunConfig)
    monkeypatch.setattr(pipeline, "run", lambda conn, cfg=None: pytest.fail("must not run"))
    code, _, err = invoke(capsys, ["--db", cli_db.path, "run", "--workers", "1", *flag])
    assert code == 1
    assert err.strip() == f"error: this build does not support {flag[0]}"


def test_run_flags_are_mapped_onto_runconfig(cli_db, capsys, monkeypatch):
    seen: dict = {}

    def fake_run(conn, cfg=None):
        seen["cfg"] = cfg
        return cli_db.run_id

    monkeypatch.setattr(pipeline, "RunConfig", _FullRunConfig)
    monkeypatch.setattr(pipeline, "run", fake_run)
    argv = ["--db", cli_db.path, "run", "--horizon", "7", "--folds", "3", "--workers", "1"]
    argv += ["--service-level", "0.9", "--review-period", "5", "--cutoff", "2024-06-01"]
    argv += ["--models", "seasonal_naive, moving_average", "--interval", "0.9"]
    argv += ["--interval-method", "normal", "--criterion", "wape", "--no-promo"]
    argv += ["--order-budget", "2500", "--service-level-by-class", "A=0.98,B=0.95"]
    code, out, err = invoke(capsys, argv)
    assert code == 0, err
    cfg = seen["cfg"]
    assert isinstance(cfg, _FullRunConfig)
    assert (cfg.horizon_days, cfg.n_folds, cfg.workers) == (7, 3, 1)
    assert (cfg.service_level, cfg.review_period_days) == (0.9, 5)
    assert cfg.cutoff_day == date(2024, 6, 1)
    assert cfg.models == ("seasonal_naive", "moving_average")
    assert (cfg.interval_level, cfg.interval_method, cfg.criterion) == (0.9, "normal", "wape")
    assert cfg.promo_aware is False
    assert cfg.order_budget == 2500.0
    assert cfg.service_level_overrides == {"A": 0.98, "B": 0.95}
    assert json.loads(out)["run_id"] == cli_db.run_id  # output shape: the forecast_runs row
    # flags that are not given are left to the dataclass defaults
    code, _, err = invoke(capsys, ["--db", cli_db.path, "run", "--workers", "1"])
    assert code == 0, err
    assert seen["cfg"] == _FullRunConfig(workers=1)


@pytest.mark.parametrize(
    ("flag", "needle"),
    [
        (["--horizon", "0"], "--horizon"),
        (["--folds", "0"], "--folds"),
        (["--service-level", "1.0"], "--service-level"),
        (["--interval", "1.5"], "--interval"),
        (["--order-budget", "-5"], "--order-budget"),
        (["--models", "nope"], "nope"),
        (["--service-level-by-class", "Z=0.9"], "--service-level-by-class"),
        (["--service-level-by-class", "A=1.2"], "--service-level-by-class"),
    ],
)
def test_run_rejects_bad_values_before_starting_a_run(cli_db, capsys, flag, needle):
    before = _scalar(cli_db.path, "SELECT COUNT(*) FROM forecast_runs")
    code, _, err = invoke(capsys, ["--db", cli_db.path, "run", "--workers", "1", *flag])
    assert code == 1
    assert err.startswith("error:")
    assert needle in err
    assert "Traceback" not in err
    assert _scalar(cli_db.path, "SELECT COUNT(*) FROM forecast_runs") == before


def test_run_service_level_parser_contract(monkeypatch):
    assert cli.parse_service_levels("A=0.98,B=0.95,C=0.90") == {"A": 0.98, "B": 0.95, "C": 0.90}
    for bad in ("A", "A=abc", "D=0.9", "A=1.0", "A=0.4"):
        with pytest.raises(ValueError):
            cli.parse_service_levels(bad)
    # the local fallback has the same rules (used until replenish.parse_service_levels exists)
    monkeypatch.delattr(replenish, "parse_service_levels", raising=False)
    assert cli.parse_service_levels(" A=0.98 , B=0.95 ") == {"A": 0.98, "B": 0.95}
    for bad in ("", "A", "A=", "A=abc", "D=0.9", "A=1.0", "A=0.4", "A=0.9,,B=0.9"):
        with pytest.raises(ValueError):
            cli.parse_service_levels(bad)
    # the shared implementation is preferred when present
    monkeypatch.setattr(replenish, "parse_service_levels", lambda spec: {"A": 0.5}, raising=False)
    assert cli.parse_service_levels("A=0.98") == {"A": 0.5}


# --------------------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------------------


def test_runs_lists_the_run(cli_db, capsys):
    code, out, err = invoke(capsys, ["--db", cli_db.path, "runs"])
    assert code == 0, err
    header = out.splitlines()[0].split()
    assert header[:2] == ["run_id", "status"]
    assert "succeeded" in out
    code, out, err = invoke(capsys, ["--db", cli_db.path, "runs", "--json"])
    assert code == 0, err
    rows = json.loads(out)
    ids = [r["run_id"] for r in rows]
    assert cli_db.run_id in ids
    assert ids == sorted(ids, reverse=True)
    row = next(r for r in rows if r["run_id"] == cli_db.run_id)
    assert row["status"] == "succeeded"
    assert row["series_count"] == N_SERIES
    assert row["horizon_days"] == 14
    for key in ("started_at", "finished_at", "cutoff_day", "notes"):
        assert key in row


def test_runs_prefers_pipeline_list_runs_when_present(cli_db, capsys, monkeypatch):
    calls: dict = {}

    def fake_list_runs(conn, limit=20):
        calls["limit"] = limit
        return [{"run_id": 99, "status": "succeeded", "started_at": "t", "notes": None}]

    monkeypatch.setattr(pipeline, "list_runs", fake_list_runs, raising=False)
    code, out, err = invoke(capsys, ["--db", cli_db.path, "runs", "--json", "--limit", "5"])
    assert code == 0, err
    assert json.loads(out) == [
        {"run_id": 99, "status": "succeeded", "started_at": "t", "notes": None}
    ]
    assert calls["limit"] == 5


# --------------------------------------------------------------------------------------
# query: formats, params, listing
# --------------------------------------------------------------------------------------


def test_query_formats_csv_and_json_parse(cli_db, capsys):
    base = ["--db", cli_db.path, "query", "forecast_accuracy_leaderboard"]
    code, out, err = invoke(capsys, [*base, "--format", "csv"])
    assert code == 0, err
    rows = list(csv.DictReader(io.StringIO(out)))
    assert rows
    assert {"model_name", "series_won", "avg_mae"} <= set(rows[0])
    assert sum(int(r["series_won"]) for r in rows) == N_SERIES
    code, out_json, err = invoke(capsys, [*base, "--format", "json"])
    assert code == 0, err
    assert sum(r["series_won"] for r in json.loads(out_json)) == N_SERIES
    code, out_alias, err = invoke(capsys, [*base, "--json"])
    assert code == 0, err
    assert out_alias == out_json
    # legacy id flags still work and feed the query parameters
    argv = ["--db", cli_db.path, "query", "replenishment_summary", "--run-id", str(cli_db.run_id)]
    code, out, err = invoke(capsys, [*argv, "--format", "csv", "--limit", "0"])
    assert code == 0, err
    assert len(list(csv.DictReader(io.StringIO(out)))) == N_SERIES


def test_query_table_format_is_the_default(cli_db, capsys):
    code, out, err = invoke(capsys, ["--db", cli_db.path, "query", "forecast_accuracy_leaderboard"])
    assert code == 0, err
    lines = out.splitlines()
    assert lines[0].split()[0] == "model_name"
    assert set(lines[1]) == {"-", " "}


def test_query_param_values_are_coerced_int_float_str(cli_db, capsys, monkeypatch):
    monkeypatch.setitem(db.QUERIES, "cli_probe", "SELECT :a AS a, :b AS b, :c AS c")
    argv = ["--db", cli_db.path, "query", "cli_probe", "--format", "json"]
    argv += ["--param", "a=7", "--param", "b=0.5", "--param", "c=2024-01-01"]
    code, out, err = invoke(capsys, argv)
    assert code == 0, err
    assert json.loads(out) == [{"a": 7, "b": 0.5, "c": "2024-01-01"}]
    # --param also drives the real queries
    argv = ["--db", cli_db.path, "query", "series_history", "--format", "json", "--limit", "0"]
    code, out, err = invoke(capsys, [*argv, "--param", "store_id=1", "--param", "product_id=2"])
    assert code == 0, err
    assert len(json.loads(out)) == N_DAYS


def test_query_param_bad_format_is_an_error(cli_db, capsys):
    code, _, err = invoke(
        capsys, ["--db", cli_db.path, "query", "series_history", "--param", "nokv"]
    )
    assert code == 1
    assert err.startswith("error:")
    assert "KEY=VALUE" in err


def test_query_list_shows_required_params_without_touching_the_db(tmp_path, capsys):
    absent = tmp_path / "absent.db"
    code, out, err = invoke(capsys, ["--db", str(absent), "query", "--list"])
    assert code == 0, err
    assert not absent.exists()
    lines = {line.split()[0]: line for line in out.strip().splitlines()}
    assert set(lines) == set(db.QUERIES)
    assert ":run_id" in lines["forecast_accuracy_leaderboard"]
    assert ":store_id" in lines["series_history"]
    assert ":product_id" in lines["series_history"]
    assert ":" not in lines["abc_classification"]


def test_query_list_uses_db_query_params_when_present(cli_db, capsys, monkeypatch):
    """Schema v2 (C5): db.query_params is authoritative over the regex fallback."""
    monkeypatch.setattr(db, "query_params", lambda name: frozenset({"custom_param"}), raising=False)
    code, out, err = invoke(capsys, ["--db", cli_db.path, "query", "--list"])
    assert code == 0, err
    assert all(":custom_param" in line for line in out.strip().splitlines())
    code, _, err = invoke(capsys, ["--db", cli_db.path, "query", "abc_classification"])
    assert code == 1
    assert ":custom_param" in err


def test_query_without_name_lists_names(cli_db, capsys):
    code, out, err = invoke(capsys, ["--db", cli_db.path, "query"])
    assert code == 0, err
    assert out.split() == sorted(db.QUERIES)


# --------------------------------------------------------------------------------------
# dashboard & stats (output shapes preserved)
# --------------------------------------------------------------------------------------


def test_dashboard_and_stats_shapes(cli_db, tmp_path, capsys):
    out_file = tmp_path / "site" / "index.html"
    code, out, err = invoke(capsys, ["--db", cli_db.path, "dashboard", "--out", str(out_file)])
    assert code == 0, err
    assert out.strip() == f"wrote {out_file}"
    text = out_file.read_text(encoding="utf-8")
    assert "Model leaderboard" in text
    assert "Replenishment order book" in text
    code, out, err = invoke(capsys, ["--db", cli_db.path, "stats"])
    assert code == 0, err
    counts = json.loads(out)
    assert counts["sales_daily"] == N_SERIES * N_DAYS
    assert counts["forecasts"] >= N_SERIES * 14


# --------------------------------------------------------------------------------------
# entry points, usage errors, --quiet
# --------------------------------------------------------------------------------------


def test_main_python_m_demandcast_version_via_runpy(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["demandcast", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("demandcast", run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"demandcast {demandcast.__version__}"


def test_main_python_m_demandcast_version_in_a_subprocess():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "demandcast", "--version"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"demandcast {demandcast.__version__}"


def test_main_help_lists_all_commands(capsys):
    code, out, _ = invoke(capsys, ["--help"])
    assert code == 0
    for command in ("init", "run", "runs", "query", "dashboard", "stats"):
        assert f"\n    {command} " in out or f"\n  {command} " in out
    assert "--quiet" in out


def test_main_usage_errors_exit_2(cli_db, capsys):
    assert invoke(capsys, [])[0] == 2
    assert invoke(capsys, ["bogus"])[0] == 2
    assert (
        invoke(capsys, ["--db", cli_db.path, "query", "abc_classification", "--format", "xml"])[0]
        == 2
    )
    assert invoke(capsys, ["--db", cli_db.path, "run", "--criterion", "rmse"])[0] == 2
    assert invoke(capsys, ["--db", cli_db.path, "run", "--cutoff", "not-a-date"])[0] == 2


def test_main_quiet_flag_silences_info_logging(cli_db, capsys, caplog):
    log = logging.getLogger("demandcast")
    code, _, err = invoke(capsys, ["--db", cli_db.path, "--quiet", "stats"])
    assert code == 0, err
    log.info("quiet-probe")
    assert "quiet-probe" not in caplog.text
    code, _, err = invoke(capsys, ["--db", cli_db.path, "stats"])
    assert code == 0, err
    log.info("loud-probe")
    assert "loud-probe" in caplog.text


# --------------------------------------------------------------------------------------
# host-review repair: `query --format csv` is spreadsheet-safe by default, `--cells raw` opts out
# --------------------------------------------------------------------------------------

# Text a spreadsheet would evaluate when the CSV is opened: the formula triggers = + - @, and a
# zero-width space in front of one (ingest strips ordinary whitespace; U+200B survives it).
HOSTILE_CITY = {1: "=1+1", 2: "+1"}
HOSTILE_NAME = {1: "@SUM(A1:A9)", 2: "-1", 3: "​=1+1"}


@dataclass(frozen=True)
class HostileDb:
    path: str
    run_id: int
    store_codes: dict[int, str]
    skus: dict[int, str]


def _rows(path: str, sql: str) -> list[dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


@pytest.fixture(scope="module")
def hostile_db(tmp_path_factory: pytest.TempPathFactory) -> HostileDb:
    """init -> `load` upserts of formula-looking master data -> run, all through the CLI."""
    root = tmp_path_factory.mktemp("hostile")
    path = str(root / "hostile.db")
    code, out = _capture(["--db", path, *INIT_ARGS])
    assert code == 0, out
    stores = _rows(path, "SELECT * FROM stores ORDER BY store_id")
    products = _rows(path, "SELECT * FROM products ORDER BY product_id")
    for row in stores:
        row["city"] = HOSTILE_CITY.get(row["store_id"], row["city"])
    for row in products:
        row["name"] = HOSTILE_NAME.get(row["product_id"], row["name"])
    stores_csv = _write_csv(root / "stores.csv", stores)
    products_csv = _write_csv(root / "products.csv", products)
    argv = ["--db", path, "load", "--stores", str(stores_csv), "--products", str(products_csv)]
    code, out = _capture(argv)
    assert code == 0, out
    report = json.loads(out)
    assert report["ok"] is True
    loads = [(r["table"], r["updated"], r["rejected"]) for r in report["loads"]]
    assert loads == [("stores", 2, 0), ("products", 4, 0)]
    code, out = _capture(["--db", path, *RUN_ARGS])
    assert code == 0, out
    return HostileDb(
        path,
        json.loads(out)["run_id"],
        {r["store_id"]: r["store_code"] for r in stores},
        {r["product_id"]: r["sku"] for r in products},
    )


def test_repair_hostile_master_data_went_through_the_real_ingest_path(hostile_db):
    assert _scalar(hostile_db.path, "SELECT city FROM stores WHERE store_id = 1") == "=1+1"
    assert _scalar(hostile_db.path, "SELECT name FROM products WHERE product_id = 3") == "​=1+1"
    assert _scalar(hostile_db.path, "SELECT COUNT(*) FROM data_loads") == 2


def test_repair_cli_uses_the_shared_csvsafe_policy():
    assert getattr(cli, "safe_rows", None) is csvsafe.safe_rows
    assert getattr(cli, "CELL_POLICIES", None) == csvsafe.CELL_POLICIES == ("safe", "raw")


def test_repair_query_csv_is_spreadsheet_safe_by_default(hostile_db, capsys):
    argv = ["--db", hostile_db.path, "query", "stockout_rate_by_store", "--format", "csv"]
    code, out, err = invoke(capsys, argv)
    assert code == 0, err
    by_code = {r["store_code"]: r for r in _csv_rows(out)}
    assert by_code[hostile_db.store_codes[1]]["city"] == "'=1+1"
    assert by_code[hostile_db.store_codes[2]]["city"] == "'+1"
    for row in by_code.values():  # numeric columns stay bare numbers
        assert not row["stockout_pct"].startswith("'")
        float(row["stockout_pct"])
        int(row["stockout_days"])
    # the apostrophe lives in the output only: database values are never mutated
    assert _scalar(hostile_db.path, "SELECT city FROM stores WHERE store_id = 1") == "=1+1"


def test_repair_query_csv_prefixes_hostile_names_but_not_numbers_or_benign_text(hostile_db, capsys):
    argv = ["--db", hostile_db.path, "query", "replenishment_summary", "--format", "csv"]
    code, out, err = invoke(capsys, [*argv, "--limit", "0"])
    assert code == 0, err
    rows = _csv_rows(out)
    assert len(rows) == N_SERIES
    names = {r["sku"]: r["name"] for r in rows}
    assert names[hostile_db.skus[1]] == "'@SUM(A1:A9)"
    assert names[hostile_db.skus[2]] == "'-1"
    assert names[hostile_db.skus[3]] == "'​=1+1"
    assert not names[hostile_db.skus[4]].startswith("'")  # benign name untouched
    for r in rows:
        assert not r["reason"].startswith("'")
        float(r["order_cost"])
        int(r["order_qty"])


def test_repair_query_csv_numbers_none_and_dates_are_untouched(cli_db, capsys, monkeypatch):
    monkeypatch.setitem(
        db.QUERIES,
        "cli_probe",
        "SELECT -0.14 AS bias, -5 AS qty, '2024-01-01' AS day, NULL AS note, "
        "'=1+1' AS sku, ' -5' AS padded, 'x+y' AS benign",
    )
    base = ["--db", cli_db.path, "query", "cli_probe", "--format", "csv"]
    code, out, err = invoke(capsys, base)
    assert code == 0, err
    assert out.splitlines() == [
        "bias,qty,day,note,sku,padded,benign",
        "-0.14,-5,2024-01-01,,'=1+1,' -5,x+y",
    ]
    code, out, err = invoke(capsys, [*base, "--cells", "raw"])
    assert code == 0, err
    assert out.splitlines()[1] == "-0.14,-5,2024-01-01,,=1+1, -5,x+y"


def test_repair_query_csv_matches_json_when_nothing_is_hostile(hostile_db, capsys):
    base = ["--db", hostile_db.path, "query", "forecast_accuracy_leaderboard"]
    code, out_csv, err = invoke(capsys, [*base, "--format", "csv"])
    assert code == 0, err
    code, out_json, err = invoke(capsys, [*base, "--format", "json"])
    assert code == 0, err
    expected = [
        {k: "" if v is None else str(v) for k, v in r.items()} for r in json.loads(out_json)
    ]
    assert expected
    assert _csv_rows(out_csv) == expected  # numbers (incl. negatives) are written as-is


def test_repair_query_cells_raw_writes_cells_verbatim(hostile_db, capsys):
    base = ["--db", hostile_db.path, "query", "stockout_rate_by_store", "--format", "csv"]
    code, out_raw, err = invoke(capsys, [*base, "--cells", "raw"])
    assert code == 0, err
    by_code = {r["store_code"]: r for r in _csv_rows(out_raw)}
    assert by_code[hostile_db.store_codes[1]]["city"] == "=1+1"
    assert by_code[hostile_db.store_codes[2]]["city"] == "+1"
    code, out_default, err = invoke(capsys, base)
    assert code == 0, err
    code, out_safe, err = invoke(capsys, [*base, "--cells", "safe"])
    assert code == 0, err
    assert out_default == out_safe
    assert out_safe != out_raw


@pytest.mark.parametrize("cells", [None, "safe", "raw"])
def test_repair_query_json_and_table_output_are_never_prefixed(hostile_db, capsys, cells):
    extra = [] if cells is None else ["--cells", cells]
    base = ["--db", hostile_db.path, "query", "stockout_rate_by_store"]
    code, out, err = invoke(capsys, [*base, "--format", "json", *extra])
    assert code == 0, err
    cities = {r["store_code"]: r["city"] for r in json.loads(out)}
    assert cities[hostile_db.store_codes[1]] == "=1+1"
    assert "'=1+1" not in out
    code, out, err = invoke(capsys, [*base, *extra])  # table is the default format
    assert code == 0, err
    assert "=1+1" in out
    assert "'=1+1" not in out


def test_repair_query_invalid_cells_value_is_a_usage_error(hostile_db, capsys):
    argv = ["--db", hostile_db.path, "query", "stockout_rate_by_store", "--format", "csv"]
    code, out, err = invoke(capsys, [*argv, "--cells", "bogus"])
    assert code == 2
    assert out == ""
    assert "--cells" in err
    assert "safe" in err
    assert "raw" in err


def test_repair_query_help_documents_the_cells_policy(capsys):
    code, out, _ = invoke(capsys, ["query", "--help"])
    assert code == 0
    assert "--cells {safe,raw}" in out
    options = out.split("options:", 1)[1]
    cells_help = options.split("--cells {safe,raw}", 1)[1]
    assert "csv" in cells_help
    assert "verbatim" in cells_help


def test_repair_query_csv_routes_rows_through_csvsafe_safe_rows(cli_db, capsys, monkeypatch):
    seen: list[int] = []

    def spy(rows):
        seen.append(len(rows))
        return [{**r, "model_name": "SPY"} for r in rows]

    monkeypatch.setattr(cli, "safe_rows", spy, raising=False)
    base = ["--db", cli_db.path, "query", "forecast_accuracy_leaderboard"]
    code, out, err = invoke(capsys, [*base, "--format", "csv"])
    assert code == 0, err
    rows = _csv_rows(out)
    assert seen == [len(rows)]
    assert {r["model_name"] for r in rows} == {"SPY"}
    for argv in ([*base, "--format", "csv", "--cells", "raw"], [*base, "--format", "json"], base):
        code, out, err = invoke(capsys, argv)
        assert code == 0, err
        assert "SPY" not in out
    assert seen == [len(rows)]  # raw / json / table never call the policy
