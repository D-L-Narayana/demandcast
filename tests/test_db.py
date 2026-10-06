"""Database layer (contract C5): schema v2 bootstrap, migration of v0.3.0 databases, named-query
helpers, bulk writes, nest-safe transactions and diagnostics."""

from __future__ import annotations

import pickle
import re
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from demandcast import db

SCHEMA_V1 = Path(__file__).parent / "fixtures" / "schema_v1.sql"

V2_COLUMNS = {
    "forecast_runs": {"config_json", "interval_level", "engine_version"},
    "forecasts": {"promo_flag"},
    "replenishment_orders": {"stockout_risk", "priority", "requested_qty"},
}
V2_TABLES = {"forecast_evaluations", "data_loads"}
V2_INDEXES = {"idx_forecasts_series_day", "idx_promotions_product"}
CORE_TABLES = {
    "stores",
    "products",
    "calendar",
    "promotions",
    "sales_daily",
    "inventory_snapshots",
    "forecast_runs",
    "forecasts",
    "backtest_metrics",
    "replenishment_orders",
}
BASELINE_QUERY_PARAMS: dict[str, set[str]] = {
    "weekly_sales_trend": set(),
    "abc_classification": set(),
    "stockout_rate_by_store": set(),
    "promo_lift": set(),
    "forecast_accuracy_leaderboard": {"run_id"},
    "replenishment_summary": {"run_id"},
    "series_history": {"store_id", "product_id"},
    "days_of_cover": {"run_id"},
}
PRODUCT_1 = "INSERT INTO products VALUES (1, 'SKU-1', 'Item 1', 'Grocery', 10.0, 15.0, 6, 4, NULL)"


# --- helpers ---------------------------------------------------------------------------------


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]


def _names(conn: sqlite3.Connection, kind: str) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


def _master(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    return [
        (r[0], r[1], r[2])
        for r in conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name")
    ]


def _table_info(conn: sqlite3.Connection, table: str) -> list[tuple]:
    """(cid, name, type, notnull, dflt_value, pk) for every column - the structural identity."""
    return [tuple(r) for r in conn.execute(f'PRAGMA table_info("{table}")')]


def _ddl(conn: sqlite3.Connection, name: str) -> str:
    """CREATE statement of a schema object with comments stripped and whitespace collapsed."""
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (name,)).fetchone()[0]
    return " ".join(re.sub(r"--[^\n]*", "", sql).split())


def _add_store(conn: sqlite3.Connection, store_id: int) -> None:
    conn.execute(
        "INSERT INTO stores VALUES (?, ?, ?, 'South', 'standard', '2020-01-01')",
        (store_id, f"S{store_id}", f"City {store_id}"),
    )


def _store_ids(conn: sqlite3.Connection) -> list[int]:
    return [r[0] for r in conn.execute("SELECT store_id FROM stores ORDER BY store_id")]


def _v1_database() -> sqlite3.Connection:
    """A v0.3.0 database (no user_version) holding a row in every table that v2 alters."""
    conn = db.connect(":memory:")
    conn.executescript(SCHEMA_V1.read_text())
    _add_store(conn, 1)
    conn.execute(PRODUCT_1)
    conn.execute("INSERT INTO calendar VALUES ('2024-01-01', 0, 1, 1, 2024, 0, NULL)")
    conn.execute("INSERT INTO sales_daily VALUES (1, 1, '2024-01-01', 5, 75.0, 0)")
    conn.execute(
        "INSERT INTO forecast_runs (started_at, cutoff_day, horizon_days, status) "
        "VALUES ('2024-01-02T00:00:00+00:00', '2024-01-01', 7, 'succeeded')"
    )
    conn.execute(
        "INSERT INTO forecasts VALUES (1, 1, 1, '2024-01-02', 'seasonal_naive', 5.0, 3.0, 7.0)"
    )
    conn.execute(
        "INSERT INTO replenishment_orders (run_id, store_id, product_id, order_day, expected_arrival, "
        "on_hand, on_order, lead_time_demand, safety_stock, reorder_point, order_up_to, order_qty, "
        "service_level, reason) VALUES (1, 1, 1, '2024-01-02', '2024-01-06', 10, 0, 20.0, 5.0, "
        "25.0, 40.0, 30, 0.95, 'below reorder point')"
    )
    conn.commit()
    return conn


def _insert_then_fail(conn: sqlite3.Connection, store_id: int) -> None:
    with db.transaction(conn):
        _add_store(conn, store_id)
        raise ValueError("inner failure")


def _outer_fails_after_inner_succeeds(conn: sqlite3.Connection) -> None:
    with db.transaction(conn):
        _add_store(conn, 1)
        with db.transaction(conn):
            _add_store(conn, 2)
        raise RuntimeError("outer failure")


# --- schema v2 on a fresh database -----------------------------------------------------------


def test_fresh_database_is_schema_version_2_with_all_v2_objects(fresh_db):
    assert db.SCHEMA_VERSION == 2
    assert db.schema_version(fresh_db) == db.SCHEMA_VERSION
    for table, cols in V2_COLUMNS.items():
        assert cols <= set(_columns(fresh_db, table)), table
    tables = _names(fresh_db, "table")
    assert tables >= CORE_TABLES
    assert tables >= V2_TABLES
    indexes = _names(fresh_db, "index")
    assert indexes >= V2_INDEXES
    assert indexes >= {"idx_sales_day", "idx_sales_product_day"}  # baseline indexes retained


def test_schema_file_is_versioned_and_ends_with_the_version_stamp():
    text = (db.SQL_DIR / "schema.sql").read_text()
    body = "\n".join(line for line in text.splitlines() if not line.strip().startswith("--"))
    statements = [s.strip() for s in body.split(";") if s.strip()]
    assert statements[0] == "PRAGMA foreign_keys = ON"
    assert statements[-1] == f"PRAGMA user_version = {db.SCHEMA_VERSION}"


def test_v2_tables_enforce_their_constraints(fresh_db):
    assert _names(fresh_db, "table") >= V2_TABLES
    _add_store(fresh_db, 1)
    fresh_db.execute(PRODUCT_1)
    fresh_db.execute(
        "INSERT INTO forecast_runs (started_at, cutoff_day, horizon_days) VALUES ('t0', '2024-01-01', 7)"
    )
    ins = "INSERT INTO forecast_evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    good = (
        1,
        1,
        1,
        "seasonal_naive",
        7,
        1.5,
        0.2,
        -0.1,
        0.8,
        10.5,
        50.0,
        0,
        "2024-01-09T00:00:00+00:00",
    )
    with pytest.raises(sqlite3.IntegrityError):  # coverage outside [0, 1]
        fresh_db.execute(ins, (*good[:8], 1.5, *good[9:]))
    with pytest.raises(sqlite3.IntegrityError):  # n_days must be positive
        fresh_db.execute(ins, (*good[:4], 0, *good[5:]))
    with pytest.raises(sqlite3.IntegrityError):  # unknown store (FK)
        fresh_db.execute(ins, (1, 99, *good[2:]))
    fresh_db.execute(ins, good)
    with pytest.raises(sqlite3.IntegrityError):  # one evaluation per run and series (PK)
        fresh_db.execute(ins, good)
    loads = (
        "INSERT INTO data_loads (loaded_at, source, table_name, mode, rows_inserted) "
        "VALUES (?, ?, ?, ?, ?)"
    )
    with pytest.raises(sqlite3.IntegrityError):  # mode enum
        fresh_db.execute(loads, ("t0", "sales.csv", "sales_daily", "merge", 10))
    fresh_db.execute(loads, ("t0", "sales.csv", "sales_daily", "upsert", 10))
    row = fresh_db.execute("SELECT * FROM data_loads").fetchone()
    assert row["load_id"] == 1
    assert (row["rows_updated"], row["rows_rejected"], row["notes"]) == (0, 0, None)


def test_table_counts_covers_v2_tables_and_excludes_sqlite_internals(fresh_db):
    fresh_db.execute(
        "INSERT INTO forecast_runs (started_at, cutoff_day, horizon_days) VALUES ('t0', '2024-01-01', 7)"
    )
    counts = db.table_counts(fresh_db)
    assert "sqlite_sequence" not in counts
    assert set(counts) >= V2_TABLES
    assert counts["forecast_runs"] == 1


def test_init_schema_is_idempotent_on_a_current_database(fresh_db):
    _add_store(fresh_db, 1)
    fresh_db.commit()
    before = _master(fresh_db)
    db.init_schema(fresh_db)
    db.init_schema(fresh_db)
    assert db.schema_version(fresh_db) == 2
    assert _master(fresh_db) == before
    assert _store_ids(fresh_db) == [1]


# --- migration of a v0.3.0 database ----------------------------------------------------------


@pytest.mark.parametrize("upgrade", ["migrate", "init_schema"])
def test_v1_database_migrates_preserving_rows(upgrade):
    conn = _v1_database()
    assert db.schema_version(conn) == 0  # the fixture really is the pre-versioning schema
    for table, cols in V2_COLUMNS.items():
        assert not cols & set(_columns(conn, table)), f"fixture already has v2 columns in {table}"
    before = db.table_counts(conn)

    if upgrade == "migrate":
        assert db.migrate(conn) == 2
    else:
        db.init_schema(conn)

    assert db.schema_version(conn) == 2
    for table, cols in V2_COLUMNS.items():
        assert cols <= set(_columns(conn, table)), table
    assert _names(conn, "table") >= V2_TABLES
    assert _names(conn, "index") >= V2_INDEXES
    after = db.table_counts(conn)
    assert {k: after[k] for k in before} == before  # every pre-existing row survived
    fc = conn.execute("SELECT * FROM forecasts").fetchone()
    assert (fc["model_name"], fc["yhat"], fc["promo_flag"]) == ("seasonal_naive", 5.0, 0)
    order = conn.execute("SELECT * FROM replenishment_orders").fetchone()
    assert order["order_qty"] == 30
    assert (order["stockout_risk"], order["priority"], order["requested_qty"]) == (None, None, None)
    run = conn.execute("SELECT * FROM forecast_runs").fetchone()
    assert run["status"] == "succeeded"
    assert (run["config_json"], run["interval_level"], run["engine_version"]) == (None, None, None)
    conn.execute(
        "UPDATE forecast_runs SET config_json = '{}', interval_level = 0.8, engine_version = '0.4.0'"
    )
    assert conn.execute("SELECT interval_level FROM forecast_runs").fetchone()[0] == 0.8


def test_migrate_is_idempotent_and_matches_a_fresh_schema():
    conn = _v1_database()
    assert db.migrate(conn) == 2
    snapshot = _master(conn)
    assert db.migrate(conn) == 2
    assert db.migrate(conn) == 2
    assert _master(conn) == snapshot  # the re-runs changed nothing
    fresh = db.connect(":memory:")
    db.init_schema(fresh)
    assert _names(conn, "table") == _names(fresh, "table")
    assert _names(conn, "index") == _names(fresh, "index")
    for table in sorted(_names(fresh, "table")):
        # same columns in the same order with the same type / NOT NULL / default / PK flags
        assert _table_info(conn, table) == _table_info(fresh, table), table
    for obj in sorted(V2_TABLES | V2_INDEXES):
        assert _ddl(conn, obj) == _ddl(fresh, obj), obj  # migration DDL == schema.sql DDL


def test_migrate_on_an_empty_database_raises_schema_missing():
    conn = db.connect(":memory:")
    with pytest.raises(db.SchemaMissingError, match="demandcast init"):
        db.migrate(conn)
    assert _names(conn, "table") == set()  # nothing half-created
    assert db.schema_version(conn) == 0


def test_migrate_refuses_a_database_from_a_newer_release(fresh_db):
    fresh_db.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="99"):
        db.migrate(fresh_db)
    assert db.schema_version(fresh_db) == 99  # left untouched


def test_v1_database_runs_the_pipeline_after_migration():
    from demandcast import pipeline
    from demandcast.simulate import SimConfig, generate

    conn = db.connect(":memory:")
    conn.executescript(SCHEMA_V1.read_text())
    generate(conn, SimConfig(n_stores=2, n_products=4, start=date(2024, 1, 1), days=120, seed=3))
    assert db.ensure_schema(conn) == 2
    run_id = pipeline.run(conn, pipeline.RunConfig(horizon_days=7, n_folds=2, workers=1))
    run = conn.execute(
        "SELECT status, series_count FROM forecast_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert run["status"] == "succeeded"
    assert run["series_count"] == 8
    n_fc = conn.execute("SELECT COUNT(*) FROM forecasts WHERE run_id = ?", (run_id,)).fetchone()[0]
    assert n_fc == 8 * 7


# --- ensure_schema ---------------------------------------------------------------------------


def test_ensure_schema_fails_fast_without_the_core_tables():
    conn = db.connect(":memory:")
    with pytest.raises(db.SchemaMissingError, match="demandcast init"):
        db.ensure_schema(conn)
    assert issubclass(db.SchemaMissingError, RuntimeError)
    db.init_schema(conn)
    assert db.ensure_schema(conn) == 2


def test_ensure_schema_upgrades_a_v1_database():
    conn = _v1_database()
    assert db.ensure_schema(conn) == 2
    assert "config_json" in _columns(conn, "forecast_runs")
    assert db.ensure_schema(conn) == 2


# --- named queries ---------------------------------------------------------------------------


def test_all_baseline_queries_still_load():
    assert set(db.QUERIES) >= set(BASELINE_QUERY_PARAMS)


@pytest.mark.parametrize(("name", "expected"), sorted(BASELINE_QUERY_PARAMS.items()))
def test_query_params_for_baseline_queries(name, expected):
    params = db.query_params(name)
    assert isinstance(params, frozenset)
    assert params == expected


def test_query_params_ignores_literals_and_comments(monkeypatch):
    monkeypatch.setitem(
        db.QUERIES,
        "_probe",
        "SELECT ':not_a_param', '12:30' AS t, \"x:y\" FROM z\n"
        "WHERE a = :real AND b = :other -- trailing :comment\n"
        "/* block :comment */ AND c = :real",
    )
    assert db.query_params("_probe") == {"real", "other"}
    with pytest.raises(KeyError):
        db.query_params("no_such_query")


def test_run_query_unknown_name_lists_available_queries(fresh_db):
    with pytest.raises(KeyError) as info:
        db.run_query(fresh_db, "does_not_exist")
    assert "does_not_exist" in str(info.value)
    assert "abc_classification" in str(info.value)
    assert not isinstance(info.value, db.MissingParameterError)
    assert isinstance(info.value, db.UnknownQueryError)
    assert not str(info.value).startswith(("'", '"'))  # readable `error: ...` line for the CLI


def test_run_query_missing_parameters_raises_before_executing(fresh_db):
    executed: list[str] = []
    fresh_db.set_trace_callback(executed.append)
    with pytest.raises(db.MissingParameterError) as info:
        db.run_query(fresh_db, "series_history", {"store_id": 1})
    fresh_db.set_trace_callback(None)
    assert executed == []  # nothing reached SQLite
    assert info.value.missing == ("product_id",)
    assert "series_history" in str(info.value)
    assert ":product_id" in str(info.value)
    assert not str(info.value).startswith(("'", '"'))  # KeyError's repr-quoting suppressed
    assert isinstance(info.value, KeyError)
    with pytest.raises(db.MissingParameterError, match="run_id"):
        db.run_query(fresh_db, "forecast_accuracy_leaderboard")
    # complete parameters (plus an unused extra one) execute normally
    rows = db.run_query(fresh_db, "series_history", {"store_id": 1, "product_id": 1, "extra": 0})
    assert rows == []


def test_load_queries_rejects_duplicate_names(tmp_path):
    good = tmp_path / "ok.sql"
    good.write_text("-- name: a\nSELECT 1;\n-- name: b\nSELECT 2;\n")
    assert set(db.load_queries(good)) == {"a", "b"}
    dup = tmp_path / "dup.sql"
    dup.write_text("-- name: a\nSELECT 1;\n-- name: b\nSELECT 2;\n-- name: a\nSELECT 3;\n")
    with pytest.raises(ValueError, match=r"duplicate.*'a'"):
        db.load_queries(dup)


def test_db_errors_survive_pickling():
    """Errors may cross the process-pool boundary; message and payload must round-trip."""
    samples = [
        db.MissingParameterError("series_history", {"product_id", "store_id"}),
        db.UnknownQueryError("nope", ["b", "a"]),
        db.SchemaMissingError(),
        db.SchemaVersionError(7),
    ]
    for exc in samples:
        clone = pickle.loads(pickle.dumps(exc))
        assert type(clone) is type(exc)
        assert str(clone) == str(exc)
    assert pickle.loads(pickle.dumps(samples[0])).missing == ("product_id", "store_id")
    assert pickle.loads(pickle.dumps(samples[1])).available == ("a", "b")
    assert pickle.loads(pickle.dumps(samples[3])).found == 7


# --- bulk writes -----------------------------------------------------------------------------


def test_insert_many_quotes_identifiers(fresh_db):
    fresh_db.execute(
        'CREATE TABLE "order" ("id" INTEGER PRIMARY KEY, "when" TEXT NOT NULL, "select" REAL)'
    )
    try:
        inserted = db.insert_many(
            fresh_db, "order", ["id", "when", "select"], [(1, "now", 0.5), (2, "later", 1.5)]
        )
    except sqlite3.OperationalError as exc:  # unquoted keywords are a syntax error
        pytest.fail(f"insert_many does not quote identifiers: {exc}")
    assert inserted == 2
    got = [r[0] for r in fresh_db.execute('SELECT "when" FROM "order" ORDER BY "id"')]
    assert got == ["now", "later"]
    assert db.insert_many(fresh_db, "order", ["id", "when"], []) == 0


def test_upsert_many_inserts_then_updates(fresh_db):
    cols = ["store_id", "store_code", "city", "region", "format", "opened_on"]
    rows = [
        (1, "BLR", "Bengaluru", "South", "flagship", "2020-01-01"),
        (2, "HYD", "Hyderabad", "South", "standard", "2021-01-01"),
    ]
    assert db.upsert_many(fresh_db, "stores", cols, rows, ["store_id"]) == 2
    changed = [
        (1, "BLR", "Bengaluru", "South", "express", "2020-01-01"),  # update
        (3, "CHN", "Chennai", "South", "standard", "2022-01-01"),  # insert
    ]
    assert db.upsert_many(fresh_db, "stores", cols, changed, ["store_id"]) == 2
    got = {
        r["store_id"]: (r["format"], r["city"]) for r in fresh_db.execute("SELECT * FROM stores")
    }
    assert got == {
        1: ("express", "Bengaluru"),
        2: ("standard", "Hyderabad"),
        3: ("standard", "Chennai"),
    }


def test_upsert_many_composite_key_and_do_nothing_branch(fresh_db):
    _add_store(fresh_db, 1)
    fresh_db.execute(PRODUCT_1)
    cols = ["store_id", "product_id", "snapshot_day", "on_hand", "on_order"]
    key = ["store_id", "product_id", "snapshot_day"]
    assert (
        db.upsert_many(fresh_db, "inventory_snapshots", cols, [(1, 1, "2024-01-01", 10, 0)], key)
        == 1
    )
    assert (
        db.upsert_many(fresh_db, "inventory_snapshots", cols, [(1, 1, "2024-01-01", 7, 12)], key)
        == 1
    )
    snap = fresh_db.execute("SELECT on_hand, on_order FROM inventory_snapshots").fetchall()
    assert [tuple(r) for r in snap] == [(7, 12)]
    fresh_db.execute("CREATE TABLE tags (tag TEXT PRIMARY KEY)")
    assert db.upsert_many(fresh_db, "tags", ["tag"], [("a",), ("b",)], ["tag"]) == 2
    assert db.upsert_many(fresh_db, "tags", ["tag"], [("a",), ("c",)], ["tag"]) == 1  # 'a' skipped


def test_upsert_many_validates_conflict_columns(fresh_db):
    with pytest.raises(ValueError, match="city_x"):
        db.upsert_many(fresh_db, "stores", ["store_id", "city"], [], ["city_x"])
    with pytest.raises(ValueError, match="conflict_columns"):
        db.upsert_many(fresh_db, "stores", ["store_id"], [], [])


# --- transactions ----------------------------------------------------------------------------


def test_transaction_top_level_commits_or_rolls_back(fresh_db):
    with db.transaction(fresh_db):
        _add_store(fresh_db, 1)
    assert not fresh_db.in_transaction
    with pytest.raises(ValueError, match="inner failure"):
        _insert_then_fail(fresh_db, 2)
    assert not fresh_db.in_transaction
    assert _store_ids(fresh_db) == [1]


def test_nested_transaction_rolls_back_only_the_inner_savepoint(fresh_db):
    try:
        with db.transaction(fresh_db):
            _add_store(fresh_db, 1)
            with pytest.raises(ValueError, match="inner failure"):
                _insert_then_fail(fresh_db, 2)
            assert fresh_db.in_transaction  # the outer block is still open
            _add_store(fresh_db, 3)
    except sqlite3.OperationalError as exc:
        # a non-nest-safe transaction() issues a second BEGIN and dies here
        pytest.fail(f"transaction() is not nest-safe: {type(exc).__name__}: {exc}")
    assert not fresh_db.in_transaction
    assert _store_ids(fresh_db) == [1, 3]


def test_nested_transaction_outer_failure_discards_released_inner_work(fresh_db):
    with pytest.raises(RuntimeError, match="outer failure"):
        _outer_fails_after_inner_succeeds(fresh_db)
    assert not fresh_db.in_transaction
    assert _store_ids(fresh_db) == []


def test_transaction_inside_an_implicit_transaction_uses_a_savepoint(fresh_db):
    _add_store(fresh_db, 1)  # the sqlite3 module opened an implicit transaction here
    assert fresh_db.in_transaction
    with db.transaction(fresh_db):
        _add_store(fresh_db, 2)
    assert fresh_db.in_transaction  # still the caller's transaction to commit or roll back
    fresh_db.rollback()
    assert _store_ids(fresh_db) == []


def test_savepoints_nest_three_levels_deep(fresh_db):
    with db.transaction(fresh_db):
        _add_store(fresh_db, 1)
        with db.transaction(fresh_db):
            _add_store(fresh_db, 2)
            with pytest.raises(ValueError, match="inner failure"):
                _insert_then_fail(fresh_db, 3)
            _add_store(fresh_db, 4)
    assert not fresh_db.in_transaction
    assert _store_ids(fresh_db) == [1, 2, 4]


# --- connect & diagnostics -------------------------------------------------------------------


def test_connect_sets_busy_timeout_and_keeps_pragmas(tmp_path):
    mem = db.connect(":memory:")
    assert mem.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    assert mem.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    file_conn = db.connect(tmp_path / "t.db", timeout=2.5)
    assert file_conn.execute("PRAGMA busy_timeout").fetchone()[0] == 2500
    assert file_conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert file_conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    file_conn.close()


def test_explain_query_returns_plan_lines(fresh_db, monkeypatch):
    # a probe whose plan is known: PK lookup on sales_daily (independent of analytics.sql text)
    monkeypatch.setitem(
        db.QUERIES, "_probe", "SELECT day FROM sales_daily WHERE store_id = :store_id"
    )
    plan = db.explain_query(fresh_db, "_probe")
    assert isinstance(plan, list)
    assert plan
    assert all(isinstance(line, str) and line for line in plan)
    assert any("sales_daily" in line and "SEARCH" in line for line in plan)
    # every baseline query explains without parameter values (unbound params become NULL)
    for name in BASELINE_QUERY_PARAMS:
        assert db.explain_query(fresh_db, name), name
    assert db.explain_query(fresh_db, "forecast_accuracy_leaderboard", {"run_id": 1})
    with pytest.raises(KeyError):
        db.explain_query(fresh_db, "no_such_query")


def test_integrity_check_reports_foreign_key_violations(fresh_db):
    assert db.integrity_check(fresh_db) == []
    fresh_db.execute("PRAGMA foreign_keys = OFF")
    _add_store(fresh_db, 1)
    fresh_db.execute(PRODUCT_1)
    fresh_db.execute("INSERT INTO calendar VALUES ('2024-01-01', 0, 1, 1, 2024, 0, NULL)")
    fresh_db.execute(
        "INSERT INTO sales_daily VALUES (99, 1, '2024-01-01', 1, 1.0, 0)"
    )  # no store 99
    fresh_db.commit()
    fresh_db.execute("PRAGMA foreign_keys = ON")
    problems = db.integrity_check(fresh_db)
    assert len(problems) == 1
    assert "sales_daily" in problems[0]
    assert "stores" in problems[0]
