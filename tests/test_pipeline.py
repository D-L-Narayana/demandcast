"""Pipeline tests: RunConfig, densified loading (D7), backdated runs, promo flags, ABC as-of,
run listing, failed-run status (D8), pool equality, backtest intervals, model restriction,
per-class service levels and budgeted orders.

Every fixture here is function-scoped and tiny (2-3 stores x 4-8 products x <= 300 days);
runs use workers=1 except the single pool-equality test.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, fields
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

import demandcast
from demandcast import db, pipeline
from demandcast.backtest import select_model
from demandcast.models import make_model
from demandcast.replenish import BudgetItem, allocate_budget
from demandcast.simulate import SimConfig, generate

START = date(2024, 1, 1)
FAST = {"horizon_days": 14, "n_folds": 2, "workers": 1}
SCHEMA_V1 = Path(__file__).parent / "fixtures" / "schema_v1.sql"  # the v0.3.0 schema (W05)

FORECAST_COLUMNS = (
    "SELECT store_id, product_id, target_day, model_name, yhat, yhat_lower, yhat_upper, "
    "promo_flag FROM forecasts WHERE run_id=? ORDER BY 1, 2, 3"
)
METRIC_COLUMNS = (
    "SELECT store_id, product_id, model_name, folds, mae, wape, bias, mase, selected "
    "FROM backtest_metrics WHERE run_id=? ORDER BY 1, 2, 3"
)
ORDER_COLUMNS = (
    "SELECT store_id, product_id, order_day, expected_arrival, on_hand, on_order, "
    "lead_time_demand, safety_stock, reorder_point, order_up_to, order_qty, service_level, "
    "reason, stockout_risk, priority, requested_qty FROM replenishment_orders "
    "WHERE run_id=? ORDER BY 1, 2"
)


def _make_db(stores: int, products: int, days: int, seed: int = 7, **sim) -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.init_schema(conn)
    generate(
        conn,
        SimConfig(n_stores=stores, n_products=products, start=START, days=days, seed=seed, **sim),
    )
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _clone(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Independent in-memory copy of a database (SQLite online-backup API)."""
    other = db.connect(":memory:")
    conn.backup(other)
    return other


def _rows(conn: sqlite3.Connection, sql: str, *params) -> list[tuple]:
    return [tuple(r) for r in conn.execute(sql, params)]


def _scalar(conn: sqlite3.Connection, sql: str, *params):
    return conn.execute(sql, params).fetchone()[0]


def _notes(conn: sqlite3.Connection, run_id: int) -> dict[str, str]:
    text = _scalar(conn, "SELECT notes FROM forecast_runs WHERE run_id=?", run_id)
    return dict(part.strip().split("=", 1) for part in text.split(";"))


def _config(conn: sqlite3.Connection, run_id: int) -> dict:
    return json.loads(_scalar(conn, "SELECT config_json FROM forecast_runs WHERE run_id=?", run_id))


def _zero_inventory(conn: sqlite3.Connection) -> None:
    """Empty every shelf so each series sits below its reorder point (one order per series)."""
    conn.execute("UPDATE inventory_snapshots SET on_hand = 0, on_order = 0")
    conn.commit()


def _v1_database(stores: int, products: int, days: int, seed: int) -> sqlite3.Connection:
    """A database exactly as v0.3.0 would have created it (no user_version, no v2 columns)."""
    conn = db.connect(":memory:")
    conn.executescript(SCHEMA_V1.read_text(encoding="utf-8"))
    generate(
        conn, SimConfig(n_stores=stores, n_products=products, start=START, days=days, seed=seed)
    )
    return conn


@pytest.fixture
def db_2x4() -> sqlite3.Connection:
    """2 stores x 4 products x 200 days (2024-01-01 .. 2024-07-18)."""
    return _make_db(2, 4, 200)


@pytest.fixture
def db_3x8() -> sqlite3.Connection:
    """3 stores x 8 products x 300 days (2024-01-01 .. 2024-10-26)."""
    return _make_db(3, 8, 300)


# ---- RunConfig ----------------------------------------------------------------------------------
def test_runconfig_defaults_match_contract():
    cfg = pipeline.RunConfig()
    assert (cfg.horizon_days, cfg.n_folds, cfg.min_history_days, cfg.review_period_days) == (
        28,
        4,
        84,
        7,
    )
    assert cfg.service_level == 0.95
    assert cfg.interval_level == 0.8 and cfg.interval_method == "empirical"
    assert cfg.workers == 0
    assert cfg.cutoff_day is None and cfg.models is None and cfg.promo_aware is True
    assert cfg.criterion == "mae" and cfg.order_budget is None
    assert cfg.service_level_overrides is None
    assert not hasattr(cfg, "interval_z")  # replaced by interval_level


def test_runconfig_normalises_and_serialises():
    cfg = pipeline.RunConfig(
        cutoff_day="2024-10-12",
        models=["seasonal_naive", "holt_winters"],
        service_level_overrides={"A": 0.98},
    )
    assert cfg.cutoff_day == date(2024, 10, 12)
    assert cfg.models == ("seasonal_naive", "holt_winters")
    text = json.dumps(asdict(cfg), default=str)
    assert json.loads(text)["cutoff_day"] == "2024-10-12"
    assert json.loads(text)["service_level_overrides"] == {"A": 0.98}


@pytest.mark.parametrize(
    "bad",
    [
        {"interval_level": 1.0},
        {"interval_level": 0.0},
        {"interval_method": "bootstrap"},
        {"criterion": "rmse"},
        {"horizon_days": 0},
        {"n_folds": 0},
        {"order_budget": -1.0},
        {"service_level": 1.0},
        {"service_level_overrides": {"A": 1.5}},
        {"workers": -1},
    ],
)
def test_runconfig_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        pipeline.RunConfig(**bad)


def test_run_persists_config_json_interval_level_and_engine_version(db_2x4):
    cfg = pipeline.RunConfig(
        cutoff_day=date(2024, 7, 11),
        interval_level=0.9,
        models=("seasonal_naive", "holt_winters"),
        service_level_overrides={"A": 0.98, "B": 0.95},
        **FAST,
    )
    run_id = pipeline.run(db_2x4, cfg)
    row = db_2x4.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
    assert row["status"] == "succeeded"
    assert row["config_json"] is not None, "config_json must be written with the run row"
    assert row["interval_level"] == pytest.approx(0.9)
    assert row["engine_version"] == demandcast.__version__
    stored = json.loads(row["config_json"])
    assert set(stored) == {f.name for f in fields(pipeline.RunConfig)}
    assert stored["cutoff_day"] == "2024-07-11"
    # round-trips to the exact RunConfig (date comes back as ISO text, tuple as list)
    assert pipeline.RunConfig(**stored) == cfg


# ---- schema handling ----------------------------------------------------------------------------
def test_run_migrates_a_v1_database_and_rejects_an_uninitialised_one():
    bare = db.connect(":memory:")
    with pytest.raises(db.SchemaMissingError):
        pipeline.run(bare, pipeline.RunConfig(**FAST))

    v1 = _v1_database(2, 4, 120, seed=3)
    assert db.schema_version(v1) == 0
    assert "config_json" not in _columns(v1, "forecast_runs")
    run_id = pipeline.run(v1, pipeline.RunConfig(horizon_days=7, n_folds=2, workers=1))
    assert db.schema_version(v1) == 2, "run() must migrate the database before writing"
    row = v1.execute(
        "SELECT status, series_count, config_json, engine_version FROM forecast_runs "
        "WHERE run_id=?",
        (run_id,),
    ).fetchone()
    assert row["status"] == "succeeded" and row["series_count"] == 8
    assert json.loads(row["config_json"])["horizon_days"] == 7
    assert row["engine_version"] == demandcast.__version__
    assert _scalar(v1, "SELECT COUNT(*) FROM forecasts WHERE run_id=?", run_id) == 8 * 7
    assert (
        _scalar(
            v1,
            "SELECT COUNT(*) FROM replenishment_orders WHERE run_id=? AND requested_qty IS NULL",
            run_id,
        )
        == 0
    )


# ---- densified series loading (D7) --------------------------------------------------------------
def test_load_series_densifies_missing_days(db_2x4):
    conn = db_2x4
    deleted = ["2024-02-10", "2024-02-11", "2024-02-12", "2024-04-03", "2024-04-04"]
    conn.execute(
        "DELETE FROM sales_daily WHERE store_id=1 AND product_id=1 AND day IN (?,?,?,?,?)",
        deleted,
    )
    # a second series that only starts 30 days into the calendar
    conn.execute("DELETE FROM sales_daily WHERE store_id=1 AND product_id=2 AND day < '2024-01-31'")
    conn.commit()
    truth = {
        r["day"]: r["units_sold"]
        for r in conn.execute(
            "SELECT day, units_sold FROM sales_daily WHERE store_id=1 AND product_id=1"
        )
    }
    assert len(truth) == 195

    series, axis = pipeline._load_series(conn, "2024-07-18")

    assert len(axis) == 200 and axis[0] == "2024-01-01" and axis[-1] == "2024-07-18"
    y = series[(1, 1)]
    assert y.shape == (200,), "array must span the calendar, not collapse the missing days"
    for i, day in enumerate(axis):
        assert y[i] == (0 if day in deleted else truth[day]), day
    # weekday profile aligned with the calendar: weekend positions of the array are real
    # Saturdays/Sundays (the synthetic data sells more at weekends)
    weekday = np.array([date.fromisoformat(d).weekday() for d in axis])
    weekend_units = conn.execute(
        """SELECT SUM(units_sold) FROM sales_daily s JOIN calendar c ON c.day = s.day
           WHERE s.store_id=1 AND s.product_id=1 AND c.is_weekend=1"""
    ).fetchone()[0]
    assert y[weekday >= 5].sum() == weekend_units
    assert y[weekday >= 5].mean() > y[weekday < 5].mean()
    # late starter: begins at its first observed day and is right-aligned with the axis
    y2 = series[(1, 2)]
    assert y2.shape == (170,)
    assert axis[-len(y2)] == "2024-01-31"
    # an untouched series is dense already and unchanged
    assert series[(2, 4)].shape == (200,)
    # the run itself copes with the gaps: every series is processed on the densified axis
    run_id = pipeline.run(conn, pipeline.RunConfig(**FAST))
    row = conn.execute(
        "SELECT status, series_count FROM forecast_runs WHERE run_id=?", (run_id,)
    ).fetchone()
    assert (row["status"], row["series_count"]) == ("succeeded", 8)
    assert _scalar(conn, "SELECT MIN(target_day) FROM forecasts WHERE run_id=?", run_id) == (
        "2024-07-19"
    )


# ---- backdated runs -----------------------------------------------------------------------------
def test_cutoff_run_uses_only_history_up_to_cutoff(db_3x8):
    cutoff = date(2024, 10, 12)  # 14 days before the last generated day (2024-10-26)
    db_3x8.executemany(
        "INSERT INTO inventory_snapshots VALUES (?,?,?,?,?)",
        [
            (1, 1, "2024-10-12", 5, 3),  # exactly on the cutoff -> used
            (1, 2, "2024-10-05", 50, 0),  # older but the latest <= cutoff -> used
        ],
    )
    db_3x8.commit()
    truncated = _clone(db_3x8)
    truncated.execute("DELETE FROM sales_daily WHERE day > ?", (cutoff.isoformat(),))
    truncated.commit()
    # only the full DB sees a post-cutoff snapshot; it must be ignored by the backdated run
    db_3x8.execute("INSERT INTO inventory_snapshots VALUES (1, 1, '2024-10-20', 999, 999)")
    db_3x8.commit()

    rid_full = pipeline.run(db_3x8, pipeline.RunConfig(cutoff_day=cutoff, **FAST))
    rid_trunc = pipeline.run(truncated, pipeline.RunConfig(**FAST))

    row = db_3x8.execute("SELECT * FROM forecast_runs WHERE run_id=?", (rid_full,)).fetchone()
    assert row["cutoff_day"] == "2024-10-12", "forecast_runs.cutoff_day stores the requested cutoff"
    assert row["status"] == "succeeded" and row["series_count"] == 24
    assert _notes(db_3x8, rid_full)["inventory_missing"] == "22"
    days = [
        r[0]
        for r in db_3x8.execute(
            "SELECT DISTINCT target_day FROM forecasts WHERE run_id=? ORDER BY 1", (rid_full,)
        )
    ]
    assert days[0] == "2024-10-13" and days[-1] == "2024-10-26" and len(days) == 14
    bad = _scalar(
        db_3x8,
        "SELECT COUNT(*) FROM forecasts WHERE run_id=? "
        "AND NOT (0 <= yhat_lower AND yhat_lower <= yhat AND yhat <= yhat_upper)",
        rid_full,
    )
    assert bad == 0

    # identical to a run on a database where the later days never existed
    assert _rows(db_3x8, FORECAST_COLUMNS, rid_full) == _rows(
        truncated, FORECAST_COLUMNS, rid_trunc
    )
    assert _rows(db_3x8, METRIC_COLUMNS, rid_full) == _rows(truncated, METRIC_COLUMNS, rid_trunc)
    assert _rows(db_3x8, ORDER_COLUMNS, rid_full) == _rows(truncated, ORDER_COLUMNS, rid_trunc)
    inv = dict(
        _rows(
            db_3x8,
            "SELECT product_id, on_hand FROM replenishment_orders WHERE run_id=? AND store_id=1",
            rid_full,
        )
    )
    assert inv[1] == 5 and inv[2] == 50 and inv[3] == 0


def test_cutoff_outside_sales_range_rejected_before_run_row(db_2x4):
    with pytest.raises(ValueError):
        pipeline.run(db_2x4, pipeline.RunConfig(cutoff_day=date(2025, 1, 1), **FAST))
    with pytest.raises(ValueError):
        pipeline.run(db_2x4, pipeline.RunConfig(cutoff_day=date(2023, 12, 31), **FAST))
    assert _scalar(db_2x4, "SELECT COUNT(*) FROM forecast_runs") == 0


# ---- model candidates, selection criterion and intervals ----------------------------------------
def test_unknown_model_name_rejected_before_run_row(db_2x4):
    with pytest.raises(ValueError, match="nope") as info:
        pipeline.run(db_2x4, pipeline.RunConfig(models=("nope",), **FAST))
    assert "seasonal_naive" in str(info.value)  # the error lists the valid names
    with pytest.raises(ValueError):
        pipeline.run(db_2x4, pipeline.RunConfig(models=(), **FAST))
    assert _scalar(db_2x4, "SELECT COUNT(*) FROM forecast_runs") == 0


def test_models_restriction_limits_candidates(db_2x4):
    only_sn = _clone(db_2x4)
    run_id = pipeline.run(only_sn, pipeline.RunConfig(models=("seasonal_naive",), **FAST))
    for table in ("backtest_metrics", "forecasts"):
        names = {
            r[0]
            for r in only_sn.execute(
                f"SELECT DISTINCT model_name FROM {table} WHERE run_id=?", (run_id,)
            )
        }
        assert names == {"seasonal_naive"}, table
    n_sel = _scalar(
        only_sn, "SELECT COUNT(*) FROM backtest_metrics WHERE run_id=? AND selected=1", run_id
    )
    assert n_sel == 8
    assert _config(only_sn, run_id)["models"] == ["seasonal_naive"]

    # a pair: croston only joins for intermittent series, moving_average covers the rest
    pair = _clone(db_2x4)
    rid = pipeline.run(pair, pipeline.RunConfig(models=("moving_average", "croston_sba"), **FAST))
    names = {
        r[0]
        for r in pair.execute(
            "SELECT DISTINCT model_name FROM backtest_metrics WHERE run_id=?", (rid,)
        )
    }
    assert "moving_average" in names and names <= {"moving_average", "croston_sba"}

    # a restriction that leaves a series without any candidate fails the run loudly
    with pytest.raises(ValueError, match="no candidate model"):
        pipeline.run(db_2x4, pipeline.RunConfig(models=("croston_sba",), **FAST))
    assert _rows(db_2x4, "SELECT status FROM forecast_runs ORDER BY run_id") == [("failed",)]


@pytest.mark.parametrize("criterion", ["mae", "wape", "mase"])
def test_criterion_selects_the_minimum_of_that_metric(db_2x4, criterion):
    run_id = pipeline.run(db_2x4, pipeline.RunConfig(criterion=criterion, **FAST))
    rows = db_2x4.execute(
        f"SELECT store_id, product_id, model_name, {criterion} AS metric, bias, selected "
        "FROM backtest_metrics WHERE run_id=?",
        (run_id,),
    ).fetchall()
    by_series: dict[tuple[int, int], list] = {}
    for r in rows:
        by_series.setdefault((r["store_id"], r["product_id"]), []).append(r)
    assert len(by_series) == 8
    for key, scores in by_series.items():
        chosen = [r for r in scores if r["selected"]]
        assert len(chosen) == 1, key
        best = min(  # C3 ranking: lowest metric (None last), then |bias|, then name
            scores,
            key=lambda r: (
                r["metric"] is None,
                0.0 if r["metric"] is None else r["metric"],
                abs(r["bias"]),
                r["model_name"],
            ),
        )
        assert chosen[0]["model_name"] == best["model_name"], key
    assert _config(db_2x4, run_id)["criterion"] == criterion


def test_persisted_intervals_come_from_the_backtest(db_2x4):
    # promo_aware=False so the per-series recomputation below mirrors the pipeline exactly
    # (no promotion flags enter the backtest or the fits)
    emp_db, norm_db = _clone(db_2x4), _clone(db_2x4)
    rid_e = pipeline.run(emp_db, pipeline.RunConfig(promo_aware=False, **FAST))  # "empirical"
    rid_n = pipeline.run(
        norm_db, pipeline.RunConfig(promo_aware=False, interval_method="normal", **FAST)
    )
    series, _axis = pipeline._load_series(db_2x4, "2024-07-18")
    model_horizon = max(14, _scalar(db_2x4, "SELECT MAX(lead_time_days) FROM products") + 7)
    sql = (
        "SELECT model_name, yhat, yhat_lower, yhat_upper FROM forecasts "
        "WHERE run_id=? AND store_id=? AND product_id=? ORDER BY target_day"
    )
    for key in [(1, 1), (2, 3)]:
        y = series[key]
        for conn, rid, method in ((emp_db, rid_e, "empirical"), (norm_db, rid_n, "normal")):
            winner, _ = select_model(y, 14, n_folds=2, interval_level=0.8, interval_method=method)
            assert winner.interval is not None
            yhat = make_model(winner.model_name).fit(y).predict(model_horizon)[:14]
            lower, upper = winner.interval.apply(yhat)
            rows = _rows(conn, sql, rid, *key)
            assert [r[0] for r in rows] == [winner.model_name] * 14, (key, method)
            np.testing.assert_allclose([r[1] for r in rows], yhat, err_msg=f"{key} {method}")
            np.testing.assert_allclose([r[2] for r in rows], lower, err_msg=f"{key} {method}")
            np.testing.assert_allclose([r[3] for r in rows], upper, err_msg=f"{key} {method}")
    # the two methods really produce different bands (D11: no single fixed ±z·σ any more)
    bands = (
        "SELECT yhat_lower, yhat_upper FROM forecasts WHERE run_id=? "
        "ORDER BY store_id, product_id, target_day"
    )
    assert _rows(emp_db, bands, rid_e) != _rows(norm_db, bands, rid_n)
    # the normal band is symmetric around yhat wherever the lower bound is not clipped at 0
    asymmetric = _scalar(
        norm_db,
        "SELECT COUNT(*) FROM forecasts WHERE run_id=? AND yhat_lower > 0 "
        "AND ABS((yhat_upper - yhat) - (yhat - yhat_lower)) > 1e-9",
        rid_n,
    )
    assert asymmetric == 0
    for conn, rid in ((emp_db, rid_e), (norm_db, rid_n)):
        assert _config(conn, rid)["interval_method"] == (
            "empirical" if conn is emp_db else "normal"
        )


def test_interval_brackets_forecast_and_widens_with_level(db_2x4):
    narrow_db, wide_db = _clone(db_2x4), _clone(db_2x4)
    rid_n = pipeline.run(narrow_db, pipeline.RunConfig(interval_level=0.5, **FAST))
    rid_w = pipeline.run(wide_db, pipeline.RunConfig(interval_level=0.95, **FAST))
    for conn, rid in ((narrow_db, rid_n), (wide_db, rid_w)):
        bad = _scalar(
            conn,
            "SELECT COUNT(*) FROM forecasts WHERE run_id=? "
            "AND NOT (0 <= yhat_lower AND yhat_lower <= yhat AND yhat <= yhat_upper)",
            rid,
        )
        assert bad == 0
    width = "SELECT AVG(yhat_upper - yhat_lower) FROM forecasts WHERE run_id=?"
    assert _scalar(wide_db, width, rid_w) > _scalar(narrow_db, width, rid_n)
    # the point forecasts do not depend on the interval level
    pts = (
        "SELECT store_id, product_id, target_day, yhat FROM forecasts WHERE run_id=? ORDER BY 1,2,3"
    )
    assert _rows(narrow_db, pts, rid_n) == _rows(wide_db, pts, rid_w)


# ---- promo flags --------------------------------------------------------------------------------
def _master_rows(conn: sqlite3.Connection, n_stores: int, n_products: int) -> None:
    conn.executemany(
        "INSERT INTO stores VALUES (?,?,?,?,?,?)",
        [
            (i, f"S{i}", f"City {i}", "South", "standard", "2020-01-01")
            for i in range(1, n_stores + 1)
        ],
    )
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (j, f"SKU-{j}", f"Item {j}", "Grocery", 10.0, 20.0, 1, 3, None)
            for j in range(1, n_products + 1)
        ],
    )


def test_load_promo_flags_chain_wide_and_store_specific(fresh_db):
    conn = fresh_db
    _master_rows(conn, 2, 3)
    conn.executemany(
        "INSERT INTO promotions VALUES (?,?,?,?,?,?)",
        [
            (1, 1, None, "2024-03-28", "2024-04-03", 0.2),  # chain-wide, straddles the cutoff
            (2, 2, 2, "2024-03-30", "2024-04-02", 0.1),  # store 2 only, straddles the cutoff
            (3, 1, None, "2024-03-01", "2024-03-05", 0.2),  # entirely before the axis
            (4, 2, None, "2024-04-10", "2024-04-12", 0.2),  # entirely after the horizon
            (5, 3, 1, "2024-03-20", "2024-03-23", 0.3),  # clipped at the axis start
        ],
    )
    conn.commit()
    hist = [f"2024-03-{d:02d}" for d in range(22, 32)]  # 10 days ending at the cutoff 03-31
    future = [f"2024-04-{d:02d}" for d in range(1, 6)]  # 5 forecast days
    keys = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]

    flags = pipeline.load_promo_flags(conn, keys, hist, future)

    assert set(flags) == set(keys)
    for key in keys:
        h, f = flags[key]
        assert h.shape == (10,) and f.shape == (5,), key
        assert set(np.unique(np.concatenate([h, f]))) <= {0.0, 1.0}, key
    chain_h, chain_f = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0]
    np.testing.assert_array_equal(flags[(1, 1)][0], chain_h)
    np.testing.assert_array_equal(flags[(1, 1)][1], chain_f)
    np.testing.assert_array_equal(flags[(2, 1)][0], chain_h)  # NULL store applies to every store
    np.testing.assert_array_equal(flags[(2, 1)][1], chain_f)
    np.testing.assert_array_equal(flags[(2, 2)][0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(flags[(2, 2)][1], [1, 1, 0, 0, 0])
    assert not flags[(1, 2)][0].any() and not flags[(1, 2)][1].any()  # other store: no flags
    np.testing.assert_array_equal(flags[(1, 3)][0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert not flags[(2, 3)][0].any() and not flags[(2, 3)][1].any()  # absent series -> zeros
    # a series that starts late takes the tail of the axis-aligned history flags
    np.testing.assert_array_equal(flags[(1, 1)][0][-3:], [1, 1, 1])


def test_run_persists_promo_flag_for_future_promo_days(db_2x4):
    db_2x4.executemany(
        "INSERT INTO promotions VALUES (?,?,?,?,?,?)",
        [
            (1001, 1, None, "2024-07-15", "2024-07-22", 0.2),  # chain-wide, 4 days past the cutoff
            (1002, 2, 2, "2024-07-25", "2024-07-27", 0.1),  # store 2 only, inside the horizon
        ],
    )
    db_2x4.commit()
    run_id = pipeline.run(db_2x4, pipeline.RunConfig(**FAST))

    def flagged(sid: int, pid: int) -> list[str]:
        return [
            r[0]
            for r in db_2x4.execute(
                "SELECT target_day FROM forecasts WHERE run_id=? AND store_id=? AND product_id=? "
                "AND promo_flag = 1 ORDER BY target_day",
                (run_id, sid, pid),
            )
        ]

    assert flagged(1, 1) == ["2024-07-19", "2024-07-20", "2024-07-21", "2024-07-22"]
    assert flagged(2, 1) == ["2024-07-19", "2024-07-20", "2024-07-21", "2024-07-22"]
    assert flagged(2, 2) == ["2024-07-25", "2024-07-26", "2024-07-27"]
    assert flagged(1, 2) == []
    assert int(_notes(db_2x4, run_id)["promo_series"]) >= 3


def test_promo_flag_matches_generated_future_promotions():
    conn = _make_db(3, 8, 300, future_promo_days=14)
    last = _scalar(conn, "SELECT MAX(day) FROM sales_daily")
    assert _scalar(conn, "SELECT COUNT(*) FROM promotions WHERE start_day > ?", last) >= 1
    run_id = pipeline.run(conn, pipeline.RunConfig(**FAST))
    mismatched = _scalar(
        conn,
        """
        SELECT COUNT(*) FROM forecasts f
        WHERE f.run_id = ?
          AND f.promo_flag <> EXISTS (
              SELECT 1 FROM promotions pr
              WHERE pr.product_id = f.product_id
                AND (pr.store_id IS NULL OR pr.store_id = f.store_id)
                AND f.target_day BETWEEN pr.start_day AND pr.end_day)
        """,
        run_id,
    )
    assert mismatched == 0
    assert (
        _scalar(conn, "SELECT COUNT(*) FROM forecasts WHERE run_id=? AND promo_flag=1", run_id) >= 1
    )
    assert int(_notes(conn, run_id)["promo_series"]) >= 1


def _promo_everywhere(conn: sqlite3.Connection) -> None:
    """Give every product a 10-day chain-wide promo in history (>= 7 promo days per series, so
    `has_promo_signal` holds everywhere) with visibly lifted sales on those days, and product 1
    a chain-wide promo inside the forecast horizon (2024-07-21 .. 2024-07-24)."""
    conn.executemany(
        "INSERT INTO promotions VALUES (?,?,?,?,?,?)",
        [(900 + pid, pid, None, "2024-06-01", "2024-06-10", 0.2) for pid in range(1, 5)]
        + [(999, 1, None, "2024-07-21", "2024-07-24", 0.25)],
    )
    conn.execute(
        "UPDATE sales_daily SET units_sold = units_sold * 3 + 5 "
        "WHERE day BETWEEN '2024-06-01' AND '2024-06-10'"
    )
    conn.commit()


def test_promo_aware_false_loads_no_flags_and_no_promo_candidates(db_2x4, monkeypatch):
    _promo_everywhere(db_2x4)
    aware = _clone(db_2x4)
    calls: list[tuple] = []
    original = pipeline.load_promo_flags

    def spy(conn, keys, hist_days, future_days):
        calls.append((len(hist_days), len(future_days)))
        return original(conn, keys, hist_days, future_days)

    monkeypatch.setattr(pipeline, "load_promo_flags", spy)
    run_id = pipeline.run(db_2x4, pipeline.RunConfig(promo_aware=False, **FAST))
    assert calls == []
    assert (
        _scalar(db_2x4, "SELECT COUNT(*) FROM forecasts WHERE run_id=? AND promo_flag=1", run_id)
        == 0
    )
    assert _notes(db_2x4, run_id)["promo_series"] == "0"
    names = "SELECT DISTINCT model_name FROM backtest_metrics WHERE run_id=? ORDER BY 1"
    assert not any(n.startswith("promo_") for (n,) in _rows(db_2x4, names, run_id))
    assert _config(db_2x4, run_id)["promo_aware"] is False

    # the same data with promo_aware=True: flags are loaded once and promo_* models compete
    rid = pipeline.run(aware, pipeline.RunConfig(**FAST))
    assert len(calls) == 1 and calls[0] == (
        200,
        max(14, 7 + _scalar(aware, "SELECT MAX(lead_time_days) FROM products")),
    )
    assert _notes(aware, rid)["promo_series"] == "8"
    competed = {n for (n,) in _rows(aware, names, rid)}
    assert {"promo_moving_average", "promo_holt_winters", "promo_theta"} <= competed


def test_promo_flags_are_forwarded_to_fit_and_predict(db_2x4):
    from demandcast import promo

    _promo_everywhere(db_2x4)
    run_id = pipeline.run(db_2x4, pipeline.RunConfig(models=("promo_moving_average",), **FAST))
    used = {
        n
        for (n,) in _rows(
            db_2x4, "SELECT DISTINCT model_name FROM forecasts WHERE run_id=?", run_id
        )
    }
    assert used == {"promo_moving_average"}
    series, axis = pipeline._load_series(db_2x4, "2024-07-18")
    model_horizon = max(14, _scalar(db_2x4, "SELECT MAX(lead_time_days) FROM products") + 7)
    future_days = [
        (date(2024, 7, 18) + timedelta(days=i + 1)).isoformat() for i in range(model_horizon)
    ]
    flags = pipeline.load_promo_flags(db_2x4, list(series), axis, future_days)
    for key in [(1, 1), (2, 1), (2, 4)]:
        y = series[key]
        hist, fut = flags[key][0][-len(y) :], flags[key][1]
        assert promo.has_promo_signal(hist), key
        expected = (
            make_model("promo_moving_average")
            .fit(y, promo_flags=hist)
            .predict(model_horizon, future_flags=fut)[:14]
        )
        persisted = [
            r[0]
            for r in db_2x4.execute(
                "SELECT yhat FROM forecasts WHERE run_id=? AND store_id=? AND product_id=? "
                "ORDER BY target_day",
                (run_id, *key),
            )
        ]
        np.testing.assert_allclose(persisted, expected, err_msg=str(key))
        if key[1] == 1:  # product 1 is on promotion 2024-07-21..24 inside the horizon
            flagged = [
                r[0]
                for r in db_2x4.execute(
                    "SELECT target_day FROM forecasts WHERE run_id=? AND store_id=? AND product_id=? "
                    "AND promo_flag=1 ORDER BY target_day",
                    (run_id, *key),
                )
            ]
            assert flagged == ["2024-07-21", "2024-07-22", "2024-07-23", "2024-07-24"]
            # the promo days were boosted by _promo_everywhere, so the lift is real and the
            # flag-less base forecast cannot reproduce the persisted promo-day values
            assert promo.estimate_lift(y, hist) > 1.0, key
            blind = make_model("moving_average").fit(y).predict(model_horizon)[:14]
            assert not np.allclose(persisted, blind), key


def test_promo_only_restriction_needs_signal_on_every_series(db_2x4):
    # without promo history there is no promo candidate at all -> the restriction empties the
    # candidate list and the run fails loudly (C3 semantics), recorded as a failed run
    bare = _clone(db_2x4)
    bare.execute("DELETE FROM promotions")
    bare.commit()
    with pytest.raises(ValueError, match="no candidate model"):
        pipeline.run(bare, pipeline.RunConfig(models=("promo_moving_average",), **FAST))
    assert _rows(bare, "SELECT status FROM forecast_runs") == [("failed",)]
    # a base model alongside the promo wrapper always leaves a candidate
    rid = pipeline.run(
        bare, pipeline.RunConfig(models=("promo_moving_average", "moving_average"), **FAST)
    )
    selected = {
        n for (n,) in _rows(bare, "SELECT DISTINCT model_name FROM forecasts WHERE run_id=?", rid)
    }
    assert selected == {"moving_average"}


@pytest.mark.parametrize("name", ["promo_seasonal_naive", "promo_nope", "PROMO_THETA"])
def test_unknown_promo_model_name_rejected_before_run_row(db_2x4, name):
    with pytest.raises(ValueError, match="valid names") as info:
        pipeline.run(db_2x4, pipeline.RunConfig(models=(name,), **FAST))
    assert "promo_moving_average" in str(info.value) and "theta" in str(info.value)
    assert _scalar(db_2x4, "SELECT COUNT(*) FROM forecast_runs") == 0


# ---- ABC classes as of a cutoff -----------------------------------------------------------------
def test_abc_classes_as_of_hand_built_revenue(fresh_db):
    conn = fresh_db
    _master_rows(conn, 1, 6)
    days = [
        "2024-04-01",
        "2024-04-02",
        "2024-06-01",
        "2024-06-02",
        "2024-06-03",
        "2024-06-04",
        "2024-06-05",
        "2024-07-01",
    ]
    conn.executemany(
        "INSERT INTO calendar VALUES (?,?,?,?,?,?,NULL)",
        [(d, date.fromisoformat(d).weekday(), 1, int(d[5:7]), 2024, 0) for d in days],
    )
    conn.executemany(
        "INSERT INTO sales_daily VALUES (1,?,?,1,?,0)",
        [
            (1, "2024-06-01", 50.0),
            (2, "2024-06-02", 25.0),
            (3, "2024-06-03", 15.0),
            (4, "2024-06-04", 6.0),
            (5, "2024-06-05", 4.0),
            (6, "2024-04-01", 1000.0),  # exactly 90 days before the cutoff: outside the window
            (6, "2024-07-01", 1000.0),  # after the cutoff: must not leak into the classes
            (1, "2024-04-02", 1.0),  # inside the window (day > cutoff - 90)
        ],
    )
    conn.commit()
    # cumulative shares at cutoff 2024-06-30: 51/101, 76/101, 91/101 (> 0.90 -> C), ...
    classes = pipeline.abc_classes_as_of(conn, "2024-06-30")
    assert classes == {1: "A", 2: "B", 3: "C", 4: "C", 5: "C"}
    assert 6 not in classes
    # cutoff 2024-06-03, 30-day window (2024-05-04 < day <= 2024-06-03): 50/90, 75/90, 90/90
    # -> the later June rows and the April rows are both outside the window
    assert pipeline.abc_classes_as_of(conn, "2024-06-03", window_days=30) == {
        1: "A",
        2: "B",
        3: "C",
    }
    # a 30-day window drops the April sale: 50/100, 75/100, 90/100 (<= 0.90 -> B)
    assert pipeline.abc_classes_as_of(conn, "2024-06-30", window_days=30) == {
        1: "A",
        2: "B",
        3: "B",
        4: "C",
        5: "C",
    }
    assert pipeline.abc_classes_as_of(conn, "2023-01-01") == {}


def test_service_level_overrides_apply_per_abc_class(db_2x4):
    base_db = _clone(db_2x4)
    rid_base = pipeline.run(base_db, pipeline.RunConfig(**FAST))
    overrides = {"A": 0.99, "C": 0.90}
    rid = pipeline.run(db_2x4, pipeline.RunConfig(service_level_overrides=overrides, **FAST))
    classes = pipeline.abc_classes_as_of(db_2x4, "2024-07-18")  # same window as the run
    assert "C" in classes.values()  # the last-ranked product always closes the share at 1.0
    sql = (
        "SELECT store_id, product_id, service_level, safety_stock FROM replenishment_orders "
        "WHERE run_id=? ORDER BY 1, 2"
    )
    base = {(sid, pid): (lvl, ss) for sid, pid, lvl, ss in base_db.execute(sql, (rid_base,))}
    rows = _rows(db_2x4, sql, rid)
    assert len(rows) == len(base) == 8
    for sid, pid, level, ss in rows:
        expected = overrides.get(classes.get(pid, ""), 0.95)
        assert level == pytest.approx(expected), (sid, pid)
        base_level, base_ss = base[(sid, pid)]
        assert base_level == pytest.approx(0.95)
        # same data, same model, same σ: only z(service_level) moves the safety stock
        if expected == 0.95 or base_ss == 0:
            assert ss == pytest.approx(base_ss), (sid, pid)
        elif expected > 0.95:
            assert ss > base_ss, (sid, pid)
        else:
            assert ss < base_ss, (sid, pid)
    assert _config(db_2x4, rid)["service_level_overrides"] == overrides


# ---- order book: risk, priority, budget ---------------------------------------------------------
BOOK = (
    "SELECT r.store_id, r.product_id, r.order_qty, r.requested_qty, r.stockout_risk, "
    "r.priority, p.unit_cost, r.reason FROM replenishment_orders r "
    "JOIN products p USING (product_id) WHERE r.run_id=? ORDER BY 1, 2"
)


def test_orders_carry_risk_priority_and_requested_qty(db_2x4):
    run_id = pipeline.run(db_2x4, pipeline.RunConfig(**FAST))
    rows = _rows(db_2x4, BOOK, run_id)
    assert len(rows) == 8
    for _sid, _pid, qty, requested, risk, priority, _cost, reason in rows:
        assert requested == qty  # no budget: everything the policy asked for is ordered
        assert risk is not None and 0.0 <= risk <= 1.0
        assert priority is not None and priority >= 0.0
        assert "deferred" not in reason
    assert _notes(db_2x4, run_id)["deferred"] == "0"


def test_order_budget_defers_low_priority_orders(db_2x4):
    _zero_inventory(db_2x4)  # every series is below its reorder point -> 8 candidate orders
    free_db = _clone(db_2x4)
    rid_free = pipeline.run(free_db, pipeline.RunConfig(**FAST))
    free = _rows(free_db, BOOK, rid_free)
    assert len(free) == 8 and all(qty > 0 and req == qty for _, _, qty, req, *_ in free)
    total = sum(qty * cost for _, _, qty, _, _, _, cost, _ in free)
    budget = total / 2

    rid = pipeline.run(db_2x4, pipeline.RunConfig(order_budget=budget, **FAST))
    rows = _rows(db_2x4, BOOK, rid)
    assert len(rows) == 8
    spent = sum(qty * cost for _, _, qty, _, _, _, cost, _ in rows)
    assert 0 < spent <= budget
    deferred = [r for r in rows if "deferred: order budget exhausted" in r[7]]
    approved = [r for r in rows if r not in deferred]
    assert deferred and approved
    assert all(qty == 0 and req > 0 for _, _, qty, req, *_ in deferred)
    assert all(req == qty for _, _, qty, req, *_ in approved)
    assert all(r[7].startswith("inventory position below reorder point") for r in deferred)
    # requested quantities, risks and priorities are the policy's and unaffected by the budget
    assert [(r[0], r[1], r[3], r[4], r[5]) for r in rows] == [
        (r[0], r[1], r[3], r[4], r[5]) for r in free
    ]
    # the approved set is exactly the greedy first-fit allocation over the persisted priorities
    items = [BudgetItem((sid, pid), req, cost, prio) for sid, pid, _, req, _, prio, cost, _ in rows]
    assert {(sid, pid): qty for sid, pid, qty, *_ in rows} == allocate_budget(items, budget)
    notes = _notes(db_2x4, rid)
    assert notes["deferred"] == str(len(deferred)) and notes["orders"] == str(len(approved))
    assert _config(db_2x4, rid)["order_budget"] == pytest.approx(budget)


# ---- run listing and failed runs (D8) -----------------------------------------------------------
def test_failed_run_status_recorded_and_error_reraised(db_2x4, monkeypatch):
    def boom(task):
        raise RuntimeError("boom in worker")

    monkeypatch.setattr(pipeline, "process_series", boom)
    with pytest.raises(RuntimeError, match="boom in worker"):
        pipeline.run(db_2x4, pipeline.RunConfig(**FAST))
    rows = db_2x4.execute("SELECT * FROM forecast_runs ORDER BY run_id").fetchall()
    assert len(rows) == 1, "the run row must be created before the series are processed"
    failed = rows[0]
    assert failed["status"] == "failed"
    assert failed["finished_at"] is not None
    assert "RuntimeError('boom in worker')" in failed["notes"]
    assert _scalar(db_2x4, "SELECT COUNT(*) FROM forecasts") == 0
    assert not db_2x4.in_transaction
    assert pipeline.latest_successful_run(db_2x4) is None

    monkeypatch.undo()  # the connection is reusable: the next run succeeds
    run_id = pipeline.run(db_2x4, pipeline.RunConfig(**FAST))
    assert pipeline.latest_successful_run(db_2x4) == run_id
    assert [r["status"] for r in pipeline.list_runs(db_2x4)] == ["succeeded", "failed"]


def test_list_runs_newest_first_with_limit(db_2x4):
    first = pipeline.run(db_2x4, pipeline.RunConfig(horizon_days=7, n_folds=2, workers=1))
    second = pipeline.run(db_2x4, pipeline.RunConfig(interval_level=0.9, **FAST))
    runs = pipeline.list_runs(db_2x4)
    assert [r["run_id"] for r in runs] == [second, first]
    expected_keys = {
        "run_id",
        "status",
        "started_at",
        "finished_at",
        "cutoff_day",
        "horizon_days",
        "series_count",
        "notes",
        "interval_level",
    }
    assert all(isinstance(r, dict) and expected_keys <= set(r) for r in runs)
    assert runs[0]["interval_level"] == pytest.approx(0.9)
    assert runs[1]["interval_level"] == pytest.approx(0.8)
    assert runs[0]["horizon_days"] == 14 and runs[1]["horizon_days"] == 7
    assert runs[0]["status"] == "succeeded" and runs[0]["series_count"] == 8
    assert runs[0]["cutoff_day"] == "2024-07-18"
    assert _notes(db_2x4, second)["skipped"] == "0"
    assert _notes(db_2x4, second)["inventory_missing"] == "0"
    assert [r["run_id"] for r in pipeline.list_runs(db_2x4, limit=1)] == [second]
    assert len(pipeline.list_runs(db_2x4, limit=0)) == 2

    # a v0.3.0 database is migrated on the fly; its old rows report interval_level NULL
    v1 = db.connect(":memory:")
    v1.executescript(SCHEMA_V1.read_text(encoding="utf-8"))
    v1.execute(
        "INSERT INTO forecast_runs (started_at, cutoff_day, horizon_days, status) "
        "VALUES ('2024-01-01T00:00:00+00:00', '2024-01-01', 7, 'failed')"
    )
    v1.commit()
    old = pipeline.list_runs(v1)
    assert len(old) == 1 and old[0]["status"] == "failed" and old[0]["interval_level"] is None
    assert db.schema_version(v1) == 2


# ---- process pool -------------------------------------------------------------------------------
def test_pool_workers_produce_identical_results(db_3x8):
    """The only test allowed to use workers=2 (24 series > the 8-series pool threshold)."""
    serial_db, pool_db = _clone(db_3x8), _clone(db_3x8)
    rid_1 = pipeline.run(serial_db, pipeline.RunConfig(horizon_days=14, n_folds=2, workers=1))
    rid_2 = pipeline.run(pool_db, pipeline.RunConfig(horizon_days=14, n_folds=2, workers=2))
    assert _notes(pool_db, rid_2)["workers"] == "2"
    assert _rows(serial_db, FORECAST_COLUMNS, rid_1) == _rows(pool_db, FORECAST_COLUMNS, rid_2)
    assert _rows(serial_db, METRIC_COLUMNS, rid_1) == _rows(pool_db, METRIC_COLUMNS, rid_2)
    assert _rows(serial_db, ORDER_COLUMNS, rid_1) == _rows(pool_db, ORDER_COLUMNS, rid_2)
    assert len(_rows(pool_db, FORECAST_COLUMNS, rid_2)) == 24 * 14
