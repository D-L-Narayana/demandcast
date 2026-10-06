"""Realised-accuracy evaluation (demandcast/evaluate.py, contract C7).

Fixture: same construction as tests/test_analytics.py (kept in sync; a shared conftest factory may
replace both) — a 3 x 8 x 300 database at schema v2 with a current run (nothing realised yet) and a
run backdated by 14 days whose horizon is fully covered by re-inserted actuals.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date

import pytest

from demandcast import db, evaluate, pipeline
from demandcast.db import insert_many
from demandcast.simulate import SimConfig, generate

HOLDOUT_DAYS = 14
SALES_COLUMNS = "store_id product_id day units_sold revenue stockout_flag".split()

# ---- schema v2 (plan §3 C5; statements verbatim, applied only when still missing) ----------------
V2_ALTERS = """
ALTER TABLE forecast_runs        ADD COLUMN config_json    TEXT;
ALTER TABLE forecast_runs        ADD COLUMN interval_level REAL;
ALTER TABLE forecast_runs        ADD COLUMN engine_version TEXT;
ALTER TABLE forecasts            ADD COLUMN promo_flag     INTEGER NOT NULL DEFAULT 0;
ALTER TABLE replenishment_orders ADD COLUMN stockout_risk  REAL;
ALTER TABLE replenishment_orders ADD COLUMN priority       REAL;
ALTER TABLE replenishment_orders ADD COLUMN requested_qty  INTEGER;   -- qty before budget allocation (NULL on old rows)
"""
_ALTER_RE = re.compile(r"ALTER TABLE (\w+)\s+ADD COLUMN (\w+)")
V2_SCRIPT = """
CREATE TABLE IF NOT EXISTS forecast_evaluations (
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
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS data_loads (
    load_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    loaded_at       TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    table_name      TEXT    NOT NULL,
    mode            TEXT    NOT NULL CHECK (mode IN ('insert', 'upsert', 'replace')),
    rows_inserted   INTEGER NOT NULL,
    rows_updated    INTEGER NOT NULL DEFAULT 0,
    rows_rejected   INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_forecasts_series_day ON forecasts(store_id, product_id, target_day);
CREATE INDEX IF NOT EXISTS idx_promotions_product   ON promotions(product_id, start_day, end_day);
PRAGMA user_version = 2;
"""


def apply_schema_v2(conn: sqlite3.Connection) -> None:
    """Bring a v1 (baseline) database to the C5 v2 shape; a no-op on a database that already is v2."""
    for ddl in V2_ALTERS.strip().splitlines():
        match = _ALTER_RE.match(ddl)
        assert match is not None, ddl
        table, column = match.groups()
        present = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in present:
            conn.execute(ddl)
    conn.executescript(V2_SCRIPT)
    conn.commit()


@dataclass
class V2Fixture:
    conn: sqlite3.Connection
    current_run: int  # cutoff = last day of data -> nothing realised yet
    backdated_run: int  # cutoff = last day - 14 -> 14 realised days per series
    cutoff: str  # cutoff day of the backdated run (ISO)
    data_end: str  # MAX(sales_daily.day)


def make_v2_db() -> V2Fixture:
    conn = db.connect(":memory:")
    db.init_schema(conn)
    generate(conn, SimConfig(n_stores=3, n_products=8, start=date(2024, 1, 1), days=300, seed=7))
    apply_schema_v2(conn)
    cfg = pipeline.RunConfig(horizon_days=HOLDOUT_DAYS, n_folds=2, workers=1)
    current_run = pipeline.run(conn, cfg)
    data_end = conn.execute("SELECT MAX(day) FROM sales_daily").fetchone()[0]
    cutoff = conn.execute("SELECT DATE(?, ?)", (data_end, f"-{HOLDOUT_DAYS} days")).fetchone()[0]
    held_out = [
        tuple(r)
        for r in conn.execute(
            f"SELECT {', '.join(SALES_COLUMNS)} FROM sales_daily WHERE day > ? ORDER BY 1, 2, 3",
            (cutoff,),
        )
    ]
    conn.execute("DELETE FROM sales_daily WHERE day > ?", (cutoff,))
    conn.commit()
    backdated_run = pipeline.run(conn, cfg)
    insert_many(conn, "sales_daily", SALES_COLUMNS, held_out)
    conn.commit()
    return V2Fixture(conn, current_run, backdated_run, cutoff, data_end)


@pytest.fixture(scope="module")
def v2() -> V2Fixture:
    return make_v2_db()


def _count_evaluations(conn: sqlite3.Connection, run_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM forecast_evaluations WHERE run_id = ?", (run_id,)
    ).fetchone()[0]


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="demandcast")
    sub = parser.add_subparsers(dest="cmd", required=True)
    evaluate.register_cli(sub)
    return parser.parse_args(argv)


# ---- pure maths -----------------------------------------------------------------------------------
def test_series_evaluations_maths_on_hand_made_rows():
    def row(day, yhat, lo, hi, actual, stockout=0, series=(1, 1)):
        return {
            "store_id": series[0],
            "product_id": series[1],
            "target_day": day,
            "model_name": "m",
            "yhat": yhat,
            "yhat_lower": lo,
            "yhat_upper": hi,
            "actual_units": actual,
            "stockout_flag": stockout,
        }

    rows = [
        row("2024-01-01", 10.0, 8.0, 12.0, 12),  # e = -2, inside the interval
        row("2024-01-02", 10.0, 8.0, 12.0, 6, stockout=1),  # e = +4, below the interval
        row("2024-01-03", 5.0, 4.0, 6.0, 5),  # e = 0, inside
        row("2024-01-01", 1.0, 0.0, 2.0, 0, series=(2, 1)),  # a series with zero actuals
    ]
    evals = evaluate.series_evaluations(rows)
    assert [(e.store_id, e.product_id) for e in evals] == [(1, 1), (2, 1)]
    first, zero = evals
    assert isinstance(first, evaluate.SeriesEvaluation)
    assert first.model_name == "m" and first.n_days == 3
    assert first.abs_error_sum == pytest.approx(6.0)
    assert first.actual_sum == pytest.approx(23.0)
    assert first.mae == pytest.approx(2.0)
    assert first.wape == pytest.approx(6 / 23)
    assert first.bias == pytest.approx(2 / 3)  # mean(yhat - y) = (-2 + 4 + 0) / 3
    assert first.coverage == pytest.approx(2 / 3)
    assert first.stockout_days == 1
    assert zero.n_days == 1 and zero.wape is None
    assert zero.mae == pytest.approx(1.0) and zero.coverage == 1.0 and zero.actual_sum == 0.0


# ---- evaluate_run ---------------------------------------------------------------------------------
def test_evaluate_run_matches_forecast_vs_actual_recomputed_in_python(v2: V2Fixture):
    summary = evaluate.evaluate_run(v2.conn, v2.backdated_run)
    assert isinstance(summary, evaluate.EvaluationSummary)
    assert summary.run_id == v2.backdated_run
    assert (summary.cutoff_day, summary.horizon_days) == (v2.cutoff, HOLDOUT_DAYS)
    assert (summary.n_series, summary.n_days_available) == (24, HOLDOUT_DAYS)

    rows = db.run_query(v2.conn, "forecast_vs_actual", {"run_id": v2.backdated_run})
    assert len(rows) == 24 * HOLDOUT_DAYS
    abs_err = sum(r["abs_error"] for r in rows)
    actual = sum(abs(r["actual_units"]) for r in rows)
    assert summary.wape == pytest.approx(abs_err / actual, rel=1e-9)
    assert summary.mae == pytest.approx(abs_err / len(rows), rel=1e-9)
    assert summary.bias == pytest.approx(
        sum(r["yhat"] - r["actual_units"] for r in rows) / len(rows), rel=1e-9, abs=1e-12
    )
    assert summary.coverage == pytest.approx(
        sum(r["in_interval"] for r in rows) / len(rows), rel=1e-9
    )
    assert 0.0 <= summary.coverage <= 1.0
    assert summary.interval_level is not None and 0.0 < summary.interval_level < 1.0
    assert sum(m["n_series"] for m in summary.by_model) == 24
    assert {m["model_name"] for m in summary.by_model} == {r["model_name"] for r in rows}
    assert sum(m["total_days"] for m in summary.by_model) == len(rows)


def test_evaluate_run_persists_idempotently_and_feeds_the_sql_views(v2: V2Fixture):
    rid = v2.backdated_run
    first = evaluate.evaluate_run(v2.conn, rid)
    assert _count_evaluations(v2.conn, rid) == 24
    second = evaluate.evaluate_run(v2.conn, rid)
    assert _count_evaluations(v2.conn, rid) == 24  # DELETE + INSERT per run, not a second copy
    assert (second.mae, second.wape, second.bias, second.coverage) == (
        first.mae,
        first.wape,
        first.bias,
        first.coverage,
    )
    assert second.by_model == db.run_query(v2.conn, "evaluation_summary", {"run_id": rid})
    hist = [h for h in db.run_query(v2.conn, "evaluation_history") if h["run_id"] == rid]
    assert len(hist) == 1
    assert hist[0]["n_series"] == 24
    assert hist[0]["mae"] == pytest.approx(first.mae, abs=5e-4)
    assert hist[0]["wape"] == pytest.approx(first.wape, abs=5e-4)
    assert hist[0]["bias"] == pytest.approx(first.bias, abs=5e-4)
    assert hist[0]["coverage"] == pytest.approx(first.coverage, abs=5e-4)
    row = v2.conn.execute(
        "SELECT * FROM forecast_evaluations WHERE run_id = ? ORDER BY store_id, product_id LIMIT 1",
        (rid,),
    ).fetchone()
    assert row["n_days"] == HOLDOUT_DAYS and 0.0 <= row["coverage"] <= 1.0
    assert row["mae"] == pytest.approx(row["abs_error_sum"] / row["n_days"])
    assert not v2.conn.in_transaction


def test_evaluate_run_defaults_to_the_latest_successful_run(v2: V2Fixture):
    assert pipeline.latest_successful_run(v2.conn) == v2.backdated_run
    summary = evaluate.evaluate_run(v2.conn)
    assert isinstance(summary, evaluate.EvaluationSummary)
    assert summary.run_id == v2.backdated_run


def test_evaluate_run_raises_no_actuals_for_a_run_whose_cutoff_is_the_data_end(v2: V2Fixture):
    assert issubclass(evaluate.NoActualsError, RuntimeError)
    with pytest.raises(evaluate.NoActualsError) as info:
        evaluate.evaluate_run(v2.conn, v2.current_run)
    message = str(info.value)
    assert "--cutoff" in message and v2.data_end in message
    assert _count_evaluations(v2.conn, v2.current_run) == 0


def test_evaluate_run_rejects_missing_runs(fresh_db):
    with pytest.raises(RuntimeError, match="run"):
        evaluate.evaluate_run(fresh_db)  # no successful run at all
    with pytest.raises(RuntimeError, match="999"):
        evaluate.evaluate_run(fresh_db, 999)


# ---- CLI plugin (contract C10 shape) -----------------------------------------------------------------
def test_cli_handler_prints_valid_json(v2: V2Fixture, capsys):
    args = _parse(["evaluate", "--run-id", str(v2.backdated_run), "--json"])
    assert args.creates_db is False and callable(args.handler)
    assert args.handler(v2.conn, args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == v2.backdated_run and payload["n_series"] == 24
    assert payload["cutoff_day"] == v2.cutoff and payload["horizon_days"] == HOLDOUT_DAYS
    assert 0.0 <= payload["coverage"] <= 1.0
    assert payload["interval_level"] is not None
    assert isinstance(payload["by_model"], list) and payload["by_model"]
    assert {"model_name", "n_series", "wape", "coverage"} <= set(payload["by_model"][0])


def test_cli_handler_table_output_defaults_to_latest_run(v2: V2Fixture, capsys):
    args = _parse(["evaluate"])
    assert args.run_id is None
    assert args.handler(v2.conn, args) == 0
    out = capsys.readouterr().out
    assert f"#{v2.backdated_run}" in out
    assert "WAPE" in out and "coverage" in out.lower()
    assert "nominal" in out  # the run's interval level is shown next to the realised coverage


def test_cli_handler_reports_missing_actuals_as_an_error(v2: V2Fixture, capsys):
    args = _parse(["evaluate", "--run-id", str(v2.current_run)])
    assert args.handler(v2.conn, args) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:") and "--cutoff" in captured.err
    assert captured.out == ""
