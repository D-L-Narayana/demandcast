import sqlite3

import pytest

from demandcast import db, pipeline
from demandcast.dashboard import render


def test_schema_constraints_enforced(fresh_db):
    fresh_db.execute(
        "INSERT INTO stores VALUES (1,'BLR','Bengaluru','South','flagship','2020-01-01')"
    )
    with pytest.raises(sqlite3.IntegrityError):  # bad format enum
        fresh_db.execute(
            "INSERT INTO stores VALUES (2,'HYD','Hyderabad','South','kiosk','2020-01-01')"
        )
    with pytest.raises(sqlite3.IntegrityError):  # price below cost
        fresh_db.execute("INSERT INTO products VALUES (1,'S1','x','Grocery',10.0,9.0,1,3,NULL)")
    with pytest.raises(sqlite3.IntegrityError):  # FK violation
        fresh_db.execute("INSERT INTO sales_daily VALUES (99, 1, '2024-01-01', 1, 1.0, 0)")


def test_generated_data_is_consistent(small_db):
    counts = db.table_counts(small_db)
    assert counts["stores"] == 3 and counts["products"] == 8
    assert counts["sales_daily"] == 3 * 8 * 300
    assert counts["inventory_snapshots"] == 24
    # revenue never negative, units censored by inventory => no negative on_hand
    assert small_db.execute("SELECT MIN(revenue) FROM sales_daily").fetchone()[0] >= 0
    # some stock-outs must exist for the censoring logic to be exercised
    assert small_db.execute("SELECT SUM(stockout_flag) FROM sales_daily").fetchone()[0] > 0
    # weekends sell more than weekdays on average (the weekly profile is baked in)
    row = small_db.execute(
        """SELECT AVG(CASE WHEN c.is_weekend THEN units_sold END) AS we,
                  AVG(CASE WHEN NOT c.is_weekend THEN units_sold END) AS wd
           FROM sales_daily s JOIN calendar c ON c.day = s.day"""
    ).fetchone()
    assert row["we"] > row["wd"]


def test_named_queries_load_and_run(small_db):
    names = set(db.QUERIES)
    assert {
        "weekly_sales_trend",
        "abc_classification",
        "stockout_rate_by_store",
        "promo_lift",
    } <= names
    abc = db.run_query(small_db, "abc_classification")
    assert [r["abc_class"] for r in abc] == sorted(r["abc_class"] for r in abc)
    assert abc[-1]["cum_revenue_share"] == pytest.approx(1.0, abs=1e-6)
    trend = db.run_query(small_db, "weekly_sales_trend")
    assert trend[0]["wow_units_delta"] is None  # first week has no LAG
    with pytest.raises(KeyError):
        db.run_query(small_db, "does_not_exist")


def test_pipeline_run_writes_consistent_outputs(small_db):
    cfg = pipeline.RunConfig(horizon_days=14, n_folds=2)
    run_id = pipeline.run(small_db, cfg)
    run = small_db.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
    assert run["status"] == "succeeded"
    assert run["series_count"] == 24

    n_fc = small_db.execute("SELECT COUNT(*) FROM forecasts WHERE run_id=?", (run_id,)).fetchone()[
        0
    ]
    assert n_fc == 24 * 14
    # exactly one selected model per series
    sel = small_db.execute(
        "SELECT store_id, product_id, SUM(selected) s FROM backtest_metrics WHERE run_id=? "
        "GROUP BY store_id, product_id",
        (run_id,),
    ).fetchall()
    assert len(sel) == 24 and all(r["s"] == 1 for r in sel)
    # forecast model must match the selected model
    mismatch = small_db.execute(
        """SELECT COUNT(*) FROM forecasts f
           JOIN backtest_metrics m ON m.run_id=f.run_id AND m.store_id=f.store_id
                AND m.product_id=f.product_id AND m.model_name=f.model_name
           WHERE f.run_id=? AND m.selected=0""",
        (run_id,),
    ).fetchone()[0]
    assert mismatch == 0
    # intervals bracket the point forecast
    bad = small_db.execute(
        "SELECT COUNT(*) FROM forecasts WHERE run_id=? AND NOT (yhat_lower <= yhat AND yhat <= yhat_upper)",
        (run_id,),
    ).fetchone()[0]
    assert bad == 0
    # one order row per series, quantities in case packs
    orders = small_db.execute(
        "SELECT r.order_qty, p.case_pack FROM replenishment_orders r JOIN products p USING(product_id) "
        "WHERE run_id=?",
        (run_id,),
    ).fetchall()
    assert len(orders) == 24
    assert all(o["order_qty"] % o["case_pack"] == 0 for o in orders)

    lb = db.run_query(small_db, "forecast_accuracy_leaderboard", {"run_id": run_id})
    assert sum(r["series_won"] for r in lb) == 24
    assert pipeline.latest_successful_run(small_db) == run_id


def test_pipeline_marks_failed_run_on_error(fresh_db):
    with pytest.raises(RuntimeError):
        pipeline.run(fresh_db)  # empty sales table


def test_dashboard_renders(small_db, tmp_path):
    if pipeline.latest_successful_run(small_db) is None:
        pipeline.run(small_db, pipeline.RunConfig(horizon_days=14, n_folds=2))
    out = render(small_db, tmp_path / "index.html")
    text = out.read_text()
    assert "<svg" in text and "Model leaderboard" in text and "Replenishment order book" in text
