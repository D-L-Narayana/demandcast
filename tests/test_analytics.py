"""Analytics SQL (demandcast/sql/analytics.sql): schema-v2 queries, the ISO-week fix (D4), the
set-based promo_lift rewrite and series_history.on_promo.

Fixture: a 3 x 8 x 300 synthetic database brought to the schema-v2 shape (the DDL from the plan
is applied idempotently, so this is a no-op once db.init_schema itself provides v2), with one
pipeline run on the full data (nothing realised yet) and one run backdated by 14 days.  The
backdated run emulates `run --cutoff` on any pipeline version: the last 14 days of sales are held
out, the pipeline runs, and the held-out rows are re-inserted so actuals exist after the cutoff.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import pytest

from demandcast import db, pipeline
from demandcast.db import insert_many
from demandcast.simulate import SimConfig, build_calendar, generate

HOLDOUT_DAYS = 14
SALES_COLUMNS = "store_id product_id day units_sold revenue stockout_flag".split()
CALENDAR_COLUMNS = "day day_of_week week_of_year month year is_weekend holiday_name".split()

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


@contextmanager
def rolled_back(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run test-local mutations inside a savepoint that is always rolled back."""
    conn.execute("SAVEPOINT test_scope")
    try:
        yield conn
    finally:
        conn.execute("ROLLBACK TO test_scope")
        conn.execute("RELEASE test_scope")


# ---- contract constants (plan §3 C9 + §1.2 invariant 4) -----------------------------------------
BASELINE_COLUMNS: dict[str, list[str]] = {
    "weekly_sales_trend": "category week_start units revenue wow_units_delta wow_pct".split(),
    "abc_classification": "product_id sku name category revenue_90d rev_rank cum_revenue_share abc_class".split(),
    "stockout_rate_by_store": "store_code city region format stockout_pct stockout_days rank_in_region".split(),
    "promo_lift": "sku name category promo_days baseline_units_per_day promo_units_per_day lift_multiplier".split(),
    "forecast_accuracy_leaderboard": "model_name series_won share_pct avg_mae avg_wape avg_mase avg_bias".split(),
    "replenishment_summary": "order_id store_code city sku name category on_hand on_order lead_time_demand safety_stock reorder_point order_qty order_cost expected_arrival reason".split(),
    "series_history": "day units_sold stockout_flag ma7".split(),
    "days_of_cover": "store_code sku name on_hand on_order forecast_7d days_of_cover health".split(),
}
C9_COLUMNS: dict[str, list[str]] = {
    "forecast_vs_actual": "store_id product_id target_day model_name yhat yhat_lower yhat_upper actual_units stockout_flag abs_error in_interval".split(),
    "evaluation_summary": "model_name n_series total_days avg_mae wape avg_bias coverage".split(),
    "evaluation_history": "run_id cutoff_day horizon_days n_series mae wape bias coverage evaluated_at".split(),
    "inventory_health_distribution": "health n_series share_pct".split(),
    "order_cost_by_category": "category n_orders units order_cost n_deferred".split(),
    "forecast_rollup": "level key horizon_units".split(),
    "run_history": "run_id status started_at finished_at cutoff_day horizon_days series_count interval_level notes".split(),
    "stockout_risk_top": "store_code sku name category on_hand on_order stockout_risk priority order_qty expected_arrival".split(),
    "promo_calendar_upcoming": "promo_id sku store_code start_day end_day discount_pct".split(),
    # optional query from the W08 "SHOULD" list
    "forecast_bias_by_category": "category n_series avg_bias avg_wape".split(),
}
EXPECTED_QUERY_NAMES = set(BASELINE_COLUMNS) | set(
    C9_COLUMNS
)  # 8 baseline + 9 contract + 1 optional
# Legitimately empty on the fixture (no evaluation rows persisted; no promo may overlap the window).
MAY_BE_EMPTY = {
    "evaluation_summary",
    "evaluation_history",
    "forecast_bias_by_category",
    "promo_calendar_upcoming",
}
HEALTH_ORDER = ["CRITICAL", "LOW", "OK", "OVERSTOCK", "NO_DEMAND"]
EVAL_COLUMNS = "run_id store_id product_id model_name n_days mae wape bias coverage abs_error_sum actual_sum stockout_days evaluated_at".split()

_PARAM_RE = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")


def query_params(sql: str) -> set[str]:
    return set(_PARAM_RE.findall(sql))


def params_for(fx: V2Fixture, name: str, run_id: int | None = None) -> dict[str, int]:
    values = {"run_id": run_id or fx.backdated_run, "store_id": 1, "product_id": 1}
    return {k: values[k] for k in query_params(db.QUERIES[name])}


# The baseline formulation (correlated EXISTS per sales row) kept verbatim so the set-based
# rewrite can be proven row-identical.
OLD_PROMO_LIFT_SQL = """
WITH flagged AS (
    SELECT s.store_id, s.product_id, s.day, s.units_sold,
           EXISTS (
               SELECT 1 FROM promotions pr
               WHERE pr.product_id = s.product_id
                 AND (pr.store_id IS NULL OR pr.store_id = s.store_id)
                 AND s.day BETWEEN pr.start_day AND pr.end_day
           ) AS on_promo
    FROM sales_daily s
),
agg AS (
    SELECT product_id,
           AVG(CASE WHEN on_promo THEN units_sold END)     AS promo_avg,
           AVG(CASE WHEN NOT on_promo THEN units_sold END) AS base_avg,
           SUM(on_promo)                                   AS promo_days
    FROM flagged
    GROUP BY product_id
    HAVING promo_days > 0
)
SELECT p.sku, p.name, p.category, a.promo_days,
       ROUND(a.base_avg, 2)  AS baseline_units_per_day,
       ROUND(a.promo_avg, 2) AS promo_units_per_day,
       ROUND(a.promo_avg / NULLIF(a.base_avg, 0), 2) AS lift_multiplier
FROM agg a JOIN products p ON p.product_id = a.product_id
ORDER BY lift_multiplier DESC;
"""


# ---- fixture self-check -------------------------------------------------------------------------
def test_apply_schema_v2_is_idempotent(fresh_db):
    apply_schema_v2(fresh_db)
    apply_schema_v2(fresh_db)  # second pass is a no-op: the post-migration situation
    cols = {
        table: {r["name"] for r in fresh_db.execute(f"PRAGMA table_info({table})")}
        for table in ("forecast_runs", "forecasts", "replenishment_orders")
    }
    assert {"config_json", "interval_level", "engine_version"} <= cols["forecast_runs"]
    assert "promo_flag" in cols["forecasts"]
    assert {"stockout_risk", "priority", "requested_qty"} <= cols["replenishment_orders"]
    tables = {
        r["name"] for r in fresh_db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"forecast_evaluations", "data_loads"} <= tables
    assert fresh_db.execute("PRAGMA user_version").fetchone()[0] == 2


# ---- query inventory ----------------------------------------------------------------------------
def test_load_queries_finds_exactly_the_expected_names():
    loaded = db.load_queries()
    assert len(loaded) == 18
    assert set(loaded) == EXPECTED_QUERY_NAMES
    assert set(db.QUERIES) == EXPECTED_QUERY_NAMES


def test_every_query_block_has_a_technique_comment():
    text = (Path(db.__file__).with_name("sql") / "analytics.sql").read_text(encoding="utf-8")
    parts = re.split(r"^--\s*name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    names = parts[1::2]
    assert set(names) == EXPECTED_QUERY_NAMES
    for name, body in zip(names, parts[2::2], strict=True):
        comments = [ln for ln in body.splitlines() if ln.strip().startswith("--")]
        assert comments, f"{name}: no `--` comment line explaining the SQL technique"


@pytest.mark.parametrize("name", sorted(EXPECTED_QUERY_NAMES))
def test_every_named_query_executes_on_a_completed_run(v2: V2Fixture, name: str):
    assert name in db.QUERIES, f"query {name!r} missing from analytics.sql"
    assert query_params(db.QUERIES[name]) <= {"run_id", "store_id", "product_id"}
    rows = db.run_query(v2.conn, name, params_for(v2, name))
    assert isinstance(rows, list)
    if name not in MAY_BE_EMPTY:
        assert rows, f"{name} returned no rows on the fixture"


@pytest.mark.parametrize("name", sorted(C9_COLUMNS))
def test_new_query_columns_match_the_contract(v2: V2Fixture, name: str):
    assert name in db.QUERIES, f"query {name!r} missing from analytics.sql"
    cur = v2.conn.execute(db.QUERIES[name], params_for(v2, name))
    assert [d[0] for d in cur.description] == C9_COLUMNS[name]


@pytest.mark.parametrize("name", sorted(BASELINE_COLUMNS))
def test_baseline_query_columns_are_stable(v2: V2Fixture, name: str):
    cur = v2.conn.execute(db.QUERIES[name], params_for(v2, name))
    columns = [d[0] for d in cur.description]
    if name == "series_history":
        assert columns == [*BASELINE_COLUMNS[name], "on_promo"]
    elif name == "weekly_sales_trend":
        assert columns == BASELINE_COLUMNS[name]
    else:
        assert set(BASELINE_COLUMNS[name]) <= set(columns)


# ---- D4: ISO weeks keyed by their Monday ---------------------------------------------------------
def test_weekly_sales_trend_does_not_split_the_week_at_the_year_boundary(fresh_db):
    conn = fresh_db
    conn.execute("INSERT INTO stores VALUES (1,'BLR','Bengaluru','South','flagship','2020-01-01')")
    conn.execute("INSERT INTO products VALUES (1,'SKU-1','x','Grocery',10.0,12.0,1,3,NULL)")
    start = date(2024, 12, 30)  # Monday of ISO week 2025-W01
    insert_many(conn, "calendar", CALENDAR_COLUMNS, build_calendar(start, 8))
    units = [10, 11, 12, 13, 14, 15, 16, 40]  # Mon 30 Dec .. Sun 5 Jan, then Mon 6 Jan
    rows = [
        (1, 1, (start + timedelta(days=i)).isoformat(), u, 12.0 * u, 0) for i, u in enumerate(units)
    ]
    insert_many(conn, "sales_daily", SALES_COLUMNS, rows)
    conn.commit()

    trend = db.run_query(conn, "weekly_sales_trend")
    assert [r["week_start"] for r in trend] == ["2024-12-30", "2025-01-06"]
    first, second = trend
    assert first["units"] == sum(units[:7])
    assert first["revenue"] == pytest.approx(12.0 * sum(units[:7]))
    assert first["wow_units_delta"] is None and first["wow_pct"] is None
    assert second["units"] == 40
    assert second["wow_units_delta"] == 40 - sum(units[:7])
    assert second["wow_pct"] == pytest.approx(
        round(100.0 * (40 - sum(units[:7])) / sum(units[:7]), 1)
    )
    assert list(first) == BASELINE_COLUMNS["weekly_sales_trend"]


def test_weekly_sales_trend_buckets_are_contiguous_mondays(v2: V2Fixture):
    trend = db.run_query(v2.conn, "weekly_sales_trend")
    by_cat: dict[str, list[dict]] = {}
    for r in trend:
        by_cat.setdefault(r["category"], []).append(r)
    n_cats = v2.conn.execute("SELECT COUNT(DISTINCT category) FROM products").fetchone()[0]
    assert len(by_cat) == n_cats
    for rows in by_cat.values():
        starts = [date.fromisoformat(r["week_start"]) for r in rows]
        assert all(d.weekday() == 0 for d in starts)
        assert all((b - a).days == 7 for a, b in pairwise(starts))
        assert len(rows) == 43  # 300 days from Monday 2024-01-01 = 42 full weeks + 6 days
        assert rows[0]["wow_units_delta"] is None
        for prev, cur in pairwise(rows):
            assert cur["wow_units_delta"] == cur["units"] - prev["units"]
    total = v2.conn.execute("SELECT SUM(units_sold) FROM sales_daily").fetchone()[0]
    assert sum(r["units"] for r in trend) == total


# ---- series_history.on_promo --------------------------------------------------------------------
def test_series_history_flags_promo_days(v2: V2Fixture):
    conn = v2.conn
    promo = conn.execute(
        "SELECT product_id FROM promotions WHERE store_id IS NULL ORDER BY promo_id LIMIT 1"
    ).fetchone()
    assert promo is not None  # the generator makes chain-wide promotions
    pid, sid = promo["product_id"], 1
    rows = db.run_query(conn, "series_history", {"store_id": sid, "product_id": pid})
    assert list(rows[0]) == [*BASELINE_COLUMNS["series_history"], "on_promo"]
    expected: set[str] = set()
    for pr in conn.execute(
        "SELECT start_day, end_day FROM promotions "
        "WHERE product_id = ? AND (store_id IS NULL OR store_id = ?)",
        (pid, sid),
    ):
        d, end = date.fromisoformat(pr["start_day"]), date.fromisoformat(pr["end_day"])
        while d <= end:
            expected.add(d.isoformat())
            d += timedelta(days=1)
    days = {r["day"] for r in rows}
    flagged = {r["day"] for r in rows if r["on_promo"] == 1}
    assert flagged == expected & days
    assert flagged and flagged < days
    assert {r["on_promo"] for r in rows} <= {0, 1}
    assert [r["day"] for r in rows] == sorted(days)


# ---- promo_lift: set-based rewrite must be row-identical to the correlated EXISTS -----------------
def test_promo_lift_rewrite_returns_identical_rows(v2: V2Fixture):
    old = [dict(r) for r in v2.conn.execute(OLD_PROMO_LIFT_SQL)]
    new = db.run_query(v2.conn, "promo_lift")
    assert old and len(new) == len(old)
    assert sorted(new, key=lambda r: r["sku"]) == sorted(old, key=lambda r: r["sku"])
    lifts = [r["lift_multiplier"] for r in new if r["lift_multiplier"] is not None]
    assert lifts == sorted(lifts, reverse=True)
    plan = [r["detail"] for r in v2.conn.execute("EXPLAIN QUERY PLAN " + db.QUERIES["promo_lift"])]
    assert not any("CORRELATED" in step for step in plan), plan


# ---- new queries ----------------------------------------------------------------------------------
def test_forecast_vs_actual_joins_realised_days_only(v2: V2Fixture):
    rows = db.run_query(v2.conn, "forecast_vs_actual", {"run_id": v2.backdated_run})
    assert len(rows) == 24 * HOLDOUT_DAYS
    assert all(v2.cutoff < r["target_day"] <= v2.data_end for r in rows)
    for r in rows:
        assert None not in (r["model_name"], r["yhat"], r["yhat_lower"], r["yhat_upper"])
        assert r["actual_units"] is not None and r["actual_units"] >= 0
        assert 0.0 <= r["yhat_lower"] <= r["yhat"] <= r["yhat_upper"]  # interval brackets yhat
        assert r["abs_error"] == pytest.approx(abs(r["actual_units"] - r["yhat"]))
        assert r["in_interval"] == int(r["yhat_lower"] <= r["actual_units"] <= r["yhat_upper"])
        assert r["stockout_flag"] in (0, 1)
    assert 0 < sum(r["in_interval"] for r in rows) < len(rows)  # a real, non-degenerate interval
    keys = [(r["store_id"], r["product_id"], r["target_day"]) for r in rows]
    assert keys == sorted(keys)
    assert db.run_query(v2.conn, "forecast_vs_actual", {"run_id": v2.current_run}) == []


def test_evaluation_queries_use_day_weighted_aggregates(v2: V2Fixture):
    rid = v2.backdated_run
    stamp = "2024-10-27T00:00:00+00:00"
    rows = [
        # run, store, product, model, n_days, mae, wape, bias, coverage, |e| sum, y sum, stockouts
        (rid, 1, 1, "model_a", 14, 1.0, 0.5, 0.2, 0.8, 14.0, 28.0, 0, stamp),
        (rid, 1, 2, "model_a", 7, 2.0, None, -0.1, 1.0, 14.0, 0.0, 2, stamp),
        (rid, 2, 1, "model_b", 14, 0.5, 0.25, 0.0, 0.5, 7.0, 28.0, 1, stamp),
    ]
    with rolled_back(v2.conn):
        insert_many(v2.conn, "forecast_evaluations", EVAL_COLUMNS, rows)
        summary = db.run_query(v2.conn, "evaluation_summary", {"run_id": rid})
        history = db.run_query(v2.conn, "evaluation_history")
        by_cat = db.run_query(v2.conn, "forecast_bias_by_category", {"run_id": rid})

    assert [r["model_name"] for r in summary] == ["model_a", "model_b"]
    a, b = summary
    assert (a["n_series"], a["total_days"]) == (2, 21)
    assert a["avg_mae"] == pytest.approx(1.5)
    assert a["wape"] == pytest.approx(28.0 / 28.0)  # Σ|e| / Σ|y|, not the mean of (0.5, NULL)
    assert a["avg_bias"] == pytest.approx(0.05)
    assert a["coverage"] == pytest.approx((0.8 * 14 + 1.0 * 7) / 21, abs=5e-4)
    assert (b["n_series"], b["total_days"]) == (1, 14)
    assert (b["avg_mae"], b["wape"], b["avg_bias"], b["coverage"]) == (0.5, 0.25, 0.0, 0.5)

    assert len(history) == 1
    h = history[0]
    assert (h["run_id"], h["cutoff_day"], h["horizon_days"]) == (rid, v2.cutoff, HOLDOUT_DAYS)
    assert h["n_series"] == 3
    assert h["mae"] == pytest.approx(35.0 / 35)
    assert h["wape"] == pytest.approx(35.0 / 56, abs=5e-4)
    assert h["bias"] == pytest.approx((0.2 * 14 - 0.1 * 7) / 35, abs=5e-4)
    assert h["coverage"] == pytest.approx((0.8 * 14 + 1.0 * 7 + 0.5 * 14) / 35, abs=5e-4)
    assert h["evaluated_at"] == stamp

    assert [(r["category"], r["n_series"]) for r in by_cat] == [("Grocery", 2), ("Household", 1)]
    assert by_cat[0]["avg_bias"] == pytest.approx(0.1) and by_cat[0]["avg_wape"] == pytest.approx(
        0.375
    )
    assert by_cat[1]["avg_bias"] == pytest.approx(-0.1) and by_cat[1]["avg_wape"] is None
    # the rollback left nothing behind
    assert db.run_query(v2.conn, "evaluation_history") == []


def test_inventory_health_distribution_matches_days_of_cover(v2: V2Fixture):
    dist = db.run_query(v2.conn, "inventory_health_distribution", {"run_id": v2.backdated_run})
    cover = db.run_query(v2.conn, "days_of_cover", {"run_id": v2.backdated_run})
    expected = Counter(r["health"] for r in cover)
    assert {r["health"]: r["n_series"] for r in dist} == dict(expected)
    assert sum(r["n_series"] for r in dist) == 24
    assert sum(r["share_pct"] for r in dist) == pytest.approx(100.0, abs=0.3)
    assert [r["health"] for r in dist] == [h for h in HEALTH_ORDER if h in expected]
    assert len(dist) >= 2


def test_order_cost_by_category_reconciles_with_the_order_book(v2: V2Fixture):
    conn, rid = v2.conn, v2.backdated_run
    # Untouched run (no order budget): every line has requested_qty == order_qty, so nothing is
    # deferred and the category units equal the requested units.
    untouched = db.run_query(conn, "order_cost_by_category", {"run_id": rid})
    assert untouched and all(r["n_deferred"] == 0 for r in untouched)
    requested = conn.execute(
        "SELECT SUM(requested_qty), SUM(requested_qty IS NULL), SUM(requested_qty <> order_qty) "
        "FROM replenishment_orders WHERE run_id = ?",
        (rid,),
    ).fetchone()
    assert tuple(requested[1:]) == (0, 0)
    assert sum(r["units"] for r in untouched) == requested[0] > 0
    with rolled_back(conn):
        deferred = conn.execute(
            "SELECT r.order_id, p.category FROM replenishment_orders r "
            "JOIN products p ON p.product_id = r.product_id "
            "WHERE r.run_id = ? ORDER BY r.order_id LIMIT 2",
            (rid,),
        ).fetchall()
        for r in deferred:  # emulate two budget-deferred lines
            conn.execute(
                "UPDATE replenishment_orders SET requested_qty = 12, order_qty = 0 WHERE order_id = ?",
                (r["order_id"],),
            )
        result = db.run_query(conn, "order_cost_by_category", {"run_id": rid})
        book = db.run_query(conn, "replenishment_summary", {"run_id": rid})

    by_cat = {r["category"]: r for r in result}
    assert set(by_cat) == {
        r["category"] for r in conn.execute("SELECT DISTINCT category FROM products")
    }
    for cat, n in Counter(r["category"] for r in deferred).items():
        assert by_cat[cat]["n_deferred"] == n
    assert sum(r["n_deferred"] for r in result) == 2
    assert sum(r["units"] for r in result) == sum(b["order_qty"] for b in book)
    assert sum(r["n_orders"] for r in result) == sum(1 for b in book if b["order_qty"] > 0)
    assert sum(r["order_cost"] for r in result) == pytest.approx(
        sum(b["order_cost"] for b in book), abs=0.1
    )
    costs = [r["order_cost"] for r in result]
    assert costs == sorted(costs, reverse=True)


def test_forecast_rollup_levels_add_up_to_the_chain_total(v2: V2Fixture):
    rows = db.run_query(v2.conn, "forecast_rollup", {"run_id": v2.backdated_run})
    total = v2.conn.execute(
        "SELECT SUM(yhat) FROM forecasts WHERE run_id = ?", (v2.backdated_run,)
    ).fetchone()[0]
    chain = [r for r in rows if r["level"] == "chain"]
    regions = [r for r in rows if r["level"] == "region"]
    cats = [r for r in rows if r["level"] == "category"]
    assert len(chain) == 1 and chain[0]["horizon_units"] == pytest.approx(total, abs=0.06)
    assert {r["key"] for r in regions} == {
        r[0] for r in v2.conn.execute("SELECT DISTINCT region FROM stores")
    }
    assert {r["key"] for r in cats} == {
        r[0] for r in v2.conn.execute("SELECT DISTINCT category FROM products")
    }
    assert sum(r["horizon_units"] for r in regions) == pytest.approx(total, abs=0.06 * len(regions))
    assert sum(r["horizon_units"] for r in cats) == pytest.approx(total, abs=0.06 * len(cats))
    assert [r["level"] for r in rows] == ["chain"] + ["region"] * len(regions) + ["category"] * len(
        cats
    )
    assert [r["key"] for r in cats] == sorted(r["key"] for r in cats)


def test_run_history_lists_runs_newest_first(v2: V2Fixture):
    rows = db.run_query(v2.conn, "run_history")
    ids = [r["run_id"] for r in rows]
    assert ids == sorted(ids, reverse=True)
    assert {v2.current_run, v2.backdated_run} <= set(ids)
    by_id = {r["run_id"]: r for r in rows}
    assert by_id[v2.backdated_run]["cutoff_day"] == v2.cutoff
    assert by_id[v2.current_run]["cutoff_day"] == v2.data_end
    assert all(r["status"] == "succeeded" and r["horizon_days"] == HOLDOUT_DAYS for r in rows)
    assert all(r["series_count"] == 24 for r in rows)
    # runs made by the current pipeline record their nominal interval level (C6)
    assert all(r["interval_level"] is not None and 0.0 < r["interval_level"] < 1.0 for r in rows)
    assert all(r["started_at"] and r["finished_at"] and r["notes"] for r in rows)


def test_stockout_risk_top_orders_by_priority_with_nulls_last(v2: V2Fixture):
    conn, rid = v2.conn, v2.backdated_run
    # Rows without a risk score (pre-0.4.0 runs) must sink to the end: blank the run's real
    # scores, give three lines synthetic ones and check they lead in priority order.
    with rolled_back(conn):
        conn.execute(
            "UPDATE replenishment_orders SET priority = NULL, stockout_risk = NULL WHERE run_id = ?",
            (rid,),
        )
        ids = [
            r["order_id"]
            for r in conn.execute(
                "SELECT order_id FROM replenishment_orders WHERE run_id = ? ORDER BY order_id LIMIT 3",
                (rid,),
            )
        ]
        for oid, priority, risk in zip(ids, (5.0, 10.0, 2.5), (0.3, 0.9, 0.1), strict=True):
            conn.execute(
                "UPDATE replenishment_orders SET priority = ?, stockout_risk = ? WHERE order_id = ?",
                (priority, risk, oid),
            )
        rows = db.run_query(conn, "stockout_risk_top", {"run_id": rid})
    assert len(rows) == 10
    assert [r["priority"] for r in rows[:3]] == [10.0, 5.0, 2.5]
    assert [r["stockout_risk"] for r in rows[:3]] == [0.9, 0.3, 0.1]
    assert all(r["priority"] is None for r in rows[3:])
    assert all(r["order_qty"] >= 0 and r["expected_arrival"] > v2.cutoff for r in rows)


def test_stockout_risk_top_ranks_the_real_risk_scores(v2: V2Fixture):
    """Untouched run: every line carries a score (C4/C6) and the top 10 come in priority order."""
    conn, rid = v2.conn, v2.backdated_run
    rows = db.run_query(conn, "stockout_risk_top", {"run_id": rid})
    assert len(rows) == 10
    assert all(r["priority"] is not None and r["stockout_risk"] is not None for r in rows)
    priorities = [r["priority"] for r in rows]
    assert priorities == sorted(priorities, reverse=True)
    assert all(0.0 <= r["stockout_risk"] <= 1.0 for r in rows)
    top = conn.execute(
        "SELECT MAX(priority) FROM replenishment_orders WHERE run_id = ?", (rid,)
    ).fetchone()[0]
    assert priorities[0] == pytest.approx(top, abs=0.005)  # the query rounds to 2 dp
    assert top > 0
    n_null = conn.execute(
        "SELECT COUNT(*) FROM replenishment_orders "
        "WHERE run_id = ? AND (priority IS NULL OR stockout_risk IS NULL OR requested_qty IS NULL)",
        (rid,),
    ).fetchone()[0]
    assert n_null == 0


def test_promo_calendar_upcoming_filters_by_cutoff_and_horizon(v2: V2Fixture):
    conn = v2.conn
    # Backdated run: compare against the overlap rule evaluated in Python on the generated promos.
    window_end = (date.fromisoformat(v2.cutoff) + timedelta(days=HOLDOUT_DAYS)).isoformat()
    expected = {
        r["promo_id"]
        for r in conn.execute("SELECT promo_id, start_day, end_day FROM promotions")
        if r["end_day"] > v2.cutoff and r["start_day"] <= window_end
    }
    rows = db.run_query(conn, "promo_calendar_upcoming", {"run_id": v2.backdated_run})
    assert {r["promo_id"] for r in rows} == expected

    # Current run (cutoff = data end, horizon 14): generated promos are all over by then, so only
    # the hand-made ones below can appear. Window = (data_end, data_end + 14 days].
    end = date.fromisoformat(v2.data_end)
    day = lambda k: (end + timedelta(days=k)).isoformat()  # noqa: E731
    next_id = conn.execute("SELECT MAX(promo_id) + 1 FROM promotions").fetchone()[0]
    promos = [  # promo_id, product_id, store_id, start, end, discount, expected in window?
        (next_id, 1, None, day(4), day(10), 0.2, True),  # fully inside, chain-wide -> 'ALL'
        (next_id + 1, 2, 1, day(14), day(25), 0.1, True),  # starts on the last horizon day
        (next_id + 2, 3, None, day(15), day(20), 0.1, False),  # starts after the horizon
        (next_id + 3, 4, 2, day(-6), day(0), 0.3, False),  # ends on the cutoff: already over
        (next_id + 4, 5, None, day(-6), day(1), 0.3, True),  # still running the day after cutoff
    ]
    with rolled_back(conn):
        insert_many(
            conn,
            "promotions",
            ["promo_id", "product_id", "store_id", "start_day", "end_day", "discount_pct"],
            [p[:6] for p in promos],
        )
        rows = db.run_query(conn, "promo_calendar_upcoming", {"run_id": v2.current_run})
    assert {r["promo_id"] for r in rows} == {p[0] for p in promos if p[6]}
    by_id = {r["promo_id"]: r for r in rows}
    assert by_id[next_id]["store_code"] == "ALL"
    assert by_id[next_id + 1]["store_code"] == "BLR"
    assert (
        by_id[next_id]["sku"]
        == conn.execute("SELECT sku FROM products WHERE product_id = 1").fetchone()[0]
    )
    assert [r["promo_id"] for r in rows] == [
        p[0] for p in sorted(promos, key=lambda p: (p[3], p[0])) if p[6]
    ]
    assert all(r["discount_pct"] > 0 for r in rows)
