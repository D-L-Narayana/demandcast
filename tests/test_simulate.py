"""Generator tests: default-dataset stability fingerprints, holiday coverage (D2), promo-day
revenue (D9) and the opt-in `snapshot_every_days` / `future_promo_days` features.

The fingerprints below were measured on the v0.3.0 generator for
``SimConfig(3, 8, date(2024, 1, 1), 300, 7)`` (the `small_db` fixture).  They guard the
promise that the default synthetic dataset is byte-identical across releases: any change to
the RNG draw order before or inside the sales loop breaks them.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import date, timedelta

import pytest

from demandcast import db
from demandcast.simulate import HOLIDAYS, SimConfig, generate

BASELINE_CFG = SimConfig(3, 8, date(2024, 1, 1), 300, 7)

UNITS_STOCKOUT_SHA256 = "bfe3cd8179160a0b4398445dc0b2ca075db59980105cfa36a2854a04b84b3c42"
PRODUCTS_SHA256 = "4f9a40c021444cb67f5bbbc70d80b2ae18553e0ef4e195046d5dc69506ff1978"
PROMOTIONS_SHA256 = "2eb02167b20c797115a7d8774c2c7408b68d20ca6d8a462159e0cf6f6fdab2b7"
SNAPSHOTS_SHA256 = "23b98593134eb100ae7095219936b50eda38c9a3cdf339fda66a1d512b1a1a2b"
# Revenue: the v0.3.0 generator priced every promo day with a hard-coded 20 % discount (D9).
# Since 0.4.0 promo-day revenue uses the real discount_pct of the active promotion(s), so the
# revenue fingerprint legitimately changed from the v0.3.0 value to the current one.  Units,
# stock-out flags, master data, promotions and snapshots are untouched (same RNG draw order);
# non-promo-day revenue is still exactly units_sold * unit_price (asserted below).
REVENUE_SHA256_V030 = "3b35c3fa149ee23f62b28d0dd59d6e836ce9aa22bf86187b469eaf88f5a4fb3a"
REVENUE_SHA256 = "02b1b0c9b505d5cb9155928023369cb9492511b009464021a8a17fea96b7f61e"

SQL_UNITS = (
    "SELECT store_id, product_id, day, units_sold, stockout_flag FROM sales_daily ORDER BY 1,2,3"
)
SQL_REVENUE = "SELECT store_id, product_id, day, revenue FROM sales_daily ORDER BY 1,2,3"
SQL_SNAPSHOTS = (
    "SELECT store_id, product_id, snapshot_day, on_hand, on_order "
    "FROM inventory_snapshots ORDER BY 1,2,3"
)
SQL_PRODUCTS = "SELECT * FROM products ORDER BY product_id"
SQL_PROMOTIONS = "SELECT * FROM promotions ORDER BY promo_id"


def fingerprint(conn: sqlite3.Connection, sql: str) -> str:
    """SHA-256 over the result rows, each rendered as '|'.join(str(v)) + newline (utf-8)."""
    h = hashlib.sha256()
    for row in conn.execute(sql):
        h.update(("|".join(str(v) for v in row) + "\n").encode("utf-8"))
    return h.hexdigest()


def make_db(cfg: SimConfig) -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.init_schema(conn)
    generate(conn, cfg)
    return conn


@pytest.fixture(scope="module")
def baseline_db() -> sqlite3.Connection:
    """Private copy of the small fixture dataset (never mutated by pipeline tests)."""
    return make_db(BASELINE_CFG)


# ---- dataset stability (invariant #2) ------------------------------------------------------


def test_units_and_stockout_fingerprint_is_stable(baseline_db):
    assert fingerprint(baseline_db, SQL_UNITS) == UNITS_STOCKOUT_SHA256


def test_master_data_fingerprints_are_stable(baseline_db):
    assert fingerprint(baseline_db, SQL_PRODUCTS) == PRODUCTS_SHA256
    assert fingerprint(baseline_db, SQL_PROMOTIONS) == PROMOTIONS_SHA256
    assert fingerprint(baseline_db, SQL_SNAPSHOTS) == SNAPSHOTS_SHA256


def test_revenue_fingerprint_is_stable(baseline_db):
    fp = fingerprint(baseline_db, SQL_REVENUE)
    assert fp == REVENUE_SHA256
    assert fp != REVENUE_SHA256_V030  # D9: promo days are no longer priced at a flat 20 % off


def test_non_promo_day_revenue_is_units_times_price(baseline_db):
    rows = baseline_db.execute(
        """
        SELECT s.units_sold, s.revenue, p.unit_price
        FROM sales_daily s JOIN products p ON p.product_id = s.product_id
        WHERE NOT EXISTS (
            SELECT 1 FROM promotions pr
            WHERE pr.product_id = s.product_id
              AND (pr.store_id IS NULL OR pr.store_id = s.store_id)
              AND s.day BETWEEN pr.start_day AND pr.end_day)
        """
    ).fetchall()
    assert len(rows) > 5000  # most days are not on promotion
    off = [r for r in rows if abs(r["revenue"] - r["units_sold"] * r["unit_price"]) > 0.005 + 1e-6]
    assert off == []


# ---- D9: promo-day revenue uses the promotion's discount --------------------------------------


def test_promo_day_revenue_uses_promotion_discount(baseline_db):
    """revenue = round(units * price * (1 - d), 2) with d = largest active discount_pct."""
    rows = baseline_db.execute(
        """
        SELECT s.units_sold, s.revenue, p.unit_price, MAX(pr.discount_pct) AS disc
        FROM sales_daily s
        JOIN products p ON p.product_id = s.product_id
        JOIN promotions pr ON pr.product_id = s.product_id
             AND (pr.store_id IS NULL OR pr.store_id = s.store_id)
             AND s.day BETWEEN pr.start_day AND pr.end_day
        GROUP BY s.store_id, s.product_id, s.day
        """
    ).fetchall()
    assert len(rows) > 100
    assert {r["disc"] for r in rows} - {0.2}, "fixture must contain discounts other than 20 %"
    off = [
        (r["units_sold"], r["revenue"], r["unit_price"], r["disc"])
        for r in rows
        if abs(r["revenue"] - r["units_sold"] * r["unit_price"] * (1 - r["disc"])) > 0.005 + 1e-6
    ]
    assert off == []


# ---- D2: holidays for every calendar year in the range ----------------------------------------


def test_holidays_cover_every_calendar_year():
    conn = make_db(SimConfig(1, 1, date(2023, 6, 1), 730, 1))  # 2023-06-01 .. 2025-05-30
    years = {
        r[0]
        for r in conn.execute("SELECT DISTINCT year FROM calendar WHERE holiday_name IS NOT NULL")
    }
    assert years == {2023, 2024, 2025}
    names = {r[0]: r[1] for r in conn.execute("SELECT day, holiday_name FROM calendar")}
    for year in (2023, 2024, 2025):
        for m, d, name, _mult, _span in HOLIDAYS:
            iso = date(year, m, d).isoformat()
            if iso in names:  # anchor inside the generated range -> must carry its name
                assert names[iso] == name, (iso, names[iso])


# ---- SimConfig additions: snapshot cadence and future promotions -------------------------------


def test_simconfig_new_fields_default_off():
    cfg = SimConfig()
    assert cfg.snapshot_every_days is None
    assert cfg.future_promo_days == 0


@pytest.mark.parametrize(
    "kwargs", [{"snapshot_every_days": 0}, {"snapshot_every_days": -3}, {"future_promo_days": -1}]
)
def test_simconfig_rejects_invalid_new_fields(kwargs):
    with pytest.raises(ValueError):
        SimConfig(2, 3, date(2024, 1, 1), 30, 1, **kwargs)


def test_snapshot_every_days_adds_cadenced_snapshots_and_keeps_final_day():
    start, days, k = date(2024, 1, 1), 120, 7
    plain = make_db(SimConfig(2, 3, start, days, 3))
    cadenced = make_db(SimConfig(2, 3, start, days, 3, snapshot_every_days=k))
    # snapshot on day i whenever (days - 1 - i) % k == 0  ->  i = 119, 112, ..., 0  (18 days)
    expected_days = sorted(
        (start + timedelta(days=i)).isoformat() for i in range(days) if (days - 1 - i) % k == 0
    )
    assert len(expected_days) == 18
    got_days = [
        r[0]
        for r in cadenced.execute(
            "SELECT DISTINCT snapshot_day FROM inventory_snapshots ORDER BY 1"
        )
    ]
    assert got_days == expected_days
    assert cadenced.execute("SELECT COUNT(*) FROM inventory_snapshots").fetchone()[0] == 6 * 18
    last_day = (start + timedelta(days=days - 1)).isoformat()
    final_plain = [tuple(r) for r in plain.execute(SQL_SNAPSHOTS)]
    final_cadenced = [
        tuple(r)
        for r in cadenced.execute(
            "SELECT store_id, product_id, snapshot_day, on_hand, on_order FROM inventory_snapshots "
            "WHERE snapshot_day = ? ORDER BY 1,2,3",
            (last_day,),
        )
    ]
    assert final_cadenced == final_plain
    # the extra snapshots change nothing else
    assert fingerprint(cadenced, SQL_UNITS) == fingerprint(plain, SQL_UNITS)
    assert fingerprint(cadenced, SQL_REVENUE) == fingerprint(plain, SQL_REVENUE)
    assert fingerprint(cadenced, SQL_PROMOTIONS) == fingerprint(plain, SQL_PROMOTIONS)


def test_future_promos_start_after_calendar_and_leave_history_untouched(baseline_db):
    start, days, horizon = date(2024, 1, 1), 300, 28
    conn = make_db(SimConfig(3, 8, start, days, 7, future_promo_days=horizon))
    last_day = (start + timedelta(days=days - 1)).isoformat()  # 2024-10-26
    window_end = (start + timedelta(days=days - 1 + horizon)).isoformat()  # 2024-11-23
    # everything drawn before/inside the sales loop is untouched
    assert fingerprint(conn, SQL_UNITS) == UNITS_STOCKOUT_SHA256
    assert fingerprint(conn, SQL_REVENUE) == fingerprint(baseline_db, SQL_REVENUE)
    assert fingerprint(conn, SQL_SNAPSHOTS) == SNAPSHOTS_SHA256
    n_hist = baseline_db.execute("SELECT COUNT(*) FROM promotions").fetchone()[0]
    assert n_hist == 26
    hist_sql = f"SELECT * FROM promotions WHERE promo_id <= {n_hist} ORDER BY promo_id"
    assert fingerprint(conn, hist_sql) == PROMOTIONS_SHA256
    future = conn.execute(
        "SELECT * FROM promotions WHERE promo_id > ? ORDER BY promo_id", (n_hist,)
    ).fetchall()
    assert len(future) >= 1
    assert [r["promo_id"] for r in future] == list(range(n_hist + 1, n_hist + 1 + len(future)))
    assert all(r["start_day"] > last_day for r in future)
    assert all(r["start_day"] <= r["end_day"] <= window_end for r in future)
    assert all(0 < r["discount_pct"] < 1 for r in future)
    assert all(1 <= r["product_id"] <= 8 for r in future)
    assert all(r["store_id"] is None or 1 <= r["store_id"] <= 3 for r in future)
    # the calendar itself is not extended; the future promotions are the only post-calendar rows
    assert conn.execute("SELECT MAX(day) FROM calendar").fetchone()[0] == last_day
    n_after = conn.execute(
        "SELECT COUNT(*) FROM promotions WHERE start_day > ?", (last_day,)
    ).fetchone()[0]
    assert n_after == len(future)
