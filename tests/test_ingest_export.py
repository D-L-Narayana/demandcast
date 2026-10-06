"""CSV ingest (demandcast.ingest) and exports (demandcast.export) — contract C8."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from demandcast import db, export, ingest, pipeline
from demandcast.simulate import SimConfig, generate

REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_DIR = REPO_ROOT / "examples" / "mini"
MINI_CFG = SimConfig(2, 3, date(2024, 1, 1), 120, 3)

# Schema-v2 provenance table (C5 DDL, verbatim).  `IF NOT EXISTS` makes this a no-op once
# db.init_schema() creates the table itself.
DATA_LOADS_DDL = """
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
"""

STORES_HEADER = ["store_id", "store_code", "city", "region", "format", "opened_on"]
GOOD_STORES = [
    [1, "BLR", "Bengaluru", "South", "flagship", "2020-01-15"],
    [2, "HYD", "Hyderabad", "South", "standard", "2021-06-01"],
    [3, "CHN", "Chennai", "South", "express", "2019-03-10"],
]
# appended after the 3 good rows -> file lines 5..9 ("row N" = line N of the file, header = 1)
BAD_STORES = [
    (
        ["x", "BAD", "Nowhere", "South", "standard", "2020-01-01"],
        "row 5: store_id must be an integer",
    ),
    (
        [4, "KIO", "Kochi", "South", "kiosk", "2020-01-01"],
        "row 6: format must be one of flagship, standard, express",
    ),
    ([5, "PUN", "Pune", "West", "standard", "01/02/2020"], "row 7: opened_on must be an ISO date"),
    (
        [2, "DUP", "Duplicate", "South", "standard", "2020-01-01"],
        "row 8: duplicate key (store_id=2) first seen at row 3",
    ),
    ([6, "", "Mumbai", "West", "flagship", "2020-01-01"], "row 9: store_code is required"),
]
PRODUCTS_HEADER = [
    "product_id", "sku", "name", "category", "unit_cost", "unit_price",
    "case_pack", "lead_time_days", "shelf_life_days",
]  # fmt: skip
GOOD_PRODUCTS = [
    [1, "SKU-GRO-0001", "Rice 5kg", "Grocery", 100.0, 160.0, 6, 4, 21],
    [2, "SKU-HOU-0002", "Detergent", "Household", 50.5, 95.0, 12, 7, ""],
    [3, "SKU-TOY-0003", "Puzzle", "Toys", 80.0, 170.0, 1, 14, ""],
]
SALES_HEADER = ["store_id", "product_id", "day", "units_sold", "stockout_flag"]
GOOD_SALES = [[1, 1, f"2024-01-{d:02d}", d, d % 2] for d in range(1, 15)]  # 14 days, Jan 1-14


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def table_rows(conn: sqlite3.Connection, sql: str, params=()) -> list[tuple]:
    return [tuple(r) for r in conn.execute(sql, params)]


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


@pytest.fixture
def ingest_db() -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.executescript(DATA_LOADS_DDL)
    return conn


@pytest.fixture
def master_db(ingest_db, tmp_path) -> sqlite3.Connection:
    """ingest_db with the 3 good stores and 3 good products loaded."""
    stores = write_csv(tmp_path / "stores.csv", STORES_HEADER, GOOD_STORES)
    products = write_csv(tmp_path / "products.csv", PRODUCTS_HEADER, GOOD_PRODUCTS)
    assert ingest.load_csv(ingest_db, "stores", stores).rejected == 0
    assert ingest.load_csv(ingest_db, "products", products).rejected == 0
    return ingest_db


@pytest.fixture(scope="module")
def run_db() -> tuple[sqlite3.Connection, int]:
    """Small generated dataset with one completed pipeline run (never mutated by tests)."""
    conn = db.connect(":memory:")
    db.init_schema(conn)
    generate(conn, MINI_CFG)
    run_id = pipeline.run(conn, pipeline.RunConfig(horizon_days=14, n_folds=2, workers=1))
    return conn, run_id


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="demandcast")
    sub = p.add_subparsers(dest="cmd", required=True)
    ingest.register_cli(sub)
    export.register_cli(sub)
    return p


# ---- load_csv: validation, rejection report, modes ------------------------------------------


def test_load_csv_rejects_bad_rows_with_reasons_and_loads_good_ones(ingest_db, tmp_path):
    path = write_csv(
        tmp_path / "stores.csv", STORES_HEADER, [*GOOD_STORES, *(r for r, _ in BAD_STORES)]
    )
    report = ingest.load_csv(ingest_db, "stores", path)
    assert isinstance(report, ingest.LoadReport)
    assert (report.table, report.inserted, report.updated, report.rejected) == ("stores", 3, 0, 5)
    assert len(report.errors) == 5
    for msg, (_, expected_prefix) in zip(report.errors, BAD_STORES, strict=True):
        assert msg.startswith(expected_prefix), (msg, expected_prefix)
    assert table_rows(ingest_db, "SELECT store_id, city FROM stores ORDER BY 1") == [
        (1, "Bengaluru"),
        (2, "Hyderabad"),
        (3, "Chennai"),
    ]
    # provenance row
    loads = table_rows(
        ingest_db,
        "SELECT table_name, mode, rows_inserted, rows_updated, rows_rejected FROM data_loads",
    )
    assert loads == [("stores", "upsert", 3, 0, 5)]
    assert ingest_db.execute("SELECT source FROM data_loads").fetchone()[0] == str(path)
    assert not ingest_db.in_transaction  # the load commits its own work


def test_rejection_report_keeps_first_20_messages_but_counts_all(ingest_db, tmp_path):
    rows = [[i, f"S{i}", "City", "Region", "kiosk", "2020-01-01"] for i in range(1, 26)]
    report = ingest.load_csv(
        ingest_db, "stores", write_csv(tmp_path / "s.csv", STORES_HEADER, rows)
    )
    assert report.rejected == 25
    assert report.inserted == 0
    assert len(report.errors) == 20
    assert report.errors[0].startswith("row 2: format must be one of")


def test_strict_mode_validates_whole_file_then_raises_and_loads_nothing(ingest_db, tmp_path):
    path = write_csv(
        tmp_path / "stores.csv", STORES_HEADER, [*GOOD_STORES, *(r for r, _ in BAD_STORES)]
    )
    with pytest.raises(ValueError) as exc_info:
        ingest.load_csv(ingest_db, "stores", path, strict=True)
    report = exc_info.value.report  # type: ignore[attr-defined]
    assert report.rejected == 5  # whole file validated
    assert len(report.errors) == 5
    assert report.errors[-1].startswith("row 9: store_code is required")
    assert ingest_db.execute("SELECT COUNT(*) FROM stores").fetchone()[0] == 0
    assert ingest_db.execute("SELECT COUNT(*) FROM data_loads").fetchone()[0] == 0
    assert not ingest_db.in_transaction
    # a clean file loads fine in strict mode
    clean = write_csv(tmp_path / "clean.csv", STORES_HEADER, GOOD_STORES)
    assert ingest.load_csv(ingest_db, "stores", clean, strict=True).inserted == 3


def test_dry_run_validates_without_writing(ingest_db, tmp_path):
    path = write_csv(
        tmp_path / "stores.csv", STORES_HEADER, [*GOOD_STORES, *(r for r, _ in BAD_STORES)]
    )
    report = ingest.load_csv(ingest_db, "stores", path, dry_run=True)
    assert report.dry_run is True
    assert (report.inserted, report.updated, report.rejected) == (3, 0, 5)
    assert ingest_db.execute("SELECT COUNT(*) FROM stores").fetchone()[0] == 0
    assert ingest_db.execute("SELECT COUNT(*) FROM data_loads").fetchone()[0] == 0
    assert not ingest_db.in_transaction


def test_dry_run_sales_reports_but_does_not_fill_calendar(master_db, tmp_path):
    path = write_csv(tmp_path / "sales.csv", SALES_HEADER, GOOD_SALES)
    report = ingest.load_csv(master_db, "sales_daily", path, dry_run=True)
    assert (report.inserted, report.rejected, report.calendar_days_added) == (14, 0, 14)
    assert master_db.execute("SELECT COUNT(*) FROM calendar").fetchone()[0] == 0
    assert master_db.execute("SELECT COUNT(*) FROM sales_daily").fetchone()[0] == 0
    n_loads = master_db.execute(
        "SELECT COUNT(*) FROM data_loads WHERE table_name = 'sales_daily'"
    ).fetchone()[0]
    assert n_loads == 0
    assert not master_db.in_transaction


def test_products_validation(ingest_db, tmp_path):
    rows = [
        *GOOD_PRODUCTS,
        [4, "SKU-4", "Price below cost", "Grocery", 100.0, 90.0, 1, 4, ""],
        [5, "SKU-5", "Zero cost", "Grocery", 0, 10.0, 1, 4, ""],
        [6, "SKU-6", "Bad case pack", "Grocery", 10.0, 20.0, 0, 4, ""],
        [7, "SKU-7", "Negative lead time", "Grocery", 10.0, 20.0, 1, -1, ""],
        [8, "SKU-8", "Zero shelf life", "Grocery", 10.0, 20.0, 1, 4, 0],
        [9, "", "Missing sku", "Grocery", 10.0, 20.0, 1, 4, ""],
        [10, "SKU-10", "Not a number", "Grocery", "ten", 20.0, 1, 4, ""],
    ]
    report = ingest.load_csv(
        ingest_db, "products", write_csv(tmp_path / "p.csv", PRODUCTS_HEADER, rows)
    )
    assert (report.inserted, report.rejected) == (3, 7)
    reasons = "\n".join(report.errors)
    assert "row 5: unit_price must be >= unit_cost" in reasons
    assert "row 6: unit_cost must be > 0" in reasons
    assert "row 7: case_pack must be >= 1" in reasons
    assert "row 8: lead_time_days must be >= 0" in reasons
    assert "row 9: shelf_life_days must be > 0" in reasons
    assert "row 10: sku is required" in reasons
    assert "row 11: unit_cost must be a number" in reasons
    got = table_rows(
        ingest_db, "SELECT product_id, case_pack, shelf_life_days FROM products ORDER BY 1"
    )
    assert got == [(1, 6, 21), (2, 12, None), (3, 1, None)]


def test_sales_validation_revenue_default_and_calendar_autofill(master_db, tmp_path):
    bad = [
        [99, 1, "2024-01-01", 2, 0],  # row 16
        [1, 99, "2024-01-01", 2, 0],  # row 17
        [1, 1, "2024-13-01", 2, 0],  # row 18
        [1, 1, "2024-01-15", -2, 0],  # row 19
        [1, 1, "2024-01-16", 2, 2],  # row 20
        [1, 1, "2024-01-17", "", 0],  # row 21
        [1, 1, "2024-01-01", 7, 0],  # row 22 duplicate of row 2
    ]
    path = write_csv(tmp_path / "sales.csv", SALES_HEADER, [*GOOD_SALES, *bad])
    report = ingest.load_csv(master_db, "sales_daily", path)
    assert (report.inserted, report.updated, report.rejected) == (14, 0, 7)
    assert report.errors == [
        "row 16: unknown store_id 99",
        "row 17: unknown product_id 99",
        "row 18: day must be an ISO date YYYY-MM-DD (got '2024-13-01')",
        "row 19: units_sold must be >= 0 (got -2)",
        "row 20: stockout_flag must be 0 or 1 (got '2')",
        "row 21: units_sold is required",
        "row 22: duplicate key (store_id=1, product_id=1, day=2024-01-01) first seen at row 2",
    ]
    # revenue defaults to units_sold x unit_price (product 1 sells at 160.0)
    rows = table_rows(
        master_db, "SELECT day, units_sold, revenue, stockout_flag FROM sales_daily ORDER BY day"
    )
    assert len(rows) == 14
    assert all(rev == pytest.approx(units * 160.0) for _, units, rev, _ in rows)
    assert rows[4] == ("2024-01-05", 5, 800.0, 1)
    # calendar auto-filled for exactly the loaded (valid) day range, weekends/weekdays correct
    cal = table_rows(
        master_db,
        "SELECT day, day_of_week, week_of_year, month, year, is_weekend, holiday_name FROM calendar ORDER BY day",
    )
    assert [c[0] for c in cal] == [f"2024-01-{d:02d}" for d in range(1, 15)]
    assert cal[0] == ("2024-01-01", 0, 1, 1, 2024, 0, None)  # Monday, ISO week 1
    assert [c[5] for c in cal] == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    assert [c[2] for c in cal] == [1] * 7 + [2] * 7
    assert report.calendar_days_added == 14
    assert table_rows(
        master_db,
        "SELECT table_name, rows_inserted, rows_rejected FROM data_loads ORDER BY load_id",
    ) == [
        ("stores", 3, 0),
        ("products", 3, 0),
        ("sales_daily", 14, 7),
    ]


def test_sales_optional_revenue_column_is_used_when_present(master_db, tmp_path):
    header = [*SALES_HEADER, "revenue"]
    rows = [[1, 1, "2024-02-01", 2, 0, 250.5], [1, 1, "2024-02-02", 1, 0, -5.0]]
    report = ingest.load_csv(master_db, "sales_daily", write_csv(tmp_path / "s.csv", header, rows))
    assert (report.inserted, report.rejected) == (1, 1)
    assert report.errors == ["row 3: revenue must be >= 0 (got -5.0)"]
    assert table_rows(master_db, "SELECT revenue FROM sales_daily") == [(250.5,)]


def test_promotions_validation(master_db, tmp_path):
    header = ["promo_id", "product_id", "store_id", "start_day", "end_day", "discount_pct"]
    rows = [
        [1, 1, "", "2024-01-05", "2024-01-10", 0.2],  # chain-wide, ok
        [2, 2, 3, "2024-01-05", "2024-01-05", 0.1],  # one-day store promo, ok
        [3, 1, "", "2024-01-05", "2024-01-10", 1.0],  # row 4
        [4, 1, "", "2024-01-05", "2024-01-10", 0],  # row 5
        [5, 1, "", "2024-01-10", "2024-01-05", 0.2],  # row 6
        [6, 42, "", "2024-01-05", "2024-01-10", 0.2],  # row 7
        [7, 1, 42, "2024-01-05", "2024-01-10", 0.2],  # row 8
    ]
    report = ingest.load_csv(master_db, "promotions", write_csv(tmp_path / "pr.csv", header, rows))
    assert (report.inserted, report.rejected) == (2, 5)
    assert report.errors == [
        "row 4: discount_pct must be > 0 and < 1 (got 1.0)",
        "row 5: discount_pct must be > 0 and < 1 (got 0.0)",
        "row 6: end_day must be >= start_day (got 2024-01-05 < 2024-01-10)",
        "row 7: unknown product_id 42",
        "row 8: unknown store_id 42",
    ]
    assert table_rows(master_db, "SELECT promo_id, store_id FROM promotions ORDER BY 1") == [
        (1, None),
        (2, 3),
    ]


def test_inventory_validation_and_on_order_default(master_db, tmp_path):
    header = ["store_id", "product_id", "snapshot_day", "on_hand"]
    rows = [[1, 1, "2024-01-14", 40], [2, 1, "2024-01-14", -1], [1, 3, "2024/01/14", 4]]
    report = ingest.load_csv(
        master_db, "inventory_snapshots", write_csv(tmp_path / "i.csv", header, rows)
    )
    assert (report.inserted, report.rejected) == (1, 2)
    assert report.errors == [
        "row 3: on_hand must be >= 0 (got -1)",
        "row 4: snapshot_day must be an ISO date YYYY-MM-DD (got '2024/01/14')",
    ]
    assert table_rows(
        master_db,
        "SELECT store_id, product_id, snapshot_day, on_hand, on_order FROM inventory_snapshots",
    ) == [(1, 1, "2024-01-14", 40, 0)]


def test_calendar_csv_derives_missing_fields_and_checks_given_ones(ingest_db, tmp_path):
    header = ["day", "day_of_week", "holiday_name"]
    rows = [
        ["2024-01-26", "", "Republic Day"],  # derived fields
        ["2024-01-27", 5, ""],  # Saturday, consistent
        ["2024-01-28", 3, ""],  # row 4: Sunday is 6, not 3
        ["2024-02-30", "", ""],  # row 5: not a date
    ]
    report = ingest.load_csv(ingest_db, "calendar", write_csv(tmp_path / "c.csv", header, rows))
    assert (report.inserted, report.rejected) == (2, 2)
    assert report.errors == [
        "row 4: day_of_week does not match day 2024-01-28 (expected 6, got 3)",
        "row 5: day must be an ISO date YYYY-MM-DD (got '2024-02-30')",
    ]
    assert table_rows(
        ingest_db,
        "SELECT day, day_of_week, week_of_year, month, year, is_weekend, holiday_name FROM calendar ORDER BY day",
    ) == [("2024-01-26", 4, 4, 1, 2024, 0, "Republic Day"), ("2024-01-27", 5, 4, 1, 2024, 1, None)]


def test_ensure_calendar_fills_gaps_only(ingest_db):
    assert ingest.ensure_calendar(ingest_db, "2023-12-30", "2024-01-05") == 7
    assert ingest.ensure_calendar(ingest_db, "2023-12-30", "2024-01-05") == 0  # idempotent
    assert ingest.ensure_calendar(ingest_db, "2024-01-04", "2024-01-08") == 3  # 6th, 7th, 8th
    rows = table_rows(
        ingest_db,
        "SELECT day, day_of_week, week_of_year, month, year, is_weekend, holiday_name FROM calendar ORDER BY day",
    )
    assert [r[0] for r in rows] == [
        "2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02", "2024-01-03",
        "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08",
    ]  # fmt: skip
    assert rows[0] == ("2023-12-30", 5, 52, 12, 2023, 1, None)  # Saturday, ISO week 52
    assert rows[1] == ("2023-12-31", 6, 52, 12, 2023, 1, None)
    assert rows[2] == ("2024-01-01", 0, 1, 1, 2024, 0, None)
    assert rows[7] == ("2024-01-06", 5, 1, 1, 2024, 1, None)
    assert rows[9] == ("2024-01-08", 0, 2, 1, 2024, 0, None)
    assert all(r[6] is None for r in rows)
    assert not ingest_db.in_transaction
    with pytest.raises(ValueError):
        ingest.ensure_calendar(ingest_db, "2024-01-08", "2024-01-01")
    with pytest.raises(ValueError):
        ingest.ensure_calendar(ingest_db, "2024-1-8", "2024-01-09")


def test_upsert_updates_existing_rows_and_inserts_new_ones(master_db, tmp_path):
    rows = [
        [1, "BLR", "Bengaluru", "South", "flagship", "2020-01-15"],  # unchanged
        [2, "HYD", "Secunderabad", "South", "standard", "2021-06-01"],  # city changed
        [4, "MUM", "Mumbai", "West", "flagship", "2018-11-20"],  # new
    ]
    report = ingest.load_csv(
        master_db, "stores", write_csv(tmp_path / "s2.csv", STORES_HEADER, rows)
    )
    assert (report.inserted, report.updated, report.rejected) == (1, 2, 0)
    assert report.mode == "upsert"
    assert table_rows(master_db, "SELECT store_id, city FROM stores ORDER BY 1") == [
        (1, "Bengaluru"),
        (2, "Secunderabad"),
        (3, "Chennai"),
        (4, "Mumbai"),
    ]
    assert table_rows(
        master_db,
        "SELECT rows_inserted, rows_updated FROM data_loads WHERE table_name='stores' ORDER BY load_id",
    ) == [(3, 0), (1, 2)]


def test_insert_mode_rejects_existing_keys(master_db, tmp_path):
    rows = [
        [2, "HYD", "Elsewhere", "South", "standard", "2021-06-01"],  # exists -> rejected
        [5, "DEL", "New Delhi", "North", "flagship", "2017-05-05"],  # new
    ]
    path = write_csv(tmp_path / "s3.csv", STORES_HEADER, rows)
    report = ingest.load_csv(master_db, "stores", path, mode="insert")
    assert (report.inserted, report.updated, report.rejected) == (1, 0, 1)
    assert report.errors == [
        "row 2: key (store_id=2) already exists (use mode 'upsert' or 'replace')"
    ]
    assert table_rows(master_db, "SELECT city FROM stores WHERE store_id = 2") == [("Hyderabad",)]
    assert master_db.execute("SELECT COUNT(*) FROM stores").fetchone()[0] == 4
    with pytest.raises(ValueError):
        ingest.load_csv(master_db, "stores", path, mode="insert", strict=True)
    assert master_db.execute("SELECT COUNT(*) FROM stores").fetchone()[0] == 4


def test_replace_mode_deletes_then_inserts_and_keeps_referencing_rows(master_db, tmp_path):
    sales = write_csv(tmp_path / "sales.csv", SALES_HEADER, GOOD_SALES)  # reference store 1
    assert ingest.load_csv(master_db, "sales_daily", sales).inserted == 14
    rows = [
        [1, "BLR", "Bengaluru Central", "South", "flagship", "2020-01-15"],  # replaced
        [9, "KOL", "Kolkata", "East", "standard", "2022-02-02"],  # new
    ]
    report = ingest.load_csv(
        master_db, "stores", write_csv(tmp_path / "s4.csv", STORES_HEADER, rows), mode="replace"
    )
    assert (report.inserted, report.updated, report.rejected) == (1, 1, 0)
    assert report.mode == "replace"
    assert table_rows(master_db, "SELECT store_id, city FROM stores ORDER BY 1") == [
        (1, "Bengaluru Central"),
        (2, "Hyderabad"),
        (3, "Chennai"),
        (9, "Kolkata"),
    ]
    assert (
        master_db.execute("SELECT COUNT(*) FROM sales_daily WHERE store_id = 1").fetchone()[0] == 14
    )
    assert master_db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_unique_columns_are_checked_before_writing(master_db, tmp_path):
    rows = [
        [7, "BLR", "Clash with store 1", "South", "standard", "2020-01-01"],  # row 2
        [8, "NEW", "Fresh", "South", "standard", "2020-01-01"],  # row 3 ok
        [9, "NEW", "In-file duplicate code", "South", "standard", "2020-01-01"],  # row 4
    ]
    report = ingest.load_csv(
        master_db, "stores", write_csv(tmp_path / "u.csv", STORES_HEADER, rows)
    )
    assert (report.inserted, report.rejected) == (1, 2)
    assert report.errors == [
        "row 2: store_code 'BLR' is already used by store_id 1",
        "row 4: duplicate store_code 'NEW' first seen at row 3",
    ]
    assert master_db.execute("SELECT COUNT(*) FROM stores").fetchone()[0] == 4


def test_bom_crlf_header_case_and_unknown_columns_are_tolerated(ingest_db, tmp_path, caplog):
    text = (
        "﻿STORE_ID, Store_Code ,city,region,format,opened_on,notes\r\n"
        "1,BLR,Bengaluru,South,flagship,2020-01-15,ignored\r\n"
        "2,HYD,Hyderabad,South,standard,2021-06-01,\r\n"
    )
    path = tmp_path / "stores_excel.csv"
    path.write_bytes(text.encode("utf-8"))
    with caplog.at_level(logging.WARNING, logger="demandcast"):
        report = ingest.load_csv(ingest_db, "stores", path)
    assert (report.inserted, report.rejected) == (2, 0)
    assert any(
        "unknown column" in r.getMessage() and "notes" in r.getMessage() for r in caplog.records
    )
    assert table_rows(ingest_db, "SELECT store_id, store_code FROM stores ORDER BY 1") == [
        (1, "BLR"),
        (2, "HYD"),
    ]


def test_missing_required_column_is_a_file_level_error(ingest_db, tmp_path):
    path = write_csv(
        tmp_path / "s.csv", ["store_id", "store_code", "city"], [[1, "BLR", "Bengaluru"]]
    )
    with pytest.raises(ValueError, match="missing required column"):
        ingest.load_csv(ingest_db, "stores", path)
    assert ingest_db.execute("SELECT COUNT(*) FROM stores").fetchone()[0] == 0


def test_unknown_table_or_mode_raise(ingest_db, tmp_path):
    path = write_csv(tmp_path / "s.csv", STORES_HEADER, GOOD_STORES)
    with pytest.raises(ValueError, match="forecast_runs"):
        ingest.load_csv(ingest_db, "forecast_runs", path)
    with pytest.raises(ValueError, match="mode"):
        ingest.load_csv(ingest_db, "stores", path, mode="merge")
    assert ingest.LOADABLE_TABLES == (
        "stores",
        "products",
        "calendar",
        "promotions",
        "sales_daily",
        "inventory_snapshots",
    )


def test_progress_logging_every_n_rows(master_db, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(ingest, "PROGRESS_EVERY_ROWS", 5)
    path = write_csv(tmp_path / "sales.csv", SALES_HEADER, GOOD_SALES)
    with caplog.at_level(logging.INFO, logger="demandcast"):
        ingest.load_csv(master_db, "sales_daily", path)
    progress = [r.getMessage() for r in caplog.records if "rows read" in r.getMessage()]
    assert len(progress) == 2  # after 5 and 10 of 14 rows
    assert "sales_daily" in progress[0]


# ---- load_dataset / export_dataset round trip -----------------------------------------------


def test_export_dataset_then_load_dataset_round_trips(ingest_db, tmp_path):
    src = db.connect(":memory:")
    db.init_schema(src)
    src_counts = generate(src, MINI_CFG)
    out_dir = tmp_path / "dataset"
    written = export.export_dataset(src, out_dir)
    assert written == src_counts
    for table in ingest.LOADABLE_TABLES:
        path = out_dir / f"{table}.csv"
        assert path.is_file()
        with path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        assert header == [r[1] for r in src.execute(f"PRAGMA table_info({table})")]

    reports = ingest.load_dataset(ingest_db, out_dir)
    assert [r.table for r in reports] == list(ingest.LOADABLE_TABLES)  # FK order
    assert all(r.rejected == 0 and r.updated == 0 for r in reports)
    assert {r.table: r.inserted for r in reports} == src_counts
    for table in ingest.LOADABLE_TABLES:
        assert ingest_db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == src_counts[table]
    agg = "SELECT SUM(units_sold), SUM(revenue), SUM(stockout_flag) FROM sales_daily"
    a, b = src.execute(agg).fetchone(), ingest_db.execute(agg).fetchone()
    assert a[0] == b[0]
    assert a[2] == b[2]
    assert a[1] == pytest.approx(b[1], abs=1e-6)
    sql = "SELECT * FROM sales_daily ORDER BY store_id, product_id, day"
    assert table_rows(src, sql) == table_rows(ingest_db, sql)
    for table in ("stores", "products", "promotions", "calendar", "inventory_snapshots"):
        order = ", ".join([r[1] for r in src.execute(f"PRAGMA table_info({table})")][:3])
        q = f"SELECT * FROM {table} ORDER BY {order}"
        assert table_rows(src, q) == table_rows(ingest_db, q)
    assert ingest_db.execute("SELECT COUNT(*) FROM data_loads").fetchone()[0] == 6
    # a second load in upsert mode is a no-op update
    again = ingest.load_dataset(ingest_db, out_dir)
    assert all(r.inserted == 0 and r.updated == src_counts[r.table] for r in again)
    with pytest.raises(ValueError):
        ingest.load_dataset(ingest_db, tmp_path / "empty-dir-does-not-exist")


# ---- run exports ------------------------------------------------------------------------------

ORDER_COLUMNS = {
    "order_id", "store_code", "city", "sku", "name", "category", "on_hand", "on_order",
    "lead_time_demand", "safety_stock", "reorder_point", "order_qty", "order_cost",
    "expected_arrival", "reason",
}  # fmt: skip


def test_export_orders_csv_header_and_json(run_db, tmp_path):
    conn, run_id = run_db
    out = export.export_orders(conn, run_id, tmp_path / "orders.csv")
    assert out == tmp_path / "orders.csv"
    assert out.is_file()
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    risk_cols = {"stockout_risk", "priority", "requested_qty"}
    present = {r[1] for r in conn.execute("PRAGMA table_info(replenishment_orders)")}
    expected = ORDER_COLUMNS | (risk_cols & present)  # query columns may grow; these must stay
    assert expected <= set(rows[0].keys())
    assert all(int(r["order_qty"]) >= 0 for r in rows)
    # json variant: same keys, typed values
    out_json = export.export_orders(conn, run_id, tmp_path / "orders.json", fmt="json")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 6
    assert expected <= set(data[0].keys())
    assert isinstance(data[0]["order_qty"], int)
    # run_id None -> latest successful run
    out2 = export.export_orders(conn, None, tmp_path / "orders2.csv")
    assert out2.read_bytes() == out.read_bytes()


def test_export_forecasts_and_metrics(run_db, tmp_path):
    conn, run_id = run_db
    fc = export.export_forecasts(conn, run_id, tmp_path / "fc.csv")
    assert fc.is_file()
    with fc.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 * 14
    assert {
        "run_id",
        "store_id",
        "product_id",
        "target_day",
        "model_name",
        "yhat",
        "yhat_lower",
        "yhat_upper",
        "store_code",
        "sku",
    } <= set(rows[0])
    assert all(float(r["yhat_lower"]) <= float(r["yhat"]) <= float(r["yhat_upper"]) for r in rows)
    assert rows[0]["target_day"] == "2024-04-30"  # data ends 2024-04-29

    mt = export.export_metrics(conn, run_id, tmp_path / "metrics.json", fmt="json")
    data = json.loads(mt.read_text(encoding="utf-8"))
    assert len(data) >= 6 * 3
    assert {
        "store_id",
        "product_id",
        "model_name",
        "folds",
        "mae",
        "wape",
        "bias",
        "mase",
        "selected",
        "store_code",
        "sku",
    } <= set(data[0])
    assert sum(d["selected"] for d in data) == 6


def test_export_errors(run_db, tmp_path):
    conn, run_id = run_db
    with pytest.raises(ValueError, match="999"):
        export.export_orders(conn, 999, tmp_path / "x.csv")
    with pytest.raises(ValueError, match="format"):
        export.export_orders(conn, run_id, tmp_path / "x.xml", fmt="xml")
    empty = db.connect(":memory:")
    db.init_schema(empty)
    with pytest.raises(ValueError, match="run"):
        export.export_forecasts(empty, None, tmp_path / "x.csv")


# ---- examples/mini -----------------------------------------------------------------------------


def test_examples_mini_matches_generator_and_is_small(tmp_path):
    src = db.connect(":memory:")
    db.init_schema(src)
    generate(src, MINI_CFG)
    export.export_dataset(src, tmp_path)
    total = 0
    for table in ingest.LOADABLE_TABLES:
        committed = MINI_DIR / f"{table}.csv"
        assert committed.is_file(), committed
        # cell-exact comparison (not raw bytes) so a line-ending normalisation by git cannot
        # masquerade as a generator change
        assert read_rows(committed) == read_rows(tmp_path / f"{table}.csv"), table
        total += committed.stat().st_size
    assert total <= 100 * 1024
    assert (MINI_DIR.parent / "README.md").is_file()
    assert "load" in (MINI_DIR.parent / "README.md").read_text(encoding="utf-8")


def test_examples_mini_loads_and_pipeline_runs(ingest_db):
    reports = ingest.load_dataset(ingest_db, MINI_DIR)
    assert {r.table: (r.inserted, r.rejected) for r in reports} == {
        "stores": (2, 0),
        "products": (3, 0),
        "calendar": (120, 0),
        "promotions": (9, 0),
        "sales_daily": (720, 0),
        "inventory_snapshots": (6, 0),
    }
    run_id = pipeline.run(ingest_db, pipeline.RunConfig(horizon_days=14, n_folds=2, workers=1))
    run = ingest_db.execute(
        "SELECT status, series_count FROM forecast_runs WHERE run_id=?", (run_id,)
    ).fetchone()
    assert (run["status"], run["series_count"]) == ("succeeded", 6)
    assert (
        ingest_db.execute(
            "SELECT COUNT(*) FROM replenishment_orders WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        == 6
    )


# ---- CLI handlers through register_cli ---------------------------------------------------------


def test_cli_load_handler_reports_json_and_exit_codes(ingest_db, tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_csv(data_dir / "stores.csv", STORES_HEADER, GOOD_STORES)
    write_csv(data_dir / "products.csv", PRODUCTS_HEADER, GOOD_PRODUCTS)
    write_csv(data_dir / "sales_daily.csv", SALES_HEADER, GOOD_SALES)
    parser = make_parser()
    args = parser.parse_args(["load", "--dir", str(data_dir)])
    assert args.creates_db is True
    assert args.mode == "upsert"
    rc = args.handler(ingest_db, args)
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert [r["table"] for r in payload["loads"]] == ["stores", "products", "sales_daily"]
    assert payload["loads"][2]["inserted"] == 14
    assert payload["loads"][2]["calendar_days_added"] == 14
    assert ingest_db.execute("SELECT COUNT(*) FROM data_loads").fetchone()[0] == 3

    # explicit per-table flags, strict mode with bad rows -> report printed, exit 1, nothing loaded
    bad = write_csv(
        tmp_path / "bad_inv.csv",
        ["store_id", "product_id", "snapshot_day", "on_hand"],
        [[1, 1, "2024-01-14", -4]],
    )
    args = parser.parse_args(["load", "--inventory", str(bad), "--strict"])
    rc = args.handler(ingest_db, args)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    assert payload["loads"][0]["table"] == "inventory_snapshots"
    assert payload["loads"][0]["errors"] == ["row 2: on_hand must be >= 0 (got -4)"]
    assert ingest_db.execute("SELECT COUNT(*) FROM inventory_snapshots").fetchone()[0] == 0

    # nothing to load -> exit 1
    args = parser.parse_args(["load"])
    rc = args.handler(ingest_db, args)
    assert rc == 1
    # dry run flag is honoured
    args = parser.parse_args(
        ["load", "--sales", str(data_dir / "sales_daily.csv"), "--dry-run", "--mode", "replace"]
    )
    rc = args.handler(ingest_db, args)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["dry_run"] is True
    assert payload["loads"][0]["mode"] == "replace"
    assert ingest_db.execute("SELECT COUNT(*) FROM data_loads").fetchone()[0] == 3  # unchanged


def test_cli_export_handler(run_db, tmp_path, capsys):
    conn, run_id = run_db
    parser = make_parser()
    args = parser.parse_args(
        ["export", "orders", "--out", str(tmp_path / "o.json"), "--format", "json"]
    )
    assert args.creates_db is False
    rc = args.handler(conn, args)
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["export"] == "orders"
    assert payload["run_id"] == run_id
    assert payload["rows"] == 6
    assert Path(payload["path"]).is_file()
    assert len(json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))) == 6

    args = parser.parse_args(["export", "dataset", "--out", str(tmp_path / "ds")])
    rc = args.handler(conn, args)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["export"] == "dataset"
    assert payload["rows"]["sales_daily"] == 720
    assert sorted(p.name for p in (tmp_path / "ds").iterdir()) == sorted(
        f"{t}.csv" for t in ingest.LOADABLE_TABLES
    )

    args = parser.parse_args(
        ["export", "forecasts", "--out", str(tmp_path / "fc.csv"), "--run-id", str(run_id)]
    )
    assert args.handler(conn, args) == 0
    assert (tmp_path / "fc.csv").is_file()

    args = parser.parse_args(
        ["export", "dataset", "--out", str(tmp_path / "ds2"), "--format", "json"]
    )
    with pytest.raises(ValueError, match="CSV only"):
        args.handler(conn, args)
    assert not (tmp_path / "ds2").exists()


# ---- host-review repair: spreadsheet-safe reporting CSV ----------------------------------------
#
# Strings that a spreadsheet would evaluate as a formula enter the database through the REAL
# ingest path (CSV upserts of stores/products) and reach the reporting exports.  The reporting
# CSVs must be inert by default (`'` + text), JSON and the dataset export must stay raw.

HOSTILE_STORES = [
    # store_id, store_code, city, region, format, opened_on
    [1, "BLR", "=1+1", "South", "flagship", "2017-03-26"],
    [2, "@x", "+1", "South", "standard", "2022-10-19"],
]
HOSTILE_PRODUCTS = [
    # product_id, sku, name, category, unit_cost, unit_price, case_pack, lead_time_days, shelf_life
    [1, "SKU-GRO-0001", "-1", "Grocery", 15.53, 27.34, 6, 2, 21],
    [2, "SKU-HOU-0002", "\t=1+1", " =1+1", 100.99, 195.67, 1, 5, ""],
    [3, "​=1+1", "ok, with comma\nsecond line", "Beauty", 60.06, 133.83, 6, 12, 365],
]
MULTILINE_NAME = "ok, with comma\nsecond line"
ZWSP_SKU = "​=1+1"  # zero-width space + formula: survives ingest, invisible in a spreadsheet


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_raw(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def hostile_run_db(tmp_path_factory) -> tuple[sqlite3.Connection, int]:
    """examples/mini, then hostile master-data strings upserted through ``ingest.load_csv``
    (the real bring-your-own-data path), then one pipeline run.  Read-only for the tests."""
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.executescript(DATA_LOADS_DDL)
    ingest.load_dataset(conn, MINI_DIR)
    tmp = tmp_path_factory.mktemp("hostile")
    stores = ingest.load_csv(
        conn, "stores", write_csv(tmp / "stores.csv", STORES_HEADER, HOSTILE_STORES)
    )
    products = ingest.load_csv(
        conn, "products", write_csv(tmp / "products.csv", PRODUCTS_HEADER, HOSTILE_PRODUCTS)
    )
    assert (stores.updated, stores.rejected, products.updated, products.rejected) == (2, 0, 3, 0)
    run_id = pipeline.run(conn, pipeline.RunConfig(horizon_days=14, n_folds=2, workers=1))
    return conn, run_id


def test_hostile_strings_reach_the_database_through_ingest(hostile_run_db):
    conn, _ = hostile_run_db
    assert table_rows(conn, "SELECT store_id, store_code, city FROM stores ORDER BY 1") == [
        (1, "BLR", "=1+1"),
        (2, "@x", "+1"),
    ]
    # ingest strips surrounding whitespace (the tab / space variants arrive as bare formulas)
    # but keeps invisible format characters such as the zero-width space
    assert table_rows(conn, "SELECT product_id, sku, name, category FROM products ORDER BY 1") == [
        (1, "SKU-GRO-0001", "-1", "Grocery"),
        (2, "SKU-HOU-0002", "=1+1", "=1+1"),
        (3, ZWSP_SKU, MULTILINE_NAME, "Beauty"),
    ]


def test_write_primitive_is_safe_by_default_host_reproduction(tmp_path):
    """The host's reproduction: ``_write(['sku'], [{'sku': '=1+1'}], …, 'csv')``."""
    path = export._write(["sku"], [{"sku": "=1+1"}], tmp_path / "x.csv", "csv")
    assert path.read_text(encoding="utf-8") == "sku\n'=1+1\n"
    raw = export._write(["sku"], [{"sku": "=1+1"}], tmp_path / "y.csv", "csv", cells="raw")
    assert raw.read_text(encoding="utf-8") == "sku\n=1+1\n"


def test_export_orders_csv_is_spreadsheet_safe_by_default(hostile_run_db, tmp_path):
    conn, run_id = hostile_run_db
    out = export.export_orders(conn, run_id, tmp_path / "orders.csv")
    rows = read_csv_rows(out)
    assert len(rows) == 6
    # every hostile text cell is written as "'" + text (visible or invisible lead-in)
    assert sorted(r["city"] for r in rows) == ["'+1"] * 3 + ["'=1+1"] * 3
    assert sorted({r["store_code"] for r in rows}) == ["'@x", "BLR"]
    assert sorted({r["sku"] for r in rows}) == ["'" + ZWSP_SKU, "SKU-GRO-0001", "SKU-HOU-0002"]
    assert sorted({r["name"] for r in rows}) == ["'-1", "'=1+1", MULTILINE_NAME]
    assert sorted({r["category"] for r in rows}) == ["'=1+1", "Beauty", "Grocery"]
    # the comma/newline cell is quoted in the file and round-trips unchanged
    assert f'"{MULTILINE_NAME}"' in read_raw(out)
    # numbers are never prefixed and stay parseable; dates start with a digit and are untouched
    numeric = ("on_hand", "on_order", "lead_time_demand", "safety_stock", "reorder_point")
    numeric += ("order_qty", "order_cost", "stockout_risk", "priority", "requested_qty")
    for r in rows:
        for col in numeric:
            assert not r[col].startswith("'"), (col, r[col])
            float(r[col])
        assert len(r["expected_arrival"]) == 10
        assert r["expected_arrival"][:4].isdigit()
        assert r["reason"].startswith("inventory position")
    # the database itself is never modified by the export
    assert table_rows(conn, "SELECT city FROM stores ORDER BY store_id") == [("=1+1",), ("+1",)]
    assert table_rows(conn, "SELECT sku FROM products WHERE product_id = 3") == [(ZWSP_SKU,)]


def test_export_metrics_and_forecasts_keep_negative_numbers_bare(hostile_run_db, tmp_path):
    conn, run_id = hostile_run_db
    n_negative = conn.execute(
        "SELECT COUNT(*) FROM backtest_metrics WHERE run_id = ? AND bias < 0", (run_id,)
    ).fetchone()[0]
    assert n_negative > 0  # the fixture run really has negative-bias series
    metrics = read_csv_rows(export.export_metrics(conn, run_id, tmp_path / "metrics.csv"))
    negatives = [r["bias"] for r in metrics if r["bias"].startswith("-")]
    assert len(negatives) == n_negative
    assert all(float(v) < 0 for v in negatives)
    assert not any(r["bias"].startswith("'") for r in metrics)
    assert sorted({r["sku"] for r in metrics}) == ["'" + ZWSP_SKU, "SKU-GRO-0001", "SKU-HOU-0002"]
    assert sorted({r["store_code"] for r in metrics}) == ["'@x", "BLR"]

    forecasts = read_csv_rows(export.export_forecasts(conn, run_id, tmp_path / "fc.csv"))
    assert len(forecasts) == 6 * 14
    assert sorted({r["sku"] for r in forecasts}) == ["'" + ZWSP_SKU, "SKU-GRO-0001", "SKU-HOU-0002"]
    assert sorted({r["store_code"] for r in forecasts}) == ["'@x", "BLR"]
    assert all(float(r["yhat"]) >= 0 and not r["yhat"].startswith("'") for r in forecasts)
    assert all(r["target_day"][:4].isdigit() for r in forecasts)


def test_export_json_is_never_prefixed_regardless_of_cells(hostile_run_db, tmp_path):
    conn, run_id = hostile_run_db
    expected = json.loads(json.dumps(export._orders(conn, run_id)[1], default=str))
    for cells in ("safe", "raw"):
        out = export.export_orders(conn, run_id, tmp_path / f"o-{cells}.json", "json", cells=cells)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == expected  # identical to the query rows: raw strings, typed numbers
        assert sorted({d["city"] for d in data}) == ["+1", "=1+1"]
        assert sorted({d["sku"] for d in data}) == ["SKU-GRO-0001", "SKU-HOU-0002", ZWSP_SKU]
    assert (tmp_path / "o-safe.json").read_bytes() == (tmp_path / "o-raw.json").read_bytes()


def test_export_cells_raw_writes_verbatim(hostile_run_db, tmp_path):
    conn, run_id = hostile_run_db
    raw = export.export_orders(conn, run_id, tmp_path / "raw.csv", cells="raw")
    rows = read_csv_rows(raw)
    assert sorted({r["city"] for r in rows}) == ["+1", "=1+1"]
    assert sorted({r["sku"] for r in rows}) == ["SKU-GRO-0001", "SKU-HOU-0002", ZWSP_SKU]
    # raw == exactly what csv.DictWriter writes for the query rows (the pre-repair behaviour)
    columns, qrows = export._orders(conn, run_id)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    writer.writerows(qrows)
    assert read_raw(raw) == buf.getvalue()
    safe = export.export_orders(conn, run_id, tmp_path / "safe.csv")
    assert safe.read_bytes() != raw.read_bytes()
    assert len(read_csv_rows(safe)) == len(rows)


@pytest.mark.parametrize("fn", ["export_orders", "export_forecasts", "export_metrics"])
def test_export_rejects_unknown_cell_policy(hostile_run_db, tmp_path, fn):
    conn, run_id = hostile_run_db
    with pytest.raises(ValueError, match="cells"):
        getattr(export, fn)(conn, run_id, tmp_path / "x.csv", cells="bogus")
    assert not (tmp_path / "x.csv").exists()


def test_export_dataset_stays_raw_and_round_trips_hostile_strings(
    hostile_run_db, ingest_db, tmp_path
):
    conn, _ = hostile_run_db
    counts = export.export_dataset(conn, tmp_path / "ds")
    assert counts["stores"] == 2
    assert counts["products"] == 3
    stores_text = read_raw(tmp_path / "ds" / "stores.csv")
    assert ",=1+1," in stores_text  # machine data: verbatim, no spreadsheet marker
    assert "'" not in stores_text
    products_text = read_raw(tmp_path / "ds" / "products.csv")
    assert ZWSP_SKU in products_text
    assert "'" not in products_text
    reports = ingest.load_dataset(ingest_db, tmp_path / "ds")
    assert all(r.rejected == 0 for r in reports)
    for table, key in (("stores", "store_id"), ("products", "product_id")):
        sql = f"SELECT * FROM {table} ORDER BY {key}"
        assert table_rows(ingest_db, sql) == table_rows(conn, sql)
    assert table_rows(ingest_db, "SELECT sku, name FROM products WHERE product_id = 3") == [
        (ZWSP_SKU, MULTILINE_NAME)
    ]


def test_cli_export_cells_flag(hostile_run_db, tmp_path, capsys):
    conn, _ = hostile_run_db
    parser = make_parser()
    # default: spreadsheet-safe
    args = parser.parse_args(["export", "orders", "--out", str(tmp_path / "o.csv")])
    assert args.handler(conn, args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("cells") == "safe"
    assert sorted({r["city"] for r in read_csv_rows(tmp_path / "o.csv")}) == ["'+1", "'=1+1"]
    # explicit raw escape hatch
    args = parser.parse_args(
        ["export", "orders", "--out", str(tmp_path / "r.csv"), "--cells", "raw"]
    )
    assert args.handler(conn, args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("cells") == "raw"
    assert sorted({r["city"] for r in read_csv_rows(tmp_path / "r.csv")}) == ["+1", "=1+1"]
    # JSON output is outside the policy
    args = parser.parse_args(["export", "orders", "--out", str(tmp_path / "o.json")])
    assert args.handler(conn, args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("format") == "json"
    assert payload.get("cells") is None
    data = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert sorted({d["city"] for d in data}) == ["+1", "=1+1"]
    # dataset export is always raw machine data: --cells is rejected, not silently ignored
    for policy in ("raw", "safe"):
        args = parser.parse_args(
            ["export", "dataset", "--out", str(tmp_path / f"ds-{policy}"), "--cells", policy]
        )
        with pytest.raises(ValueError, match="always raw"):
            args.handler(conn, args)
        assert not (tmp_path / f"ds-{policy}").exists()
    args = parser.parse_args(["export", "dataset", "--out", str(tmp_path / "ds")])
    assert args.handler(conn, args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("cells") == "raw"
    # an unknown policy is an argparse usage error
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["export", "orders", "--out", "x.csv", "--cells", "bogus"])
    assert exc_info.value.code == 2
    capsys.readouterr()
    # --help explains the policy and labels the dataset export as machine data
    with pytest.raises(SystemExit):
        parser.parse_args(["export", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())  # undo argparse line wrapping
    assert "--cells {safe,raw}" in help_text
    assert "spreadsheet" in help_text.lower()
    assert "machine data" in help_text.lower()
    assert "(default: safe)" in help_text.lower()
    assert "verbatim" in help_text.lower()
