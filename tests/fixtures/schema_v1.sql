-- DemandCast relational schema (SQLite dialect, portable to PostgreSQL with minor changes).
-- Third-normal-form core tables + append-only forecast/replenishment outputs.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS stores (
    store_id        INTEGER PRIMARY KEY,
    store_code      TEXT    NOT NULL UNIQUE,
    city            TEXT    NOT NULL,
    region          TEXT    NOT NULL,
    format          TEXT    NOT NULL CHECK (format IN ('flagship', 'standard', 'express')),
    opened_on       DATE    NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id      INTEGER PRIMARY KEY,
    sku             TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    unit_cost       REAL    NOT NULL CHECK (unit_cost > 0),
    unit_price      REAL    NOT NULL CHECK (unit_price >= unit_cost),
    case_pack       INTEGER NOT NULL DEFAULT 1 CHECK (case_pack >= 1),
    lead_time_days  INTEGER NOT NULL CHECK (lead_time_days >= 0),
    shelf_life_days INTEGER          CHECK (shelf_life_days IS NULL OR shelf_life_days > 0)
);

CREATE TABLE IF NOT EXISTS calendar (
    day             DATE    PRIMARY KEY,
    day_of_week     INTEGER NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    week_of_year    INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    year            INTEGER NOT NULL,
    is_weekend      INTEGER NOT NULL CHECK (is_weekend IN (0, 1)),
    holiday_name    TEXT
);

CREATE TABLE IF NOT EXISTS promotions (
    promo_id        INTEGER PRIMARY KEY,
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    store_id        INTEGER          REFERENCES stores(store_id),   -- NULL => chain-wide
    start_day       DATE    NOT NULL,
    end_day         DATE    NOT NULL,
    discount_pct    REAL    NOT NULL CHECK (discount_pct > 0 AND discount_pct < 1),
    CHECK (end_day >= start_day)
);

-- One row per store × product × day. units_sold is *observed* sales (censored by stock-outs).
CREATE TABLE IF NOT EXISTS sales_daily (
    store_id        INTEGER NOT NULL REFERENCES stores(store_id),
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    day             DATE    NOT NULL REFERENCES calendar(day),
    units_sold      INTEGER NOT NULL CHECK (units_sold >= 0),
    revenue         REAL    NOT NULL CHECK (revenue >= 0),
    stockout_flag   INTEGER NOT NULL DEFAULT 0 CHECK (stockout_flag IN (0, 1)),
    PRIMARY KEY (store_id, product_id, day)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_sales_day ON sales_daily(day);
CREATE INDEX IF NOT EXISTS idx_sales_product_day ON sales_daily(product_id, day);

CREATE TABLE IF NOT EXISTS inventory_snapshots (
    store_id        INTEGER NOT NULL REFERENCES stores(store_id),
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    snapshot_day    DATE    NOT NULL,
    on_hand         INTEGER NOT NULL CHECK (on_hand >= 0),
    on_order        INTEGER NOT NULL DEFAULT 0 CHECK (on_order >= 0),
    PRIMARY KEY (store_id, product_id, snapshot_day)
) WITHOUT ROWID;

-- Every forecasting job is a run; forecasts are append-only and keyed by run.
CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    cutoff_day      DATE    NOT NULL,
    horizon_days    INTEGER NOT NULL CHECK (horizon_days > 0),
    series_count    INTEGER,
    status          TEXT    NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'succeeded', 'failed')),
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id          INTEGER NOT NULL REFERENCES forecast_runs(run_id),
    store_id        INTEGER NOT NULL REFERENCES stores(store_id),
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    target_day      DATE    NOT NULL,
    model_name      TEXT    NOT NULL,
    yhat            REAL    NOT NULL CHECK (yhat >= 0),
    yhat_lower      REAL    NOT NULL CHECK (yhat_lower >= 0),
    yhat_upper      REAL    NOT NULL,
    PRIMARY KEY (run_id, store_id, product_id, target_day)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS backtest_metrics (
    run_id          INTEGER NOT NULL REFERENCES forecast_runs(run_id),
    store_id        INTEGER NOT NULL,
    product_id      INTEGER NOT NULL,
    model_name      TEXT    NOT NULL,
    folds           INTEGER NOT NULL,
    mae             REAL    NOT NULL,
    wape            REAL,
    bias            REAL    NOT NULL,
    mase            REAL,
    selected        INTEGER NOT NULL DEFAULT 0 CHECK (selected IN (0, 1)),
    PRIMARY KEY (run_id, store_id, product_id, model_name)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS replenishment_orders (
    order_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER NOT NULL REFERENCES forecast_runs(run_id),
    store_id            INTEGER NOT NULL REFERENCES stores(store_id),
    product_id          INTEGER NOT NULL REFERENCES products(product_id),
    order_day           DATE    NOT NULL,
    expected_arrival    DATE    NOT NULL,
    on_hand             INTEGER NOT NULL,
    on_order            INTEGER NOT NULL,
    lead_time_demand    REAL    NOT NULL,
    safety_stock        REAL    NOT NULL,
    reorder_point       REAL    NOT NULL,
    order_up_to         REAL    NOT NULL,
    order_qty           INTEGER NOT NULL CHECK (order_qty >= 0),
    service_level       REAL    NOT NULL,
    reason              TEXT    NOT NULL,
    UNIQUE (run_id, store_id, product_id)
);
