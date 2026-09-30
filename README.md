# DemandCast — store-level demand forecasting & replenishment

[![CI](https://github.com/D-L-Narayana/demandcast/actions/workflows/ci.yml/badge.svg)](https://github.com/D-L-Narayana/demandcast/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Live dashboard](https://img.shields.io/badge/dashboard-live-CC0000.svg)](https://demandcast.vercel.app)

A retail supply-chain engine that answers two questions for every **store × SKU** pair:

1. **How many units will we sell over the next 28 days?** — four forecasting models are
   backtested with rolling-origin cross-validation and the best one is picked *per series*.
2. **What should we order today?** — a periodic-review `(R, s, S)` policy turns the forecast and
   its backtest error into lead-time demand, safety stock, a reorder point and a case-pack-rounded
   order quantity.

Everything lives in **SQL** (SQLite, PostgreSQL-portable schema) and **Python** (NumPy only).
No ML frameworks — every model is implemented from the equations so the behaviour is
transparent and unit-testable.

```
  sales_daily ─┐                 ┌─ seasonal_naive ─┐
  promotions   ├─ SQL ─▶ series ─┼─ moving_average ─┼─ rolling-origin ─▶ winner ─▶ 28-day forecast
  inventory    ┘  (per store×SKU)├─ holt_winters ───┤   backtest (MAE)     │        + 80% interval
                                 └─ croston_sba ────┘                      ▼
                                                              (R,s,S) replenishment ─▶ order book
                                                                        │
                        forecasts · backtest_metrics · replenishment_orders  ─▶  8 analytics queries ─▶ HTML dashboard
```

## Results (10 stores × 40 SKUs × 730 days = 292 000 sales rows, 2-vCPU sandbox)

| Step | Output | Time |
|---|---|---|
| `demandcast init` | 292 000 daily sales rows with weekly/yearly seasonality, holidays, 149 promotions and inventory-censored stock-outs | 3.5 s |
| `demandcast run` | 400 series × 4 candidate models × 4 backtest folds → 11 200 forecasts, 1 257 metric rows, 400 order decisions | **9.0 s** (2 workers) / 17.4 s (1 worker) |
| `demandcast dashboard` | self-contained HTML with inline SVG charts | < 1 s |

Backtest leaderboard from that run (which model won each series, and how accurate it was):

| model | series won | share | avg MAE | avg WAPE | avg MASE | avg bias |
|---|---|---|---|---|---|---|
| moving_average (weekday-profiled) | 343 | 85.8 % | 1.87 | 0.60 | 0.76 | +0.11 |
| holt_winters (damped trend) | 37 | 9.3 % | 1.86 | 0.64 | 0.79 | −0.03 |
| croston_sba (intermittent) | 14 | 3.5 % | 0.50 | 1.50 | 0.94 | −0.02 |
| seasonal_naive | 6 | 1.5 % | 0.37 | 1.25 | 0.77 | −0.14 |

MASE < 1 across the board means every selected model beats the seasonal-naive benchmark on
its own series. Croston only competes on series that are ≥ 50 % zeros, which is why it wins the
slow-moving Electronics SKUs.

## Quick start

```bash
git clone https://github.com/D-L-Narayana/demandcast && cd demandcast
pip install -e ".[dev]"

demandcast init                       # create schema + synthetic dataset  -> demandcast.db
demandcast run                        # backtest, forecast, replenish     -> run_id 1
demandcast query                      # list analytics queries
demandcast query replenishment_summary --limit 10
demandcast query series_history --store-id 1 --product-id 9
demandcast dashboard --out dashboard/index.html
pytest                                # 35 tests, ~1 s
```

Smaller dataset for a quick look: `demandcast init --stores 3 --products 10 --days 365`.

## What's inside

| Path | Purpose |
|---|---|
| `demandcast/sql/schema.sql` | 3NF schema: `stores`, `products`, `calendar`, `promotions`, `sales_daily` (composite PK, `WITHOUT ROWID`), `inventory_snapshots`, plus append-only `forecast_runs` / `forecasts` / `backtest_metrics` / `replenishment_orders`. CHECK constraints and FKs enforce invariants at the database layer. |
| `demandcast/sql/analytics.sql` | 8 named queries: WoW trend (`LAG`), ABC/Pareto classification (running `SUM() OVER`), stock-out rate ranked per region (`DENSE_RANK`), promo lift (correlated `EXISTS`), model leaderboard, order book, series history with 7-day moving average, days-of-cover health. |
| `demandcast/db.py` | Connection factory (WAL, FK enforcement), `-- name:` query loader, bulk insert, explicit transaction context manager. |
| `demandcast/simulate.py` | Multiplicative demand generator (base × weekday × yearly × trend × promo × holiday × Poisson noise) plus an `(s, S)` inventory simulation that *censors* sales, so stock-outs look like real POS data. |
| `demandcast/models.py` | `SeasonalNaive`, `MovingAverage` (weekday re-profiled), `HoltWinters` (additive, damped trend, grid-searched), `Croston` (SBA-corrected). One `fit / predict` interface. |
| `demandcast/backtest.py` | Rolling-origin evaluation, MAE / WAPE / bias / MASE, per-series model selection. |
| `demandcast/replenish.py` | `(R, s, S)` policy: `SS = z·σ·√(L+R)`, case-pack rounding, shelf-life cap. Pure function, fully unit-tested. |
| `demandcast/pipeline.py` | Orchestrates a run: fan-out over a `ProcessPoolExecutor`, single-transaction write-back, run status tracking (`running → succeeded/failed`). |
| `demandcast/dashboard.py` | Renders `dashboard/index.html` — KPIs, forecast chart with interval band, leaderboard, order book, inventory health — from the SQL queries above. Zero JS dependencies. |
| `tests/` | 35 pytest cases: model maths, metric definitions, no-future-leakage in backtests, replenishment edge cases, schema constraints, end-to-end pipeline consistency. |

## Design notes

**Why per-series model selection?** Retail demand is heterogeneous: a fast-moving grocery
SKU is well served by a weekday-profiled moving average, a Q4-seasonal toy needs trend
handling, and a slow-moving electronics item sells on 1 day in 5 and needs an intermittent
model. Backtesting each candidate on the last four 28-day windows and picking the lowest MAE
is simple, explainable and cheap enough to run for every series nightly.

**Why is safety stock tied to backtest error?** `σ` in `SS = z·σ·√(L+R)` is the residual
standard deviation of the *selected* model on held-out folds. Series that are harder to
forecast automatically carry more buffer, which is exactly what a service-level target means.

**Why append-only forecast tables?** Each run is immutable and keyed by `run_id`; consumers
read the latest `status = 'succeeded'` run. A crashed run can never leave half-written
forecasts visible, and yesterday's forecast is always available for accuracy tracking.

**Scaling path.** Series are independent, so the pipeline fans out with a process pool
(2× on 2 vCPUs, linear up to core count). The same `process_series` function could be
submitted to Spark `mapPartitions` or a Kafka consumer group without change; the schema is
PostgreSQL-compatible apart from `WITHOUT ROWID` and `AUTOINCREMENT`.

## Roadmap

- [ ] Promo-aware regressors (the `promotions` table is already joined in `promo_lift`)
- [ ] Hierarchical reconciliation (store → region → chain)
- [ ] Quantile forecasts instead of a symmetric interval
- [ ] PostgreSQL backend via `psycopg` with identical named queries

## License

MIT © D L Narayana
