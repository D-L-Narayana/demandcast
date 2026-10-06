# DemandCast — store-level demand forecasting & replenishment

[![CI](https://github.com/D-L-Narayana/demandcast/actions/workflows/ci.yml/badge.svg)](https://github.com/D-L-Narayana/demandcast/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Live dashboard](https://img.shields.io/badge/dashboard-live-CC0000.svg)](https://demandcast-bay.vercel.app)

A retail supply-chain engine that answers three questions for every **store × SKU** pair:

1. **How many units will we sell over the next 28 days?** — five forecasting models (plus
   promotion-aware variants) are backtested with rolling-origin cross-validation and the best one
   is picked *per series*, with a prediction interval calibrated on the backtest residuals.
2. **What should we order today?** — a periodic-review `(R, s, S)` policy turns the forecast and
   its backtest error into lead-time demand, safety stock, a reorder point, a case-pack-rounded
   order quantity, a **stock-out risk** and a cost-weighted **priority** — optionally within an
   order budget and with per-ABC-class service levels.
3. **How good were yesterday's forecasts?** — backdated runs are scored against what really sold
   (MAE, WAPE, bias and interval coverage) and the history is kept in SQL.

Everything lives in **SQL** (SQLite, PostgreSQL-portable schema) and **Python** (NumPy only).
No ML frameworks — every model is implemented from the equations so the behaviour is
transparent and unit-testable. Run it on the built-in synthetic retailer or on your own CSV
extracts.

```
  sales_daily ─┐                 ┌─ seasonal_naive ─┐
  promotions   ├─ SQL ─▶ series ─┼─ moving_average ─┼─ rolling-origin ─▶ winner ─▶ 28-day forecast
  inventory    ┘  (per store×SKU)├─ holt_winters ───┤   backtest (MAE)     │        + empirical interval
                                 ├─ theta ──────────┤                      ▼
                                 ├─ croston_sba ────┤        (R,s,S) replenishment ─▶ order book
                                 └─ promo_* ────────┘        + stock-out risk · priority · budget
                                                                        │
   forecasts · backtest_metrics · replenishment_orders · forecast_evaluations ─▶ 18 SQL queries ─▶ HTML dashboard
```

## Results (10 stores × 40 SKUs × 730 days = 292 000 sales rows, 2-vCPU sandbox)

| Step | Output | Time |
|---|---|---|
| `demandcast init --snapshot-every 7 --future-promo-days 28` | 292 000 daily sales rows with weekly/yearly seasonality, holidays and inventory-censored stock-outs, 170 promotions (28 of them scheduled after the last sales day), 42 000 inventory snapshots | 11.2 s (3.9 s without the weekly snapshots) |
| `demandcast run --cutoff 2025-12-02` | backdated run: 400 series × up to 8 candidate models × 4 folds → 11 200 forecasts with intervals, 101 orders | 5.1 s (2 workers) |
| `demandcast evaluate` | realised accuracy of that run over the 28 days that followed | 0.7 s |
| `demandcast run` | current run (cutoff 2025-12-30): 11 200 forecasts, 105 orders | 5.4 s (2 workers) |
| `demandcast dashboard` | self-contained HTML with inline SVG charts | 0.8 s |

Timings are wall-clock seconds from one run on a shared 2-vCPU sandbox; repeated runs varied by
about 2× (the two `run` steps took 2.9–5.4 s). The forecasts, orders and accuracy figures below are
deterministic for the default seed.

Backtest leaderboard of the current run (which model won each series, and how accurate it was on
the held-out folds):

| model | series won | share | avg MAE | avg WAPE | avg MASE | avg bias |
|---|---|---|---|---|---|---|
| promo_theta | 144 | 36.0 % | 1.79 | 0.55 | 0.75 | +0.08 |
| moving_average (weekday-profiled) | 85 | 21.3 % | 2.09 | 0.59 | 0.77 | +0.10 |
| theta | 85 | 21.3 % | 1.89 | 0.57 | 0.74 | +0.09 |
| promo_moving_average | 39 | 9.8 % | 1.83 | 0.73 | 0.79 | +0.12 |
| croston_sba (intermittent) | 21 | 5.3 % | 0.62 | 1.34 | 0.91 | −0.01 |
| holt_winters (damped trend) | 14 | 3.5 % | 1.11 | 0.81 | 0.72 | −0.09 |
| promo_holt_winters | 6 | 1.5 % | 1.90 | 0.59 | 0.70 | −0.17 |
| seasonal_naive | 6 | 1.5 % | 0.37 | 1.25 | 0.77 | −0.14 |

MASE < 1 across the board means every selected model beats the seasonal-naive benchmark on its
own series. The promotion-aware variants win most of the series that carry scheduled promotions;
Croston only competes on series that are ≥ 50 % zeros, which is why it wins the slow-moving
Electronics SKUs.

**Realised accuracy** of the backdated run, scored by `demandcast evaluate` against the 28 days of
sales that followed its cutoff: MAE 1.76 units/day, WAPE 37.3 %, bias −0.14, and **80.2 % of
actuals inside the 80 % prediction interval** (nominal 80 %).

| model | series | realised WAPE | bias | interval coverage |
|---|---|---|---|---|
| promo_theta | 150 | 0.38 | −0.06 | 80.8 % |
| theta | 110 | 0.41 | +0.09 | 79.7 % |
| moving_average | 62 | 0.29 | −0.58 | 80.8 % |
| promo_moving_average | 36 | 0.35 | −0.38 | 81.0 % |
| croston_sba | 16 | 1.30 | −0.03 | 78.6 % |
| holt_winters | 11 | 0.76 | −0.23 | 72.4 % |
| promo_holt_winters | 9 | 0.53 | −0.27 | 73.8 % |
| seasonal_naive | 6 | 1.23 | −0.14 | 90.5 % |

Inventory health of the current run: 74 store × SKU series CRITICAL (18.5 %), 42 LOW, 243 OK,
38 OVERSTOCK, 3 NO_DEMAND; order book 105 orders, ₹387 797 at cost.

Compared with 0.3.0 on the same machine (plain `init` → `run --workers 2` → `dashboard`), the
full run took 2.8–5.4 s across repeated runs instead of 18.6 s, although it now backtests up to
eight candidates per series and fits intervals, and the dashboard renders in under a second
instead of 6.8 s (the `promo_lift` query is set-based instead of a correlated `EXISTS`). The vectorised Holt-Winters grid search runs ~30× faster than
the 0.3.0 scalar loop and picks the same parameters (results agree to about 1e-12 relative;
`python benchmarks/bench.py`).

## Quick start

```bash
git clone https://github.com/D-L-Narayana/demandcast && cd demandcast
pip install -e ".[dev]"

demandcast init                                   # schema + synthetic dataset -> demandcast.db
demandcast run                                    # backtest, forecast, replenish -> run_id 1
demandcast query --list                           # the analytics queries and their parameters
demandcast query replenishment_summary --limit 10
demandcast query series_history --param store_id=1 --param product_id=9 --format csv
demandcast dashboard --out dashboard/index.html
pytest                                            # unit, SQL, CLI and dashboard tests
```

Smaller dataset for a quick look: `demandcast init --stores 3 --products 10 --days 365`.
`python -m demandcast …` works too, and `make check` runs lint, type-check and tests.

### Validate forecasts against what really happened

```bash
demandcast init --snapshot-every 7 --future-promo-days 28
demandcast run --cutoff 2025-12-02            # pretend it is 2025-12-02; later sales are not used
demandcast evaluate                           # realised MAE / WAPE / bias / interval coverage
demandcast query evaluation_history           # one row per evaluated run
```

### Bring your own data

```bash
demandcast --db shop.db load --dir examples/mini          # stores, products, promotions, sales, inventory CSVs
demandcast --db shop.db run --order-budget 25000 --service-level-by-class A=0.98,B=0.95,C=0.90
demandcast --db shop.db export orders --out orders.csv    # also: forecasts, metrics, dataset; --format json
```

The CSV layout is documented in [`docs/data-format.md`](docs/data-format.md); rows that fail
validation are reported and skipped (or fail the whole file with `--strict`). `examples/mini`
is a tiny loadable dataset; `examples/README.md` walks through the round trip.

Reporting CSVs (`export orders|forecasts|metrics`, `query --format csv`) are spreadsheet-safe by
default: text cells that a spreadsheet would evaluate as a formula (`=`, `+`, `-`, `@` as the
first visible character) are written with a leading `'`; numbers are untouched and JSON is
unchanged. Use `--cells raw` for verbatim machine output. `export dataset` is always raw — it is
the lossless input for `load`, not a spreadsheet report.

### Operator options

`run --models seasonal_naive,theta` restricts the candidates, `--criterion wape|mase` changes the
selection metric, `--interval 0.9` the interval level, `--no-promo` disables promotion-aware
models, `--order-budget` defers the lowest-priority orders; `runs` lists past runs; `query
--format table|csv|json`; every error is a one-line `error: …` with exit code 1. Full reference:
[`docs/cli.md`](docs/cli.md).

## What's inside

| Path | Purpose |
|---|---|
| `demandcast/sql/schema.sql` | 3NF schema: `stores`, `products`, `calendar`, `promotions`, `sales_daily` (composite PK, `WITHOUT ROWID`), `inventory_snapshots`, plus append-only `forecast_runs` / `forecasts` / `backtest_metrics` / `replenishment_orders` / `forecast_evaluations` / `data_loads`. CHECK constraints and FKs enforce invariants; the schema is versioned with `PRAGMA user_version` and `db.migrate()` upgrades 0.3.0 databases in place. |
| `demandcast/sql/analytics.sql` | 18 named queries: WoW trend (`LAG`), ABC/Pareto (running `SUM() OVER`), stock-out rate per region (`DENSE_RANK`), promo lift (set-based range join), model leaderboard, order book, series history with promo flags, days of cover, forecast-vs-actual, evaluation summary/history, inventory health distribution, order cost by category, bottom-up rollup (`UNION ALL`), run history, stock-out risk top-10, upcoming promotions, bias by category. |
| `demandcast/db.py` | Connection factory (WAL, FK enforcement, busy timeout), schema bootstrap + migrations, `-- name:` query loader with parameter introspection, bulk insert/upsert, nest-safe transactions. |
| `demandcast/simulate.py` | Multiplicative demand generator (base × weekday × yearly × trend × promo × holiday × Poisson noise) plus an `(s, S)` inventory simulation that *censors* sales, so stock-outs look like real POS data; optional periodic snapshots and scheduled future promotions. |
| `demandcast/models.py` | `SeasonalNaive`, `MovingAverage` (weekday re-profiled), `HoltWinters` (additive, damped trend, vectorised grid search), `Theta`, `Croston` (SBA-corrected). One `fit / predict` interface with optional promotion flags. |
| `demandcast/promo.py` | `PromoAdjusted` wrapper: estimates a shrunk promotional lift per series, fits any base model on de-promoted history and re-applies the lift on scheduled promo days. |
| `demandcast/backtest.py` | Rolling-origin evaluation, MAE / WAPE / bias / MASE, empirical prediction intervals, per-series model selection with a pluggable criterion. |
| `demandcast/replenish.py` | `(R, s, S)` policy: `SS = z·σ·√(L+R)`, case-pack / min / max rounding, shelf-life cap, stock-out probability and expected shortfall (normal loss function), priority, greedy budget allocation, per-ABC-class service levels. Pure functions, fully unit-tested. |
| `demandcast/pipeline.py` | Orchestrates a run: dense series loading, backdated cutoffs, promo flags, fan-out over a `ProcessPoolExecutor`, single-transaction write-back, run status tracking and the full `RunConfig` persisted per run. |
| `demandcast/evaluate.py` | Scores a run against realised sales (per series and per model) and persists the result. |
| `demandcast/ingest.py` · `export.py` · `csvsafe.py` | Validated CSV loading (rejection report, calendar auto-fill, provenance), CSV/JSON exports, and the spreadsheet-safe cell policy applied to reporting CSVs. |
| `demandcast/dashboard.py` · `charts.py` | Renders `dashboard/index.html` — KPIs, forecast chart with interval band and promo shading, leaderboard, realised accuracy, inventory health, risk top-10, order book, … — as one self-contained file with inline SVG, dark mode and print styles. Zero JavaScript, strict Content-Security-Policy. |
| `scripts/verify_dashboard.py` · `vercel.json` | The production response headers and a stdlib checker that serves the dashboard with exactly those headers (optionally in a real browser). |
| `benchmarks/` · `docs/` · `examples/` | Timing script; CLI, methodology, architecture, data-format and deployment docs; a tiny loadable dataset. |
| `tests/` | pytest cases for model maths, metric and interval definitions, no-future-leakage in backtests, replenishment edge cases, schema constraints and migrations, CSV round-trips, CLI error paths, dashboard structure and CSP compliance, packaging/compatibility guards and end-to-end pipeline consistency. |

## Design notes

**Why per-series model selection?** Retail demand is heterogeneous: a fast-moving grocery
SKU is well served by a weekday-profiled moving average, a Q4-seasonal toy needs trend
handling, a promoted SKU needs its lift separated from its baseline, and a slow-moving
electronics item sells on 1 day in 5 and needs an intermittent model. Backtesting each candidate
on the last four 28-day windows and picking the lowest MAE is simple, explainable and cheap
enough to run for every series nightly.

**Why is safety stock tied to backtest error?** `σ` in `SS = z·σ·√(L+R)` is the residual
standard deviation of the *selected* model on held-out folds. Series that are harder to
forecast automatically carry more buffer, which is exactly what a service-level target means.
The same `σ` gives the stock-out risk `1 − Φ((IP − μ_{L+R}) / σ√(L+R))` and the expected
shortfall that prioritises the order book when a budget is tight.

**Why empirical intervals?** A symmetric `±z·σ` band is wrong for count data near zero. The
interval is built from the quantiles of the backtest residuals, so it is asymmetric, never
negative and its coverage can be checked with `demandcast evaluate` — on the synthetic retailer
it covers 80.2 % of the following four weeks at a nominal 80 %.

**Why append-only forecast tables?** Each run is immutable and keyed by `run_id`; consumers
read the latest `status = 'succeeded'` run. A crashed run can never leave half-written
forecasts visible, yesterday's forecast is always available, and `forecast_evaluations` closes
the loop by scoring a backdated run against the sales that followed.

**Scaling path.** Series are independent, so the pipeline fans out with a process pool
(2× on 2 vCPUs, linear up to core count). The same `process_series` function could be
submitted to Spark `mapPartitions` or a Kafka consumer group without change; the schema is
PostgreSQL-compatible apart from `WITHOUT ROWID` and `AUTOINCREMENT`.

## Deployment

The dashboard is a static file hosted on Vercel; `vercel.json` adds a strict
Content-Security-Policy and the usual hardening headers, and
`python scripts/verify_dashboard.py --html dashboard/index.html --vercel vercel.json` reproduces
them locally (static checks, a local server with the same headers, and `--browser` for a real
Chromium run when Playwright is installed). See [`docs/deployment.md`](docs/deployment.md).

## Roadmap

- [ ] PostgreSQL backend via `psycopg` with identical named queries
- [ ] Hierarchical reconciliation (store → region → chain) beyond the bottom-up rollup
- [ ] Holiday regressor as a second exogenous channel next to promotions
- [ ] Quantile regression for the prediction intervals

## License

MIT © D L Narayana
