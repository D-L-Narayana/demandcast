# Changelog

All notable changes to DemandCast are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses semantic versioning.

## [0.4.0] — 2026-10-05

Theme: from demo to operable tool — bring your own data, validate forecasts against what really
happened, prioritise and budget replenishment, and ship the dashboard under a strict content
security policy. The runtime dependency is still NumPy only; Python 3.10+.

### Added
- **Theta forecaster** (`theta`) and a **promotion-aware wrapper** (`promo_moving_average`,
  `promo_holt_winters`, `promo_theta`): the wrapper estimates a shrunk per-series promo lift,
  fits the base model on de-promoted history and re-applies the lift on scheduled promo days.
  Promo variants compete in the same rolling-origin backtest and are only used when they win.
- **Empirical prediction intervals** built from backtest residual quantiles (asymmetric, never
  negative, bracket guaranteed, mild horizon widening) with `run --interval LEVEL` and
  `--interval-method empirical|normal`; `select_model` accepts `--criterion mae|wape|mase` and a
  `--models` restriction.
- **Backdated runs and realised accuracy**: `run --cutoff DATE` uses only history up to the
  cutoff (inventory as of the cutoff); `demandcast evaluate` scores a run against the sales that
  arrived afterwards (MAE, WAPE, bias, interval coverage per series) and persists the result in
  the new `forecast_evaluations` table; `evaluation_summary` / `evaluation_history` queries and a
  "Realised accuracy" dashboard section expose it.
- **Risk-aware, budgeted replenishment**: every order line now carries `stockout_risk`
  (probability that demand over lead time + review period exceeds the inventory position),
  expected shortfall (normal loss function) and a cost-weighted `priority`; optional
  `min_order_qty` / `max_order_qty`; `run --order-budget X` approves orders greedily by priority
  and records deferred lines (`requested_qty` kept, `order_qty = 0`);
  `run --service-level-by-class A=0.98,B=0.95,C=0.90` sets per-ABC-class service levels.
- **Bring your own data**: `demandcast load` ingests validated CSVs (stores, products, calendar,
  promotions, sales, inventory) with a rejection report, `--strict`, `--dry-run`, upsert /
  insert / replace modes, automatic calendar fill and a `data_loads` provenance table;
  `demandcast export orders|forecasts|metrics|dataset` writes CSV or JSON. A tiny loadable
  dataset lives in `examples/mini`; the format is documented in `docs/data-format.md`.
- **Schema v2 with migrations**: `PRAGMA user_version`, additive columns (`forecast_runs.config_json`,
  `interval_level`, `engine_version`; `forecasts.promo_flag`; `replenishment_orders.stockout_risk`,
  `priority`, `requested_qty`), new tables `forecast_evaluations` and `data_loads`, two indexes.
  Databases created by 0.3.0 are migrated in place by `db.migrate()` / `db.ensure_schema()`.
- **Analytics v2**: nine new named queries — `forecast_vs_actual`, `evaluation_summary`,
  `evaluation_history`, `inventory_health_distribution`, `order_cost_by_category`,
  `forecast_rollup`, `run_history`, `stockout_risk_top`, `promo_calendar_upcoming` — plus
  `forecast_bias_by_category`; `series_history` gains `on_promo`.
- **Dashboard v2**: pure SVG chart library (`demandcast/charts.py`), new sections (run
  configuration, realised accuracy, inventory health distribution, stock-out risk top 10, order
  cost by category, bottom-up rollup, upcoming promotions), date axes, promotion shading, interval
  level label, dark mode (`prefers-color-scheme`), print styles, accessibility fixes (table
  captions, column scope, labelled charts, keyboard-reachable scroll regions, AA colour contrast),
  a Content-Security-Policy `<meta>` tag and a `data:` favicon. Still one self-contained file with
  zero JavaScript.
- **Deployment hardening**: `vercel.json` adds `Content-Security-Policy`, `X-Content-Type-Options`,
  `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, `Cross-Origin-Opener-Policy` and
  `Cross-Origin-Resource-Policy` headers; `scripts/verify_dashboard.py` checks the document and
  serves it locally with those exact headers (optional real-browser mode). See
  `docs/deployment.md`.
- **CLI**: plugin sub-commands (`evaluate`, `load`, `export`), `runs`, `query --format
  table|csv|json`, `query --param k=v`, `query --list`, `--quiet`, `python -m demandcast`,
  `init --no-data --snapshot-every N --future-promo-days N`.
- **Engineering**: `mypy` configuration and `py.typed`, stricter ruff rule set, CI on Python
  3.10 / 3.12 / 3.13 / 3.14 with the full CLI smoke sequence and the dashboard verifier,
  `Makefile` targets (`check`, `typecheck`, `smoke`, `demo`, `verify-dashboard`, `bench`),
  `benchmarks/bench.py`, `docs/` (CLI, methodology, architecture, data format, deployment),
  `CONTRIBUTING.md`, compatibility and portability guard tests.

### Security
- **Spreadsheet-safe reporting CSV.** `demandcast export orders|forecasts|metrics` (CSV) and
  `demandcast query --format csv` now neutralise text cells that a spreadsheet would evaluate as a
  formula: a string whose first visible character is `=`, `+`, `-` or `@` — or that starts with a
  tab, carriage return, line feed or invisible whitespace before such a character — is written with
  a leading `'` (the conventional "treat as text" marker). Numbers are never changed, so negative
  values stay numeric; JSON output is unchanged; database values are never modified. Pass
  `--cells raw` (Python: `cells="raw"`) for verbatim machine output. `export dataset` is always raw
  because it is the lossless round trip for `demandcast load`; it is labelled as machine data and
  should not be opened in a spreadsheet without care. Found by an independent review of the
  release; reproduced test-first (`demandcast/csvsafe.py`, `tests/test_csvsafe.py`).

### Changed
- Holt-Winters grid search evaluates all parameter candidates in one vectorised state-space pass
  (agrees with the previous scalar loop to about 1e-12 relative and selects the same candidate;
  ~30× faster at 730 days).
- `promo_lift` is set-based instead of a correlated `EXISTS` per sales row (identical rows; from
  ~20 s to well under a second on the 292 000-row default dataset), which is what made the
  dashboard render in under a second instead of 6.8 s; the default full run went from 18.6 s to
  roughly 3–5 s with two workers on the same 2-vCPU machine.
- Forecast intervals are no longer a fixed symmetric `±1.2816·σ`; `RunConfig.interval_z` was
  replaced by `interval_level` / `interval_method`.
- Series are loaded on a dense daily axis: missing days count as zero sales instead of silently
  shifting the weekday profile.
- Every run persists its full `RunConfig` as JSON together with the engine version; run notes
  include `inventory_missing`, `deferred` and `promo_series` counters.
- Read-only CLI commands fail with a one-line `error: …` and exit code 1 instead of a traceback,
  and never create an empty database file; schema checks migrate 0.3.0 databases on first use.
- `MovingAverage` estimates its weekday profile from the last eight weeks when at least eight
  weeks of history exist.

### Fixed
- Dashboard "Units by category (last 8 weeks)" drew only the alphabetically last category.
- Holidays were missing for the middle calendar years of multi-year datasets.
- Croston's first inter-demand interval was over-counted by one day.
- `weekly_sales_trend` split the ISO week that straddles New Year across two calendar years.
- `query --limit 0` crashed; unknown query names, missing parameters and uninitialised databases
  produced raw tracebacks.
- Failed pipeline runs are covered by a real test of the `status = 'failed'` path.
- Synthetic promo-day revenue used a hard-coded 20 % discount instead of the promotion's
  `discount_pct` (units, stock-outs, master data and inventory snapshots of the default dataset
  are byte-identical to 0.3.0; only promo-day revenue changed).
- `HoltWinters.predict()` before `fit()` raised `AttributeError`; all models now raise a clear
  `RuntimeError`.
- Six pre-existing `mypy` errors.

## [0.3.0]

Initial public release: synthetic retail dataset generator, four NumPy forecasting models with
rolling-origin backtests and per-series selection, `(R, s, S)` replenishment, eight SQL analytics
queries and a self-contained HTML dashboard.
