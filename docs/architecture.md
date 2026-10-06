# Architecture

DemandCast is a SQL-centric batch engine: SQLite holds the data and every analytical result,
NumPy does the maths, and one HTML file renders the outcome. There is no server, no
background process and no JavaScript.

```
 CSV / generator ──► SQLite (schema v2) ──► pipeline.run ──► forecasts, metrics, orders ──► analytics.sql ──► dashboard / exports
                                             │                     ▲
                                             ├─ backtest.select_model (models, promo)      │
                                             ├─ replenish.decide / allocate_budget         │
                                             └─ evaluate.evaluate_run (realised accuracy) ─┘
```

## Module map

| Module | Responsibility | Depends on |
|---|---|---|
| `demandcast/cli.py` | argparse front end: `init`, `run`, `runs`, `query`, `dashboard`, `stats`; plugin hook (`COMMAND_PLUGINS`) that mounts `evaluate`, `load`, `export`; error wrapping and exit codes | everything below |
| `demandcast/__main__.py` | `python -m demandcast` | `cli` |
| `demandcast/db.py` | connection factory (WAL, foreign keys, `busy_timeout`), `init_schema` + `migrate` (`PRAGMA user_version`), `ensure_schema`, named-query loader (`-- name:` markers), `run_query` with parameter validation, `insert_many` / `upsert_many`, savepoint-safe `transaction`, `table_counts` | sqlite3 |
| `demandcast/sql/schema.sql` | DDL (3NF master data, append-only run tables, v2 columns and tables, indexes) | - |
| `demandcast/sql/analytics.sql` | 17 named queries (8 baseline + 9 new) | - |
| `demandcast/simulate.py` | synthetic dataset generator with inventory censoring; opt-in periodic snapshots and future promotions | `db` |
| `demandcast/ingest.py` | validated CSV ingest (`load_csv`, `load_dataset`, `ensure_calendar`), rejection report, `data_loads` provenance; CLI plugin `load` | `db` |
| `demandcast/export.py` | `export_orders` / `export_forecasts` / `export_metrics` / `export_dataset` (csv / json); CLI plugin `export` | `db` |
| `demandcast/models.py` | `Forecaster` interface, `SeasonalNaive`, `MovingAverage`, `HoltWinters` (vectorised grid), `Theta`, `Croston`, `MODEL_REGISTRY`, `make_model`, `is_intermittent` | numpy |
| `demandcast/promo.py` | promo lift estimation, deflate / inflate, `PromoAdjusted` wrapper | `models` |
| `demandcast/backtest.py` | rolling-origin folds, MAE / WAPE / bias / MASE, `IntervalModel` (empirical or normal), `candidate_models`, `select_model` | `models`, `promo` |
| `demandcast/replenish.py` | $(R, s, S)$ policy, safety stock, stock-out risk, expected shortfall, priority, min/max order, `allocate_budget`, per-class service levels | stdlib |
| `demandcast/pipeline.py` | `RunConfig`, series loading on a dense day axis (as of an optional cutoff), promo flags, ABC classes as of cutoff, process-pool fan-out over `process_series`, single-transaction write-back, run status | all of the above |
| `demandcast/evaluate.py` | realised accuracy and interval coverage of a run, persisted to `forecast_evaluations`; CLI plugin `evaluate` | `db` |
| `demandcast/charts.py` | pure SVG helpers (`line_chart`, `bar_chart`, `hbar_chart`, `stacked_bar`) with HTML escaping | stdlib |
| `demandcast/dashboard.py` | `render(conn, out_path, run_id=None)`: KPIs, featured series, leaderboard, order book, health, risk, evaluation history, run config; CSP-compliant single file | `db`, `charts` |
| `scripts/verify_dashboard.py` | static / served / browser verification of the dashboard against `vercel.json` (see [deployment.md](deployment.md)) | stdlib |
| `benchmarks/bench.py` | model, selection and end-to-end timings | package |

Runtime dependency: NumPy only. Everything else is the standard library (sqlite3, argparse,
csv, json, statistics, concurrent.futures, http.server for the verifier).

## Run lifecycle (`pipeline.run`)

1. `db.ensure_schema` - raise `SchemaMissingError` on an empty database, otherwise migrate to v2.
2. Insert the `forecast_runs` row (`status = 'running'`, `cutoff_day`, `horizon_days`,
   `config_json`, `interval_level`, `engine_version`) and commit, so a crash leaves a visible
   failed run instead of nothing.
3. Load series: sales up to the cutoff are placed on a **dense** day axis ending at the cutoff
   (missing days become 0, a series starts at its first observed day) so weekday alignment is
   never shifted by gaps. Inventory is the latest snapshot on or before the cutoff. Promo flags
   (chain-wide and store-specific) are built for history and horizon. ABC classes are computed
   from the 90 days before the cutoff.
4. Build one `SeriesTask` per store x product with at least `min_history_days` observations.
   The model horizon is $\max(H, \max L + R)$ so the replenishment policy is always covered.
5. Fan out `process_series` (pure, picklable, module-level) over a `ProcessPoolExecutor`
   (`workers`; sequential when only one worker or at most 8 tasks). Each task: candidate models
   -> rolling-origin backtest -> winner -> refit on the full history -> forecast -> interval ->
   replenishment decision with risk, shortfall and priority.
6. Budget allocation (optional): greedy by priority, deferred orders keep `requested_qty`.
7. Write back in a single transaction: `forecasts` (with `promo_flag`), `backtest_metrics` (one
   `selected = 1` row per series), `replenishment_orders` (`stockout_risk`, `priority`,
   `requested_qty`); then mark the run `succeeded` with timing notes. Any exception rolls back,
   marks the run `failed` with the error text and re-raises.
8. Consumers (`query`, `dashboard`, `export`, `evaluate`) read the latest `status = 'succeeded'`
   run unless a `run_id` is given.

## Schema v2

`PRAGMA user_version = 2`. `db.migrate()` upgrades v0/v1 files additively (guarded by
`PRAGMA table_info`), so a 0.3.0 database keeps working.

Core tables (unchanged): `stores`, `products`, `calendar`, `promotions` (`store_id NULL` =
chain-wide), `sales_daily` (PK store/product/day, `WITHOUT ROWID`, censored units + flag),
`inventory_snapshots`.

Run tables and the v2 additions:

| Table | v2 columns added | Notes |
|---|---|---|
| `forecast_runs` | `config_json TEXT`, `interval_level REAL`, `engine_version TEXT` | a run is reproducible from its row |
| `forecasts` | `promo_flag INTEGER NOT NULL DEFAULT 0` | `0 <= yhat_lower <= yhat <= yhat_upper` |
| `backtest_metrics` | - | one `selected = 1` row per series |
| `replenishment_orders` | `stockout_risk REAL`, `priority REAL`, `requested_qty INTEGER` | `requested_qty` is the quantity before budget allocation (NULL on old rows); `order_qty % case_pack == 0` |
| `forecast_evaluations` (new) | `run_id, store_id, product_id, model_name, n_days, mae, wape, bias, coverage, abs_error_sum, actual_sum, stockout_days, evaluated_at`, PK (run, store, product) | written by `evaluate`, replaced per run |
| `data_loads` (new) | `load_id, loaded_at, source, table_name, mode, rows_inserted, rows_updated, rows_rejected, notes` | provenance of every `load` |

New indexes: `idx_forecasts_series_day (store_id, product_id, target_day)`,
`idx_promotions_product (product_id, start_day, end_day)`.

## Analytics catalogue (`demandcast/sql/analytics.sql`)

Baseline (names and columns stable): `weekly_sales_trend` (ISO-week buckets since 0.4.0),
`abc_classification`, `stockout_rate_by_store`, `promo_lift`, `forecast_accuracy_leaderboard`,
`replenishment_summary`, `series_history` (+ `on_promo`), `days_of_cover`.

New in 0.4.0: `forecast_vs_actual`, `evaluation_summary`, `evaluation_history`,
`inventory_health_distribution`, `order_cost_by_category`, `forecast_rollup`, `run_history`,
`stockout_risk_top`, `promo_calendar_upcoming`. Each block starts with a `-- name:` marker and a
comment explaining the SQL technique it demonstrates; parameters use SQLite named style
(`:run_id`, `:store_id`, `:product_id`).

## Extension points

### Adding a forecasting model

1. Subclass `Forecaster` in `demandcast/models.py` (or a new module): set `name`, implement
   `fit(self, y, promo_flags=None) -> self` and `predict(self, h, future_flags=None) -> np.ndarray`
   (shape `(h,)`, non-negative, no NaN; raise `RuntimeError("call fit() first")` when unfitted).
   Keep it pure NumPy and picklable (it runs inside worker processes).
2. Register it in `MODEL_REGISTRY` - the dict order is the candidate order used by
   `backtest.candidate_models`. If it should only compete on some series, add the gate there
   (as Croston does with `is_intermittent`).
3. Add tests in `tests/test_models.py` (shape / non-negativity via the parametrised registry
   test, plus a behavioural test on a constructed series). The leaderboard, dashboard and
   benchmarks pick the new name up automatically.

### Adding an analytics query

1. Append a block to `demandcast/sql/analytics.sql`:

   ```sql
   -- name: my_query
   -- One-line explanation of the technique (window function, CTE, ...).
   SELECT ... WHERE run_id = :run_id;
   ```

   Names must be unique (`load_queries` rejects duplicates; `tests/test_packaging.py` guards it).
2. `demandcast query my_query` works immediately; `query --list` shows the `:params` it needs
   (`db.query_params`). Add a test in `tests/test_analytics.py` that executes it against the
   small fixture database.
3. To show it on the dashboard, add a section in `dashboard.py` that degrades gracefully when
   the query is absent or returns no rows, and keep the page free of scripts and external
   resources (see [deployment.md](deployment.md)).

### Adding a CLI sub-command (plugin)

1. Create `demandcast/<feature>.py` with

   ```python
   def register_cli(sub) -> None:
       p = sub.add_parser("feature", help="one line shown by `demandcast --help`")
       p.add_argument("--flag", type=int, default=0)
       p.set_defaults(handler=_handle, creates_db=False)


   def _handle(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
       # Do the work; raise ValueError / RuntimeError for user errors - the CLI prints
       # "error: ..." and exits 1. Return the exit code.
       return 0
   ```

   `creates_db=True` is reserved for commands that may create the database file (`init`,
   `load`); every other command fails cleanly when `--db` does not exist.
2. Add the module path to `COMMAND_PLUGINS` in `demandcast/cli.py`. Import errors are
   tolerated (the command simply does not appear), so the core CLI never breaks.
3. Document the flags in [cli.md](cli.md) and add a `cli.main([...])` test with a temporary
   database.

## Testing strategy

* Unit tests per module (`tests/test_<module>.py`) on tiny seeded data; pipeline tests use
  at most 3 x 8 x 300 and `workers <= 2`.
* Guard tests: `tests/test_compat.py` (NumPy-only imports, no Python 3.11+ constructs,
  portable public files) and `tests/test_packaging.py` (version, dependencies, package data,
  `vercel.json`, verifier behaviour).
* CI (`.github/workflows/ci.yml`) runs ruff, mypy, pytest and the full CLI smoke sequence on
  Python 3.10-3.14, then verifies the generated dashboard under the production headers.
