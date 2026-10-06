# Command-line reference

```
demandcast [--db PATH] [-v] [--quiet] [--version] COMMAND [options]
python -m demandcast ...        # identical; works without the console script (0.4.0)
```

Flags marked **0.4.0** are new in that release; everything else behaves exactly as in 0.3.0.

## Global options

| Option | Default | Meaning |
|---|---|---|
| `--db PATH` | `demandcast.db` | SQLite database file. Only `init` and `load` may create it; every other command fails with exit 1 (and creates nothing) when the file does not exist. **0.4.0** |
| `-v`, `--verbose` | off | DEBUG logging on stderr |
| `--quiet` | off | suppress INFO logging **0.4.0** |
| `--version` | | print `demandcast <version>` |

Exit codes: `0` success - `1` a reported error (`error: ...` on stderr: missing schema,
unknown query, missing `:param`, no realised sales to evaluate, invalid flag values) -
`2` argparse usage error. Tracebacks are reserved for genuine bugs. **0.4.0**

## Commands

### `init` - create the schema and (optionally) a synthetic dataset

```
demandcast --db demo.db init [--stores 10] [--products 40] [--days 730] [--start 2024-01-01]
                             [--seed 42] [--no-data] [--snapshot-every N] [--future-promo-days N]
```

| Option | Default | Meaning |
|---|---|---|
| `--stores`, `--products`, `--days` | 10 / 40 / 730 | dataset size (default: 292 000 sales rows) |
| `--start` | 2024-01-01 | first calendar day |
| `--seed` | 42 | generator seed (the default dataset is byte-identical across releases) |
| `--no-data` | off | create the schema only, for `load` **0.4.0** |
| `--snapshot-every N` | last day only | write an inventory snapshot every N days (enables backdated runs with as-of inventory) **0.4.0** |
| `--future-promo-days N` | 0 | schedule promotions for N days after the last sales day (exercises promo-aware forecasting) **0.4.0** |

Prints the row counts per table as JSON. Refuses to overwrite a database that already contains
sales.

### `load` - bring your own data (CSV) **0.4.0**

```
demandcast --db my.db load --dir DIR
demandcast --db my.db load [--stores F] [--products F] [--calendar F] [--promotions F]
                           [--sales F] [--inventory F] [--mode upsert|insert|replace]
                           [--dry-run] [--strict]
```

Loads `<table>.csv` files (column names = table columns, ISO dates, UTF-8 with header) in
foreign-key order, auto-creates the schema, fills missing calendar rows, validates every row
(types, ranges, enums, foreign keys, duplicates) and prints a per-table report (inserted /
updated / rejected with the first reasons). `--strict` turns any rejected row into exit 1;
`--dry-run` validates without writing. Every load is recorded in `data_loads`. Column
specifications live in [`data-format.md`](data-format.md).

### `run` - backtest, forecast and replenish

```
demandcast --db demo.db run [--horizon 28] [--folds 4] [--service-level 0.95] [--review-period 7]
                            [--workers 0] [--cutoff DATE] [--models a,b] [--interval 0.8]
                            [--interval-method empirical|normal] [--criterion mae|wape|mase]
                            [--no-promo] [--order-budget X]
                            [--service-level-by-class A=0.98,B=0.95,C=0.90]
```

| Option | Default | Meaning |
|---|---|---|
| `--horizon` | 28 | forecast days persisted per series |
| `--folds` | 4 | rolling-origin backtest folds |
| `--service-level` | 0.95 | cycle service level for safety stock |
| `--review-period` | 7 | days between orders (R) |
| `--workers` | 0 = all CPUs | process-pool size |
| `--cutoff DATE` | last sales day | backdated run: only sales up to DATE are used and inventory is taken from the latest snapshot on or before DATE; forecasts start at DATE+1 so `evaluate` can score them against what really happened **0.4.0** |
| `--models a,b` | all | restrict the candidates (names from `MODEL_REGISTRY` and `promo_*`) **0.4.0** |
| `--interval LEVEL` | 0.8 | nominal prediction-interval level **0.4.0** |
| `--interval-method` | empirical | `empirical` quantiles of backtest residuals or symmetric `normal` **0.4.0** |
| `--criterion` | mae | selection metric (`mae`, `wape`, `mase`) **0.4.0** |
| `--no-promo` | off | ignore promotions (no `promo_*` candidates, no promo flags) **0.4.0** |
| `--order-budget X` | none | cap the total order cost; orders are approved greedily by priority, the rest are deferred (`order_qty = 0`, `requested_qty` kept) **0.4.0** |
| `--service-level-by-class` | none | per-ABC-class service levels, e.g. `A=0.98,B=0.95,C=0.90` **0.4.0** |

Prints the `forecast_runs` row as JSON (status, cutoff, horizon, series count, notes with
skipped / deferred / timing counters). The run configuration is stored in `config_json`.

### `evaluate` - realised accuracy of a backdated run **0.4.0**

```
demandcast --db demo.db evaluate [--run-id N] [--json]
```

Joins the run's forecasts with the sales observed since the cutoff and persists per-series MAE,
WAPE, bias and interval coverage into `forecast_evaluations` (re-running replaces the rows).
Prints the aggregate summary (`n_series`, `n_days_available`, `mae`, `wape`, `bias`,
`coverage`, per-model breakdown). Fails with exit 1 when no realised sales exist yet - use
`run --cutoff` or load newer sales first.

### `runs` - list forecast runs **0.4.0**

```
demandcast --db demo.db runs [--limit 20] [--json]
```

### `query` - execute a named analytics query

```
demandcast --db demo.db query                       # list query names
demandcast --db demo.db query --list                # names with their required :params
demandcast --db demo.db query NAME [--run-id N] [--store-id N] [--product-id N]
                                   [--limit 20] [--json] [--format table|csv|json]
                                   [--cells safe|raw] [--param key=value]...
```

| Option | Default | Meaning |
|---|---|---|
| `--run-id` | latest successful run | fills `:run_id` |
| `--store-id`, `--product-id` | | fill `:store_id` / `:product_id` (`series_history`) |
| `--limit N` | 20 | rows to print; `0` prints all rows **0.4.0** (0.3.0 crashed on 0) |
| `--json` | off | same as `--format json` |
| `--format` | table | `table`, `csv` or `json` **0.4.0** |
| `--cells` | safe | `safe` prefixes formula-like text cells with `'` in CSV output, `raw` writes them verbatim; accepted with every format but only affects `--format csv` (see "Spreadsheet-safe CSV" under `export`) **0.4.0** |
| `--param k=v` | | any other `:param` the query needs **0.4.0** |

Unknown names and missing parameters produce a one-line `error:` message, exit 1. The query
catalogue (8 baseline + 9 new) is documented in [`architecture.md`](architecture.md).

### `export` - write run outputs or the whole dataset **0.4.0**

```
demandcast --db demo.db export orders    --out orders.csv    [--run-id N] [--format csv|json] [--cells safe|raw]
demandcast --db demo.db export forecasts --out forecasts.csv [--run-id N] [--format csv|json] [--cells safe|raw]
demandcast --db demo.db export metrics   --out metrics.csv   [--run-id N] [--format csv|json] [--cells safe|raw]
demandcast --db demo.db export dataset   --out DIR            # <table>.csv files: raw machine data for `load`
```

| Option | Default | Meaning |
|---|---|---|
| `--out PATH` | required | output file (`orders`, `forecasts`, `metrics`) or directory (`dataset`) |
| `--run-id N` | latest successful run | which run to export (not used by `dataset`) |
| `--format` | from the `--out` suffix (`.json` → json, otherwise csv) | `csv` or `json` for `orders`, `forecasts`, `metrics`; `dataset` is always CSV (`--format json` is an error) |
| `--cells` | safe | `safe` = spreadsheet-safe text cells, `raw` = cells verbatim; CSV only, see below; rejected by `export dataset` **0.4.0** |

`orders` carries the order book (`replenishment_summary` columns plus `stockout_risk`,
`priority`, `requested_qty`). `dataset` writes one `<table>.csv` per core table
(`stores`, `products`, `calendar`, `promotions`, `sales_daily`, `inventory_snapshots`) exactly as
stored, so `load --dir` reproduces the database byte for byte.

#### Spreadsheet-safe CSV (`--cells`) **0.4.0**

Text that reaches the reporting exports and `query --format csv` can come from outside -
store codes, SKUs, product names, cities and categories loaded with `load`, or free-text
`reason` columns. A spreadsheet application that opens such a CSV would evaluate a cell like
`=1+1`, `+1`, `-1` or `@x` as a formula; CSV quoting does not prevent that. The reporting
outputs are therefore written **spreadsheet-safe by default**:

* A text cell is prefixed with a single quote (`'`, the conventional "treat as text" marker)
  when it begins with a tab, carriage return or line feed, or when its first visible character -
  after any leading whitespace or invisible Unicode characters (control, format, space, line and
  paragraph separators) - is `=`, `+`, `-` or `@`. Example: `=1+1` is written as `'=1+1`.
* Numbers are never changed: negative values such as a bias of `-0.14` or a negative
  `priority` stay numeric; dates start with a digit; empty cells stay empty. Only the exported
  text is affected - the database is never modified.
* JSON output (`--format json`, `query --json`) is unchanged and always carries the raw strings.
* `--cells raw` writes every cell verbatim for machine consumers that parse the CSV themselves
  and never open it in a spreadsheet. Any other value is rejected as a usage error.
* `export dataset` is raw machine data for `load` (a lossless round trip, byte-equal values); it
  does not accept `--cells` (an error, never silently ignored) and must not be opened in a
  spreadsheet without care.
* Some spreadsheet applications display the leading `'`; it is a marker, not part of the data.

The same policy applies to `demandcast query --format csv` (`--cells safe|raw`); `--format table`
and `--format json` are unaffected. The Python API mirrors the flag
(`export_orders(..., cells="safe")`, likewise `export_forecasts` / `export_metrics`; an unknown
policy raises `ValueError`). Column specifications and the export policy table live in
[`data-format.md`](data-format.md).

### `dashboard` - render the self-contained HTML report

```
demandcast --db demo.db dashboard [--out dashboard/index.html] [--run-id N]
```

One HTML file, inline SVG, zero JavaScript; see [`deployment.md`](deployment.md) for the
content-security policy it must satisfy and how to verify it.

### `stats` - row counts per table

```
demandcast --db demo.db stats [--json]
```

The output is JSON already; `--json` (**0.4.0**) is accepted for symmetry with `runs`.

## Typical sessions

```bash
# Demo (UW-A)
demandcast init && demandcast run && demandcast query replenishment_summary --limit 10
demandcast dashboard --out dashboard/index.html

# Own data (UW-B)
demandcast --db shop.db init --no-data
demandcast --db shop.db load --dir ./csv            # stores, products, sales_daily, inventory_snapshots, [promotions]
demandcast --db shop.db run --workers 2
demandcast --db shop.db export orders --out orders.csv            # spreadsheet-safe cells by default
demandcast --db shop.db export orders --out feed.csv --cells raw  # verbatim cells for a downstream parser

# Validation loop (UW-C)
demandcast --db shop.db run --cutoff 2025-12-02     # forecast as of that day
demandcast --db shop.db evaluate --json             # realised MAE / WAPE / bias / coverage
demandcast --db shop.db query evaluation_history

# Budgeted, prioritised ordering (UW-D)
demandcast --db shop.db run --order-budget 300000 --service-level-by-class A=0.98,B=0.95,C=0.90
demandcast --db shop.db query stockout_risk_top
```

`make smoke` runs the complete CI sequence locally; `make demo` runs the validation loop on a
fresh synthetic database.
