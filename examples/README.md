# Examples

## `mini/` — a tiny bring-your-own-data set

Six CSV files in exactly the layout `demandcast load` expects (column names = table columns,
ISO dates, UTF-8 — see [`docs/data-format.md`](../docs/data-format.md)): 2 stores × 3 products
× 120 days (2024-01-01 … 2024-04-29), about 24 KB in total.

| file | rows | contents |
|---|---|---|
| `stores.csv` | 2 | store master data (`format` ∈ flagship / standard / express) |
| `products.csv` | 3 | SKU master data: cost, price, case pack, lead time, shelf life |
| `calendar.csv` | 120 | one row per day with weekday / ISO week / weekend / holiday flags |
| `promotions.csv` | 9 | chain-wide (`store_id` empty) and store-specific promotions with `discount_pct` |
| `sales_daily.csv` | 720 | observed units, revenue and stock-out flag per store × product × day |
| `inventory_snapshots.csv` | 6 | on-hand / on-order position on the last day |

The files were produced by the synthetic generator
(`SimConfig(n_stores=2, n_products=3, start=date(2024, 1, 1), days=120, seed=3)`) through
`demandcast.export.export_dataset`; regenerate them with

```bash
python examples/make_mini.py
```

## Load → run → export

All commands run from the repository root and write only to the paths you pass.

```bash
# 1. create the database and load the dataset (tables are loaded in foreign-key order;
#    rows that fail validation are reported and skipped — add --strict to fail instead)
python -m demandcast --db mini.db load --dir examples/mini

# 2. backtest, forecast and build the replenishment order book
python -m demandcast --db mini.db run --horizon 14 --folds 2 --workers 1

# 3. export the results (format follows the --out suffix, or pass --format csv|json;
#    reporting CSVs are spreadsheet-safe by default — add --cells raw for verbatim cells)
python -m demandcast --db mini.db export orders    --out out/orders.csv
python -m demandcast --db mini.db export forecasts --out out/forecasts.json
python -m demandcast --db mini.db export metrics   --out out/metrics.csv

# 4. round-trip: dump the six core tables and reload them into a fresh database
python -m demandcast --db mini.db export dataset --out out/dataset
python -m demandcast --db copy.db load --dir out/dataset

# 5. optional: the self-contained HTML dashboard
python -m demandcast --db mini.db dashboard --out out/index.html
```

Reporting CSVs (`orders`, `forecasts`, `metrics`) are spreadsheet-safe by default: text cells
that a spreadsheet would read as a formula (`=…`, `+…`, `-…`, `@…`) get a leading `'`; pass
`--cells raw` for verbatim cells, while `export dataset` is always raw machine data for `load`
(see [`docs/data-format.md`](../docs/data-format.md#spreadsheet-safety-the---cells-policy)).

`load` prints a JSON report per table (`inserted`, `updated`, `rejected`, the first 20 error
messages as `"row N: reason"`, `calendar_days_added`) and records a provenance row in
`data_loads`.  Useful variants:

```bash
python -m demandcast --db mini.db load --sales new_sales.csv --dry-run        # validate only
python -m demandcast --db mini.db load --sales new_sales.csv --mode replace   # re-deliver a day
python -m demandcast --db mini.db load --stores stores.csv --products products.csv --strict
```

## Bringing your own data

Only `stores`, `products` and `sales_daily` are needed to run a forecast; `calendar` rows are
filled automatically for the days you load, `revenue` defaults to `units_sold × unit_price`,
and `inventory_snapshots` / `promotions` are optional (without a snapshot the replenishment
policy assumes zero stock on hand).  The same workflow is available from Python:

```python
from demandcast import db, export, ingest, pipeline

conn = db.connect("mini.db")
db.init_schema(conn)
reports = ingest.load_dataset(conn, "examples/mini")  # list[LoadReport], FK order
run_id = pipeline.run(conn, pipeline.RunConfig(horizon_days=14, n_folds=2, workers=1))
export.export_orders(conn, run_id, "out/orders.csv")
```
