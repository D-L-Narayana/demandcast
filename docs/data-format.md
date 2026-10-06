# Data format — loading your own data

`demandcast load` ingests CSV files into the six core tables, validating every row before
anything is written.  This page is the column specification, the load semantics and the shape
of the rejection report.  The ready-made [`examples/mini`](../examples/mini) dataset follows
this format exactly and `demandcast export dataset` writes it.

## File conventions

* One CSV file per table, UTF-8 (a BOM is tolerated), comma separated, **header row required**.
  Column names are the table's column names (case-insensitive, surrounding spaces ignored);
  unknown columns are ignored with a warning, missing *required* columns reject the whole file.
* Dates are ISO `YYYY-MM-DD`.  Integers may be written as `12` or `12.0`; numbers use `.` as
  the decimal separator.  An empty field is NULL / "use the default".
* Error messages refer to **`row N` = line N of the file, the header being line 1**.
* Default file names: `stores.csv`, `products.csv`, `calendar.csv`, `promotions.csv`,
  `sales_daily.csv`, `inventory_snapshots.csv` (what `--dir` looks for and what
  `export dataset` writes).  Individual flags accept any file name:
  `--stores F --products F --calendar F --promotions F --sales F --inventory F`.

## Load order (foreign keys)

Tables are loaded in this order so that references resolve:

```
stores → products → calendar → promotions → sales_daily → inventory_snapshots
```

`sales_daily`, `inventory_snapshots` and `promotions` rows whose `store_id` / `product_id` is
not already in the database are rejected (`unknown store_id 99`).  `sales_daily.day` references
`calendar.day`, but you do not have to supply a calendar: before sales rows are written the
loader **auto-fills the missing calendar days** between the earliest and latest valid day in
the file (`day_of_week`, ISO `week_of_year`, `month`, `year`, `is_weekend` derived from the
date; `holiday_name` NULL).  Load a `calendar.csv` first if you want holiday names.

## Column specification

Required columns are marked **R**; the others may be omitted from the file or left empty.

### `stores` — key `store_id`

| column | type | rules |
|---|---|---|
| `store_id` **R** | integer ≥ 1 | primary key |
| `store_code` **R** | text | unique across stores |
| `city` **R** | text | |
| `region` **R** | text | |
| `format` **R** | enum | `flagship`, `standard` or `express` |
| `opened_on` **R** | date | |

```csv
store_id,store_code,city,region,format,opened_on
1,BLR,Bengaluru,South,flagship,2017-11-06
2,HYD,Hyderabad,South,standard,2021-05-27
```

### `products` — key `product_id`

| column | type | rules |
|---|---|---|
| `product_id` **R** | integer ≥ 1 | primary key |
| `sku` **R** | text | unique across products |
| `name` **R** | text | |
| `category` **R** | text | free text (the demo uses Grocery, Household, Beauty, Apparel, Electronics, Toys) |
| `unit_cost` **R** | number > 0 | |
| `unit_price` **R** | number ≥ `unit_cost` | drives revenue defaults and ABC classification |
| `case_pack` | integer ≥ 1 | default 1; order quantities are rounded up to it |
| `lead_time_days` **R** | integer ≥ 0 | supplier lead time |
| `shelf_life_days` | integer > 0 | empty = non-perishable |

```csv
product_id,sku,name,category,unit_cost,unit_price,case_pack,lead_time_days,shelf_life_days
1,SKU-GRO-0001,Grocery item 1,Grocery,15.53,27.34,6,2,21
2,SKU-HOU-0002,Household item 2,Household,100.99,195.67,1,5,
```

### `calendar` — key `day`

| column | type | rules |
|---|---|---|
| `day` **R** | date | primary key |
| `day_of_week` | integer | Monday = 0 … Sunday = 6; derived when empty, must match `day` when given |
| `week_of_year` | integer 1–53 | ISO week when empty; your own numbering is kept when given |
| `month` | integer | derived when empty, must match `day` when given |
| `year` | integer | derived when empty, must match `day` when given |
| `is_weekend` | 0/1 | Saturday/Sunday when empty, must match `day` when given |
| `holiday_name` | text | empty = no holiday |

```csv
day,day_of_week,week_of_year,month,year,is_weekend,holiday_name
2024-01-26,4,4,1,2024,0,Republic Day
2024-01-27,,,,,,
```

### `promotions` — key `promo_id`

| column | type | rules |
|---|---|---|
| `promo_id` **R** | integer ≥ 1 | primary key |
| `product_id` **R** | integer | must exist in `products` |
| `store_id` | integer | must exist in `stores`; **empty = chain-wide** |
| `start_day` **R** | date | may lie after the last sales day (future promotions) |
| `end_day` **R** | date | ≥ `start_day`, inclusive |
| `discount_pct` **R** | number | strictly between 0 and 1 (`0.2` = 20 % off) |

```csv
promo_id,product_id,store_id,start_day,end_day,discount_pct
1,1,,2024-02-14,2024-02-22,0.1
3,1,2,2024-02-03,2024-02-13,0.15
```

### `sales_daily` — key (`store_id`, `product_id`, `day`)

| column | type | rules |
|---|---|---|
| `store_id` **R** | integer | must exist in `stores` |
| `product_id` **R** | integer | must exist in `products` |
| `day` **R** | date | calendar rows are auto-filled |
| `units_sold` **R** | integer ≥ 0 | *observed* sales (censored by stock-outs) |
| `revenue` | number ≥ 0 | default `round(units_sold × unit_price, 2)` |
| `stockout_flag` | 0/1 | default 0; 1 = the product ran out that day |

```csv
store_id,product_id,day,units_sold,revenue,stockout_flag
1,1,2024-01-01,25,683.5,0
1,1,2024-01-02,27,,0
```

Days with no sales may be omitted: the pipeline densifies each series onto a common daily axis
(missing days count as zero units).

### `inventory_snapshots` — key (`store_id`, `product_id`, `snapshot_day`)

| column | type | rules |
|---|---|---|
| `store_id` **R** | integer | must exist in `stores` |
| `product_id` **R** | integer | must exist in `products` |
| `snapshot_day` **R** | date | the replenishment step uses the latest snapshot per series |
| `on_hand` **R** | integer ≥ 0 | |
| `on_order` | integer ≥ 0 | default 0; quantity already ordered but not received |

```csv
store_id,product_id,snapshot_day,on_hand,on_order
1,1,2024-04-29,140,0
```

## Load modes

The key of a row is the table's primary key (see the headings above).

| `--mode` | existing key in the database | new key |
|---|---|---|
| `upsert` (default) | row is updated (`INSERT … ON CONFLICT(key) DO UPDATE SET col = excluded.col`) — counted as `updated` | inserted |
| `insert` | row is rejected: `key (store_id=2) already exists (use mode 'upsert' or 'replace')` | inserted |
| `replace` | existing rows whose keys appear in the file are deleted, then the file's rows are inserted — counted as `updated`; rows in other tables that reference them are kept (foreign keys are checked when the load commits) | inserted |

Rows that are not mentioned in the file are never touched; to delete data use SQL.

Each `load_csv` call is atomic: either all valid rows of a file are written or none are.
Within one file the **first** occurrence of a key wins; later duplicates are rejected
(`duplicate key (store_id=1, product_id=1, day=2024-01-01) first seen at row 2`).  Clashes
on UNIQUE columns are rejected the same way (`store_code 'BLR' is already used by store_id 1`,
`duplicate sku 'X' first seen at row 3`).

### Strict and dry-run

* `--strict`: the whole file is validated first; if **any** row is rejected nothing is written
  and the command exits with status 1 — the printed report still lists the errors.  From Python,
  `load_csv(..., strict=True)` raises `IngestError` (a `ValueError`) whose `.report` attribute
  holds the full `LoadReport`.
* `--dry-run`: validate and report only; the database is left untouched (no calendar fill, no
  provenance row).

## Rejection report

`load_csv` returns a `LoadReport` and the CLI prints one JSON object:

```json
{
  "ok": true,
  "mode": "upsert",
  "dry_run": false,
  "strict": false,
  "loads": [
    {
      "table": "sales_daily",
      "inserted": 14,
      "updated": 0,
      "rejected": 2,
      "errors": [
        "row 16: unknown store_id 99",
        "row 19: units_sold must be >= 0 (got -2)"
      ],
      "rows_read": 16,
      "mode": "upsert",
      "source": "data/sales_daily.csv",
      "dry_run": false,
      "calendar_days_added": 14
    }
  ]
}
```

`errors` holds the first 20 messages; `rejected` always counts every rejected row.  Message
patterns:

| message | meaning |
|---|---|
| `<col> is required` | required column empty |
| `<col> must be an integer (got 'x')` / `must be a number (got 'ten')` | type error |
| `<col> must be an ISO date YYYY-MM-DD (got '01/02/2020')` | date format |
| `<col> must be 0 or 1 (got '2')` | flag columns (`stockout_flag`, `is_weekend`) |
| `format must be one of flagship, standard, express (got 'kiosk')` | enumeration |
| `<col> must be >= 0 (got -2)`, `> 0`, `> 0 and < 1`, `>= 1 and <= 53` | range checks |
| `unit_price must be >= unit_cost (…)`, `end_day must be >= start_day (…)` | cross-column rules |
| `day_of_week does not match day 2024-01-28 (expected 6, got 3)` | calendar consistency |
| `unknown store_id 99` / `unknown product_id 42` | foreign key missing |
| `duplicate key (…) first seen at row N`, `duplicate sku '…' first seen at row N` | duplicates inside the file |
| `sku '…' is already used by product_id N` | UNIQUE clash with an existing row |
| `key (…) already exists (use mode 'upsert' or 'replace')` | `--mode insert` on an existing key |

File-level problems (unknown table, unknown mode, missing required column, empty file, missing
directory) raise `ValueError` / print `error: …` and exit 1 without loading anything.

## Provenance

Every non-dry-run load appends a row to `data_loads` (`loaded_at`, `source`, `table_name`,
`mode`, `rows_inserted`, `rows_updated`, `rows_rejected`, `notes`), so you can audit what was
loaded when:

```sql
SELECT loaded_at, table_name, mode, rows_inserted, rows_updated, rows_rejected
FROM data_loads ORDER BY load_id;
```

Large files log progress every 50 000 rows (`sales_daily: 50000 rows read`, logger
`demandcast.ingest`).

## Exports

| command | contents |
|---|---|
| `export orders --out F [--run-id N] [--format csv\|json] [--cells safe\|raw]` | the `replenishment_summary` query for the run (store code, SKU, on-hand/on-order, lead-time demand, safety stock, reorder point, order quantity and cost, expected arrival, reason) plus `stockout_risk`, `priority`, `requested_qty` when the database has those columns |
| `export forecasts --out F [--cells safe\|raw]` | one row per series × target day: `run_id, store_id, product_id, target_day, model_name, yhat, yhat_lower, yhat_upper, …, store_code, sku` |
| `export metrics --out F [--cells safe\|raw]` | every backtest candidate per series (`selected = 1` marks the winner) with its MAE / WAPE / bias / MASE |
| `export dataset --out DIR` | the six tables above as `<table>.csv` — header = table columns, rows ordered by primary key, NULL as an empty field; **raw machine data** for `load --dir DIR` (no `--cells`) |

`--run-id` defaults to the latest successful run; `--format` defaults to `json` for a `.json`
output path and `csv` otherwise.  A dataset export followed by a load into a fresh database
reproduces every table row for row.

### Spreadsheet safety: the `--cells` policy

Text that arrives through CSV ingest (store codes, cities, SKUs, product names, categories)
is stored verbatim, so a value such as `=1+1`, `+1`, `-1` or `@SUM(A1:A9)` can reach the
reporting exports.  A spreadsheet importing such a CSV evaluates those cells as formulas —
CSV quoting does not prevent it.  The reporting exports therefore apply a cell policy (shared
with `query --format csv`):

| output | cell policy |
|---|---|
| `export orders\|forecasts\|metrics --format csv` (default `--cells safe`) | **spreadsheet-safe**: a text cell is written as `'` + text when its first character is a tab / CR / LF, or when its first *visible* character — skipping whitespace and invisible Unicode control, format and separator characters such as a zero-width space — is `=`, `+`, `-` or `@`.  Numbers are never touched (a bias of `-0.14` stays a negative number), dates start with a digit and are unchanged, empty cells stay empty.  Commas and newlines inside text are quoted as usual and round-trip through any CSV reader. |
| `… --cells raw` | every cell verbatim — for machine consumers that parse the file themselves.  Identical to the pre-0.4.0 output. |
| `… --format json` | never transformed: raw strings and typed numbers, whatever `--cells` says (the option only affects CSV). |
| `export dataset` | **always raw machine data** for `load --dir`: a lossless round trip of the six tables, including any hostile-looking strings.  `--cells` is rejected for it rather than silently ignored.  Treat these files as data, not reports — do not open them in a spreadsheet that evaluates formulas without checking the text columns first. |

The leading `'` is the conventional spreadsheet "treat as text" marker; some applications show
the apostrophe in the cell, others hide it.  Database values are never modified by any export
— the marker exists only in the written reporting file.  From Python the same policy is the
`cells` argument of `export_orders`, `export_forecasts` and `export_metrics`
(`cells="safe"` default, `cells="raw"` verbatim; any other value raises `ValueError`).

## Python API

```python
from demandcast import db, export, ingest

conn = db.connect("shop.db")
db.init_schema(conn)
report = ingest.load_csv(conn, "sales_daily", "data/sales.csv", mode="upsert", strict=False)
print(report.inserted, report.updated, report.rejected, report.errors[:3])
reports = ingest.load_dataset(conn, "data/")  # all <table>.csv in FK order
ingest.ensure_calendar(conn, "2024-01-01", "2024-12-31")  # pre-fill a calendar range
export.export_dataset(conn, "backup/")  # {table: rows written}; raw machine data
export.export_orders(conn, None, "out/orders.csv")  # spreadsheet-safe CSV (cells="safe")
export.export_orders(conn, None, "out/orders-raw.csv", cells="raw")  # verbatim cells
```
