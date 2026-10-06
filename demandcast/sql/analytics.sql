-- Named analytical queries. Each block is separated by a `-- name: <query_name>` marker
-- and loaded by demandcast.db.load_queries(). Parameters use SQLite named style (:param).
-- Every block starts with a comment naming the SQL technique it demonstrates.

-- name: weekly_sales_trend
-- Weekly units by product category with week-over-week change (LAG window function).
-- Weeks are ISO weeks keyed by their Monday, DATE(c.day, '-' || c.day_of_week || ' days')
-- (calendar.day_of_week is 0 = Monday), so a week that straddles New Year is one bucket
-- instead of being split by calendar year. Sales are pre-aggregated per product x day first
-- (walks idx_sales_product_day) so the joins and date arithmetic touch 10x fewer rows.
WITH daily AS (
    SELECT product_id, day, SUM(units_sold) AS units, SUM(revenue) AS revenue
    FROM sales_daily
    GROUP BY product_id, day
),
weekly AS (
    SELECT p.category,
           DATE(c.day, '-' || c.day_of_week || ' days') AS week_start,
           SUM(d.units)                                 AS units,
           ROUND(SUM(d.revenue), 2)                     AS revenue
    FROM daily d
    JOIN products p  ON p.product_id = d.product_id
    JOIN calendar c  ON c.day = d.day
    GROUP BY p.category, week_start
)
SELECT category, week_start, units, revenue,
       units - LAG(units) OVER (PARTITION BY category ORDER BY week_start)            AS wow_units_delta,
       ROUND(100.0 * (units - LAG(units) OVER (PARTITION BY category ORDER BY week_start))
             / NULLIF(LAG(units) OVER (PARTITION BY category ORDER BY week_start), 0), 1) AS wow_pct
FROM weekly
ORDER BY category, week_start;

-- name: abc_classification
-- Pareto/ABC classification of SKUs by cumulative revenue share (window SUM + RANK).
WITH rev AS (
    SELECT p.product_id, p.sku, p.name, p.category, SUM(s.revenue) AS revenue
    FROM sales_daily s JOIN products p ON p.product_id = s.product_id
    WHERE s.day > (SELECT DATE(MAX(day), '-90 days') FROM sales_daily)
    GROUP BY p.product_id
),
ranked AS (
    SELECT *,
           RANK() OVER (ORDER BY revenue DESC)                                 AS rev_rank,
           SUM(revenue) OVER (ORDER BY revenue DESC
                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
               / SUM(revenue) OVER ()                                          AS cum_share
    FROM rev
)
SELECT product_id, sku, name, category, ROUND(revenue, 2) AS revenue_90d, rev_rank,
       ROUND(cum_share, 4) AS cum_revenue_share,
       CASE WHEN cum_share <= 0.70 THEN 'A'
            WHEN cum_share <= 0.90 THEN 'B'
            ELSE 'C' END AS abc_class
FROM ranked
ORDER BY rev_rank;

-- name: stockout_rate_by_store
-- Share of store-days flagged as stock-outs in the trailing 28 days, ranked per region.
WITH recent AS (
    SELECT store_id,
           SUM(stockout_flag) * 1.0 / COUNT(*) AS stockout_rate,
           SUM(stockout_flag)                  AS stockout_days
    FROM sales_daily
    WHERE day > (SELECT DATE(MAX(day), '-28 days') FROM sales_daily)
    GROUP BY store_id
)
SELECT st.store_code, st.city, st.region, st.format,
       ROUND(100.0 * r.stockout_rate, 2) AS stockout_pct,
       r.stockout_days,
       DENSE_RANK() OVER (PARTITION BY st.region ORDER BY r.stockout_rate DESC) AS rank_in_region
FROM recent r JOIN stores st ON st.store_id = r.store_id
ORDER BY st.region, rank_in_region;

-- name: promo_lift
-- Average daily units during promotions vs. the non-promo baseline for the same product.
-- Set-based instead of a correlated EXISTS per sales row: promotions are range-joined to sales
-- through idx_sales_product_day (product_id, day BETWEEN start_day AND end_day; store_id NULL =
-- chain-wide), DISTINCT folds overlapping promotions, and the non-promo baseline is derived
-- from one GROUP BY over per-product totals: base_avg = (all units - promo units) /
-- (all days - promo days). CROSS JOIN pins promotions as the outer loop so the inner side is
-- always an index range search, whatever indexes exist on promotions.
WITH promo_sales AS (
    SELECT DISTINCT s.store_id, s.product_id, s.day, s.units_sold
    FROM promotions pr
    CROSS JOIN sales_daily s ON s.product_id = pr.product_id
                            AND s.day BETWEEN pr.start_day AND pr.end_day
                            AND (pr.store_id IS NULL OR pr.store_id = s.store_id)
),
promo_agg AS (
    SELECT product_id, SUM(units_sold) AS promo_units, COUNT(*) AS promo_days
    FROM promo_sales
    GROUP BY product_id
),
totals AS (
    SELECT product_id, SUM(units_sold) AS all_units, COUNT(*) AS all_days
    FROM sales_daily
    GROUP BY product_id
),
agg AS (
    SELECT t.product_id,
           pa.promo_days,
           pa.promo_units * 1.0 / pa.promo_days                                          AS promo_avg,
           (t.all_units - pa.promo_units) * 1.0 / NULLIF(t.all_days - pa.promo_days, 0)  AS base_avg
    FROM totals t
    JOIN promo_agg pa ON pa.product_id = t.product_id
)
SELECT p.sku, p.name, p.category, a.promo_days,
       ROUND(a.base_avg, 2)  AS baseline_units_per_day,
       ROUND(a.promo_avg, 2) AS promo_units_per_day,
       ROUND(a.promo_avg / NULLIF(a.base_avg, 0), 2) AS lift_multiplier
FROM agg a JOIN products p ON p.product_id = a.product_id
ORDER BY lift_multiplier DESC, p.sku;

-- name: forecast_accuracy_leaderboard
-- Which model won most often in the latest run, and how accurate was it on average?
SELECT model_name,
       COUNT(*)                                             AS series_won,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)   AS share_pct,
       ROUND(AVG(mae), 3)                                   AS avg_mae,
       ROUND(AVG(wape), 3)                                  AS avg_wape,
       ROUND(AVG(mase), 3)                                  AS avg_mase,
       ROUND(AVG(bias), 3)                                  AS avg_bias
FROM backtest_metrics
WHERE run_id = :run_id AND selected = 1
GROUP BY model_name
ORDER BY series_won DESC;

-- name: replenishment_summary
-- Order book for a run, joined to master data, with cost exposure per order.
SELECT r.order_id, st.store_code, st.city, p.sku, p.name, p.category,
       r.on_hand, r.on_order,
       ROUND(r.lead_time_demand, 1) AS lead_time_demand,
       ROUND(r.safety_stock, 1)     AS safety_stock,
       ROUND(r.reorder_point, 1)    AS reorder_point,
       r.order_qty,
       ROUND(r.order_qty * p.unit_cost, 2) AS order_cost,
       r.expected_arrival, r.reason
FROM replenishment_orders r
JOIN stores st   ON st.store_id = r.store_id
JOIN products p  ON p.product_id = r.product_id
WHERE r.run_id = :run_id
ORDER BY order_cost DESC;

-- name: series_history
-- Full daily history for one store/product with a 7-day trailing moving average (window frame
-- ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) and an on_promo flag from an EXISTS semi-join
-- against the promotions date range (store_id NULL = chain-wide). The WHERE clause hits the
-- primary key, so the semi-join runs once per day of a single series only.
SELECT s.day, s.units_sold, s.stockout_flag,
       ROUND(AVG(s.units_sold) OVER (ORDER BY s.day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS ma7,
       EXISTS (
           SELECT 1 FROM promotions pr
           WHERE pr.product_id = s.product_id
             AND (pr.store_id IS NULL OR pr.store_id = s.store_id)
             AND s.day BETWEEN pr.start_day AND pr.end_day
       ) AS on_promo
FROM sales_daily s
WHERE s.store_id = :store_id AND s.product_id = :product_id
ORDER BY s.day;

-- name: days_of_cover
-- Inventory health: current on-hand vs. forecast demand over the next 7 days.
WITH latest_snap AS (
    SELECT store_id, product_id, on_hand, on_order,
           ROW_NUMBER() OVER (PARTITION BY store_id, product_id ORDER BY snapshot_day DESC) AS rn
    FROM inventory_snapshots
),
next7 AS (
    SELECT store_id, product_id, SUM(yhat) AS demand_7d
    FROM forecasts
    WHERE run_id = :run_id
      AND target_day <= (SELECT DATE(cutoff_day, '+7 days') FROM forecast_runs WHERE run_id = :run_id)
    GROUP BY store_id, product_id
)
SELECT st.store_code, p.sku, p.name, ls.on_hand, ls.on_order,
       ROUND(n.demand_7d, 1) AS forecast_7d,
       ROUND(7.0 * ls.on_hand / NULLIF(n.demand_7d, 0), 1) AS days_of_cover,
       CASE WHEN n.demand_7d = 0                 THEN 'NO_DEMAND'
            WHEN ls.on_hand < 0.5 * n.demand_7d THEN 'CRITICAL'
            WHEN ls.on_hand < n.demand_7d        THEN 'LOW'
            WHEN ls.on_hand > 4 * n.demand_7d    THEN 'OVERSTOCK'
            ELSE 'OK' END AS health
FROM latest_snap ls
JOIN next7 n     ON n.store_id = ls.store_id AND n.product_id = ls.product_id
JOIN stores st   ON st.store_id = ls.store_id
JOIN products p  ON p.product_id = ls.product_id
WHERE ls.rn = 1
ORDER BY days_of_cover IS NULL, days_of_cover ASC;

-- name: forecast_vs_actual
-- Realised accuracy, row level: inner equi-join of a run's forecasts to the sales that have
-- arrived since on the series key + day (primary-key lookup per forecast row). Days without an
-- actual yet are simply absent, so the result only ever covers target_day <= MAX(sales_daily.day).
SELECT f.store_id, f.product_id, f.target_day, f.model_name,
       f.yhat, f.yhat_lower, f.yhat_upper,
       s.units_sold                      AS actual_units,
       s.stockout_flag,
       ABS(s.units_sold - f.yhat)        AS abs_error,
       CASE WHEN s.units_sold BETWEEN f.yhat_lower AND f.yhat_upper THEN 1 ELSE 0 END AS in_interval
FROM forecasts f
JOIN sales_daily s ON s.store_id = f.store_id
                  AND s.product_id = f.product_id
                  AND s.day = f.target_day
WHERE f.run_id = :run_id
ORDER BY f.store_id, f.product_id, f.target_day;

-- name: evaluation_summary
-- Realised accuracy per model for one run, from the persisted per-series evaluations.
-- Ratios are re-weighted from the stored sums (wape = SUM|e| / SUM|y|, coverage weighted by
-- n_days) rather than averaging per-series ratios; avg_mae / avg_bias are plain means.
SELECT model_name,
       COUNT(*)                                                    AS n_series,
       SUM(n_days)                                                 AS total_days,
       ROUND(AVG(mae), 3)                                          AS avg_mae,
       ROUND(SUM(abs_error_sum) / NULLIF(SUM(actual_sum), 0), 3)   AS wape,
       ROUND(AVG(bias), 3)                                         AS avg_bias,
       ROUND(SUM(coverage * n_days) / SUM(n_days), 3)              AS coverage
FROM forecast_evaluations
WHERE run_id = :run_id
GROUP BY model_name
ORDER BY n_series DESC, model_name;

-- name: evaluation_history
-- One line per evaluated run (GROUP BY run joined to forecast_runs), oldest cutoff first, so
-- realised accuracy can be charted over time. All metrics are day-weighted over every series:
-- mae = SUM|e| / SUM n_days, wape = SUM|e| / SUM|y|, bias = SUM e / SUM n_days, coverage likewise.
SELECT e.run_id, r.cutoff_day, r.horizon_days,
       COUNT(*)                                                       AS n_series,
       ROUND(SUM(e.abs_error_sum) / SUM(e.n_days), 3)                 AS mae,
       ROUND(SUM(e.abs_error_sum) / NULLIF(SUM(e.actual_sum), 0), 3)  AS wape,
       ROUND(SUM(e.bias * e.n_days) / SUM(e.n_days), 3)               AS bias,
       ROUND(SUM(e.coverage * e.n_days) / SUM(e.n_days), 3)           AS coverage,
       MAX(e.evaluated_at)                                            AS evaluated_at
FROM forecast_evaluations e
JOIN forecast_runs r ON r.run_id = e.run_id
GROUP BY e.run_id
ORDER BY r.cutoff_day, e.run_id;

-- name: forecast_bias_by_category
-- Realised bias and WAPE per product category (GROUP BY on the persisted evaluations joined
-- to products); positive avg_bias = over-forecasting. avg_wape ignores series with no sales.
SELECT p.category,
       COUNT(*)                 AS n_series,
       ROUND(AVG(e.bias), 3)    AS avg_bias,
       ROUND(AVG(e.wape), 3)    AS avg_wape
FROM forecast_evaluations e
JOIN products p ON p.product_id = e.product_id
WHERE e.run_id = :run_id
GROUP BY p.category
ORDER BY avg_bias DESC, p.category;

-- name: inventory_health_distribution
-- How many series sit in each days_of_cover health bucket (same CASE thresholds), with the
-- share per bucket from SUM(COUNT(*)) OVER (); a CASE sort key orders buckets by severity.
WITH latest_snap AS (
    SELECT store_id, product_id, on_hand,
           ROW_NUMBER() OVER (PARTITION BY store_id, product_id ORDER BY snapshot_day DESC) AS rn
    FROM inventory_snapshots
),
next7 AS (
    SELECT store_id, product_id, SUM(yhat) AS demand_7d
    FROM forecasts
    WHERE run_id = :run_id
      AND target_day <= (SELECT DATE(cutoff_day, '+7 days') FROM forecast_runs WHERE run_id = :run_id)
    GROUP BY store_id, product_id
),
health AS (
    SELECT CASE WHEN n.demand_7d = 0                 THEN 'NO_DEMAND'
                WHEN ls.on_hand < 0.5 * n.demand_7d THEN 'CRITICAL'
                WHEN ls.on_hand < n.demand_7d        THEN 'LOW'
                WHEN ls.on_hand > 4 * n.demand_7d    THEN 'OVERSTOCK'
                ELSE 'OK' END AS health
    FROM latest_snap ls
    JOIN next7 n ON n.store_id = ls.store_id AND n.product_id = ls.product_id
    WHERE ls.rn = 1
)
SELECT health,
       COUNT(*)                                            AS n_series,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)  AS share_pct
FROM health
GROUP BY health
ORDER BY CASE health WHEN 'CRITICAL' THEN 1 WHEN 'LOW' THEN 2 WHEN 'OK' THEN 3
                     WHEN 'OVERSTOCK' THEN 4 ELSE 5 END;

-- name: order_cost_by_category
-- Order book rolled up to product category (GROUP BY after the join to master data), with
-- conditional aggregates: n_orders counts positive lines, n_deferred the lines a budget cut
-- to zero (requested_qty > 0 AND order_qty = 0).
SELECT p.category,
       SUM(CASE WHEN r.order_qty > 0 THEN 1 ELSE 0 END)                          AS n_orders,
       SUM(r.order_qty)                                                           AS units,
       ROUND(SUM(r.order_qty * p.unit_cost), 2)                                   AS order_cost,
       SUM(CASE WHEN r.requested_qty > 0 AND r.order_qty = 0 THEN 1 ELSE 0 END)  AS n_deferred
FROM replenishment_orders r
JOIN products p ON p.product_id = r.product_id
WHERE r.run_id = :run_id
GROUP BY p.category
ORDER BY order_cost DESC, p.category;

-- name: forecast_rollup
-- Bottom-up hierarchy: SUM(yhat) over the horizon at chain, region and category level, stacked
-- with UNION ALL inside a CTE so the final ORDER BY can sort the levels with a CASE key.
WITH fc AS (
    SELECT store_id, product_id, yhat
    FROM forecasts
    WHERE run_id = :run_id
),
rollup AS (
    SELECT 'chain' AS level, 'ALL' AS "key", SUM(yhat) AS horizon_units
    FROM fc
    UNION ALL
    SELECT 'region', st.region, SUM(fc.yhat)
    FROM fc JOIN stores st ON st.store_id = fc.store_id
    GROUP BY st.region
    UNION ALL
    SELECT 'category', p.category, SUM(fc.yhat)
    FROM fc JOIN products p ON p.product_id = fc.product_id
    GROUP BY p.category
)
SELECT level, "key", ROUND(horizon_units, 1) AS horizon_units
FROM rollup
ORDER BY CASE level WHEN 'chain' THEN 0 WHEN 'region' THEN 1 ELSE 2 END, "key";

-- name: run_history
-- Audit trail of every run, newest first: a plain projection of the append-only run table
-- (interval_level and notes carry the configuration a run was produced with).
SELECT run_id, status, started_at, finished_at, cutoff_day, horizon_days, series_count,
       interval_level, notes
FROM forecast_runs
ORDER BY run_id DESC;

-- name: stockout_risk_top
-- The ten riskiest order lines of a run: ORDER BY priority DESC with NULLs last through the
-- portable `priority IS NULL` sort key (rows without a risk score sink to the end), LIMIT 10.
SELECT st.store_code, p.sku, p.name, p.category, r.on_hand, r.on_order,
       ROUND(r.stockout_risk, 3) AS stockout_risk,
       ROUND(r.priority, 2)      AS priority,
       r.order_qty, r.expected_arrival
FROM replenishment_orders r
JOIN stores st   ON st.store_id = r.store_id
JOIN products p  ON p.product_id = r.product_id
WHERE r.run_id = :run_id
ORDER BY r.priority IS NULL, r.priority DESC, r.stockout_risk DESC, st.store_code, p.sku
LIMIT 10;

-- name: promo_calendar_upcoming
-- Promotions that overlap a run's forecast window: an interval-overlap test against the run row
-- (end_day > cutoff_day AND start_day <= cutoff_day + horizon_days), LEFT JOIN to stores so
-- chain-wide promotions (store_id NULL) show as 'ALL' via COALESCE.
SELECT pr.promo_id, p.sku,
       COALESCE(st.store_code, 'ALL') AS store_code,
       pr.start_day, pr.end_day, pr.discount_pct
FROM promotions pr
JOIN forecast_runs r ON r.run_id = :run_id
JOIN products p      ON p.product_id = pr.product_id
LEFT JOIN stores st  ON st.store_id = pr.store_id
WHERE pr.end_day > r.cutoff_day
  AND pr.start_day <= DATE(r.cutoff_day, '+' || r.horizon_days || ' days')
ORDER BY pr.start_day, pr.promo_id;
