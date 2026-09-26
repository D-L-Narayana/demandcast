-- Named analytical queries. Each block is separated by a `-- name: <query_name>` marker
-- and loaded by demandcast.db.load_queries(). Parameters use SQLite named style (:param).

-- name: weekly_sales_trend
-- Weekly units by product category with week-over-week change (LAG window function).
WITH weekly AS (
    SELECT p.category,
           c.year,
           c.week_of_year,
           MIN(c.day)                 AS week_start,
           SUM(s.units_sold)          AS units,
           ROUND(SUM(s.revenue), 2)   AS revenue
    FROM sales_daily s
    JOIN products p  ON p.product_id = s.product_id
    JOIN calendar c  ON c.day = s.day
    GROUP BY p.category, c.year, c.week_of_year
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
-- Average daily units during promotions vs. the non-promo baseline for the same store/product.
WITH flagged AS (
    SELECT s.store_id, s.product_id, s.day, s.units_sold,
           EXISTS (
               SELECT 1 FROM promotions pr
               WHERE pr.product_id = s.product_id
                 AND (pr.store_id IS NULL OR pr.store_id = s.store_id)
                 AND s.day BETWEEN pr.start_day AND pr.end_day
           ) AS on_promo
    FROM sales_daily s
),
agg AS (
    SELECT product_id,
           AVG(CASE WHEN on_promo THEN units_sold END)     AS promo_avg,
           AVG(CASE WHEN NOT on_promo THEN units_sold END) AS base_avg,
           SUM(on_promo)                                   AS promo_days
    FROM flagged
    GROUP BY product_id
    HAVING promo_days > 0
)
SELECT p.sku, p.name, p.category, a.promo_days,
       ROUND(a.base_avg, 2)  AS baseline_units_per_day,
       ROUND(a.promo_avg, 2) AS promo_units_per_day,
       ROUND(a.promo_avg / NULLIF(a.base_avg, 0), 2) AS lift_multiplier
FROM agg a JOIN products p ON p.product_id = a.product_id
ORDER BY lift_multiplier DESC;

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
-- Full daily history for one store/product with a 7-day trailing moving average.
SELECT day, units_sold, stockout_flag,
       ROUND(AVG(units_sold) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS ma7
FROM sales_daily
WHERE store_id = :store_id AND product_id = :product_id
ORDER BY day;

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
