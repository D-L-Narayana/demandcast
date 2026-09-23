"""End-to-end forecasting + replenishment job.

    load series from SQL  ->  backtest & select model per series  ->  forecast horizon
    ->  persist forecasts + metrics  ->  compute replenishment orders  ->  persist orders

Everything is written inside a single forecast_run so a failed job never leaves partial
output visible (status stays 'failed'; consumers filter on status = 'succeeded').
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .backtest import ModelScore, select_model
from .db import insert_many
from .models import make_model
from .replenish import ReplenishmentInput, decide

log = logging.getLogger("demandcast")


@dataclass(frozen=True)
class RunConfig:
    horizon_days: int = 28
    n_folds: int = 4
    min_history_days: int = 84
    review_period_days: int = 7
    service_level: float = 0.95
    interval_z: float = 1.2816  # 80 % prediction interval


@dataclass(frozen=True)
class SeriesTask:
    store_id: int
    product_id: int
    y: np.ndarray
    product: dict
    on_hand: int
    on_order: int
    model_horizon: int
    cfg: RunConfig


@dataclass
class SeriesResult:
    store_id: int
    product_id: int
    model_name: str
    yhat: np.ndarray
    residual_std: float
    scores: list[ModelScore]
    decision_fields: tuple


def _load_series(conn: sqlite3.Connection) -> dict[tuple[int, int], np.ndarray]:
    """Return {(store_id, product_id): units} with days sorted ascending."""
    rows = conn.execute(
        "SELECT store_id, product_id, day, units_sold FROM sales_daily "
        "ORDER BY store_id, product_id, day"
    ).fetchall()
    series: dict[tuple[int, int], list[int]] = {}
    for r in rows:
        series.setdefault((r["store_id"], r["product_id"]), []).append(r["units_sold"])
    return {k: np.asarray(v, dtype=float) for k, v in series.items()}


def _load_master(conn: sqlite3.Connection):
    products = {r["product_id"]: dict(r) for r in conn.execute("SELECT * FROM products")}
    inv = {}
    for r in conn.execute(
        """
        SELECT store_id, product_id, on_hand, on_order
        FROM inventory_snapshots i
        WHERE snapshot_day = (
            SELECT MAX(snapshot_day) FROM inventory_snapshots j
            WHERE j.store_id = i.store_id AND j.product_id = i.product_id)
        """
    ):
        inv[(r["store_id"], r["product_id"])] = (r["on_hand"], r["on_order"])
    return products, inv


def process_series(task: SeriesTask) -> SeriesResult:
    """Pure function: backtest, select, forecast and decide replenishment for one series."""
    cfg = task.cfg
    winner, scores = select_model(task.y, cfg.horizon_days, n_folds=cfg.n_folds)
    for s in scores:  # drop fold arrays before pickling back to the parent
        s.fold_results = []
    model = make_model(winner.model_name).fit(task.y)
    yhat = model.predict(task.model_horizon)
    dec = decide(
        ReplenishmentInput(
            on_hand=task.on_hand,
            on_order=task.on_order,
            lead_time_days=task.product["lead_time_days"],
            review_period_days=cfg.review_period_days,
            case_pack=task.product["case_pack"],
            forecast=[float(v) for v in yhat],
            residual_std=winner.residual_std,
            service_level=cfg.service_level,
            shelf_life_days=task.product["shelf_life_days"],
        )
    )
    return SeriesResult(
        task.store_id,
        task.product_id,
        winner.model_name,
        yhat,
        winner.residual_std,
        scores,
        (
            dec.lead_time_demand,
            dec.safety_stock,
            dec.reorder_point,
            dec.order_up_to,
            dec.order_qty,
            dec.reason,
        ),
    )


def run(conn: sqlite3.Connection, cfg: RunConfig | None = None) -> int:
    """Execute a full run and return its run_id. Raises on failure after marking the run failed."""
    cfg = cfg or RunConfig()
    t0 = time.perf_counter()
    cutoff = conn.execute("SELECT MAX(day) AS d FROM sales_daily").fetchone()["d"]
    if cutoff is None:
        raise RuntimeError("sales_daily is empty — generate or load data first")
    cutoff_day = date.fromisoformat(cutoff)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cur = conn.execute(
        "INSERT INTO forecast_runs (started_at, cutoff_day, horizon_days) VALUES (?, ?, ?)",
        (now, cutoff, cfg.horizon_days),
    )
    run_id = int(cur.lastrowid)
    conn.commit()

    try:
        series = _load_series(conn)
        products, inventory = _load_master(conn)
        # The replenishment policy needs cover for lead time + review period, so the model
        # horizon is extended if the configured one is too short; persisted forecasts stay
        # at cfg.horizon_days.
        max_lt = max((p["lead_time_days"] for p in products.values()), default=0)
        model_horizon = max(cfg.horizon_days, max_lt + cfg.review_period_days)

        tasks, skipped = [], 0
        for (sid, pid), y in series.items():
            if y.size < cfg.min_history_days:
                skipped += 1
                continue
            on_hand, on_order = inventory.get((sid, pid), (0, 0))
            tasks.append(
                SeriesTask(sid, pid, y, products[pid], on_hand, on_order, model_horizon, cfg)
            )
        log.info(
            "run %s: %d series (%d skipped), cutoff=%s, model_horizon=%d, workers=%d",
            run_id,
            len(tasks),
            skipped,
            cutoff,
            model_horizon,
            workers,
        )

        results = [process_series(t) for t in tasks]

        fc_rows, metric_rows, order_rows = [], [], []
        order_day = (cutoff_day + timedelta(days=1)).isoformat()
        for res in results:
            half_width = cfg.interval_z * res.residual_std
            for i in range(cfg.horizon_days):
                target = (cutoff_day + timedelta(days=i + 1)).isoformat()
                fc_rows.append(
                    (
                        run_id,
                        res.store_id,
                        res.product_id,
                        target,
                        res.model_name,
                        float(res.yhat[i]),
                        float(max(res.yhat[i] - half_width, 0.0)),
                        float(res.yhat[i] + half_width),
                    )
                )
            for s in res.scores:
                metric_rows.append(
                    _metric_row(
                        run_id, res.store_id, res.product_id, s, s.model_name == res.model_name
                    )
                )
            prod = products[res.product_id]
            on_hand, on_order = inventory.get((res.store_id, res.product_id), (0, 0))
            ltd, ss, rop, out, qty, reason = res.decision_fields
            order_rows.append(
                (
                    run_id,
                    res.store_id,
                    res.product_id,
                    order_day,
                    (cutoff_day + timedelta(days=1 + prod["lead_time_days"])).isoformat(),
                    on_hand,
                    on_order,
                    ltd,
                    ss,
                    rop,
                    out,
                    qty,
                    cfg.service_level,
                    reason,
                )
            )

        conn.execute("BEGIN")
        insert_many(
            conn,
            "forecasts",
            [
                "run_id",
                "store_id",
                "product_id",
                "target_day",
                "model_name",
                "yhat",
                "yhat_lower",
                "yhat_upper",
            ],
            fc_rows,
        )
        insert_many(
            conn,
            "backtest_metrics",
            [
                "run_id",
                "store_id",
                "product_id",
                "model_name",
                "folds",
                "mae",
                "wape",
                "bias",
                "mase",
                "selected",
            ],
            metric_rows,
        )
        insert_many(
            conn,
            "replenishment_orders",
            [
                "run_id",
                "store_id",
                "product_id",
                "order_day",
                "expected_arrival",
                "on_hand",
                "on_order",
                "lead_time_demand",
                "safety_stock",
                "reorder_point",
                "order_up_to",
                "order_qty",
                "service_level",
                "reason",
            ],
            order_rows,
        )
        elapsed = time.perf_counter() - t0
        n_orders = sum(1 for o in order_rows if o[11] > 0)
        conn.execute(
            "UPDATE forecast_runs SET finished_at=?, series_count=?, status='succeeded', notes=? "
            "WHERE run_id=?",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                len(tasks),
                f"skipped={skipped}; elapsed={elapsed:.1f}s; orders={n_orders}",
                run_id,
            ),
        )
        conn.execute("COMMIT")
        log.info(
            "run %s succeeded in %.1fs (%d forecasts, %d orders)",
            run_id,
            elapsed,
            len(fc_rows),
            n_orders,
        )
        return run_id
    except Exception as exc:  # noqa: BLE001 - we re-raise after recording failure
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        conn.execute(
            "UPDATE forecast_runs SET finished_at=?, status='failed', notes=? WHERE run_id=?",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), repr(exc)[:500], run_id),
        )
        conn.commit()
        log.exception("run %s failed", run_id)
        raise


def _metric_row(run_id: int, sid: int, pid: int, s: ModelScore, selected: bool):
    return (run_id, sid, pid, s.model_name, s.folds, s.mae, s.wape, s.bias, s.mase, int(selected))


def latest_successful_run(conn: sqlite3.Connection) -> int | None:
    r = conn.execute(
        "SELECT MAX(run_id) AS rid FROM forecast_runs WHERE status = 'succeeded'"
    ).fetchone()
    return r["rid"]
