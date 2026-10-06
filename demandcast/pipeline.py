"""End-to-end forecasting + replenishment job.

    load series from SQL  ->  backtest & select model per series  ->  forecast horizon
    ->  persist forecasts + metrics  ->  compute replenishment orders  ->  persist orders

Series are independent, so the compute-heavy step fans out over a process pool
(`RunConfig.workers`). Everything is written inside a single forecast_run so a failed job
never leaves partial output visible (status stays 'failed'; consumers filter on
status = 'succeeded').

A run can be *backdated* with `RunConfig.cutoff_day`: only sales up to the cutoff are used,
inventory is the latest snapshot on or before it and forecasts start the day after, so the
realised sales can later be compared against them (`demandcast evaluate`).

Promotions enter as 0/1 flags per series (history and horizon): promo-aware candidate models
compete in the same backtest and the scheduled promo days are persisted as
`forecasts.promo_flag`. Prediction intervals come from the backtest residuals of the winning
model (`IntervalModel`), replenishment carries a stock-out risk and a cost-weighted priority,
and an optional order budget defers the lowest-priority orders.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np

from . import __version__, db
from .backtest import ModelScore, normal_interval, select_model
from .db import insert_many
from .models import MODEL_REGISTRY, make_model
from .promo import PROMO_MODEL_NAMES
from .replenish import (
    BudgetItem,
    ReplenishmentDecision,
    ReplenishmentInput,
    allocate_budget,
    decide,
    service_level_for,
)

log = logging.getLogger("demandcast")

CRITERIA = ("mae", "wape", "mase")
INTERVAL_METHODS = ("empirical", "normal")
DEFERRED_REASON = "; deferred: order budget exhausted"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RunConfig:
    """Everything that parametrises a run; persisted as JSON in forecast_runs.config_json.

    Values are validated on construction (ValueError), `cutoff_day` accepts an ISO string and
    `models` any sequence of names, so a config read back from the database round-trips.
    """

    horizon_days: int = 28
    n_folds: int = 4
    min_history_days: int = 84
    review_period_days: int = 7
    service_level: float = 0.95
    interval_level: float = 0.8  # nominal coverage of the prediction interval
    interval_method: str = "empirical"  # or "normal"
    workers: int = 0  # 0 => os.cpu_count()
    cutoff_day: date | None = None  # backdated run: only sales with day <= cutoff
    models: tuple[str, ...] | None = None  # restrict the candidate models
    promo_aware: bool = True  # load promo flags and allow promo_* candidates
    criterion: str = "mae"
    order_budget: float | None = None
    service_level_overrides: Mapping[str, float] | None = None  # by ABC class

    def __post_init__(self) -> None:
        if isinstance(self.cutoff_day, str):
            object.__setattr__(self, "cutoff_day", date.fromisoformat(self.cutoff_day))
        if self.models is not None:
            object.__setattr__(self, "models", tuple(str(m) for m in self.models))
        if self.service_level_overrides is not None:
            object.__setattr__(self, "service_level_overrides", dict(self.service_level_overrides))
        _require(self.horizon_days >= 1, "horizon_days must be >= 1")
        _require(self.n_folds >= 1, "n_folds must be >= 1")
        _require(self.min_history_days >= 0, "min_history_days must be >= 0")
        _require(self.review_period_days >= 0, "review_period_days must be >= 0")
        _require(self.workers >= 0, "workers must be >= 0")
        _require(0.5 <= self.service_level < 1.0, "service_level must be in [0.5, 1.0)")
        _require(0.0 < self.interval_level < 1.0, "interval_level must be in (0, 1)")
        _require(
            self.interval_method in INTERVAL_METHODS,
            f"interval_method must be one of {INTERVAL_METHODS}",
        )
        _require(self.criterion in CRITERIA, f"criterion must be one of {CRITERIA}")
        _require(self.order_budget is None or self.order_budget >= 0, "order_budget must be >= 0")
        for abc_class, level in (self.service_level_overrides or {}).items():
            _require(
                0.5 <= float(level) < 1.0,
                f"service level for class {abc_class!r} must be in [0.5, 1.0)",
            )


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
    hist_flags: np.ndarray | None = None  # promo flags aligned with y (None: no promo days)
    future_flags: np.ndarray | None = None  # promo flags for the model_horizon steps
    abc_class: str | None = None  # drives the per-class service level when overrides exist


@dataclass
class SeriesResult:
    store_id: int
    product_id: int
    model_name: str
    yhat: np.ndarray  # model_horizon steps
    yhat_lower: np.ndarray  # cfg.horizon_days steps
    yhat_upper: np.ndarray
    residual_std: float
    scores: list[ModelScore]
    decision: ReplenishmentDecision
    service_level: float
    on_hand: int
    on_order: int
    future_flags: np.ndarray | None = None


# ---- loading -----------------------------------------------------------------------------
def _load_series(
    conn: sqlite3.Connection, cutoff_day: str
) -> tuple[dict[tuple[int, int], np.ndarray], list[str]]:
    """Return ({(store_id, product_id): units}, day_axis) densified on a common daily axis.

    The axis runs from the first sales day on or before `cutoff_day` up to the cutoff
    (inclusive). A series starts at its first observed day and is right-aligned with the axis
    (y covers axis[-len(y):]); days without a row inside that span count as 0 units, so the
    weekday positions stay aligned with the calendar even when the feed has gaps.
    """
    first = conn.execute(
        "SELECT MIN(day) AS d FROM sales_daily WHERE day <= ?", (cutoff_day,)
    ).fetchone()["d"]
    if first is None:
        return {}, []
    start = date.fromisoformat(first)
    n_days = (date.fromisoformat(cutoff_day) - start).days + 1
    axis = [(start + timedelta(days=i)).isoformat() for i in range(n_days)]
    offsets: dict[tuple[int, int], list[int]] = {}
    units: dict[tuple[int, int], list[int]] = {}
    for sid, pid, t, u in conn.execute(
        "SELECT store_id, product_id, CAST(julianday(day) - julianday(?) AS INTEGER), units_sold "
        "FROM sales_daily WHERE day <= ? ORDER BY store_id, product_id, day",
        (first, cutoff_day),
    ):
        offsets.setdefault((sid, pid), []).append(t)
        units.setdefault((sid, pid), []).append(u)
    series: dict[tuple[int, int], np.ndarray] = {}
    for key, ts in offsets.items():
        t = np.asarray(ts, dtype=int)
        y = np.zeros(n_days - int(t[0]))
        y[t - t[0]] = units[key]
        series[key] = y
    return series, axis


def _load_master(
    conn: sqlite3.Connection, cutoff_day: str
) -> tuple[dict[int, dict], dict[tuple[int, int], tuple[int, int]]]:
    """Products by id and the latest inventory snapshot on or before the cutoff per series."""
    products = {r["product_id"]: dict(r) for r in conn.execute("SELECT * FROM products")}
    inv: dict[tuple[int, int], tuple[int, int]] = {}
    for r in conn.execute(
        """
        SELECT store_id, product_id, on_hand, on_order
        FROM inventory_snapshots i
        WHERE snapshot_day = (
            SELECT MAX(snapshot_day) FROM inventory_snapshots j
            WHERE j.store_id = i.store_id AND j.product_id = i.product_id
              AND j.snapshot_day <= :cutoff)
        """,
        {"cutoff": cutoff_day},
    ):
        inv[(r["store_id"], r["product_id"])] = (r["on_hand"], r["on_order"])
    return products, inv


def load_promo_flags(
    conn: sqlite3.Connection,
    keys: Iterable[tuple[int, int]],
    hist_days: list[str],
    future_days: list[str],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Return {(store_id, product_id): (hist_flags, future_flags)} as 0/1 float arrays.

    `hist_days` is the common day axis of the loaded series and `future_days` the forecast
    steps that follow it. Promotions with store_id NULL apply to every store; a series without
    promotions gets zero arrays. A series shorter than the axis uses `hist_flags[-len(y):]`.
    """
    keys = list(keys)
    all_days = list(hist_days) + list(future_days)
    n_hist, n_total = len(hist_days), len(all_days)
    if not keys or n_total == 0:
        return {key: (np.zeros(n_hist), np.zeros(n_total - n_hist)) for key in keys}
    axis_start = date.fromisoformat(all_days[0])
    by_product: dict[int, list[tuple[int | None, int, int]]] = {}
    for r in conn.execute(
        "SELECT product_id, store_id, start_day, end_day FROM promotions "
        "WHERE end_day >= ? AND start_day <= ?",
        (all_days[0], all_days[-1]),
    ):
        lo = max(0, (date.fromisoformat(r["start_day"]) - axis_start).days)
        hi = min(n_total - 1, (date.fromisoformat(r["end_day"]) - axis_start).days)
        if lo <= hi:
            by_product.setdefault(r["product_id"], []).append((r["store_id"], lo, hi))
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for sid, pid in keys:
        flags = np.zeros(n_total)
        for store_id, lo, hi in by_product.get(pid, ()):
            if store_id is None or store_id == sid:
                flags[lo : hi + 1] = 1.0
        out[(sid, pid)] = (flags[:n_hist], flags[n_hist:])
    return out


def abc_classes_as_of(
    conn: sqlite3.Connection, cutoff_day: str, window_days: int = 90
) -> dict[int, str]:
    """ABC class per product from revenue over the window (cutoff - window_days, cutoff].

    Products are ranked by revenue; with cumulative share  c_k = Σ_{i<=k} rev_i / Σ rev  the
    class is 'A' when c_k <= 0.70, 'B' when c_k <= 0.90, else 'C'. Unlike the
    `abc_classification` named query (anchored at MAX(day)) this never sees post-cutoff rows,
    so a backdated run cannot leak future revenue into its service levels. Products without
    sales in the window are absent from the result.
    """
    rows = conn.execute(
        """
        WITH rev AS (
            SELECT product_id, SUM(revenue) AS revenue
            FROM sales_daily
            WHERE day > DATE(:cutoff, :back) AND day <= :cutoff
            GROUP BY product_id
        ),
        ranked AS (
            SELECT product_id,
                   SUM(revenue) OVER (ORDER BY revenue DESC, product_id
                                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                       / SUM(revenue) OVER () AS cum_share
            FROM rev
        )
        SELECT product_id,
               CASE WHEN cum_share <= 0.70 THEN 'A'
                    WHEN cum_share <= 0.90 THEN 'B'
                    ELSE 'C' END AS abc_class
        FROM ranked
        """,
        {"cutoff": cutoff_day, "back": f"-{int(window_days)} days"},
    )
    return {r["product_id"]: r["abc_class"] for r in rows}


# ---- per-series work (runs in worker processes) ------------------------------------------
def process_series(task: SeriesTask) -> SeriesResult:
    """Pure function: backtest, select, forecast and decide replenishment for one series.

    Promotion flags are forwarded to the backtest, the fit and the prediction only when the
    series has promo days (`task.hist_flags is not None`); otherwise every model is called
    exactly as in v0.3. The persisted interval is the winner's backtest interval
    (`IntervalModel.apply`, empirical residual quantiles or the normal ±z·σ band depending on
    `interval_method`); `normal_interval` is only a fallback for a score without one.
    """
    cfg = task.cfg
    flags = task.hist_flags
    winner, scores = select_model(
        task.y,
        cfg.horizon_days,
        promo_flags=flags,
        allowed=cfg.models,
        criterion=cfg.criterion,
        interval_level=cfg.interval_level,
        interval_method=cfg.interval_method,
        n_folds=cfg.n_folds,
    )
    for s in scores:  # drop fold arrays before pickling back to the parent
        s.fold_results = []
    model = make_model(winner.model_name)
    if flags is None:
        yhat = model.fit(task.y).predict(task.model_horizon)
    else:
        model.fit(task.y, promo_flags=flags)
        yhat = model.predict(task.model_horizon, future_flags=task.future_flags)
    yhat = np.asarray(yhat, dtype=float)
    interval = winner.interval or normal_interval(winner.residual_std, cfg.interval_level)
    lower, upper = interval.apply(yhat[: cfg.horizon_days])
    service_level = service_level_for(
        task.abc_class, cfg.service_level, cfg.service_level_overrides
    )
    dec = decide(
        ReplenishmentInput(
            on_hand=task.on_hand,
            on_order=task.on_order,
            lead_time_days=task.product["lead_time_days"],
            review_period_days=cfg.review_period_days,
            case_pack=task.product["case_pack"],
            forecast=[float(v) for v in yhat],
            residual_std=winner.residual_std,
            service_level=service_level,
            shelf_life_days=task.product["shelf_life_days"],
            unit_cost=float(task.product["unit_cost"]),
        )
    )
    return SeriesResult(
        store_id=task.store_id,
        product_id=task.product_id,
        model_name=winner.model_name,
        yhat=yhat,
        yhat_lower=lower,
        yhat_upper=upper,
        residual_std=winner.residual_std,
        scores=scores,
        decision=dec,
        service_level=service_level,
        on_hand=task.on_hand,
        on_order=task.on_order,
        future_flags=task.future_flags,
    )


# ---- run orchestration -------------------------------------------------------------------
def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_models(models: Sequence[str] | None) -> None:
    """Reject unknown names up front (ValueError lists the valid base and promo_* names)."""
    if models is None:
        return
    valid = list(MODEL_REGISTRY) + list(PROMO_MODEL_NAMES)
    _require(len(models) > 0, f"models must not be empty; valid names: {valid}")
    unknown = [m for m in models if m not in valid]
    _require(not unknown, f"unknown model name(s) {unknown}; valid names: {valid}")


def _resolve_cutoff(conn: sqlite3.Connection, requested: date | None) -> str:
    bounds = conn.execute("SELECT MIN(day) AS lo, MAX(day) AS hi FROM sales_daily").fetchone()
    if bounds["hi"] is None:
        raise RuntimeError("sales_daily is empty — generate or load data first")
    if requested is None:
        return str(bounds["hi"])
    cutoff = requested.isoformat()
    _require(
        bounds["lo"] <= cutoff <= bounds["hi"],
        f"cutoff_day {cutoff} is outside the loaded sales range {bounds['lo']}..{bounds['hi']}",
    )
    return cutoff


def _series_flags(
    pair: tuple[np.ndarray, np.ndarray] | None, n: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Slice the axis-aligned flags to the series length; None when there is no promo day."""
    if pair is None:
        return None, None
    hist, future = pair[0][-n:], pair[1]
    if not hist.any() and not future.any():
        return None, None
    return hist, future


def _collect(run_id: int, done: Iterable[SeriesResult], total: int) -> list[SeriesResult]:
    out: list[SeriesResult] = []
    for res in done:
        out.append(res)
        if len(out) % 100 == 0:
            log.debug("run %s: %d/%d series processed", run_id, len(out), total)
    return out


def _allocate(
    results: Sequence[SeriesResult], products: Mapping[int, dict], budget: float | None
) -> dict[tuple[int, int], int] | None:
    """Approved quantity per series under the order budget (None when no budget applies).

    Greedy first-fit by `priority` (cost-weighted expected shortfall) over the orders the
    policy asked for, so Σ approved order_qty · unit_cost <= budget.
    """
    if budget is None:
        return None
    items = [
        BudgetItem(
            key=(r.store_id, r.product_id),
            order_qty=r.decision.order_qty,
            unit_cost=float(products[r.product_id]["unit_cost"]),
            priority=r.decision.priority,
        )
        for r in results
        if r.decision.order_qty > 0
    ]
    return allocate_budget(items, budget)


def config_json(cfg: RunConfig) -> str:
    """Serialised run configuration (dates as ISO strings)."""
    return json.dumps(asdict(cfg), default=str)


def run(conn: sqlite3.Connection, cfg: RunConfig | None = None) -> int:
    """Execute a full run and return its run_id. Raises on failure after marking the run failed.

    The database is migrated to the current schema first (`db.ensure_schema`). Configuration
    errors (unknown model names, a cutoff outside the loaded sales) raise ValueError *before*
    the run row exists, so a typo never leaves a failed run behind.
    """
    cfg = cfg or RunConfig()
    t0 = time.perf_counter()
    db.ensure_schema(conn)
    _validate_models(cfg.models)
    cutoff = _resolve_cutoff(conn, cfg.cutoff_day)
    cutoff_date = date.fromisoformat(cutoff)

    cur = conn.execute(
        "INSERT INTO forecast_runs (started_at, cutoff_day, horizon_days, config_json, "
        "interval_level, engine_version) VALUES (?, ?, ?, ?, ?, ?)",
        (_utcnow(), cutoff, cfg.horizon_days, config_json(cfg), cfg.interval_level, __version__),
    )
    if cur.lastrowid is None:
        raise RuntimeError("could not create the forecast_runs row")
    run_id = int(cur.lastrowid)
    conn.commit()

    try:
        series, axis = _load_series(conn, cutoff)
        products, inventory = _load_master(conn, cutoff)
        # The replenishment policy needs cover for lead time + review period, so the model
        # horizon is extended if the configured one is too short; persisted forecasts stay
        # at cfg.horizon_days.
        max_lt = max((p["lead_time_days"] for p in products.values()), default=0)
        model_horizon = max(cfg.horizon_days, max_lt + cfg.review_period_days)
        future_days = [
            (cutoff_date + timedelta(days=i + 1)).isoformat() for i in range(model_horizon)
        ]
        flags = load_promo_flags(conn, list(series), axis, future_days) if cfg.promo_aware else {}
        abc = abc_classes_as_of(conn, cutoff) if cfg.service_level_overrides else {}

        tasks: list[SeriesTask] = []
        skipped = inventory_missing = 0
        for (sid, pid), y in series.items():
            if y.size < cfg.min_history_days:
                skipped += 1
                continue
            if (sid, pid) not in inventory:
                inventory_missing += 1
            on_hand, on_order = inventory.get((sid, pid), (0, 0))
            hist_flags, future_flags = _series_flags(flags.get((sid, pid)), y.size)
            tasks.append(
                SeriesTask(
                    sid,
                    pid,
                    y,
                    products[pid],
                    on_hand,
                    on_order,
                    model_horizon,
                    cfg,
                    hist_flags,
                    future_flags,
                    abc.get(pid),
                )
            )
        promo_series = sum(1 for t in tasks if t.hist_flags is not None)
        workers = cfg.workers or (os.cpu_count() or 1)
        log.info(
            "run %s: %d series (%d skipped), cutoff=%s, model_horizon=%d, workers=%d",
            run_id,
            len(tasks),
            skipped,
            cutoff,
            model_horizon,
            workers,
        )

        if workers > 1 and len(tasks) > 8:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = _collect(run_id, pool.map(process_series, tasks, chunksize=8), len(tasks))
        else:
            results = _collect(run_id, map(process_series, tasks), len(tasks))

        approved = _allocate(results, products, cfg.order_budget)
        fc_rows: list[tuple] = []
        metric_rows: list[tuple] = []
        order_rows: list[tuple] = []
        order_day = future_days[0]
        deferred = n_orders = 0
        for res in results:
            for i in range(cfg.horizon_days):
                promo_flag = int(res.future_flags[i]) if res.future_flags is not None else 0
                fc_rows.append(
                    (
                        run_id,
                        res.store_id,
                        res.product_id,
                        future_days[i],
                        res.model_name,
                        float(res.yhat[i]),
                        float(res.yhat_lower[i]),
                        float(res.yhat_upper[i]),
                        promo_flag,
                    )
                )
            for s in res.scores:
                metric_rows.append(
                    _metric_row(
                        run_id, res.store_id, res.product_id, s, s.model_name == res.model_name
                    )
                )
            prod = products[res.product_id]
            dec = res.decision
            requested, reason = dec.order_qty, dec.reason
            qty = requested
            if (
                approved is not None
                and requested > 0
                and approved.get((res.store_id, res.product_id), 0) == 0
            ):
                qty = 0
                reason += DEFERRED_REASON
                deferred += 1
            if qty > 0:
                n_orders += 1
            order_rows.append(
                (
                    run_id,
                    res.store_id,
                    res.product_id,
                    order_day,
                    (cutoff_date + timedelta(days=1 + prod["lead_time_days"])).isoformat(),
                    res.on_hand,
                    res.on_order,
                    dec.lead_time_demand,
                    dec.safety_stock,
                    dec.reorder_point,
                    dec.order_up_to,
                    qty,
                    res.service_level,
                    reason,
                    dec.stockout_risk,
                    dec.priority,
                    requested,
                )
            )

        elapsed = time.perf_counter() - t0
        notes = (
            f"skipped={skipped}; elapsed={elapsed:.1f}s; workers={workers}; orders={n_orders}; "
            f"inventory_missing={inventory_missing}; deferred={deferred}; "
            f"promo_series={promo_series}"
        )
        with db.transaction(conn):
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
                    "promo_flag",
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
                    "stockout_risk",
                    "priority",
                    "requested_qty",
                ],
                order_rows,
            )
            conn.execute(
                "UPDATE forecast_runs SET finished_at=?, series_count=?, status='succeeded', "
                "notes=? WHERE run_id=?",
                (_utcnow(), len(tasks), notes, run_id),
            )
        log.info(
            "run %s succeeded in %.1fs (%d forecasts, %d orders, %d deferred)",
            run_id,
            elapsed,
            len(fc_rows),
            n_orders,
            deferred,
        )
        return run_id
    except Exception as exc:  # re-raised after recording the failure on the run row
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        conn.execute(
            "UPDATE forecast_runs SET finished_at=?, status='failed', notes=? WHERE run_id=?",
            (_utcnow(), repr(exc)[:500], run_id),
        )
        conn.commit()
        log.exception("run %s failed", run_id)
        raise


def _metric_row(run_id: int, sid: int, pid: int, s: ModelScore, selected: bool) -> tuple:
    return (run_id, sid, pid, s.model_name, s.folds, s.mae, s.wape, s.bias, s.mase, int(selected))


def latest_successful_run(conn: sqlite3.Connection) -> int | None:
    r = conn.execute(
        "SELECT MAX(run_id) AS rid FROM forecast_runs WHERE status = 'succeeded'"
    ).fetchone()
    return r["rid"]


def list_runs(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """Newest-first summary of forecast runs; `limit <= 0` returns all of them.

    Works on any DemandCast database: a pre-0.4.0 file is migrated first (its old run rows
    then report `interval_level` NULL).
    """
    db.ensure_schema(conn)
    sql = (
        "SELECT run_id, status, started_at, finished_at, cutoff_day, horizon_days, "
        "series_count, notes, interval_level FROM forecast_runs ORDER BY run_id DESC"
    )
    params: tuple = ()
    if limit > 0:
        sql += " LIMIT ?"
        params = (int(limit),)
    return [dict(r) for r in conn.execute(sql, params)]
