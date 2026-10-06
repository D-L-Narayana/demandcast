"""Realised-accuracy evaluation of a forecast run against the sales that arrived afterwards.

A backdated run (``run --cutoff D``) forecasts the days D+1 … D+H. Once sales for those days
exist in ``sales_daily`` the forecast can be scored on what actually happened — the backtest only
estimated it. With e_t = yhat_t − y_t for every realised day t of a series:

    MAE      = Σ|e_t| / n_days
    WAPE     = Σ|e_t| / Σ|y_t|                            (None when Σ|y_t| = 0)
    bias     = Σ e_t  / n_days                             (positive => over-forecasting)
    coverage = #{t : yhat_lower_t <= y_t <= yhat_upper_t} / n_days

Aggregates over series are day-weighted: MAE and bias are means over all realised days, WAPE is
Σ|e| / Σ|y| over everything and coverage is weighted by n_days, i.e. the share of all realised
days that fell inside their interval (nominally the run's interval level).

Per-series results are persisted into ``forecast_evaluations`` with DELETE + INSERT per run, so
re-evaluating a run after more sales arrive replaces the previous assessment instead of
duplicating it. The row-level join lives in the named query ``forecast_vs_actual``; the per-model
breakdown is the named query ``evaluation_summary``, so SQL and Python share one definition.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from . import db
from .pipeline import latest_successful_run


class NoActualsError(RuntimeError):
    """No sales rows exist after the run's cutoff day, so nothing can be evaluated yet."""


@dataclass
class SeriesEvaluation:
    """Realised accuracy of one store x product series over its evaluated days."""

    store_id: int
    product_id: int
    model_name: str
    n_days: int
    mae: float
    wape: float | None
    bias: float
    coverage: float
    abs_error_sum: float
    actual_sum: float
    stockout_days: int


@dataclass
class EvaluationSummary:
    """Day-weighted accuracy of a whole run plus the per-model breakdown (`evaluation_summary`)."""

    run_id: int
    cutoff_day: str
    horizon_days: int
    n_series: int
    n_days_available: int
    mae: float
    wape: float | None
    bias: float
    coverage: float
    by_model: list[dict[str, Any]] = field(default_factory=list)
    interval_level: float | None = None  # nominal coverage of the run's intervals, when recorded


EVALUATION_COLUMNS = (
    "run_id",
    "store_id",
    "product_id",
    "model_name",
    "n_days",
    "mae",
    "wape",
    "bias",
    "coverage",
    "abs_error_sum",
    "actual_sum",
    "stockout_days",
    "evaluated_at",
)


def series_evaluations(rows: Iterable[Mapping[str, Any]]) -> list[SeriesEvaluation]:
    """Score every (store_id, product_id) series found in `forecast_vs_actual`-shaped rows.

    Each row needs store_id, product_id, model_name, yhat, yhat_lower, yhat_upper, actual_units
    and stockout_flag. The metrics are recomputed from those raw fields (not read from the
    query's abs_error / in_interval columns) so the SQL and Python definitions check each other.
    Series are returned in first-seen order.
    """
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["store_id"]), int(row["product_id"])), []).append(row)

    out: list[SeriesEvaluation] = []
    for (store_id, product_id), group in groups.items():
        n = len(group)
        err_sum = abs_err_sum = actual_sum = 0.0
        inside = stockouts = 0
        for row in group:
            y = float(row["actual_units"])
            e = float(row["yhat"]) - y
            err_sum += e
            abs_err_sum += abs(e)
            actual_sum += abs(y)
            inside += int(float(row["yhat_lower"]) <= y <= float(row["yhat_upper"]))
            stockouts += int(row["stockout_flag"] or 0)
        out.append(
            SeriesEvaluation(
                store_id=store_id,
                product_id=product_id,
                model_name=str(group[0]["model_name"]),
                n_days=n,
                mae=abs_err_sum / n,
                wape=abs_err_sum / actual_sum if actual_sum > 0 else None,
                bias=err_sum / n,
                coverage=inside / n,
                abs_error_sum=abs_err_sum,
                actual_sum=actual_sum,
                stockout_days=stockouts,
            )
        )
    return out


def evaluate_run(conn: sqlite3.Connection, run_id: int | None = None) -> EvaluationSummary:
    """Evaluate a succeeded run against every realised day, persist per-series rows, summarise.

    `run_id` defaults to the latest succeeded run. Raises NoActualsError when sales_daily has no
    rows after the run's cutoff (forecast target days beyond MAX(sales_daily.day) cannot be
    scored) and RuntimeError for unknown or unsuccessful runs.
    """
    ensure_schema = getattr(db, "ensure_schema", None)  # schema-v2 migration hook when available
    if ensure_schema is not None:
        ensure_schema(conn)
    if run_id is None:
        run_id = latest_successful_run(conn)
        if run_id is None:
            raise RuntimeError("no successful forecast run found — run the pipeline first")
    row = conn.execute("SELECT * FROM forecast_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"run {run_id} does not exist")
    run = dict(row)  # a plain dict so `in` tests column names (sqlite3.Row iterates values)
    if run["status"] != "succeeded":
        raise RuntimeError(
            f"run {run_id} has status '{run['status']}'; only succeeded runs can be evaluated"
        )
    cutoff, horizon = str(run["cutoff_day"]), int(run["horizon_days"])

    rows = db.run_query(conn, "forecast_vs_actual", {"run_id": run_id})
    if not rows:
        data_end = conn.execute("SELECT MAX(day) FROM sales_daily").fetchone()[0]
        raise NoActualsError(_no_actuals_message(run_id, cutoff, horizon, data_end))

    evals = series_evaluations(rows)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _persist(conn, run_id, evals, stamp)

    total_days = sum(e.n_days for e in evals)
    abs_err = sum(e.abs_error_sum for e in evals)
    actual = sum(e.actual_sum for e in evals)
    return EvaluationSummary(
        run_id=run_id,
        cutoff_day=cutoff,
        horizon_days=horizon,
        n_series=len(evals),
        n_days_available=len({r["target_day"] for r in rows}),
        mae=abs_err / total_days,
        wape=abs_err / actual if actual > 0 else None,
        bias=sum(e.bias * e.n_days for e in evals) / total_days,
        coverage=sum(e.coverage * e.n_days for e in evals) / total_days,
        by_model=db.run_query(conn, "evaluation_summary", {"run_id": run_id}),
        interval_level=run.get("interval_level"),  # None on a v1 database without the column
    )


def _no_actuals_message(run_id: int, cutoff: str, horizon: int, data_end: str | None) -> str:
    start = date.fromisoformat(cutoff)
    first, last = start + timedelta(days=1), start + timedelta(days=horizon)
    if data_end is None:
        tail = "sales_daily is empty"
        suggested = cutoff
    else:
        tail = f"sales_daily ends on {data_end}"
        suggested = (date.fromisoformat(data_end) - timedelta(days=horizon)).isoformat()
    return (
        f"run {run_id} forecasts {first.isoformat()} to {last.isoformat()} but {tail}: "
        "no realised days to evaluate yet. Backdate a run with `run --cutoff "
        f"{suggested}` (cutoff <= last sales day - horizon) or load newer sales "
        "(`load --sales FILE`) and evaluate again."
    )


def _persist(
    conn: sqlite3.Connection, run_id: int, evals: list[SeriesEvaluation], stamp: str
) -> None:
    """Replace the run's rows in forecast_evaluations (idempotent DELETE + INSERT)."""
    rows = [
        (
            run_id,
            e.store_id,
            e.product_id,
            e.model_name,
            e.n_days,
            e.mae,
            e.wape,
            e.bias,
            e.coverage,
            e.abs_error_sum,
            e.actual_sum,
            e.stockout_days,
            stamp,
        )
        for e in evals
    ]

    def write() -> None:
        conn.execute("DELETE FROM forecast_evaluations WHERE run_id = ?", (run_id,))
        db.insert_many(conn, "forecast_evaluations", list(EVALUATION_COLUMNS), rows)

    if conn.in_transaction:  # the caller owns the transaction; write inside it
        write()
    else:
        with db.transaction(conn):
            write()


# ---- CLI plugin ---------------------------------------------------------------------------------
def _cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_summary(summary: EvaluationSummary) -> str:
    """Human-readable report: headline metrics plus the per-model table."""

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.1f}%"

    nominal = ""
    if summary.interval_level is not None:
        nominal = f" (nominal {100 * summary.interval_level:.0f}%)"
    lines = [
        f"Realised accuracy of run #{summary.run_id}: cutoff {summary.cutoff_day}, "
        f"horizon {summary.horizon_days} d, {summary.n_series} series, "
        f"{summary.n_days_available} realised days",
        f"MAE {summary.mae:.3f}  WAPE {pct(summary.wape)}  bias {summary.bias:+.3f}  "
        f"interval coverage {pct(summary.coverage)}{nominal}",
    ]
    if summary.by_model:
        cols = list(summary.by_model[0])
        widths = {c: max(len(c), *(len(_cell(r[c])) for r in summary.by_model)) for c in cols}
        lines.append("  ".join(c.ljust(widths[c]) for c in cols))
        lines.append("  ".join("-" * widths[c] for c in cols))
        lines.extend(
            "  ".join(_cell(r[c]).ljust(widths[c]) for c in cols) for r in summary.by_model
        )
    return "\n".join(lines)


def evaluate_command(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    """`evaluate` handler: 0 on success; 1 with `error: ...` on stderr when nothing is realised."""
    try:
        summary = evaluate_run(conn, getattr(args, "run_id", None))
    except NoActualsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(asdict(summary), indent=2, default=str))
    else:
        print(format_summary(summary))
    return 0


def register_cli(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `evaluate [--run-id N] [--json]` on the CLI's sub-parser action."""
    parser = sub.add_parser(
        "evaluate", help="measure realised forecast accuracy of a run against actual sales"
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        dest="run_id",
        help="run to evaluate (default: the latest successful run)",
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.set_defaults(handler=evaluate_command, creates_db=False)
