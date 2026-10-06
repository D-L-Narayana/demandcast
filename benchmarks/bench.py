#!/usr/bin/env python3
"""Benchmark DemandCast's forecasting stack and a small end-to-end run.

All measurements use seeded synthetic data, so the numbers are reproducible on one machine and
comparable between machines only in relative terms (speed-ups, shares).

Sections of the JSON report
---------------------------
``fit_predict``
    Fit + 28-day predict wall time per registered model (``MODEL_REGISTRY`` plus the
    ``promo_*`` wrappers when ``demandcast.promo`` is importable) on one ``--n``-day series
    (default 730, the history length of the default dataset).
``holt_winters``
    The current ``HoltWinters.fit`` against a retained *scalar* reference: the v0.3.0
    per-candidate recursion (``HoltWinters._run`` and its 12-point grid loop), copied into
    this file. Reports the speed-up ratio and whether both reach the same best in-sample SSE.
``select_model``
    Rolling-origin selection per series (``--series`` series, 4 folds, 28-day horizon) - the
    dominant cost of ``demandcast run`` - with the winner histogram.
``full_run``
    Generate a 3 x 8 x 300 dataset in memory and execute ``pipeline.run`` with ``--workers``
    processes (default 2), i.e. the same shape as the test fixture and the CI smoke run.

The default 10 x 40 x 730 dataset is never generated here. ``--quick`` lowers the repetition
and series counts for CI; ``--out PATH`` also writes the report to a file (``benchmarks/results/``
is git-ignored).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

import demandcast
from demandcast import db
from demandcast.backtest import select_model
from demandcast.models import MODEL_REGISTRY, HoltWinters, make_model
from demandcast.pipeline import RunConfig, run
from demandcast.simulate import SimConfig, generate

SEASON = 7
HORIZON = 28


@dataclass(frozen=True)
class _RefParams:
    """Mirror of the v0.3.0 ``HWParams`` so the reference recursion reads identically."""

    alpha: float
    beta: float
    gamma: float
    phi: float


#: The v0.3.0 grid: alpha x beta x gamma with a fixed damping of 0.95 (12 candidates).
HW_GRID: list[_RefParams] = [
    _RefParams(a, b, g, phi)
    for a, b, g, phi in product([0.15, 0.3, 0.5], [0.02, 0.1], [0.1, 0.3], [0.95])
]


# ---- synthetic data -------------------------------------------------------------------------
def synthetic_series(
    n: int, seed: int, intermittent: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Daily units with weekday profile, yearly cycle, drift, promo lift and Poisson noise.

    Returns ``(y, promo_flags)``; ``promo_flags`` marks week-long promotions (about 10 % of
    days) that lift demand by 50 %. ``intermittent=True`` zeroes 70 % of the days.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    profile = np.array([0.9, 0.85, 0.88, 0.95, 1.05, 1.3, 1.25])
    flags = np.zeros(n)
    for start in rng.integers(0, max(n - 7, 1), size=max(n // 70, 1)):
        flags[start : start + 7] = 1.0
    lam = (
        12.0
        * profile[t % SEASON]
        * (1 + 0.1 * np.sin(2 * np.pi * t / 365.25))
        * np.exp(0.0003 * t)
        * np.where(flags > 0, 1.5, 1.0)
    )
    y = rng.poisson(lam).astype(float)
    if intermittent:
        y = np.where(rng.random(n) < 0.7, 0.0, y)
    return y, flags


# ---- retained scalar Holt-Winters reference (v0.3.0 `HoltWinters._run` + grid loop) ---------
def _scalar_hw_run(
    y: np.ndarray, p: _RefParams, m: int = SEASON
) -> tuple[float, float, np.ndarray, float, int]:
    """The v0.3.0 per-candidate recursion, copied verbatim (one Python loop per candidate).

    level_t = alpha (y_t - s_{t-m}) + (1 - alpha)(level_{t-1} + phi trend_{t-1})
    trend_t = beta (level_t - level_{t-1}) + (1 - beta) phi trend_{t-1}
    s_t     = gamma (y_t - level_t) + (1 - gamma) s_{t-m}
    """
    n = y.size
    # initial states from the first two seasons (or whatever is available)
    k = min(2, n // m) if n >= m else 0
    if k >= 1:
        season_means = [y[i * m : (i + 1) * m].mean() for i in range(k)]
        level = season_means[0]
        trend = (season_means[-1] - season_means[0]) / (m * (k - 1)) if k > 1 else 0.0
        seasonal = np.array([y[i] - level for i in range(m)])
    else:
        level, trend, seasonal = y.mean(), 0.0, np.zeros(m)
    sse = 0.0
    for t in range(n):
        s_idx = t % m
        fitted = level + p.phi * trend + seasonal[s_idx]
        err = y[t] - fitted
        sse += err * err
        new_level = p.alpha * (y[t] - seasonal[s_idx]) + (1 - p.alpha) * (level + p.phi * trend)
        new_trend = p.beta * (new_level - level) + (1 - p.beta) * p.phi * trend
        seasonal[s_idx] = p.gamma * (y[t] - new_level) + (1 - p.gamma) * seasonal[s_idx]
        level, trend = new_level, new_trend
    return float(level), float(trend), seasonal, float(sse), n


def scalar_hw_fit(y: np.ndarray) -> tuple[float, _RefParams]:
    """Grid search exactly like the v0.3.0 ``fit``: lowest in-sample one-step SSE wins."""
    best_sse, best_params = float("inf"), HW_GRID[0]
    for params in HW_GRID:
        sse = _scalar_hw_run(y, params)[3]
        if sse < best_sse:
            best_sse, best_params = sse, params
    return best_sse, best_params


# ---- timing helpers -------------------------------------------------------------------------
def timeit(fn: Callable[[], Any], reps: int) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "median_s": round(statistics.median(samples), 6),
        "min_s": round(min(samples), 6),
        "reps": reps,
    }


def _promo_model_names() -> list[str]:
    try:
        promo = importlib.import_module("demandcast.promo")
    except ImportError:
        return []
    return [str(n) for n in getattr(promo, "PROMO_MODEL_NAMES", ())]


def _fit_predict_closure(
    name: str, y: np.ndarray, flags: np.ndarray, future_flags: np.ndarray
) -> Callable[[], Any]:
    if name.startswith("promo_"):

        def fit_predict_promo() -> Any:
            model: Any = make_model(name)  # flag keywords exist from the 0.4.0 interface on
            model.fit(y, promo_flags=flags)
            return model.predict(HORIZON, future_flags=future_flags)

        return fit_predict_promo

    def fit_predict() -> Any:
        return make_model(name).fit(y).predict(HORIZON)

    return fit_predict


# ---- benchmark sections ---------------------------------------------------------------------
def bench_models(n: int, reps: int, seed: int) -> dict[str, Any]:
    y, flags = synthetic_series(n, seed)
    y_sparse, _ = synthetic_series(n, seed + 1, intermittent=True)
    future_flags = np.zeros(HORIZON)
    future_flags[7:14] = 1.0
    out: dict[str, Any] = {}
    for name in [*MODEL_REGISTRY, *_promo_model_names()]:
        series = y_sparse if name == "croston_sba" else y
        fn = _fit_predict_closure(name, series, flags, future_flags)
        try:
            fn()  # warm-up and capability probe (older model signatures raise TypeError)
        except TypeError as exc:
            out[name] = {"skipped": f"unsupported signature: {exc}"}
            continue
        out[name] = timeit(fn, reps)
    return out


def bench_holt_winters(n: int, reps: int, seed: int) -> dict[str, Any]:
    y, _ = synthetic_series(n, seed)
    reference = timeit(lambda: scalar_hw_fit(y), reps)
    current = timeit(lambda: HoltWinters().fit(y), reps)
    ref_sse, ref_params = scalar_hw_fit(y)
    model = HoltWinters().fit(y)
    cur_sse = float(getattr(model, "sse_", float("nan")))
    return {
        "n": n,
        "grid_candidates": len(HW_GRID),
        "scalar_reference": reference,
        "current": current,
        "speedup": round(reference["median_s"] / max(current["median_s"], 1e-12), 2),
        "reference_sse": round(ref_sse, 6),
        "current_sse": round(cur_sse, 6),
        "sse_match": bool(abs(cur_sse - ref_sse) <= 1e-6 * max(1.0, abs(ref_sse))),
        "reference_params": asdict(ref_params),
    }


def bench_selection(n: int, n_series: int, seed: int) -> dict[str, Any]:
    per_series: list[float] = []
    winners: dict[str, int] = {}
    n_candidates = 0
    for i in range(n_series):
        y, _ = synthetic_series(n, seed + 100 + i, intermittent=(i % 5 == 4))
        t0 = time.perf_counter()
        winner, scores = select_model(y, HORIZON, n_folds=4)
        per_series.append(time.perf_counter() - t0)
        winners[winner.model_name] = winners.get(winner.model_name, 0) + 1
        n_candidates = max(n_candidates, len(scores))
    return {
        "series": n_series,
        "n": n,
        "horizon": HORIZON,
        "folds": 4,
        "max_candidates": n_candidates,
        "mean_s_per_series": round(statistics.fmean(per_series), 6),
        "max_s_per_series": round(max(per_series), 6),
        "total_s": round(sum(per_series), 3),
        "winners": dict(sorted(winners.items())),
    }


def bench_full_run(workers: int, seed: int) -> dict[str, Any]:
    conn = db.connect(":memory:")
    db.init_schema(conn)
    t0 = time.perf_counter()
    counts = generate(conn, SimConfig(3, 8, date(2024, 1, 1), 300, seed))
    generate_s = time.perf_counter() - t0
    cfg = RunConfig(horizon_days=14, n_folds=2, workers=workers)
    t0 = time.perf_counter()
    run_id = run(conn, cfg)
    run_s = time.perf_counter() - t0
    row = conn.execute(
        "SELECT series_count, notes FROM forecast_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    n_forecasts = conn.execute(
        "SELECT COUNT(*) FROM forecasts WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    n_orders = conn.execute(
        "SELECT COUNT(*) FROM replenishment_orders WHERE run_id = ? AND order_qty > 0", (run_id,)
    ).fetchone()[0]
    return {
        "dataset": {"stores": 3, "products": 8, "days": 300, "sales_rows": counts["sales_daily"]},
        "workers": workers,
        "horizon_days": cfg.horizon_days,
        "n_folds": cfg.n_folds,
        "generate_s": round(generate_s, 3),
        "run_s": round(run_s, 3),
        "series_count": row["series_count"],
        "forecasts": n_forecasts,
        "orders": n_orders,
        "notes": row["notes"],
    }


# ---- CLI ------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--quick", action="store_true", help="fewer repetitions and series (CI)")
    p.add_argument("--out", help="also write the JSON report to this path")
    p.add_argument("--n", type=int, default=730, help="series length in days (default 730)")
    p.add_argument("--series", type=int, help="series for the selection benchmark (20, quick 5)")
    p.add_argument("--reps", type=int, help="timing repetitions per model (10, quick 3)")
    p.add_argument("--workers", type=int, default=2, help="process-pool size for the full run")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reps = args.reps or (3 if args.quick else 10)
    n_series = args.series or (5 if args.quick else 20)
    report: dict[str, Any] = {
        "meta": {
            "demandcast": demandcast.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "quick": args.quick,
            "n": args.n,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    }
    t0 = time.perf_counter()
    report["fit_predict"] = bench_models(args.n, reps, args.seed)
    report["holt_winters"] = bench_holt_winters(args.n, reps, args.seed)
    report["select_model"] = bench_selection(args.n, n_series, args.seed)
    report["full_run"] = bench_full_run(args.workers, args.seed)
    report["meta"]["total_s"] = round(time.perf_counter() - t0, 3)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
