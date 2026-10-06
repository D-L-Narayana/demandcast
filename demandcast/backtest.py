"""Rolling-origin backtesting, accuracy metrics, prediction intervals and model selection.

For every series the candidate models (``candidate_models``) are backtested on rolling-origin
folds (``rolling_origin``), scored (``score_model``: MAE / WAPE / bias / MASE / residual std) and
ranked (``select_model``: lowest ``criterion``, ties broken by |bias| then name).  Each score
also carries an :class:`IntervalModel` learnt from the fold residuals, so the pipeline can emit
asymmetric, non-negative prediction intervals with a verifiable nominal coverage.

Promotion flags (0/1 arrays aligned with ``y``) are optional: they are sliced per fold and
forwarded to ``fit(..., promo_flags=)`` / ``predict(..., future_flags=)`` only when supplied, so
flag-unaware models keep working unchanged.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Protocol, cast

import numpy as np

from .models import MODEL_REGISTRY, Forecaster, is_intermittent, make_model

CROSTON = "croston_sba"  # only a candidate for intermittent series
CRITERIA = ("mae", "wape", "mase")  # selection criteria accepted by select_model
INTERVAL_METHODS = ("empirical", "normal")


# ---- metrics -----------------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean error (pred - true). Positive => over-forecasting."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Weighted absolute percentage error = sum|e| / sum|y|. Undefined if sum|y| == 0."""
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return None
    return float(np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denom)


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season: int = 7
) -> float | None:
    """Mean absolute scaled error vs. in-sample seasonal-naive forecast."""
    y_train = np.asarray(y_train, dtype=float)
    if y_train.size <= season:
        return None
    scale = float(np.mean(np.abs(y_train[season:] - y_train[:-season])))
    if scale == 0:
        return None
    return mae(y_true, y_pred) / scale


# ---- rolling-origin evaluation -----------------------------------------------------------
@dataclass
class FoldResult:
    origin: int
    y_true: np.ndarray
    y_pred: np.ndarray


# ---- prediction intervals ----------------------------------------------------------------
MAX_INTERVAL_GROWTH = 0.03  # cap on the per-step widening factor
MIN_GROWTH_RESIDUALS = 20  # growth is only estimated from this many residuals ...
MIN_GROWTH_HORIZON = 4  # ... and only for horizons of at least this many steps


def _check_level(level: float) -> float:
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"interval level must be strictly between 0 and 1, got {level!r}")
    return level


@dataclass(frozen=True)
class IntervalModel:
    """Asymmetric prediction interval learnt from backtest residuals ``e = y_true - y_pred``.

    ``lo_offset <= 0 <= hi_offset`` are the (clamped) empirical residual quantiles at
    ``(1 - level) / 2`` and ``1 - (1 - level) / 2``; ``growth >= 0`` widens the band linearly
    with the 1-based horizon step ``h``::

        lower_h = max(0, yhat_h + lo_offset * (1 + growth * (h - 1)))
        upper_h =        yhat_h + hi_offset * (1 + growth * (h - 1))

    For any non-negative point forecast this guarantees ``0 <= lower <= yhat <= upper``
    (also when ``yhat`` contains zeros).
    """

    level: float
    lo_offset: float
    hi_offset: float
    growth: float = 0.0

    def __post_init__(self) -> None:
        _check_level(self.level)
        if not (self.lo_offset <= 0.0 <= self.hi_offset):
            raise ValueError(
                "interval offsets must satisfy lo_offset <= 0 <= hi_offset, got "
                f"lo_offset={self.lo_offset!r}, hi_offset={self.hi_offset!r}"
            )
        if not self.growth >= 0.0:
            raise ValueError(f"interval growth must be >= 0, got {self.growth!r}")

    def apply(self, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` arrays aligned with ``yhat`` (step h = position + 1)."""
        yhat = np.asarray(yhat, dtype=float).ravel()
        scale = 1.0 + self.growth * np.arange(yhat.size, dtype=float)  # = 1 + growth*(h-1)
        lower = np.maximum(yhat + self.lo_offset * scale, 0.0)
        upper = yhat + self.hi_offset * scale
        return lower, upper


def _horizon_growth(abs_e: np.ndarray, steps: np.ndarray, horizon: int) -> float:
    """``clamp((r2/r1 - 1) / (H/2), 0, MAX_INTERVAL_GROWTH)`` or 0.0 when a guard fails."""
    if abs_e.size < MIN_GROWTH_RESIDUALS or horizon < MIN_GROWTH_HORIZON:
        return 0.0
    if steps.shape != abs_e.shape:
        raise ValueError("steps must have one entry per residual")
    split = -(-horizon // 2)  # ceil(H / 2)
    near = abs_e[steps <= split]
    far = abs_e[steps > split]
    if near.size == 0 or far.size == 0:
        return 0.0
    r1 = float(near.mean())
    if r1 <= 0.0:
        return 0.0
    r2 = float(far.mean())
    raw = (r2 / r1 - 1.0) / (horizon / 2.0)
    return float(min(max(raw, 0.0), MAX_INTERVAL_GROWTH))


def empirical_interval(
    residuals: np.ndarray,
    level: float = 0.8,
    steps: np.ndarray | None = None,
    horizon: int | None = None,
) -> IntervalModel:
    """Build an :class:`IntervalModel` from pooled residuals ``e = y_true - y_pred``.

    Offsets are empirical quantiles (NumPy's default linear interpolation), clamped so the
    band always contains the point forecast even when the residuals are one-sided::

        lo_offset = min(0, Q_e((1 - level) / 2))
        hi_offset = max(0, Q_e(1 - (1 - level) / 2))

    Horizon growth needs the 1-based step index of every residual (``steps``) and the
    horizon ``H``.  With ``r1 = mean|e|`` over steps ``1..ceil(H/2)`` and ``r2 = mean|e|``
    over the remaining steps (both pooled over folds)::

        growth = clamp((r2 / r1 - 1) / (H / 2), 0, 0.03)

    estimated only when ``len(e) >= 20``, ``r1 > 0`` and ``H >= 4`` — otherwise ``0.0``.
    """
    level = _check_level(level)
    e = np.asarray(residuals, dtype=float).ravel()
    if e.size == 0:
        raise ValueError("cannot build a prediction interval from zero residuals")
    if not np.all(np.isfinite(e)):
        raise ValueError("residuals contain NaN or inf")
    alpha = (1.0 - level) / 2.0
    lo_q, hi_q = np.quantile(e, [alpha, 1.0 - alpha])
    growth = 0.0
    if steps is not None and horizon is not None:
        growth = _horizon_growth(np.abs(e), np.asarray(steps).ravel(), int(horizon))
    return IntervalModel(
        level=level,
        lo_offset=min(float(lo_q), 0.0),
        hi_offset=max(float(hi_q), 0.0),
        growth=growth,
    )


def interval_from_folds(
    folds: Sequence[FoldResult], level: float = 0.8, horizon: int | None = None
) -> IntervalModel:
    """Empirical interval from rolling-origin folds (residuals pooled, step index per fold)."""
    if not folds:
        raise ValueError("cannot build a prediction interval from zero folds")
    e = np.concatenate([np.asarray(f.y_true, dtype=float) - f.y_pred for f in folds])
    steps = np.concatenate([np.arange(1, np.asarray(f.y_true).size + 1) for f in folds])
    if horizon is None:
        horizon = max(np.asarray(f.y_true).size for f in folds)
    return empirical_interval(e, level, steps, horizon)


def normal_interval(residual_std: float, level: float) -> IntervalModel:
    """Symmetric Gaussian interval ``yhat ± z·sigma`` with ``z = Φ⁻¹(1 - (1 - level) / 2)``.

    This is the v0.3 behaviour (z = 1.2816 at level 0.8) expressed as an
    :class:`IntervalModel`, so it still benefits from the non-negative lower bound.
    """
    level = _check_level(level)
    sigma = float(residual_std)
    if not sigma >= 0.0:
        raise ValueError(f"residual_std must be >= 0, got {residual_std!r}")
    z = NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    return IntervalModel(level=level, lo_offset=-z * sigma, hi_offset=z * sigma)


@dataclass
class ModelScore:
    """Backtest summary of one model (metrics pooled over folds).

    ``interval`` is the prediction interval fitted on the fold residuals and
    ``coverage_backtest`` the share of fold actuals it contains (in-sample diagnostic).
    ``fold_results`` is dropped by the pipeline before results cross process boundaries.
    """

    model_name: str
    folds: int
    mae: float
    wape: float | None
    bias: float
    mase: float | None
    residual_std: float
    fold_results: list[FoldResult] = field(default_factory=list, repr=False)
    interval: IntervalModel | None = None
    coverage_backtest: float | None = None


class _PromoAwareForecaster(Protocol):
    """The promo-aware forecaster interface (contract C1) used when flags are supplied."""

    def fit(self, y: np.ndarray, promo_flags: np.ndarray | None = ...) -> Any: ...

    def predict(self, h: int, future_flags: np.ndarray | None = ...) -> np.ndarray: ...


def rolling_origin(
    y: np.ndarray,
    model_factory: Callable[[], Forecaster],
    horizon: int,
    n_folds: int = 4,
    step: int | None = None,
    min_train: int = 56,
    promo_flags: np.ndarray | None = None,
) -> list[FoldResult]:
    """Fit on y[:origin], predict `horizon`, slide origin forward. Oldest fold first.

    When ``promo_flags`` (aligned with ``y``) is given, fold ``o`` fits with
    ``promo_flags=flags[:o]`` and predicts with ``future_flags=flags[o:o+horizon]``; when it is
    ``None`` the model is called exactly as in v0.3 (no flag keywords at all).
    """
    y = np.asarray(y, dtype=float)
    flags: np.ndarray | None = None
    if promo_flags is not None:
        flags = np.asarray(promo_flags, dtype=float).ravel()
        if flags.shape != y.shape:
            raise ValueError(
                f"promo_flags must align with y: got {flags.size} flags for {y.size} observations"
            )
    step = step or horizon
    last_origin = y.size - horizon
    origins = [last_origin - k * step for k in range(n_folds)][::-1]
    origins = [o for o in origins if o >= min_train]
    if not origins:
        raise ValueError(
            f"series too short for backtest: n={y.size}, horizon={horizon}, min_train={min_train}"
        )
    out: list[FoldResult] = []
    for o in origins:
        model: Forecaster = model_factory()
        if flags is None:
            model.fit(y[:o])
            pred = model.predict(horizon)
        else:
            promo_model = cast(_PromoAwareForecaster, model)
            promo_model.fit(y[:o], promo_flags=flags[:o])
            pred = promo_model.predict(horizon, future_flags=flags[o : o + horizon])
        out.append(FoldResult(o, y[o : o + horizon], np.asarray(pred, dtype=float)))
    return out


def backtest_coverage(folds: Sequence[FoldResult], interval: IntervalModel) -> float:
    """Share of fold actuals inside ``interval.apply(y_pred)`` (in-sample coverage diagnostic)."""
    inside = 0
    total = 0
    for f in folds:
        y_true = np.asarray(f.y_true, dtype=float)
        lower, upper = interval.apply(f.y_pred)
        inside += int(np.count_nonzero((lower <= y_true) & (y_true <= upper)))
        total += y_true.size
    if total == 0:
        raise ValueError("cannot measure coverage without fold actuals")
    return inside / total


def fit_interval(
    folds: Sequence[FoldResult],
    residual_std: float,
    level: float = 0.8,
    method: str = "empirical",
    horizon: int | None = None,
) -> IntervalModel:
    """Interval for a scored model: ``"empirical"`` (fold residual quantiles) or ``"normal"``."""
    if method == "empirical":
        return interval_from_folds(folds, level, horizon)
    if method == "normal":
        return normal_interval(residual_std, level)
    raise ValueError(f"unknown interval_method {method!r}, choose from {INTERVAL_METHODS}")


def score_model(
    y: np.ndarray,
    model_name: str,
    horizon: int,
    promo_flags: np.ndarray | None = None,
    interval_level: float = 0.8,
    interval_method: str = "empirical",
    **kw: Any,
) -> ModelScore:
    """Backtest one model and summarise it (metrics as in v0.3 + a fitted prediction interval).

    ``promo_flags`` is forwarded to :func:`rolling_origin`; ``interval_level`` /
    ``interval_method`` configure :func:`fit_interval`; remaining keywords (``n_folds``,
    ``step``, ``min_train``) go to :func:`rolling_origin`.
    """
    _check_level(interval_level)
    if interval_method not in INTERVAL_METHODS:
        raise ValueError(
            f"unknown interval_method {interval_method!r}, choose from {INTERVAL_METHODS}"
        )
    folds = rolling_origin(
        y, lambda: make_model(model_name), horizon, promo_flags=promo_flags, **kw
    )
    yt = np.concatenate([f.y_true for f in folds])
    yp = np.concatenate([f.y_pred for f in folds])
    train_ref = np.asarray(y)[: folds[0].origin]
    residual_std = float(np.std(yt - yp, ddof=1)) if yt.size > 1 else 0.0
    interval = fit_interval(folds, residual_std, interval_level, interval_method, horizon)
    return ModelScore(
        model_name=model_name,
        folds=len(folds),
        mae=mae(yt, yp),
        wape=wape(yt, yp),
        bias=bias(yt, yp),
        mase=mase(yt, yp, train_ref),
        residual_std=residual_std,
        fold_results=folds,
        interval=interval,
        coverage_backtest=backtest_coverage(folds, interval),
    )


def candidate_models(
    y: np.ndarray,
    promo_flags: np.ndarray | None = None,
    allowed: Sequence[str] | None = None,
) -> list[str]:
    """Ordered candidate model names for one series.

    Base order is ``MODEL_REGISTRY`` (Croston only when :func:`is_intermittent`); when
    ``promo_flags`` carry a usable signal (``promo.has_promo_signal``) the ``promo_<base>``
    wrappers are appended for every promo base present.  ``allowed`` filters the list while
    preserving that order; an empty result raises ``ValueError``.
    """
    intermittent = is_intermittent(y)
    names = [n for n in MODEL_REGISTRY if n != CROSTON or intermittent]
    if promo_flags is not None:
        # Lazy, dynamic import of the promo module (contract C2): flag-less callers never
        # depend on it, and the module may be absent without breaking type checking.
        promo = importlib.import_module(".promo", __package__)
        if promo.has_promo_signal(np.asarray(promo_flags, dtype=float).ravel()):
            pairs = zip(promo.PROMO_BASES, promo.PROMO_MODEL_NAMES, strict=True)
            names += [promo_name for base, promo_name in pairs if base in names]
    if allowed is not None:
        wanted = set(allowed)
        filtered = [n for n in names if n in wanted]
        if not filtered:
            raise ValueError(
                f"no candidate model left for this series: allowed={sorted(wanted)}, "
                f"candidates={names}"
            )
        names = filtered
    return names


def _rank_key(criterion: str) -> Callable[[ModelScore], tuple[bool, float, float, str]]:
    def key(s: ModelScore) -> tuple[bool, float, float, str]:
        metric = getattr(s, criterion)
        return (metric is None, 0.0 if metric is None else float(metric), abs(s.bias), s.model_name)

    return key


def rank_scores(scores: Sequence[ModelScore], criterion: str = "mae") -> list[ModelScore]:
    """Sort scores best-first: lowest ``criterion`` (``None`` last), then |bias|, then name."""
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}, choose from {CRITERIA}")
    return sorted(scores, key=_rank_key(criterion))


def select_model(
    y: np.ndarray,
    horizon: int,
    promo_flags: np.ndarray | None = None,
    allowed: Sequence[str] | None = None,
    criterion: str = "mae",
    interval_level: float = 0.8,
    interval_method: str = "empirical",
    **kw: Any,
) -> tuple[ModelScore, list[ModelScore]]:
    """Backtest every candidate; return ``(winner, all_scores)`` in candidate order.

    The winner minimises ``criterion`` (``"mae"`` by default, or ``"wape"`` / ``"mase"``);
    undefined metrics sort last and ties are broken by |bias| and then by model name.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}, choose from {CRITERIA}")
    scores = [
        score_model(
            y,
            name,
            horizon,
            promo_flags=promo_flags,
            interval_level=interval_level,
            interval_method=interval_method,
            **kw,
        )
        for name in candidate_models(y, promo_flags, allowed)
    ]
    return rank_scores(scores, criterion)[0], scores
