"""Rolling-origin backtesting, accuracy metrics and per-series model selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import MODEL_REGISTRY, Forecaster, is_intermittent, make_model


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


@dataclass
class ModelScore:
    model_name: str
    folds: int
    mae: float
    wape: float | None
    bias: float
    mase: float | None
    residual_std: float
    fold_results: list[FoldResult] = field(default_factory=list, repr=False)


def rolling_origin(
    y: np.ndarray,
    model_factory,
    horizon: int,
    n_folds: int = 4,
    step: int | None = None,
    min_train: int = 56,
) -> list[FoldResult]:
    """Fit on y[:origin], predict `horizon`, slide origin forward. Oldest fold first."""
    y = np.asarray(y, dtype=float)
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
        model.fit(y[:o])
        pred = model.predict(horizon)
        out.append(FoldResult(o, y[o : o + horizon], pred))
    return out


def score_model(y: np.ndarray, model_name: str, horizon: int, **kw) -> ModelScore:
    folds = rolling_origin(y, lambda: make_model(model_name), horizon, **kw)
    yt = np.concatenate([f.y_true for f in folds])
    yp = np.concatenate([f.y_pred for f in folds])
    train_ref = np.asarray(y)[: folds[0].origin]
    return ModelScore(
        model_name=model_name,
        folds=len(folds),
        mae=mae(yt, yp),
        wape=wape(yt, yp),
        bias=bias(yt, yp),
        mase=mase(yt, yp, train_ref),
        residual_std=float(np.std(yt - yp, ddof=1)) if yt.size > 1 else 0.0,
        fold_results=folds,
    )


def candidate_models(y: np.ndarray) -> list[str]:
    names = [n for n in MODEL_REGISTRY if n != "croston_sba"]
    if is_intermittent(y):
        names.append("croston_sba")
    return names


def select_model(y: np.ndarray, horizon: int, **kw) -> tuple[ModelScore, list[ModelScore]]:
    """Backtest every candidate; return (winner, all_scores). Winner = lowest MAE."""
    scores = [score_model(y, name, horizon, **kw) for name in candidate_models(y)]
    winner = min(scores, key=lambda s: (s.mae, abs(s.bias)))
    return winner, scores
