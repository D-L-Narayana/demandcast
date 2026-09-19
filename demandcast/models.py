"""Forecasting models implemented from first principles on NumPy.

All models share one small interface so they can be swapped, backtested and selected
automatically per series:

    model = SomeModel(**params)
    model.fit(y)                 # y: 1-D array of daily units, oldest first
    yhat = model.predict(h)      # point forecast for the next h days (always >= 0)

Models included
---------------
SeasonalNaive     — repeat the same weekday from the last full week (strong retail baseline)
MovingAverage     — trailing mean, seasonally re-weighted by weekday profile
HoltWinters       — additive damped-trend + additive weekly seasonality (triple exp. smoothing)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product

import numpy as np

SEASON = 7  # weekly seasonality for daily retail data


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, y: np.ndarray) -> Forecaster: ...

    @abstractmethod
    def predict(self, h: int) -> np.ndarray: ...

    @staticmethod
    def _clean(y) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        if y.size == 0:
            raise ValueError("series is empty")
        if np.any(np.isnan(y)):
            raise ValueError("series contains NaN")
        return y


# ------------------------------------------------------------------------------------------
class SeasonalNaive(Forecaster):
    name = "seasonal_naive"

    def __init__(self, season: int = SEASON):
        self.season = season
        self._last: np.ndarray | None = None

    def fit(self, y):
        y = self._clean(y)
        if y.size < self.season:
            self._last = np.full(self.season, y.mean())
        else:
            self._last = y[-self.season :]
        return self

    def predict(self, h: int) -> np.ndarray:
        assert self._last is not None, "call fit() first"
        idx = np.arange(h) % self.season
        return np.maximum(self._last[idx], 0.0)


# ------------------------------------------------------------------------------------------
class MovingAverage(Forecaster):
    """Trailing-window mean re-scaled by a weekday profile estimated from the history."""

    name = "moving_average"

    def __init__(self, window: int = 28, season: int = SEASON):
        self.window = window
        self.season = season
        self._level = 0.0
        self._profile = np.ones(season)
        self._n = 0

    def fit(self, y):
        y = self._clean(y)
        self._n = y.size
        w = y[-self.window :]
        self._level = float(w.mean())
        if y.size >= 2 * self.season and y.mean() > 0:
            profile = np.array(
                [y[i :: self.season].mean() for i in range(self.season)]
            )  # indexed by position-in-cycle relative to y[0]
            profile = profile / profile.mean()
            # shift so that profile[0] corresponds to the day right after the series ends
            shift = y.size % self.season
            self._profile = np.roll(profile, -shift)
        else:
            self._profile = np.ones(self.season)
        return self

    def predict(self, h: int) -> np.ndarray:
        idx = np.arange(h) % self.season
        return np.maximum(self._level * self._profile[idx], 0.0)


# ------------------------------------------------------------------------------------------
@dataclass
class HWParams:
    alpha: float = 0.3  # level
    beta: float = 0.05  # trend
    gamma: float = 0.2  # season
    phi: float = 0.95  # trend damping


class HoltWinters(Forecaster):
    """Additive Holt-Winters with damped trend.

    Smoothing parameters are chosen by grid search on in-sample one-step SSE when
    `auto=True`; otherwise the supplied HWParams are used directly.
    """

    name = "holt_winters"

    def __init__(self, params: HWParams | None = None, season: int = SEASON, auto: bool = True):
        self.params = params or HWParams()
        self.season = season
        self.auto = auto
        self._level = 0.0
        self._trend = 0.0
        self._season = np.zeros(season)
        self.sse_ = float("inf")

    # --- core recursion --------------------------------------------------------------------
    def _run(self, y: np.ndarray, p: HWParams):
        m = self.season
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
        return level, trend, seasonal, sse, n

    def fit(self, y):
        y = self._clean(y)
        candidates = [self.params]
        if self.auto and y.size >= 3 * self.season:
            grid = product([0.15, 0.3, 0.5], [0.02, 0.1], [0.1, 0.3], [0.95])
            candidates = [HWParams(*g) for g in grid]
        best = None
        for p in candidates:
            level, trend, seasonal, sse, n = self._run(y, p)
            if best is None or sse < best[3]:
                best = (level, trend, seasonal.copy(), sse, n, p)
        self._level, self._trend, self._season, self.sse_, self._n, self.params = best
        return self

    def predict(self, h: int) -> np.ndarray:
        phi = self.params.phi
        out = np.empty(h)
        damp = 0.0
        for i in range(1, h + 1):
            damp += phi**i
            out[i - 1] = (
                self._level + damp * self._trend + self._season[(self._n + i - 1) % self.season]
            )
        return np.maximum(out, 0.0)


# ------------------------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[Forecaster]] = {
    SeasonalNaive.name: SeasonalNaive,
    MovingAverage.name: MovingAverage,
    HoltWinters.name: HoltWinters,
}


def make_model(name: str) -> Forecaster:
    try:
        return MODEL_REGISTRY[name]()
    except KeyError as e:
        raise KeyError(f"unknown model '{name}', choose from {sorted(MODEL_REGISTRY)}") from e


def is_intermittent(y: np.ndarray, zero_share: float = 0.5) -> bool:
    """Heuristic used to decide whether Croston is a sensible candidate."""
    y = np.asarray(y, dtype=float)
    return y.size > 0 and float(np.mean(y == 0)) >= zero_share
