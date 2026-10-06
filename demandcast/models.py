"""Forecasting models implemented from first principles on NumPy.

All models share one small interface so they can be swapped, backtested and selected
automatically per series:

    model = SomeModel(**params)
    model.fit(y)                 # y: 1-D array of daily units, oldest first
    yhat = model.predict(h)      # point forecast for the next h days (always >= 0, no NaN)

``fit`` and ``predict`` also accept promotion flags (``promo_flags`` aligned with ``y``,
``future_flags`` aligned with the forecast steps). The base models accept and ignore them;
``demandcast.promo.PromoAdjusted`` (``supports_promo = True``) wraps any base model and uses
them to estimate and re-apply a promotional lift. Calling ``predict`` before ``fit`` raises
``RuntimeError("call fit() first")`` for every model.

Models included
---------------
SeasonalNaive     — repeat the same weekday from the last full week (strong retail baseline)
MovingAverage     — trailing mean, re-weighted by a recent weekday profile
HoltWinters       — additive damped-trend + additive weekly seasonality (triple exp. smoothing)
Theta             — classical Theta method (theta = 2) on weekly-deseasonalised data
Croston           — intermittent-demand method (separate size/interval smoothing) for slow movers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

SEASON = 7  # weekly seasonality for daily retail data
PROMO_PREFIX = "promo_"  # make_model("promo_<base>") -> PromoAdjusted(<base>)
THETA_ALPHA_GRID = (0.1, 0.2, 0.3, 0.5)
HW_BATCH_BLOCK = 14  # days folded into one matrix product by HoltWinters._run_batch


class Forecaster(ABC):
    """Common interface: ``fit(y, promo_flags=None)`` then ``predict(h, future_flags=None)``."""

    name: str = "base"
    supports_promo: bool = False  # True only for demandcast.promo.PromoAdjusted
    _fitted: bool = False

    @abstractmethod
    def fit(self, y: ArrayLike, promo_flags: np.ndarray | None = None) -> Forecaster: ...

    @abstractmethod
    def predict(self, h: int, future_flags: np.ndarray | None = None) -> np.ndarray: ...

    @staticmethod
    def _clean(y: ArrayLike) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        if y.size == 0:
            raise ValueError("series is empty")
        if np.any(np.isnan(y)):
            raise ValueError("series contains NaN")
        return y

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() first")


# ------------------------------------------------------------------------------------------
class SeasonalNaive(Forecaster):
    """Repeat the last observed season: yhat_{n+h} = y_{n+h-m} (the series mean when n < m)."""

    name = "seasonal_naive"

    def __init__(self, season: int = SEASON):
        self.season = season
        self._last: np.ndarray | None = None

    def fit(self, y: ArrayLike, promo_flags: np.ndarray | None = None) -> SeasonalNaive:
        y = self._clean(y)
        if y.size < self.season:
            self._last = np.full(self.season, y.mean())
        else:
            self._last = y[-self.season :].copy()
        self._fitted = True
        return self

    def predict(self, h: int, future_flags: np.ndarray | None = None) -> np.ndarray:
        if self._last is None:
            raise RuntimeError("call fit() first")
        idx = np.arange(h) % self.season
        return np.maximum(self._last[idx], 0.0)


# ------------------------------------------------------------------------------------------
class MovingAverage(Forecaster):
    """Trailing-window mean re-scaled by a weekday profile.

        level      = mean(y[n-window : n])
        profile_j  = mean{ x_t : t = j (mod m) } / mean_j(profile)   over the profile window x
        yhat_{n+h} = max(0, level * profile_{(n+h-1) mod m})

    The profile window is the last ``profile_weeks`` full weeks (default 8) when the history is
    at least that long, so a recent weekday pattern adapts to changes in the weekly mix;
    otherwise the whole history is used. A profile needs at least two seasons with a positive
    mean; otherwise it is flat (all ones).
    """

    name = "moving_average"

    def __init__(self, window: int = 28, season: int = SEASON, profile_weeks: int = 8):
        self.window = window
        self.season = season
        self.profile_weeks = profile_weeks
        self._level = 0.0
        self._profile = np.ones(season)
        self._n = 0

    def fit(self, y: ArrayLike, promo_flags: np.ndarray | None = None) -> MovingAverage:
        y = self._clean(y)
        self._n = int(y.size)
        w = y[-self.window :]
        self._level = float(w.mean())
        span = self.profile_weeks * self.season
        tail = y[-span:] if y.size >= span else y
        if tail.size >= 2 * self.season and tail.mean() > 0:
            profile = np.array(
                [tail[i :: self.season].mean() for i in range(self.season)]
            )  # indexed by position-in-cycle relative to tail[0]
            profile = profile / profile.mean()
            # shift so that profile[0] corresponds to the day right after the series ends
            shift = tail.size % self.season
            self._profile = np.roll(profile, -shift)
        else:
            self._profile = np.ones(self.season)
        self._fitted = True
        return self

    def predict(self, h: int, future_flags: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
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
    """Additive Holt-Winters with damped trend (season length m).

        fitted_t   = l_{t-1} + phi * b_{t-1} + s_{t-m}               (one-step fitted value)
        e_t        = y_t - fitted_t
        l_t        = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + phi * b_{t-1})
        b_t        = beta * (l_t - l_{t-1}) + (1 - beta) * phi * b_{t-1}
        s_t        = gamma * (y_t - l_t) + (1 - gamma) * s_{t-m}
        yhat_{n+h} = max(0, l_n + (phi + phi^2 + ... + phi^h) * b_n + s_{n+h-m*ceil(h/m)})

    Initial states: l_0 = mean of the first season, b_0 = (mean of season 2 - mean of season 1)
    / m (0 with fewer than two seasons), s_i = y_i - l_0 over the first season; with n < m the
    level is the series mean and the seasonal states are zero.

    Smoothing parameters are chosen by grid search on the in-sample one-step SSE = sum(e_t^2)
    when ``auto=True`` and n >= 3m (12 candidates: alpha in {0.15, 0.3, 0.5}, beta in
    {0.02, 0.1}, gamma in {0.1, 0.3}, phi = 0.95; the first minimum wins ties); otherwise the
    supplied ``HWParams`` are used directly. The chosen parameters are exposed as ``params``
    and the SSE as ``sse_``.

    ``_run`` is the scalar reference recursion for one candidate. ``_run_batch`` evaluates all
    candidates in ONE pass over time using the (linear) state-space form of the same equations,
    folding ``HW_BATCH_BLOCK`` days into a single matrix product per step of the loop; the two
    paths agree to about 1e-12 relative (tested at rtol 1e-9).
    """

    name = "holt_winters"

    def __init__(self, params: HWParams | None = None, season: int = SEASON, auto: bool = True):
        self.params = params or HWParams()
        self.season = season
        self.auto = auto
        self._level = 0.0
        self._trend = 0.0
        self._season = np.zeros(season)
        self._n = 0
        self.sse_ = float("inf")

    @staticmethod
    def grid_candidates() -> list[HWParams]:
        """The 12-point parameter grid searched when ``auto=True`` (candidate order matters)."""
        grid = product([0.15, 0.3, 0.5], [0.02, 0.1], [0.1, 0.3], [0.95])
        return [HWParams(*g) for g in grid]

    # --- core recursion --------------------------------------------------------------------
    def _initial_state(self, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Initial (level, trend, seasonal) from the first two seasons (or what is available)."""
        m = self.season
        n = y.size
        k = min(2, n // m) if n >= m else 0
        if k >= 1:
            season_means = [y[i * m : (i + 1) * m].mean() for i in range(k)]
            level = float(season_means[0])
            trend = float((season_means[-1] - season_means[0]) / (m * (k - 1))) if k > 1 else 0.0
            seasonal = np.array([y[i] - level for i in range(m)])
        else:
            level, trend, seasonal = float(y.mean()), 0.0, np.zeros(m)
        return level, trend, seasonal

    def _run(self, y: np.ndarray, p: HWParams) -> tuple[float, float, np.ndarray, float, int]:
        """Scalar reference recursion for one parameter set -> (level, trend, seasonal, sse, n)."""
        m = self.season
        n = y.size
        level, trend, seasonal = self._initial_state(y)
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

    def _run_batch(
        self, y: np.ndarray, params: list[HWParams], block: int = HW_BATCH_BLOCK
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the recursion for every candidate at once: one pass over time, in blocks of days.

        Returns ``(levels, trends, seasonals, sses)`` with shapes ``(c,)``, ``(c,)``,
        ``(c, season)`` and ``(c,)`` for ``c = len(params)``; entry ``i`` agrees with
        ``_run(y, params[i])`` to about 1e-12 relative (floating-point order differs).

        The recursion is linear in its state, so it is run in state-space form with the
        time-invariant state x_t = (l, b, s_{t-1}, ..., s_{t-m}) (the seasonal part rotates so
        that the slot used next is always last). With w = (1, phi, 0, ..., 0, 1) and
        g = (alpha, alpha*beta, gamma*(1-alpha), 0, ..., 0):

            e_t     = y_t - w . x_t                       (one-step error, fitted = w . x_t)
            x_{t+1} = T x_t + g y_t                       T = S - g w^T, S = shift/trend part
            l'      = l + phi*b + alpha*e                 (error-correction form of the
            b'      = phi*b + alpha*beta*e                 equations used in ``_run``)
            s_t     = s_{t-m} + gamma*(1-alpha)*e

        Iterating k days: x_{t+k} = T^k x_t + sum_j T^{k-1-j} g y_{t+j} and
        e_{t+j} = y_{t+j} - w.T^j x_t - sum_{i<j} w.T^{j-1-i} g y_{t+i}, i.e. one matrix-vector
        product per block maps (x_t, y_t..y_{t+k-1}) to (x_{t+k}, e_t..e_{t+k-1}). The block
        matrices are built once per fit for all candidates, which amortises NumPy's per-call
        cost over ``block`` days and the whole grid. ``sse = sum_t e_t^2``.
        """
        m, n, c = self.season, int(y.size), len(params)
        d = m + 2
        alpha = np.array([p.alpha for p in params], dtype=float)
        beta = np.array([p.beta for p in params], dtype=float)
        gamma = np.array([p.gamma for p in params], dtype=float)
        phi = np.array([p.phi for p in params], dtype=float)
        level0, trend0, seasonal0 = self._initial_state(y)
        x = np.empty((c, d))
        x[:, 0], x[:, 1], x[:, 2:] = level0, trend0, seasonal0[::-1]
        w = np.zeros((c, d))
        w[:, 0], w[:, 1], w[:, -1] = 1.0, phi, 1.0
        g = np.zeros((c, d))
        g[:, 0], g[:, 1], g[:, 2] = alpha, alpha * beta, gamma * (1 - alpha)
        t_mat = np.zeros((c, d, d))
        t_mat[:, 0, 0], t_mat[:, 0, 1], t_mat[:, 1, 1], t_mat[:, 2, -1] = 1.0, phi, phi, 1.0
        t_mat[:, 3:, 2:-1] = np.eye(m - 1)  # older seasonal values move down one slot
        t_mat -= g[:, :, None] * w[:, None, :]
        k = max(1, min(block, n))
        powers = np.empty((k + 1, c, d, d))  # powers[j] = T^j
        powers[0] = np.eye(d)
        for j in range(1, k + 1):
            powers[j] = powers[j - 1] @ t_mat
        tg = np.matmul(powers[:k], g[None, :, :, None])[..., 0]  # (k, c, d): T^j g
        wt = np.matmul(w[None, :, None, :], powers[:k])[:, :, 0, :]  # (k, c, d): w^T T^j
        h = np.einsum("kcd,cd->kc", tg, w)  # (k, c): w^T T^j g

        def block_matrix(r: int) -> np.ndarray:
            # maps u = (x_t, y_t..y_{t+r-1}) to (x_{t+r}, e_t..e_{t+r-1})
            b_mat = np.zeros((c, d + r, d + r))
            b_mat[:, :d, :d] = powers[r]
            b_mat[:, :d, d:] = tg[r - 1 :: -1].transpose(1, 2, 0)
            b_mat[:, d:, :d] = -wt[:r].transpose(1, 0, 2)
            lag = np.arange(r)[:, None] - 1 - np.arange(r)[None, :]  # j - 1 - i
            low = np.where(lag >= 0, h[np.clip(lag, 0, None)].transpose(2, 0, 1), 0.0)
            b_mat[:, d:, d:] = np.eye(r) - low
            return b_mat

        errs = np.empty((c, n))
        n_full, rem = divmod(n, k)
        b_full = block_matrix(k)
        u = np.empty((c, d + k, 1))
        for i in range(n_full):
            t = i * k
            u[:, :d, 0] = x
            u[:, d:, 0] = y[t : t + k]
            out = np.matmul(b_full, u)[:, :, 0]
            x = out[:, :d]
            errs[:, t : t + k] = out[:, d:]
        if rem:
            t = n_full * k
            u = np.empty((c, d + rem, 1))
            u[:, :d, 0] = x
            u[:, d:, 0] = y[t:]
            out = np.matmul(block_matrix(rem), u)[:, :, 0]
            x = out[:, :d]
            errs[:, t:] = out[:, d:]
        slots = (n - 1 - np.arange(m)) % m  # x[:, 2 + i] holds the seasonal of day n-1-i
        seasonals = np.empty((c, m))
        seasonals[:, slots] = x[:, 2:]
        return x[:, 0].copy(), x[:, 1].copy(), seasonals, np.einsum("cn,cn->c", errs, errs)

    def fit(self, y: ArrayLike, promo_flags: np.ndarray | None = None) -> HoltWinters:
        y = self._clean(y)
        if self.auto and y.size >= 3 * self.season:
            candidates = self.grid_candidates()
            levels, trends, seasonals, sses = self._run_batch(y, candidates)
            best = int(np.argmin(sses))  # first minimum, like the per-candidate loop
            self._level, self._trend = float(levels[best]), float(trends[best])
            self._season = seasonals[best].copy()
            self.sse_ = float(sses[best])
            self.params = candidates[best]
        else:
            level, trend, seasonal, sse, _ = self._run(y, self.params)
            self._level, self._trend, self._season = float(level), float(trend), seasonal
            self.sse_ = float(sse)
        self._n = int(y.size)
        self._fitted = True
        return self

    def predict(self, h: int, future_flags: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
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
class Theta(Forecaster):
    """Classical Theta method (Assimakopoulos & Nikolopoulos, 2000) with theta = 2.

    With 0-based time t = 0..n-1 and season length m:

    1. Multiplicative weekly deseasonalisation (only when n >= 2m, otherwise I = 1):
           I_j = mean{ y_t : t = j (mod m) } / mean(y),  clipped to [0.1, 10]
           x_t = y_t / I_{t mod m}
    2. theta = 0 line: least-squares trend  L_t = a + b t  (b = 0 for a single point);
       theta = 2 line: z_t = 2 x_t - L_t.
    3. Simple exponential smoothing of the theta = 2 line:
           level_0 = z_0,   level_t = level_{t-1} + alpha (z_t - level_{t-1})
       with alpha from {0.1, 0.2, 0.3, 0.5} chosen by the in-sample one-step SSE
       sum_{t>=1} (z_t - level_{t-1})^2 (first alpha wins ties).
    4. Equal-weight combination, reseasonalisation, clipping:
           yhat_{n+h} = max(0, [0.5 * L_{n+h-1} + 0.5 * level_{n-1}] * I_{(n+h-1) mod m})

    Guards: mean(y) == 0 -> zero forecast. Fitted attributes: ``seasonal_`` (I: a 1-D float64
    array of length ``season``, indexed by position in the cycle relative to y[0]; all ones
    until ``fit`` deseasonalises), ``intercept_`` (a), ``slope_`` (b), ``alpha_``, ``level_``
    and ``sse_``.
    """

    name = "theta"

    def __init__(self, season: int = SEASON, alphas: tuple[float, ...] = THETA_ALPHA_GRID):
        self.season = season
        self.alphas = alphas
        self.seasonal_: NDArray[np.float64] = np.ones(season)
        self.intercept_ = 0.0
        self.slope_ = 0.0
        self.level_ = 0.0
        self.alpha_ = alphas[0]
        self.sse_ = 0.0
        self._n = 0
        self._zero = False

    @staticmethod
    def _ses(z: list[float], alpha: float) -> tuple[float, float]:
        """SES with level_0 = z_0 -> (final level, one-step SSE over t >= 1)."""
        level = z[0]
        sse = 0.0
        for v in z[1:]:
            e = v - level
            sse += e * e
            level += alpha * e
        return level, sse

    def fit(self, y: ArrayLike, promo_flags: np.ndarray | None = None) -> Theta:
        y = self._clean(y)
        m, n = self.season, int(y.size)
        self._n = n
        self.seasonal_ = np.ones(m)
        self.intercept_ = self.slope_ = self.level_ = self.sse_ = 0.0
        self.alpha_ = self.alphas[0]
        y_bar = float(y.mean())
        self._zero = y_bar <= 0.0
        if self._zero:
            self._fitted = True
            return self
        if n >= 2 * m:
            by_position = np.array([y[j::m].mean() for j in range(m)])
            self.seasonal_ = np.clip(by_position / y_bar, 0.1, 10.0)
        x = y / self.seasonal_[np.arange(n) % m]
        t = np.arange(n, dtype=float)
        t_bar, x_bar = t.mean(), x.mean()
        denom = float(np.sum((t - t_bar) ** 2))
        self.slope_ = float(np.sum((t - t_bar) * (x - x_bar)) / denom) if denom > 0 else 0.0
        self.intercept_ = float(x_bar - self.slope_ * t_bar)
        z = (2.0 * x - (self.intercept_ + self.slope_ * t)).tolist()
        best_alpha, best_level, best_sse = self.alphas[0], float(z[0]), float("inf")
        for alpha in self.alphas:
            level, sse = self._ses(z, alpha)
            if sse < best_sse:
                best_alpha, best_level, best_sse = alpha, level, sse
        self.alpha_, self.level_, self.sse_ = best_alpha, best_level, best_sse
        self._fitted = True
        return self

    def predict(self, h: int, future_flags: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        h = int(h)
        if self._zero:
            return np.zeros(h)
        t = self._n + np.arange(h)  # time index of each step: n + h - 1
        line = self.intercept_ + self.slope_ * t
        yhat = (0.5 * line + 0.5 * self.level_) * self.seasonal_[t % self.season]
        return np.maximum(yhat, 0.0)


# ------------------------------------------------------------------------------------------
class Croston(Forecaster):
    """Croston's method with the Syntetos-Boylan approximation (SBA) for intermittent demand.

    Non-zero demand sizes z and inter-demand intervals q are smoothed separately with the same
    alpha, updated only in periods with demand:

        z <- alpha * y_t + (1 - alpha) * z
        q <- alpha * tau_t + (1 - alpha) * q

    where tau_t is the number of periods since the previous demand (a demand in the very next
    period gives tau = 1). The first demand initialises z = y_t and q = t + 1 (periods elapsed
    from the start of the series up to and including it); the period counter restarts at 0 after
    EVERY demand, including the first one, so all intervals are counted with the same rule (the
    original implementation started the counter at 1 after the first demand and over-counted
    the first interval by one).

        forecast = (1 - alpha / 2) * z / q        (flat over the horizon; 0 for all-zero history)

    Fitted attributes: ``size_`` (z) and ``interval_`` (q).
    """

    name = "croston_sba"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._rate = 0.0
        self.size_ = 0.0
        self.interval_ = 0.0

    def fit(self, y: ArrayLike, promo_flags: np.ndarray | None = None) -> Croston:
        y = self._clean(y)
        nz = np.flatnonzero(y > 0)
        if nz.size == 0:
            self._rate, self.size_, self.interval_ = 0.0, 0.0, 0.0
            self._fitted = True
            return self
        z = float(y[nz[0]])  # size
        q = float(nz[0] + 1)  # first interval
        interval = 0.0  # periods since the last demand (same convention for every interval)
        for t in range(nz[0] + 1, y.size):
            interval += 1
            if y[t] > 0:
                z = self.alpha * y[t] + (1 - self.alpha) * z
                q = self.alpha * interval + (1 - self.alpha) * q
                interval = 0.0
        self.size_, self.interval_ = float(z), float(q)
        self._rate = (1 - self.alpha / 2) * z / max(q, 1e-9)
        self._fitted = True
        return self

    def predict(self, h: int, future_flags: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        return np.full(h, max(self._rate, 0.0))


# ------------------------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[Forecaster]] = {  # insertion order = candidate order
    SeasonalNaive.name: SeasonalNaive,
    MovingAverage.name: MovingAverage,
    HoltWinters.name: HoltWinters,
    Theta.name: Theta,
    Croston.name: Croston,
}


def make_model(name: str) -> Forecaster:
    """Instantiate a model by name; ``"promo_<base>"`` wraps ``<base>`` in ``PromoAdjusted``."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]()
    from .promo import PROMO_MODEL_NAMES, PromoAdjusted  # lazy: promo imports this module

    if name.startswith(PROMO_PREFIX) and name[len(PROMO_PREFIX) :] in MODEL_REGISTRY:
        return PromoAdjusted(MODEL_REGISTRY[name[len(PROMO_PREFIX) :]]())
    choices = list(MODEL_REGISTRY) + list(PROMO_MODEL_NAMES)
    raise KeyError(f"unknown model '{name}', choose from {choices}")


def is_intermittent(y: np.ndarray, zero_share: float = 0.5) -> bool:
    """Heuristic used to decide whether Croston is a sensible candidate."""
    y = np.asarray(y, dtype=float)
    return y.size > 0 and float(np.mean(y == 0)) >= zero_share
