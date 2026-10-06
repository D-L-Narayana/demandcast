"""Promotion-aware forecasting: estimate a per-series promo lift and wrap any base model.

Pure NumPy, no SQL. Flags are 0/1 (int or float) arrays aligned with the history ``y``
(``promo_flags``) or with the forecast steps (``future_flags``); a value > 0 marks a promo day.

    lift   = clamp( 1 + (mean(y | promo) / mean(y | base) - 1) * n_promo / (n_promo + 7), lo, hi )
    fit    : base.fit( y / where(promo, lift, 1) )              # history deflated to base demand
    predict: base.predict(h) * where(future_promo, lift, 1)     # lift re-applied on promo days

The shrinkage factor n_promo / (n_promo + 7) pulls the raw ratio toward 1 when only a few promo
days are observed; the clamp (default [1, 5]) keeps a noisy ratio from inflating or deflating
the forecast. ``PromoAdjusted`` competes in the same backtest as the base models and is only
used when it wins.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .models import Forecaster

PROMO_BASES = ("moving_average", "holt_winters", "theta")
PROMO_MODEL_NAMES = tuple(f"promo_{b}" for b in PROMO_BASES)
SHRINKAGE_DAYS = 7  # pseudo-observations pulling the lift ratio toward 1


def _as_mask(flags: ArrayLike, n: int | None = None, what: str = "flags") -> np.ndarray:
    """0/1 flags -> boolean mask; validates the length against ``n`` when given."""
    mask = np.asarray(flags, dtype=float).ravel() > 0
    if n is not None and mask.size != n:
        raise ValueError(f"{what} length {mask.size} does not match series length {n}")
    return mask


def has_promo_signal(
    flags: ArrayLike | None, min_promo_days: int = 7, min_base_days: int = 28
) -> bool:
    """True when the history holds enough promo AND base days to estimate a lift.

    Both thresholds must be met (``None`` or empty flags -> False):

        n_promo >= min_promo_days  and  n_base >= min_base_days
    """
    if flags is None:
        return False
    mask = _as_mask(flags)
    n_promo = int(mask.sum())
    n_base = int(mask.size - n_promo)
    return n_promo >= min_promo_days and n_base >= min_base_days


def estimate_lift(
    y: ArrayLike,
    flags: ArrayLike | None,
    min_promo_days: int = 7,
    lo: float = 1.0,
    hi: float = 5.0,
) -> float:
    """Shrunken, clamped multiplicative promo lift of ``y`` on promo days vs. base days.

        ratio = mean(y[flags == 1]) / mean(y[flags == 0])
        lift  = 1 + (ratio - 1) * n_promo / (n_promo + 7)        (shrinkage toward 1)
        lift  = min(max(lift, lo), hi)                            (clamp)

    Returns 1.0 (no adjustment) when flags are None, fewer than ``min_promo_days`` promo days
    or no base days are available, or when the base-day mean is zero.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if flags is None:
        return 1.0
    mask = _as_mask(flags, y_arr.size)
    n_promo = int(mask.sum())
    n_base = int(mask.size - n_promo)
    if n_promo < min_promo_days or n_promo == 0 or n_base == 0:
        return 1.0
    base_mean = float(y_arr[~mask].mean())
    if base_mean <= 0.0:
        return 1.0
    ratio = float(y_arr[mask].mean()) / base_mean
    lift = 1.0 + (ratio - 1.0) * n_promo / (n_promo + SHRINKAGE_DAYS)
    return float(min(max(lift, lo), hi))


def _factor(flags: ArrayLike | None, n: int, lift: float, what: str) -> np.ndarray:
    if lift <= 0.0:
        raise ValueError("lift must be positive")
    if flags is None:
        return np.ones(n)
    return np.where(_as_mask(flags, n, what), float(lift), 1.0)


def deflate(y: ArrayLike, flags: ArrayLike | None, lift: float) -> np.ndarray:
    """Remove the promo effect from history: ``y / where(flags, lift, 1)`` (None -> copy of y)."""
    y_arr = np.array(y, dtype=float).ravel()
    return y_arr / _factor(flags, y_arr.size, lift, "promo_flags")


def inflate(yhat: ArrayLike, future_flags: ArrayLike | None, lift: float) -> np.ndarray:
    """Re-apply the promo effect: ``yhat * where(future_flags, lift, 1)`` (None -> copy)."""
    yhat_arr = np.array(yhat, dtype=float).ravel()
    return yhat_arr * _factor(future_flags, yhat_arr.size, lift, "future_flags")


class PromoAdjusted(Forecaster):
    """Wrap a base forecaster: fit it on promo-deflated history, re-inflate future promo days.

    ``name`` is ``"promo_<base.name>"``; the estimated lift is exposed as ``lift_``. With
    ``promo_flags=None`` the lift is 1 and the wrapper behaves exactly like the base model;
    with ``future_flags=None`` the base forecast is returned unchanged.
    """

    supports_promo = True

    def __init__(self, base: Forecaster):
        self.base = base
        self.name = f"promo_{base.name}"
        self.lift_ = 1.0

    def fit(self, y: ArrayLike, promo_flags: np.ndarray | None = None) -> PromoAdjusted:
        y_arr = self._clean(y)
        if promo_flags is None:
            self.lift_ = 1.0
            self.base.fit(y_arr)
        else:
            _as_mask(promo_flags, y_arr.size, "promo_flags")
            self.lift_ = estimate_lift(y_arr, promo_flags)
            self.base.fit(deflate(y_arr, promo_flags, self.lift_))
        self._fitted = True
        return self

    def predict(self, h: int, future_flags: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        yhat = self.base.predict(h)
        if future_flags is None:
            return yhat
        return np.maximum(inflate(yhat, future_flags, self.lift_), 0.0)
