import numpy as np
import pytest

from demandcast.models import MODEL_REGISTRY, Forecaster, make_model
from demandcast.promo import (
    PROMO_BASES,
    PROMO_MODEL_NAMES,
    PromoAdjusted,
    deflate,
    estimate_lift,
    has_promo_signal,
    inflate,
)

PROFILE = np.array([0.8, 0.9, 0.9, 1.0, 1.1, 1.3, 1.0])


PROMO_BLOCKS = ((30, 44), (90, 104), (150, 164), (210, 224), (250, 264))


def promo_series(n=280, base=40.0, lift=1.8, seed=42, blocks=PROMO_BLOCKS):
    """Poisson demand with a weekly profile, multiplied by `lift` on promo days."""
    rng = np.random.default_rng(seed)
    flags = np.zeros(n)
    for a, b in blocks:
        flags[a:b] = 1
    mu = base * PROFILE[np.arange(n) % 7] * np.where(flags == 1, lift, 1.0)
    return rng.poisson(mu).astype(float), flags


def test_promo_constants():
    assert PROMO_BASES == ("moving_average", "holt_winters", "theta")
    assert PROMO_MODEL_NAMES == ("promo_moving_average", "promo_holt_winters", "promo_theta")
    assert set(PROMO_BASES) <= set(MODEL_REGISTRY)


def test_has_promo_signal_thresholds():
    assert has_promo_signal(None) is False
    assert has_promo_signal(np.zeros(0)) is False
    flags = np.zeros(100)
    assert has_promo_signal(flags) is False  # no promo days at all
    flags[:6] = 1
    assert has_promo_signal(flags) is False  # 6 < 7 promo days
    flags[:7] = 1
    assert has_promo_signal(flags) is True  # 7 promo / 93 base
    assert has_promo_signal(flags, min_promo_days=8) is False
    assert has_promo_signal(flags, min_base_days=94) is False
    just_enough = np.concatenate([np.ones(7), np.zeros(28)])
    assert has_promo_signal(just_enough) is True
    assert has_promo_signal(np.concatenate([np.ones(7), np.zeros(27)])) is False
    assert has_promo_signal(np.ones(60)) is False  # no base days
    assert has_promo_signal([1, 1, 1, 1, 1, 1, 1] + [0] * 28) is True  # plain lists accepted


def test_estimate_lift_recovers_known_lift_within_ten_percent():
    y, flags = promo_series(lift=1.8)
    assert has_promo_signal(flags)
    lift = estimate_lift(y, flags)
    assert abs(lift - 1.8) / 1.8 < 0.10
    # deterministic for the same seed and sensitive to the true lift
    assert estimate_lift(*promo_series(lift=1.8)) == lift
    y3, flags3 = promo_series(lift=3.0)
    assert estimate_lift(y3, flags3) > lift


def test_estimate_lift_shrinkage_and_clamps():
    # base days worth 10 units, promo days worth 30 => raw ratio 3; shrinkage n/(n+7) toward 1
    def series(n_promo, promo_value, base_value=10.0, n_base=40):
        y = np.concatenate([np.full(n_base, base_value), np.full(n_promo, promo_value)])
        flags = np.concatenate([np.zeros(n_base), np.ones(n_promo)])
        return y, flags

    assert estimate_lift(*series(7, 30.0)) == pytest.approx(1 + 2 * 7 / 14)  # 2.0
    assert estimate_lift(*series(21, 30.0)) == pytest.approx(1 + 2 * 21 / 28)  # 2.5
    assert estimate_lift(*series(63, 30.0)) == pytest.approx(1 + 2 * 63 / 70)  # 2.8
    # fewer promo days than min_promo_days -> no evidence -> 1.0
    assert estimate_lift(*series(6, 30.0)) == 1.0
    assert estimate_lift(*series(6, 30.0), min_promo_days=6) == pytest.approx(1 + 2 * 6 / 13)
    # upper clamp: ratio 10 with 70 promo days would give 9.18 -> hi
    assert estimate_lift(*series(70, 100.0)) == 5.0
    assert estimate_lift(*series(70, 100.0), hi=3.0) == 3.0
    # lower clamp: promo days sell less than base days -> never below lo
    assert estimate_lift(*series(21, 5.0)) == 1.0
    assert estimate_lift(*series(21, 5.0), lo=0.5) == pytest.approx(1 + (0.5 - 1) * 21 / 28)
    # base mean 0 -> 1.0; all days on promo -> 1.0; no promo days -> 1.0
    assert estimate_lift(*series(21, 30.0, base_value=0.0)) == 1.0
    assert estimate_lift(np.full(30, 12.0), np.ones(30)) == 1.0
    assert estimate_lift(np.full(30, 12.0), np.zeros(30)) == 1.0
    assert isinstance(estimate_lift(*series(21, 30.0)), float)
    with pytest.raises(ValueError):
        estimate_lift(np.ones(10), np.ones(9))


def test_deflate_inflate_round_trip():
    y, flags = promo_series()
    lift = 1.7
    d = deflate(y, flags, lift)
    assert d.shape == y.shape
    np.testing.assert_allclose(d[flags == 0], y[flags == 0])
    np.testing.assert_allclose(d[flags == 1], y[flags == 1] / lift)
    np.testing.assert_allclose(inflate(d, flags, lift), y, rtol=1e-12)
    # explicit factor semantics on a tiny example
    np.testing.assert_allclose(inflate([2.0, 2.0, 2.0], [0, 1, 0], 1.5), [2.0, 3.0, 2.0])
    np.testing.assert_allclose(deflate([3.0, 3.0], [1, 0], 1.5), [2.0, 3.0])
    # None flags leave the values unchanged and never alias the input
    out = inflate(y, None, lift)
    np.testing.assert_array_equal(out, y)
    assert out is not y
    np.testing.assert_array_equal(deflate(y, None, lift), y)
    with pytest.raises(ValueError):
        inflate(np.ones(5), np.ones(4), lift)


@pytest.mark.parametrize("base_name", PROMO_BASES)
def test_promo_adjusted_equals_base_when_flags_none(base_name):
    y, _ = promo_series()
    m = PromoAdjusted(make_model(base_name))
    assert isinstance(m, Forecaster)
    assert m.supports_promo is True
    assert m.name == f"promo_{base_name}"
    pred = m.fit(y).predict(21)
    assert m.lift_ == 1.0
    np.testing.assert_allclose(pred, make_model(base_name).fit(y).predict(21), rtol=1e-12)
    # future flags without any lift estimate are a no-op as well
    fut = np.zeros(21)
    fut[2:9] = 1
    np.testing.assert_allclose(m.predict(21, future_flags=fut), pred, rtol=1e-12)


@pytest.mark.parametrize("base_name", PROMO_BASES)
def test_promo_adjusted_scales_future_promo_days_by_lift(base_name):
    y, flags = promo_series()
    m = make_model(f"promo_{base_name}").fit(y, promo_flags=flags)
    assert isinstance(m, PromoAdjusted)
    assert 1.3 < m.lift_ < 2.2
    assert m.lift_ == pytest.approx(estimate_lift(y, flags))
    h = 14
    fut = np.zeros(h)
    fut[3:8] = 1
    base_pred = make_model(base_name).fit(deflate(y, flags, m.lift_)).predict(h)
    np.testing.assert_allclose(m.predict(h), base_pred, rtol=1e-12)
    pred = m.predict(h, future_flags=fut)
    np.testing.assert_allclose(pred, base_pred * np.where(fut == 1, m.lift_, 1.0), rtol=1e-12)
    np.testing.assert_allclose(pred[fut == 0], base_pred[fut == 0], rtol=1e-12)
    assert np.all(pred[fut == 1] > base_pred[fut == 1])
    assert pred.shape == (h,)
    assert np.all(pred >= 0)
    assert not np.any(np.isnan(pred))


def test_promo_adjusted_without_signal_matches_base():
    y, _ = promo_series()
    sparse = np.zeros(y.size)
    sparse[10:14] = 1  # 4 promo days < min_promo_days -> lift 1.0 -> identical to the base model
    m = PromoAdjusted(make_model("moving_average")).fit(y, promo_flags=sparse)
    assert m.lift_ == 1.0
    fut = np.ones(7)
    np.testing.assert_allclose(
        m.predict(7, future_flags=fut), make_model("moving_average").fit(y).predict(7)
    )


@pytest.mark.parametrize("name", PROMO_MODEL_NAMES)
def test_promo_adjusted_predict_before_fit_raises(name):
    with pytest.raises(RuntimeError, match=r"call fit\(\) first"):
        make_model(name).predict(7)


def test_promo_adjusted_rejects_misaligned_flags():
    y, flags = promo_series()
    with pytest.raises(ValueError):
        PromoAdjusted(make_model("theta")).fit(y, promo_flags=flags[:-1])
    m = PromoAdjusted(make_model("theta")).fit(y, promo_flags=flags)
    with pytest.raises(ValueError):
        m.predict(14, future_flags=np.zeros(13))
    with pytest.raises(ValueError):
        PromoAdjusted(make_model("theta")).fit([], promo_flags=None)
