import numpy as np
import pytest

from demandcast.models import (
    MODEL_REGISTRY,
    Croston,
    HoltWinters,
    HWParams,
    MovingAverage,
    SeasonalNaive,
    is_intermittent,
    make_model,
)


def weekly_series(n=140, base=20.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    profile = np.array([0.8, 0.9, 0.9, 1.0, 1.1, 1.3, 1.0])
    y = base * profile[np.arange(n) % 7]
    return y + rng.normal(0, noise, n) if noise else y


def test_seasonal_naive_repeats_last_week():
    y = weekly_series(70)
    pred = SeasonalNaive().fit(y).predict(14)
    np.testing.assert_allclose(pred[:7], y[-7:])
    np.testing.assert_allclose(pred[7:], y[-7:])


def test_seasonal_naive_short_series_uses_mean():
    pred = SeasonalNaive().fit([3, 5, 7]).predict(7)
    assert np.allclose(pred, 5.0)


def test_moving_average_recovers_weekday_profile():
    y = weekly_series(140)
    pred = MovingAverage(window=28).fit(y).predict(7)
    # Deterministic series → the re-profiled MA should reproduce the next week almost exactly.
    np.testing.assert_allclose(pred, y[:7], rtol=1e-6)


def test_holt_winters_tracks_trend_and_season():
    n = 210
    t = np.arange(n)
    y = 30 + 0.05 * t + 5 * np.sin(2 * np.pi * t / 7)
    model = HoltWinters().fit(y)
    pred = model.predict(14)
    truth = 30 + 0.05 * (n + np.arange(14)) + 5 * np.sin(2 * np.pi * (n + np.arange(14)) / 7)
    assert np.mean(np.abs(pred - truth)) < 1.5


def test_holt_winters_manual_params_not_overridden():
    p = HWParams(alpha=0.2, beta=0.02, gamma=0.1, phi=0.9)
    m = HoltWinters(params=p, auto=False).fit(weekly_series(100, noise=1.0))
    assert m.params == p


def test_croston_on_intermittent_demand():
    rng = np.random.default_rng(1)
    y = np.where(rng.random(400) < 0.2, rng.integers(1, 5, 400), 0).astype(float)
    m = Croston(alpha=0.1).fit(y)
    pred = m.predict(10)
    assert pred.shape == (10,)
    # forecast should be near the true mean demand rate (0.2 * 2.5 = 0.5)
    assert 0.25 < pred[0] < 0.9


def test_croston_all_zero_series():
    assert np.all(Croston().fit(np.zeros(50)).predict(5) == 0)


def test_is_intermittent():
    assert is_intermittent(np.array([0, 0, 0, 1, 0, 2, 0, 0]))
    assert not is_intermittent(np.array([3, 4, 5, 6, 0, 7]))


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_all_models_nonnegative_and_correct_shape(name):
    rng = np.random.default_rng(3)
    y = np.maximum(rng.normal(2, 3, 120), 0)
    pred = make_model(name).fit(y).predict(28)
    assert pred.shape == (28,)
    assert np.all(pred >= 0)
    assert not np.any(np.isnan(pred))


def test_make_model_unknown_name():
    with pytest.raises(KeyError):
        make_model("prophet")


def test_models_reject_bad_input():
    with pytest.raises(ValueError):
        SeasonalNaive().fit([])
    with pytest.raises(ValueError):
        MovingAverage().fit([1.0, np.nan, 2.0])
