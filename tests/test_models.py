import numpy as np
import pytest

from demandcast.models import (
    MODEL_REGISTRY,
    Croston,
    HoltWinters,
    HWParams,
    MovingAverage,
    SeasonalNaive,
    Theta,
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


# ---- C1 forecaster interface --------------------------------------------------------------
ALL_MODEL_CLASSES = [SeasonalNaive, MovingAverage, HoltWinters, Theta, Croston]


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES, ids=lambda c: c.name)
def test_base_models_accept_and_ignore_promo_flags(cls):
    y = weekly_series(84, noise=1.0, seed=5)
    flags = np.zeros(84)
    flags[10:30] = 1
    future = np.zeros(14)
    future[::2] = 1
    plain = cls().fit(y).predict(14)
    flagged = cls().fit(y, promo_flags=flags).predict(14, future_flags=future)
    np.testing.assert_array_equal(plain, flagged)
    assert getattr(cls, "supports_promo", None) is False
    assert plain.shape == (14,)
    assert np.all(plain >= 0)
    assert not np.any(np.isnan(plain))


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_predict_before_fit_raises_runtime_error(name):
    # D10: HoltWinters used to raise AttributeError, SeasonalNaive an AssertionError and
    # the other models silently returned zeros.
    with pytest.raises(RuntimeError, match=r"call fit\(\) first"):
        make_model(name).predict(7)


# ---- D3: Croston interval recursion --------------------------------------------------------
def test_croston_hand_computed_recursion():
    # alpha = 0.1, demand at t = 2, 6, 7, 11 (0-based). Interval = periods since the previous
    # demand, counting the demand period itself (consecutive demand => 1).
    #   t=2 : z = 3,  q = 3 (first demand in period 3)
    #   t=6 : z = 0.1*5 + 0.9*3    = 3.2,   q = 0.1*4 + 0.9*3    = 3.1
    #   t=7 : z = 0.1*2 + 0.9*3.2  = 3.08,  q = 0.1*1 + 0.9*3.1  = 2.89
    #   t=11: z = 0.1*4 + 0.9*3.08 = 3.172, q = 0.1*4 + 0.9*2.89 = 3.001
    #   rate = (1 - 0.1/2) * 3.172 / 3.001 = 1.004132
    # The pre-fix code started the counter at 1 instead of 0 after the first demand, counted
    # the t=2 -> t=6 gap as 5 and ended at q = 3.082 (rate 0.97774).
    y = np.array([0, 0, 3, 0, 0, 0, 5, 2, 0, 0, 0, 4], dtype=float)
    m = Croston(alpha=0.1).fit(y)
    assert m.predict(1)[0] == pytest.approx(0.95 * 3.172 / 3.001, rel=1e-12)
    assert m.predict(1)[0] == pytest.approx(1.004132, abs=1e-6)
    assert getattr(m, "size_", None) == pytest.approx(3.172, rel=1e-12)
    assert getattr(m, "interval_", None) == pytest.approx(3.001, rel=1e-12)
    assert np.all(m.predict(5) == m.predict(1)[0])


def test_croston_consecutive_demand_has_unit_interval():
    # Uniform interval counting: demand every period => smoothed interval stays exactly 1,
    # so the SBA forecast is (1 - alpha/2) * size.
    m = Croston(alpha=0.1).fit(np.ones(12))
    assert m.predict(1)[0] == pytest.approx(0.95, rel=1e-12)
    assert getattr(m, "interval_", None) == pytest.approx(1.0, rel=1e-12)


# ---- Theta -------------------------------------------------------------------------------
def trend_weekly_series(n, base=20.0, slope=0.05):
    profile = np.array([0.8, 0.9, 0.9, 1.0, 1.1, 1.3, 1.0])
    t = np.arange(n)
    return (base + slope * t) * profile[t % 7]


def test_theta_linear_weekly_series_mae_below_one():
    n, h = 140, 14
    y = trend_weekly_series(n + h)
    m = Theta().fit(y[:n])
    pred = m.predict(h)
    assert pred.shape == (h,)
    assert np.mean(np.abs(pred - y[n:])) < 1.0
    # the weekly pattern must be reproduced: the peak weekday (index 5) is the weekly maximum
    for week in (pred[:7], pred[7:]):
        assert int(np.argmax(week)) == int(np.argmax(y[n : n + 7]))


def test_theta_constant_series_forecasts_constant():
    pred = Theta().fit(np.full(60, 5.0)).predict(10)
    np.testing.assert_allclose(pred, 5.0, rtol=1e-9)


def test_theta_all_zero_series_returns_zeros():
    pred = Theta().fit(np.zeros(40)).predict(7)
    assert pred.shape == (7,)
    assert np.all(pred == 0)


def test_theta_short_series_skips_deseasonalisation():
    # n < 2 * season: no weekly indices, plain theta on the raw series; still finite and >= 0
    y = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    m = Theta().fit(y)
    np.testing.assert_array_equal(m.seasonal_, np.ones(7))
    pred = m.predict(5)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0)
    # positive slope recovered by the LS trend line and reflected in a rising forecast
    assert pred[-1] > pred[0]
    assert m.slope_ == pytest.approx(1.0)
    # a single observation also fits (LS slope 0, SES level = the observation)
    np.testing.assert_allclose(Theta().fit([3.0]).predict(3), 3.0)


def test_theta_indices_clipped_and_alpha_from_grid():
    y = weekly_series(98, noise=2.0, seed=9)
    y[5::7] = 0.0  # one weekday never sells -> raw index 0 -> clipped to 0.1
    m = Theta().fit(y)
    assert m.seasonal_.shape == (7,)
    assert np.all(m.seasonal_ >= 0.1)
    assert np.all(m.seasonal_ <= 10.0)
    assert m.seasonal_[5] == pytest.approx(0.1)
    assert m.alpha_ in {0.1, 0.2, 0.3, 0.5}
    pred = m.predict(28)
    assert np.all(pred >= 0)
    assert not np.any(np.isnan(pred))
    # Upper clip: with m positions a single index is bounded by m, so a 7-day season can never
    # exceed 10 on non-negative data; a 14-day season can (raw 1000 / ((13 + 1000) / 14) = 13.8).
    y14 = np.ones(56)
    y14[3::14] = 1000.0
    m14 = Theta(season=14).fit(y14)
    assert m14.seasonal_.shape == (14,)
    assert m14.seasonal_[3] == pytest.approx(10.0)
    assert np.all(m14.seasonal_ <= 10.0)
    assert np.all(m14.seasonal_ >= 0.1)


def assert_seasonal_contract(seasonal, season):
    # `Theta.seasonal_` is always a 1-D float64 array with one index per position in the cycle
    assert isinstance(seasonal, np.ndarray)
    assert seasonal.dtype == np.float64
    assert seasonal.ndim == 1
    assert seasonal.shape == (season,)


def test_theta_seasonal_indices_are_1d_float64_with_exact_clipped_values():
    # Unfitted: flat indices of the requested length
    assert_seasonal_contract(Theta().seasonal_, 7)
    np.testing.assert_array_equal(Theta().seasonal_, np.ones(7))
    assert_seasonal_contract(Theta(season=12).seasonal_, 12)
    # Two identical weeks 0, 7, ..., 42: mean 21 -> raw indices j/3 (all sums exact), the zero
    # weekday clipped up to 0.1; the fitted forecast is pinned too (values from the model as
    # shipped, so the typing of `seasonal_` can never silently change the numbers).
    m = Theta().fit(np.tile(np.arange(0.0, 49.0, 7.0), 2))
    assert_seasonal_contract(m.seasonal_, 7)
    np.testing.assert_array_equal(m.seasonal_, [0.1, 1 / 3, 2 / 3, 1.0, 4 / 3, 5 / 3, 2.0])
    assert m.alpha_ == 0.5
    assert m.slope_ == pytest.approx(0.5538461538461539, rel=1e-12)
    assert m.intercept_ == pytest.approx(14.4, rel=1e-12)
    np.testing.assert_allclose(
        m.predict(7),
        [
            2.1387186373197116,
            7.221369816706731,
            14.627355018028846,
            22.217955603966345,
            29.993171574519227,
            37.9530029296875,
            46.09744966947115,
        ],
        rtol=1e-12,
    )
    # season 12, flat 1 with a 1000 spike at position 3: mean 84.25 -> 11.87 clipped to 10,
    # 0.0119 clipped to 0.1 everywhere else
    y12 = np.ones(24)
    y12[3::12] = 1000.0
    m12 = Theta(season=12).fit(y12)
    assert_seasonal_contract(m12.seasonal_, 12)
    expected12 = np.full(12, 0.1)
    expected12[3] = 10.0
    np.testing.assert_array_equal(m12.seasonal_, expected12)
    # n < 2 * season: no deseasonalisation -> all ones, same dtype/shape contract
    short = Theta().fit(np.arange(1.0, 14.0))
    assert_seasonal_contract(short.seasonal_, 7)
    np.testing.assert_array_equal(short.seasonal_, np.ones(7))
    # all-zero history keeps the flat indices and forecasts zeros
    zero = Theta().fit(np.zeros(20))
    assert_seasonal_contract(zero.seasonal_, 7)
    np.testing.assert_array_equal(zero.seasonal_, np.ones(7))
    np.testing.assert_array_equal(zero.predict(7), np.zeros(7))


# ---- Holt-Winters: batch grid evaluation == scalar reference ------------------------------
def hw_scalar_reference_fit(model, y):
    """Baseline selection loop, kept in the test as the per-candidate reference."""
    best = None
    for p in model.grid_candidates():
        level, trend, seasonal, sse, n = model._run(y, p)
        if best is None or sse < best[3]:
            best = (level, trend, seasonal.copy(), sse, n, p)
    return best


def noisy_trend_series(seed, n):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.maximum(10 + 0.02 * t + 3 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1.5, n), 0)


# The batch path evaluates the same equations in state-space form, so floating-point rounding
# differs from the scalar recursion: compare at rtol 1e-9 with a tiny absolute floor (1e-9 on
# series of magnitude ~10-30) for states that are exactly 0 in one path and ~1e-16 in the other.
HW_RTOL, HW_ATOL = 1e-9, 1e-9


def assert_hw_batch_matches_scalar(hw, y, params, block=None):
    kwargs = {} if block is None else {"block": block}
    levels, trends, seasonals, sses = hw._run_batch(y, params, **kwargs)
    c = len(params)
    assert levels.shape == (c,)
    assert trends.shape == (c,)
    assert seasonals.shape == (c, hw.season)
    assert sses.shape == (c,)
    for i, p in enumerate(params):
        level, trend, seasonal, sse, _ = hw._run(y, p)
        np.testing.assert_allclose(levels[i], level, rtol=HW_RTOL, atol=HW_ATOL)
        np.testing.assert_allclose(trends[i], trend, rtol=HW_RTOL, atol=HW_ATOL)
        np.testing.assert_allclose(seasonals[i], seasonal, rtol=HW_RTOL, atol=HW_ATOL)
        np.testing.assert_allclose(sses[i], sse, rtol=HW_RTOL, atol=HW_ATOL)
    return levels, trends, seasonals, sses


@pytest.mark.parametrize(
    ("seed", "n"),
    [
        (0, 730),
        (1, 100),
        (2, 20),
        (3, 101),
        (4, 365),
        (5, 6),
        (6, 7),
        (8, 1),
        (9, 27),
        (10, 28),
        (11, 29),
        (12, 56),
    ],
)
def test_holt_winters_batch_matches_scalar(seed, n):
    y = noisy_trend_series(seed, n)
    hw = HoltWinters()
    params = hw.grid_candidates()
    assert len(params) == 12
    assert_hw_batch_matches_scalar(hw, y, params)


@pytest.mark.parametrize("block", [1, 5, 7, 28, 1000])
def test_holt_winters_batch_block_size_does_not_change_results(block):
    y = noisy_trend_series(3, 101)
    hw = HoltWinters()
    params = hw.grid_candidates()
    out = assert_hw_batch_matches_scalar(hw, y, params, block=block)
    ref = hw._run_batch(y, params)
    for a, b in zip(out, ref, strict=True):
        np.testing.assert_allclose(a, b, rtol=HW_RTOL, atol=HW_ATOL)


def test_holt_winters_batch_other_season_and_single_candidate():
    rng = np.random.default_rng(21)
    t = np.arange(100)
    y = np.maximum(15 + 4 * np.sin(2 * np.pi * t / 14) + rng.normal(0, 1, 100), 0)
    hw = HoltWinters(season=14)
    assert_hw_batch_matches_scalar(hw, y, [HWParams(0.4, 0.05, 0.25, 0.9)])
    assert_hw_batch_matches_scalar(hw, y, [HWParams(), HWParams(0.5, 0.1, 0.3, 0.95)])


@pytest.mark.parametrize(("seed", "n"), [(0, 730), (1, 100), (3, 101), (4, 365), (7, 21)])
def test_holt_winters_fit_matches_scalar_reference_selection(seed, n):
    y = noisy_trend_series(seed, n)
    m = HoltWinters().fit(y)
    level, trend, seasonal, sse, _, p = hw_scalar_reference_fit(HoltWinters(), y)
    assert m.params == p
    assert m.params in m.grid_candidates()
    np.testing.assert_allclose(m._level, level, rtol=HW_RTOL, atol=HW_ATOL)
    np.testing.assert_allclose(m._trend, trend, rtol=HW_RTOL, atol=HW_ATOL)
    np.testing.assert_allclose(m._season, seasonal, rtol=HW_RTOL, atol=HW_ATOL)
    np.testing.assert_allclose(m.sse_, sse, rtol=HW_RTOL, atol=HW_ATOL)
    ref = HoltWinters(params=p, auto=False).fit(y)
    np.testing.assert_allclose(m.predict(28), ref.predict(28), rtol=HW_RTOL, atol=HW_ATOL)


def test_holt_winters_grid_skipped_below_three_seasons():
    y = noisy_trend_series(2, 20)  # n < 3 * season -> default params, scalar path
    m = HoltWinters().fit(y)
    assert m.params == HWParams()
    level, _trend, seasonal, sse, _ = HoltWinters()._run(y, HWParams())
    np.testing.assert_allclose(m._level, level, rtol=1e-9, atol=0)
    np.testing.assert_allclose(m._season, seasonal, rtol=1e-9, atol=0)
    np.testing.assert_allclose(m.sse_, sse, rtol=1e-9, atol=0)


def test_holt_winters_exposes_chosen_params_and_sse():
    y = noisy_trend_series(11, 200)
    m = HoltWinters().fit(y)
    assert isinstance(m.params, HWParams)
    assert m.params in m.grid_candidates()
    assert np.isfinite(m.sse_)
    scalar_sses = [HoltWinters()._run(y, p)[3] for p in m.grid_candidates()]
    assert m.sse_ == pytest.approx(min(scalar_sses), rel=HW_RTOL)
    assert m.params == m.grid_candidates()[int(np.argmin(scalar_sses))]


# ---- registry & factory -------------------------------------------------------------------
def test_model_registry_order():
    assert list(MODEL_REGISTRY) == [
        "seasonal_naive",
        "moving_average",
        "holt_winters",
        "theta",
        "croston_sba",
    ]
    assert all(MODEL_REGISTRY[k].name == k for k in MODEL_REGISTRY)


def test_make_model_promo_names():
    from demandcast.promo import PROMO_MODEL_NAMES, PromoAdjusted

    assert PROMO_MODEL_NAMES == ("promo_moving_average", "promo_holt_winters", "promo_theta")
    for name in PROMO_MODEL_NAMES:
        m = make_model(name)
        assert isinstance(m, PromoAdjusted)
        assert m.name == name
        assert m.base.name == name[len("promo_") :]
        assert m.supports_promo is True
        assert isinstance(m.base, MODEL_REGISTRY[m.base.name])


def test_make_model_unknown_name_lists_choices():
    with pytest.raises(KeyError) as excinfo:
        make_model("prophet")
    msg = str(excinfo.value)
    for name in ("seasonal_naive", "theta", "croston_sba", "promo_theta", "promo_holt_winters"):
        assert name in msg
    with pytest.raises(KeyError):
        make_model("promo_prophet")


# ---- MovingAverage: weekday profile from the last 8 weeks ---------------------------------
P_EARLY = np.array([1.3, 1.1, 1.0, 0.9, 0.9, 0.8, 1.0])
P_LATE = np.array([0.8, 0.9, 0.9, 1.0, 1.1, 1.3, 1.0])


def regime_change_series(level=20.0, early_weeks=12, late_weeks=8):
    early = np.tile(level * P_EARLY, early_weeks)
    late = np.tile(level * P_LATE, late_weeks)
    return np.concatenate([early, late])


def test_moving_average_profile_uses_last_eight_weeks():
    y = regime_change_series()  # 20 weeks; the weekday pattern changed 8 weeks ago
    pred = MovingAverage(window=28).fit(y).predict(14)
    np.testing.assert_allclose(pred[:7], 20.0 * P_LATE, rtol=1e-9)
    np.testing.assert_allclose(pred[7:], 20.0 * P_LATE, rtol=1e-9)


def test_moving_average_profile_weeks_parameter_and_short_history():
    y = regime_change_series()
    blended = (12 * P_EARLY + 8 * P_LATE) / 20
    pred_all = MovingAverage(window=28, profile_weeks=20).fit(y).predict(7)
    np.testing.assert_allclose(pred_all, 20.0 * blended, rtol=1e-9)
    # fewer than 8 weeks of history: the whole history is used (baseline behaviour)
    short = np.tile(20.0 * P_LATE, 5)
    np.testing.assert_allclose(MovingAverage(window=28).fit(short).predict(7), 20.0 * P_LATE)
    # 8 weeks + 3 days: the profile shift must still line up with the next calendar day
    y2 = np.concatenate([y, 20.0 * P_LATE[:3]])
    pred2 = MovingAverage(window=28).fit(y2).predict(7)
    np.testing.assert_allclose(pred2, 20.0 * np.roll(P_LATE, -3), rtol=1e-9)
