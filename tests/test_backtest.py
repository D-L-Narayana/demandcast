import pickle
import sys
import types
from statistics import NormalDist

import numpy as np
import pytest

from demandcast import backtest
from demandcast.backtest import (
    FoldResult,
    IntervalModel,
    ModelScore,
    bias,
    candidate_models,
    empirical_interval,
    interval_from_folds,
    mae,
    mase,
    normal_interval,
    rank_scores,
    rolling_origin,
    score_model,
    select_model,
    wape,
)
from demandcast.models import MODEL_REGISTRY, SeasonalNaive


def test_metric_definitions():
    yt = np.array([10, 20, 30])
    yp = np.array([12, 18, 33])
    assert mae(yt, yp) == pytest.approx(7 / 3)
    assert bias(yt, yp) == pytest.approx(1.0)
    assert wape(yt, yp) == pytest.approx(7 / 60)
    assert wape(np.zeros(3), yp) is None


def test_mase_scales_by_seasonal_naive():
    train = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 4)  # perfectly seasonal => scale 0
    assert mase(np.ones(3), np.ones(3), train) is None
    train2 = np.arange(21, dtype=float)  # seasonal naive error is 7 everywhere
    assert mase(np.array([0.0]), np.array([7.0]), train2) == pytest.approx(1.0)


def test_rolling_origin_folds_do_not_leak_future():
    y = np.arange(200, dtype=float)
    folds = rolling_origin(y, SeasonalNaive, horizon=14, n_folds=3, min_train=56)
    assert len(folds) == 3
    origins = [f.origin for f in folds]
    assert origins == sorted(origins)
    assert origins[-1] == 200 - 14
    for f in folds:
        # SeasonalNaive uses the last 7 training points, so prediction max <= y[origin-1]
        assert f.y_pred.max() <= y[f.origin - 1]
        assert f.y_true[0] == y[f.origin]


def test_rolling_origin_too_short():
    with pytest.raises(ValueError):
        rolling_origin(np.arange(30.0), SeasonalNaive, horizon=28)


def test_score_model_and_select_model():
    rng = np.random.default_rng(11)
    profile = np.array([0.8, 0.9, 0.9, 1.0, 1.1, 1.3, 1.0])
    y = rng.poisson(15 * profile[np.arange(240) % 7]).astype(float)
    s = score_model(y, "moving_average", horizon=14, n_folds=3)
    assert s.folds == 3 and s.mae > 0 and s.residual_std > 0
    winner, scores = select_model(y, horizon=14, n_folds=3)
    assert winner in scores
    assert winner.mae == min(x.mae for x in scores)
    assert "croston_sba" not in {x.model_name for x in scores}


def test_candidate_models_includes_croston_for_sparse_series():
    y = np.zeros(100)
    y[::9] = 2
    assert "croston_sba" in candidate_models(y)


# ---- prediction intervals (contract C3) --------------------------------------------------
Z80 = NormalDist().inv_cdf(0.9)  # 1.2816 — the 80 % two-sided normal quantile


class _ConstantModel:
    """Flag-less stub with the baseline signature: always forecasts a fixed level."""

    def __init__(self, level: float = 50.0):
        self.level = level

    def fit(self, y):
        return self

    def predict(self, h):
        return np.full(h, self.level)


def test_interval_model_apply_brackets_and_never_negative():
    iv = IntervalModel(level=0.8, lo_offset=-3.0, hi_offset=1.5)
    yhat = np.array([0.0, 1.0, 2.5, 0.0, 10.0])
    lower, upper = iv.apply(yhat)
    assert lower.shape == upper.shape == yhat.shape
    np.testing.assert_allclose(upper, yhat + 1.5)
    np.testing.assert_allclose(lower, [0.0, 0.0, 0.0, 0.0, 7.0])
    assert np.all(lower >= 0)
    assert np.all(lower <= yhat)
    assert np.all(yhat <= upper)


def test_interval_model_growth_widens_later_steps():
    iv = IntervalModel(level=0.8, lo_offset=-2.0, hi_offset=4.0, growth=0.02)
    yhat = np.full(10, 100.0)
    lower, upper = iv.apply(yhat)
    h = np.arange(1, 11)
    np.testing.assert_allclose(upper - yhat, 4.0 * (1 + 0.02 * (h - 1)))
    np.testing.assert_allclose(yhat - lower, 2.0 * (1 + 0.02 * (h - 1)))
    assert upper[-1] - lower[-1] > upper[0] - lower[0]


def test_interval_model_rejects_invalid_offsets():
    with pytest.raises(ValueError):
        IntervalModel(level=0.8, lo_offset=1.0, hi_offset=2.0)
    with pytest.raises(ValueError):
        IntervalModel(level=0.8, lo_offset=-1.0, hi_offset=-0.5)
    with pytest.raises(ValueError):
        IntervalModel(level=1.5, lo_offset=-1.0, hi_offset=1.0)


def test_empirical_interval_uses_residual_quantiles():
    e = np.arange(-50.0, 51.0)  # symmetric, uniform: Q(0.1) = -40, Q(0.9) = 40
    iv = empirical_interval(e, level=0.8)
    assert iv.level == 0.8
    assert iv.lo_offset == pytest.approx(-40.0)
    assert iv.hi_offset == pytest.approx(40.0)
    assert iv.growth == 0.0
    iv95 = empirical_interval(e, level=0.95)
    assert iv95.lo_offset < iv.lo_offset
    assert iv95.hi_offset > iv.hi_offset


def test_empirical_interval_clamps_offsets_on_biased_residuals():
    # systematic under-forecast: every residual e = y_true - y_pred is positive
    e_pos = np.linspace(2.0, 6.0, 60)
    iv = empirical_interval(e_pos, level=0.8)
    assert iv.lo_offset == 0.0
    assert iv.hi_offset > 0.0
    yhat = np.array([0.0, 3.0, 8.0])
    lower, upper = iv.apply(yhat)
    np.testing.assert_allclose(lower, yhat)  # the band never drops below the point forecast
    assert np.all(upper > yhat)
    # systematic over-forecast: every residual negative -> hi clamps to 0, lower stays >= 0
    iv2 = empirical_interval(-e_pos, level=0.8)
    assert iv2.hi_offset == 0.0
    assert iv2.lo_offset < 0.0
    lower2, upper2 = iv2.apply(yhat)
    np.testing.assert_allclose(upper2, yhat)
    assert np.all(lower2 >= 0)
    assert np.all(lower2 <= yhat)
    assert lower2[0] == 0.0
    assert lower2[2] < 8.0


def test_normal_interval_is_symmetric_z_sigma():
    iv = normal_interval(2.0, 0.8)
    assert iv.level == 0.8
    assert iv.growth == 0.0
    assert iv.hi_offset == pytest.approx(Z80 * 2.0, rel=1e-6)
    assert iv.lo_offset == pytest.approx(-Z80 * 2.0, rel=1e-6)
    lower, upper = iv.apply(np.array([1.0, 50.0]))
    assert lower[0] == 0.0
    assert lower[1] == pytest.approx(50.0 - Z80 * 2.0, rel=1e-6)
    assert upper[1] == pytest.approx(50.0 + Z80 * 2.0, rel=1e-6)
    assert normal_interval(0.0, 0.8).hi_offset == 0.0
    with pytest.raises(ValueError):
        normal_interval(1.0, 1.0)


def _folds_with_abs_errors(horizon, n_folds, first_half, second_half):
    """Synthetic folds: |residual| is `first_half` on steps 1..ceil(H/2), `second_half` after."""
    half = -(-horizon // 2)
    mag = np.where(np.arange(1, horizon + 1) <= half, first_half, second_half).astype(float)
    folds = []
    for k in range(n_folds):
        sign = np.where((np.arange(horizon) + k) % 2 == 0, 1.0, -1.0)
        y_pred = np.full(horizon, 20.0)
        folds.append(
            FoldResult(origin=100 + k * horizon, y_true=y_pred + sign * mag, y_pred=y_pred)
        )
    return folds


def test_interval_growth_formula_and_clamp():
    # H=8, 5 folds = 40 residuals: r1 = 1.0, r2 = 1.04 -> (1.04/1.0 - 1) / (8/2) = 0.01
    iv = interval_from_folds(_folds_with_abs_errors(8, 5, 1.0, 1.04), level=0.8, horizon=8)
    assert iv.growth == pytest.approx(0.01)
    # much larger second-half error -> clamped at 0.03
    iv_big = interval_from_folds(_folds_with_abs_errors(8, 5, 1.0, 3.0), level=0.8, horizon=8)
    assert iv_big.growth == pytest.approx(0.03)
    # errors shrinking with the horizon -> clamped at 0 (never negative)
    assert interval_from_folds(_folds_with_abs_errors(8, 5, 2.0, 1.0), 0.8, 8).growth == 0.0


def test_interval_growth_guards():
    # fewer than 20 residuals (8 x 2 = 16) -> no growth even with a strong signal
    assert interval_from_folds(_folds_with_abs_errors(8, 2, 1.0, 3.0), 0.8, 8).growth == 0.0
    # horizon < 4 -> no growth (30 residuals)
    assert interval_from_folds(_folds_with_abs_errors(3, 10, 1.0, 3.0), 0.8, 3).growth == 0.0
    # r1 == 0 -> no growth
    assert interval_from_folds(_folds_with_abs_errors(8, 5, 0.0, 3.0), 0.8, 8).growth == 0.0
    # exactly at the guards (20 residuals, H = 4) the growth is computed
    iv = interval_from_folds(_folds_with_abs_errors(4, 5, 1.0, 3.0), 0.8, 4)
    assert iv.growth == pytest.approx(0.03)


def test_empirical_interval_nominal_coverage_on_gaussian_residuals():
    rng = np.random.default_rng(2024)
    y = 50.0 + rng.normal(0.0, 5.0, 700)
    folds = rolling_origin(y, _ConstantModel, horizon=14, n_folds=30, min_train=56)
    assert sum(f.y_true.size for f in folds) >= 400
    iv = interval_from_folds(folds, level=0.8, horizon=14)
    inside = []
    for f in folds:
        lower, upper = iv.apply(f.y_pred)
        inside.append((lower <= f.y_true) & (f.y_true <= upper))
    assert abs(float(np.concatenate(inside).mean()) - 0.8) <= 0.07
    # out of sample: fresh 14-step noise paths around the same constant forecast
    fresh = 50.0 + np.random.default_rng(99).normal(0.0, 5.0, (300, 14))
    lower, upper = iv.apply(np.full(14, 50.0))
    assert abs(float(np.mean((lower <= fresh) & (fresh <= upper))) - 0.8) <= 0.07


# ---- promo-flag plumbing through the folds -----------------------------------------------
def _recording_factory(log):
    """Factory for a C1-shaped stub that logs exactly the kwargs fit/predict receive."""

    class Recording:
        def fit(self, y, **kw):
            log.append(("fit", np.asarray(y).size, kw))
            return self

        def predict(self, h, **kw):
            log.append(("predict", h, kw))
            return np.zeros(h)

    return Recording


def test_rolling_origin_slices_promo_flags_per_fold():
    y = np.arange(120.0)
    flags = (np.arange(120) % 10 == 0).astype(float)
    log = []
    folds = rolling_origin(
        y, _recording_factory(log), 14, n_folds=3, min_train=56, promo_flags=flags
    )
    fits = [c for c in log if c[0] == "fit"]
    preds = [c for c in log if c[0] == "predict"]
    assert len(folds) == len(fits) == len(preds) == 3
    for f, (_, n_train, fit_kw), (_, h, pred_kw) in zip(folds, fits, preds, strict=True):
        assert n_train == f.origin
        assert h == 14
        assert set(fit_kw) == {"promo_flags"}
        assert set(pred_kw) == {"future_flags"}
        np.testing.assert_array_equal(fit_kw["promo_flags"], flags[: f.origin])
        np.testing.assert_array_equal(pred_kw["future_flags"], flags[f.origin : f.origin + 14])
        assert pred_kw["future_flags"].size == 14


def test_rolling_origin_rejects_misaligned_flags():
    with pytest.raises(ValueError, match="promo_flags"):
        rolling_origin(np.arange(120.0), _ConstantModel, 14, n_folds=2, promo_flags=np.zeros(100))


def test_rolling_origin_passes_no_flag_kwargs_when_flags_are_none():
    log = []
    folds = rolling_origin(np.arange(120.0), _recording_factory(log), 14, n_folds=2)
    assert len(folds) == 2
    assert len(log) == 4
    assert all(kw == {} for _, _, kw in log)
    # a baseline-style stub without any flag parameters keeps working
    folds2 = rolling_origin(np.arange(120.0), _ConstantModel, 14, n_folds=2)
    assert [f.origin for f in folds2] == [92, 106]


# ---- score_model: intervals, coverage, flags ---------------------------------------------
def _poisson_series(seed=5, n=240, lam=20.0):
    return np.random.default_rng(seed).poisson(lam, n).astype(float)


def test_score_model_fills_empirical_interval_and_coverage():
    y = _poisson_series()
    s = score_model(y, "moving_average", horizon=14, n_folds=3)
    assert isinstance(s.interval, IntervalModel)
    assert s.interval.level == 0.8
    assert s.interval.lo_offset <= 0 <= s.interval.hi_offset
    assert s.interval == interval_from_folds(s.fold_results, level=0.8, horizon=14)
    inside = []
    for f in s.fold_results:
        lower, upper = s.interval.apply(f.y_pred)
        inside.append((lower <= f.y_true) & (f.y_true <= upper))
    assert s.coverage_backtest == pytest.approx(float(np.concatenate(inside).mean()))
    assert abs(s.coverage_backtest - 0.8) <= 0.1
    # metric definitions are unchanged
    yt = np.concatenate([f.y_true for f in s.fold_results])
    yp = np.concatenate([f.y_pred for f in s.fold_results])
    assert s.mae == pytest.approx(mae(yt, yp))
    assert s.bias == pytest.approx(bias(yt, yp))
    assert s.wape == pytest.approx(wape(yt, yp))
    assert s.residual_std == pytest.approx(float(np.std(yt - yp, ddof=1)))


def test_score_model_normal_interval_and_custom_level():
    y = _poisson_series(6)
    s = score_model(
        y, "seasonal_naive", 14, n_folds=3, interval_level=0.9, interval_method="normal"
    )
    assert s.interval == normal_interval(s.residual_std, 0.9)
    assert s.interval.level == 0.9
    assert s.interval.growth == 0.0
    assert s.interval.hi_offset == pytest.approx(-s.interval.lo_offset)
    assert s.coverage_backtest is not None
    assert 0.0 <= s.coverage_backtest <= 1.0
    e = score_model(y, "seasonal_naive", 14, n_folds=3, interval_level=0.9)
    assert e.interval is not None
    assert e.interval.level == 0.9
    assert e.interval != s.interval
    with pytest.raises(ValueError, match="interval_method"):
        score_model(y, "seasonal_naive", 14, n_folds=3, interval_method="bootstrap")


def test_model_score_with_interval_is_picklable():
    s = score_model(_poisson_series(7), "moving_average", 14, n_folds=2)
    s.fold_results = []  # the pipeline drops fold arrays before sending results back
    clone = pickle.loads(pickle.dumps(s))
    assert isinstance(clone.interval, IntervalModel)
    assert clone.interval == s.interval
    assert clone.coverage_backtest == s.coverage_backtest
    assert clone.model_name == "moving_average"


def test_score_model_forwards_promo_flags_to_each_fold(monkeypatch):
    log = []
    monkeypatch.setitem(MODEL_REGISTRY, "recording_stub", _recording_factory(log))
    y = np.arange(120.0)
    flags = np.zeros(120)
    flags[60:70] = 1.0
    s = score_model(y, "recording_stub", 14, n_folds=2, promo_flags=flags)
    assert s.model_name == "recording_stub"
    assert s.folds == 2
    fits = [c for c in log if c[0] == "fit"]
    assert [set(kw) for _, _, kw in fits] == [{"promo_flags"}] * 2
    np.testing.assert_array_equal(fits[0][2]["promo_flags"], flags[:92])
    # without flags the same stub gets no keyword arguments at all
    log.clear()
    score_model(y, "recording_stub", 14, n_folds=2)
    assert len(log) == 4
    assert all(kw == {} for _, _, kw in log)


# ---- candidate selection -----------------------------------------------------------------
def test_candidate_models_allowed_filters_and_keeps_registry_order():
    y = 10.0 + np.arange(100.0) % 7  # never intermittent
    base = [n for n in MODEL_REGISTRY if n != "croston_sba"]
    assert candidate_models(y) == base
    assert candidate_models(y, allowed=["holt_winters", "seasonal_naive"]) == [
        "seasonal_naive",
        "holt_winters",
    ]
    # croston stays reserved for intermittent series even when explicitly allowed
    assert candidate_models(y, allowed=["croston_sba", "seasonal_naive"]) == ["seasonal_naive"]
    sparse = np.zeros(100)
    sparse[::9] = 2
    assert candidate_models(sparse, allowed=("croston_sba",)) == ["croston_sba"]
    with pytest.raises(ValueError, match="croston_sba"):
        candidate_models(y, allowed=["croston_sba"])
    with pytest.raises(ValueError):
        candidate_models(y, allowed=[])


def test_candidate_models_order_derives_from_registry(monkeypatch):
    monkeypatch.setitem(MODEL_REGISTRY, "zzz_stub", _ConstantModel)
    y = 10.0 + np.arange(100.0) % 7
    assert candidate_models(y) == [n for n in MODEL_REGISTRY if n != "croston_sba"]
    sparse = np.zeros(100)
    sparse[::9] = 2
    assert candidate_models(sparse) == list(MODEL_REGISTRY)
    assert candidate_models(sparse, allowed=["zzz_stub", "croston_sba"]) == [
        n for n in MODEL_REGISTRY if n in {"zzz_stub", "croston_sba"}
    ]


def _fake_promo_module(signal: bool):
    """Minimal stand-in for contract C2 so the lazy-import plumbing is testable pre-integration."""

    def has_promo_signal(flags, **kw):
        return signal and float(np.sum(flags)) > 0

    mod = types.ModuleType("demandcast.promo")
    mod.PROMO_BASES = ("moving_average", "holt_winters", "theta")
    mod.PROMO_MODEL_NAMES = tuple(f"promo_{b}" for b in mod.PROMO_BASES)
    mod.has_promo_signal = has_promo_signal
    return mod


def test_candidate_models_promo_plumbing_with_stub_promo_module(monkeypatch):
    y = 20.0 + np.arange(140.0) % 7
    flags = np.zeros(140)
    flags[100:114] = 1.0
    base = [n for n in MODEL_REGISTRY if n != "croston_sba"]
    # flags present but the promo module sees no usable signal -> plain candidates
    monkeypatch.setitem(sys.modules, "demandcast.promo", _fake_promo_module(signal=False))
    assert candidate_models(y, promo_flags=flags) == base
    # signal -> promo_<base> appended (only for bases that exist in the registry), base order kept
    monkeypatch.setitem(sys.modules, "demandcast.promo", _fake_promo_module(signal=True))
    promo = [f"promo_{b}" for b in ("moving_average", "holt_winters", "theta") if b in base]
    assert candidate_models(y, promo_flags=flags) == base + promo
    assert candidate_models(y, promo_flags=np.zeros(140)) == base  # all-zero flags: no signal
    assert candidate_models(
        y, promo_flags=flags, allowed=["promo_moving_average", "seasonal_naive"]
    ) == ["seasonal_naive", "promo_moving_average"]
    # with flags=None the promo module is never imported (an import here would raise)
    monkeypatch.setitem(sys.modules, "demandcast.promo", None)
    assert candidate_models(y) == base
    assert candidate_models(y, allowed=["holt_winters"]) == ["holt_winters"]


def test_select_model_threads_flags_to_restricted_stub_candidate(monkeypatch):
    log = []
    monkeypatch.setitem(MODEL_REGISTRY, "recording_stub", _recording_factory(log))
    monkeypatch.setitem(sys.modules, "demandcast.promo", _fake_promo_module(signal=False))
    y = np.arange(120.0)
    flags = (np.arange(120) >= 100).astype(float)
    winner, scores = select_model(
        y, 14, n_folds=2, promo_flags=flags, allowed=["recording_stub"], interval_level=0.9
    )
    assert winner is scores[0]
    assert winner.model_name == "recording_stub"
    preds = [c for c in log if c[0] == "predict"]
    assert len(preds) == 2
    np.testing.assert_array_equal(preds[-1][2]["future_flags"], flags[106:120])
    assert winner.interval is not None
    assert winner.interval.level == 0.9


def test_candidate_models_adds_promo_names_only_with_signal():
    from demandcast.promo import PROMO_BASES, PROMO_MODEL_NAMES, has_promo_signal

    y = 20.0 + np.arange(140.0) % 7
    quiet = np.zeros(140)
    assert not has_promo_signal(quiet)
    assert candidate_models(y, promo_flags=quiet) == candidate_models(y)
    flags = np.zeros(140)
    flags[100:114] = 1.0  # 14 promo days on 126 base days -> signal
    assert has_promo_signal(flags)
    names = candidate_models(y, promo_flags=flags)
    base = [n for n in MODEL_REGISTRY if n != "croston_sba"]
    promo = [n for b, n in zip(PROMO_BASES, PROMO_MODEL_NAMES, strict=True) if b in base]
    assert names == base + promo
    assert len(promo) == 3
    assert candidate_models(y, promo_flags=flags, allowed=["promo_theta", "seasonal_naive"]) == [
        "seasonal_naive",
        "promo_theta",
    ]
    # intermittent series: croston joins in registry position, promo names still come last
    sparse = np.zeros(140)
    sparse[::9] = 2.0
    assert candidate_models(sparse, promo_flags=flags) == list(MODEL_REGISTRY) + promo


# ---- model selection: criteria, ties, restrictions ---------------------------------------
def _score(name, mae_, wape_=None, bias_=0.0, mase_=None):
    return ModelScore(
        model_name=name, folds=2, mae=mae_, wape=wape_, bias=bias_, mase=mase_, residual_std=1.0
    )


def _fake_score_model(table):
    def fake(y, model_name, horizon, **kw):
        row = table.get(model_name, {"mae": 99.0, "wape": 99.0, "mase": 99.0, "bias": 99.0})
        return _score(
            model_name, row["mae"], row.get("wape"), row.get("bias", 0.0), row.get("mase")
        )

    return fake


def test_rank_scores_orders_by_criterion_with_none_last_and_tie_breaks():
    scores = [
        _score("b_model", mae_=1.0, wape_=None, bias_=0.2, mase_=1.0),
        _score("a_model", mae_=1.0, wape_=0.5, bias_=-0.2, mase_=None),
        _score("c_model", mae_=0.5, wape_=0.9, bias_=3.0, mase_=2.0),
        _score("d_model", mae_=1.0, wape_=0.5, bias_=0.1, mase_=None),
    ]
    by = lambda crit: [s.model_name for s in rank_scores(scores, crit)]  # noqa: E731
    assert by("mae") == ["c_model", "d_model", "a_model", "b_model"]
    assert by("wape") == ["d_model", "a_model", "c_model", "b_model"]
    assert by("mase") == ["b_model", "c_model", "d_model", "a_model"]
    with pytest.raises(ValueError, match="criterion"):
        rank_scores(scores, "rmse")


def test_select_model_criterion_changes_winner(monkeypatch):
    table = {
        "seasonal_naive": {"mae": 1.0, "wape": 0.30, "mase": 0.9},
        "moving_average": {"mae": 2.0, "wape": 0.10, "mase": 1.5},
        "holt_winters": {"mae": 3.0, "wape": 0.20, "mase": 0.5},
    }
    monkeypatch.setattr(backtest, "score_model", _fake_score_model(table))
    y = 10.0 + np.arange(120.0) % 7
    assert select_model(y, 14)[0].model_name == "seasonal_naive"
    assert select_model(y, 14, criterion="wape")[0].model_name == "moving_average"
    assert select_model(y, 14, criterion="mase")[0].model_name == "holt_winters"
    winner, scores = select_model(y, 14, criterion="wape")
    assert winner in scores
    assert [s.model_name for s in scores] == candidate_models(y)
    with pytest.raises(ValueError, match="criterion"):
        select_model(y, 14, criterion="rmse")


def test_select_model_none_metric_sorts_last_and_ties_break_on_bias_then_name(monkeypatch):
    table = {
        "seasonal_naive": {"mae": 1.0, "wape": None, "bias": -0.5, "mase": None},
        "moving_average": {"mae": 1.0, "wape": 0.2, "bias": 0.1, "mase": None},
        "holt_winters": {"mae": 1.0, "wape": 0.3, "bias": -0.1, "mase": 2.0},
    }
    monkeypatch.setattr(backtest, "score_model", _fake_score_model(table))
    y = 10.0 + np.arange(120.0) % 7
    # full tie on MAE -> |bias| 0.1 shared by two models -> alphabetical name wins
    assert select_model(y, 14)[0].model_name == "holt_winters"
    # WAPE undefined for seasonal_naive -> it sorts last; best defined WAPE wins
    assert select_model(y, 14, criterion="wape")[0].model_name == "moving_average"
    # only holt_winters has a MASE -> the None scores sort behind it
    assert select_model(y, 14, criterion="mase")[0].model_name == "holt_winters"


def test_select_model_allowed_and_interval_options_pass_through():
    y = _poisson_series(8)
    winner, scores = select_model(
        y, 14, n_folds=2, allowed=["seasonal_naive"], interval_level=0.9, interval_method="normal"
    )
    assert [s.model_name for s in scores] == ["seasonal_naive"]
    assert winner is scores[0]
    assert winner.interval == normal_interval(winner.residual_std, 0.9)
    with pytest.raises(ValueError, match="allowed"):
        select_model(y, 14, n_folds=2, allowed=["no_such_model"])


def test_select_model_default_matches_baseline_min_mae_rule():
    y = _poisson_series(21, n=200, lam=12.0)
    winner, scores = select_model(y, horizon=14, n_folds=2)
    assert [s.model_name for s in scores] == candidate_models(y)
    assert (winner.mae, abs(winner.bias)) == min((s.mae, abs(s.bias)) for s in scores)
    assert all(isinstance(s.interval, IntervalModel) for s in scores)
    assert all(s.interval.level == 0.8 for s in scores)
    for s in scores:
        yhat = s.fold_results[-1].y_pred
        lower, upper = s.interval.apply(yhat)
        assert np.all(lower >= 0)
        assert np.all(lower <= yhat)
        assert np.all(yhat <= upper)


def test_select_model_scores_promo_candidates_with_real_models():
    from demandcast.promo import PROMO_MODEL_NAMES

    rng = np.random.default_rng(13)
    y = rng.poisson(20.0, 200).astype(float)
    flags = np.zeros(200)
    flags[60:70] = 1.0
    flags[150:164] = 1.0
    y[flags == 1] *= 2.0
    winner, scores = select_model(y, horizon=14, n_folds=2, promo_flags=flags)
    names = [s.model_name for s in scores]
    assert set(PROMO_MODEL_NAMES) <= set(names)
    assert names == candidate_models(y, promo_flags=flags)
    assert winner.mae == min(s.mae for s in scores)
    assert all(isinstance(s.interval, IntervalModel) for s in scores)
    plain = [s.model_name for s in select_model(y, horizon=14, n_folds=2)[1]]
    assert not any(n.startswith("promo_") for n in plain)
