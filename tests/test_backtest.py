import numpy as np
import pytest

from demandcast.backtest import (
    bias,
    candidate_models,
    mae,
    mase,
    rolling_origin,
    score_model,
    select_model,
    wape,
)
from demandcast.models import SeasonalNaive


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
