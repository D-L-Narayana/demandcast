import math

import pytest

from demandcast.replenish import ReplenishmentInput, decide, round_up_to_pack, z_score


def base(**kw):
    d = dict(
        on_hand=10,
        on_order=0,
        lead_time_days=5,
        review_period_days=7,
        case_pack=6,
        forecast=[4.0] * 28,
        residual_std=2.0,
        service_level=0.95,
    )
    d.update(kw)
    return ReplenishmentInput(**d)


def test_z_score_matches_standard_normal():
    assert z_score(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert z_score(0.5) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        z_score(1.0)


def test_round_up_to_pack():
    assert round_up_to_pack(0, 6) == 0
    assert round_up_to_pack(1, 6) == 6
    assert round_up_to_pack(6, 6) == 6
    assert round_up_to_pack(7, 6) == 12
    assert round_up_to_pack(-3, 6) == 0


def test_order_placed_when_below_reorder_point():
    d = decide(base())
    assert d.lead_time_demand == pytest.approx(20.0)
    assert d.safety_stock == pytest.approx(1.6449 * 2.0 * math.sqrt(12), abs=1e-2)
    assert d.reorder_point == pytest.approx(d.lead_time_demand + d.safety_stock)
    assert d.order_up_to == pytest.approx(48.0 + d.safety_stock)
    assert d.inventory_position == 10
    assert d.order_qty > 0 and d.order_qty % 6 == 0
    assert d.order_qty >= d.order_up_to - 10


def test_no_order_when_inventory_position_covers_reorder_point():
    d = decide(base(on_hand=30, on_order=40))
    assert d.order_qty == 0
    assert "above" in d.reason


def test_on_order_counts_toward_inventory_position():
    d1 = decide(base(on_hand=10, on_order=0))
    d2 = decide(base(on_hand=10, on_order=12))
    assert d2.order_qty == d1.order_qty - 12


def test_higher_service_level_means_more_safety_stock():
    lo = decide(base(service_level=0.90))
    hi = decide(base(service_level=0.99))
    assert hi.safety_stock > lo.safety_stock
    assert hi.order_qty >= lo.order_qty


def test_zero_residual_std_gives_zero_safety_stock():
    d = decide(base(residual_std=0.0))
    assert d.safety_stock == 0.0


def test_shelf_life_caps_order():
    d = decide(base(on_hand=0, shelf_life_days=3, case_pack=1))
    # can only sell 12 units in 3 days → cap at 12
    assert d.order_qty == 12
    assert "shelf life" in d.reason


def test_horizon_shorter_than_protection_period_rejected():
    with pytest.raises(ValueError):
        decide(base(forecast=[4.0] * 10))
