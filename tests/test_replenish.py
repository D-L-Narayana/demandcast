import math
import random
import re
from dataclasses import fields
from itertools import pairwise

import pytest

from demandcast.replenish import (
    BudgetItem,
    ReplenishmentDecision,
    ReplenishmentInput,
    allocate_budget,
    decide,
    expected_shortfall,
    parse_service_levels,
    round_down_to_pack,
    round_up_to_pack,
    service_level_for,
    stockout_probability,
    z_score,
)


def base(**kw):
    d = {
        "on_hand": 10,
        "on_order": 0,
        "lead_time_days": 5,
        "review_period_days": 7,
        "case_pack": 6,
        "forecast": [4.0] * 28,
        "residual_std": 2.0,
        "service_level": 0.95,
    }
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


# ---------------------------------------------------------------------------
# C4 additions: stock-out risk / expected shortfall (normal loss function),
# cost-weighted priority, days of cover, min/max order quantities, greedy
# budget allocation and per-ABC-class service levels.
#
# For the base() fixture: μ_LR = Σ forecast[:L+R] = 12 · 4 = 48,
# σ_LR = residual_std · √(L+R) = 2 · √12 ≈ 6.93, IP = 10.
# ---------------------------------------------------------------------------

MU = 48.0
SIGMA = 2.0 * math.sqrt(12)
PHI0 = 1.0 / math.sqrt(2.0 * math.pi)  # φ(0) = 0.39894...


def test_stockout_probability_closed_forms():
    assert stockout_probability(MU, SIGMA, MU) == pytest.approx(0.5)
    # one σ above / below the mean → 1 − Φ(1) and 1 − Φ(−1)
    assert stockout_probability(MU, SIGMA, MU + SIGMA) == pytest.approx(0.15866, abs=1e-4)
    assert stockout_probability(MU, SIGMA, MU - SIGMA) == pytest.approx(0.84134, abs=1e-4)
    assert stockout_probability(MU, SIGMA, MU + 10 * SIGMA) == pytest.approx(0.0, abs=1e-12)
    assert stockout_probability(MU, SIGMA, MU - 10 * SIGMA) == pytest.approx(1.0, abs=1e-12)


def test_stockout_probability_monotone_decreasing_in_ip():
    ips = [MU + k * SIGMA / 2 for k in range(-6, 7)]
    risks = [stockout_probability(MU, SIGMA, ip) for ip in ips]
    assert all(a > b for a, b in pairwise(risks))
    assert all(0.0 <= r <= 1.0 for r in risks)


def test_expected_shortfall_closed_forms():
    # L(0) = φ(0) = 0.3989 → shortfall σ · L(0) when IP = μ
    assert expected_shortfall(MU, SIGMA, MU) == pytest.approx(0.3989 * SIGMA, abs=1e-3)
    assert expected_shortfall(MU, SIGMA, MU) == pytest.approx(PHI0 * SIGMA)
    # L(1) = φ(1) − 1 · (1 − Φ(1)) = 0.24197 − 0.15866 = 0.08332
    assert expected_shortfall(MU, SIGMA, MU + SIGMA) == pytest.approx(0.08332 * SIGMA, abs=1e-3)
    # IP ≫ μ → 0 ; IP ≪ μ → μ − IP
    assert expected_shortfall(MU, SIGMA, MU + 10 * SIGMA) == pytest.approx(0.0, abs=1e-9)
    assert expected_shortfall(MU, SIGMA, MU - 10 * SIGMA) == pytest.approx(10 * SIGMA, abs=1e-6)


def test_expected_shortfall_nonnegative_decreasing_and_above_deterministic_gap():
    ips = [MU + k * SIGMA / 2 for k in range(-6, 7)]
    es = [expected_shortfall(MU, SIGMA, ip) for ip in ips]
    assert all(v >= 0.0 for v in es)
    assert all(a > b for a, b in pairwise(es))
    assert all(v >= max(0.0, MU - ip) - 1e-9 for v, ip in zip(es, ips, strict=True))


def test_sigma_zero_edge_cases_both_sides():
    assert stockout_probability(MU, 0.0, MU - 10) == 1.0
    assert stockout_probability(MU, 0.0, MU) == 0.0
    assert stockout_probability(MU, 0.0, MU + 10) == 0.0
    assert expected_shortfall(MU, 0.0, MU - 10) == 10.0
    assert expected_shortfall(MU, 0.0, MU) == 0.0
    assert expected_shortfall(MU, 0.0, MU + 10) == 0.0


def test_negative_sigma_rejected():
    with pytest.raises(ValueError):
        stockout_probability(MU, -1.0, MU)
    with pytest.raises(ValueError):
        expected_shortfall(MU, -1.0, MU)


def test_decision_risk_fields_match_module_functions():
    d = decide(base())
    assert d.stockout_risk == pytest.approx(stockout_probability(MU, SIGMA, 10))
    assert d.stockout_risk > 0.99  # IP 10 against μ 48, σ 6.9 → almost certain stock-out
    assert d.expected_shortfall == pytest.approx(expected_shortfall(MU, SIGMA, 10))
    assert d.expected_shortfall == pytest.approx(38.0, abs=1e-3)
    assert d.priority == pytest.approx(d.expected_shortfall)  # unit_cost None → weight 1.0
    assert d.days_of_cover == pytest.approx(2.5)  # 10 / (48 / 12)


def test_decision_risk_computed_even_when_no_order_is_placed():
    d = decide(base(on_hand=30, on_order=40))
    assert d.order_qty == 0
    assert 0.0 < d.stockout_risk < 0.01
    assert d.stockout_risk == pytest.approx(stockout_probability(MU, SIGMA, 70))
    assert d.expected_shortfall == pytest.approx(expected_shortfall(MU, SIGMA, 70))
    assert d.days_of_cover == pytest.approx(17.5)


def test_stockout_risk_decreases_with_inventory_position():
    risks = [decide(base(on_hand=h)).stockout_risk for h in (0, 20, 40, 60, 80)]
    assert all(a > b for a, b in pairwise(risks))


def test_priority_uses_unit_cost():
    plain = decide(base())
    costed = decide(base(unit_cost=2.5))
    assert costed.priority == pytest.approx(2.5 * costed.expected_shortfall)
    assert costed.priority == pytest.approx(2.5 * plain.priority)
    assert costed.expected_shortfall == pytest.approx(plain.expected_shortfall)
    assert costed.order_qty == plain.order_qty  # cost never changes the quantity
    with pytest.raises(ValueError):
        decide(base(unit_cost=-1.0))


def test_days_of_cover_none_when_forecast_all_zero():
    d = decide(base(forecast=[0.0] * 28))
    assert d.days_of_cover is None
    assert d.stockout_risk == pytest.approx(stockout_probability(0.0, SIGMA, 10))
    assert decide(base()).days_of_cover == pytest.approx(2.5)


def test_zero_sigma_decision_edges():
    # σ_LR = 0: the risk is a step function of IP against μ_LR = 48
    assert decide(base(residual_std=0.0, on_hand=10)).stockout_risk == 1.0
    assert decide(base(residual_std=0.0, on_hand=10)).expected_shortfall == 38.0
    assert decide(base(residual_std=0.0, on_hand=48)).stockout_risk == 0.0
    assert decide(base(residual_std=0.0, on_hand=60)).expected_shortfall == 0.0


def test_default_inputs_keep_baseline_numbers_and_field_order():
    d = decide(base())
    assert d.order_qty == 54  # ceil((59.396 - 10) / 6) · 6
    assert d.reorder_point == pytest.approx(31.3964, abs=1e-3)
    assert d.order_up_to == pytest.approx(59.3964, abs=1e-3)
    assert d.reason == "inventory position below reorder point"
    names = [f.name for f in fields(ReplenishmentDecision)]
    assert names[:7] == [
        "lead_time_demand",
        "safety_stock",
        "reorder_point",
        "order_up_to",
        "inventory_position",
        "order_qty",
        "reason",
    ]
    assert names[7:] == ["stockout_risk", "expected_shortfall", "priority", "days_of_cover"]
    # positional construction of the seven baseline fields still works; new fields default
    legacy = ReplenishmentDecision(20.0, 11.4, 31.4, 59.4, 10, 54, "x")
    assert (legacy.stockout_risk, legacy.expected_shortfall, legacy.priority) == (0.0, 0.0, 0.0)
    assert legacy.days_of_cover is None
    legacy_in = ReplenishmentInput(10, 0, 5, 7, 6, [4.0] * 28, 2.0)
    assert legacy_in.unit_cost is None and legacy_in.max_order_qty is None
    assert legacy_in.min_order_qty == 0
    assert decide(legacy_in) == decide(base())


# -- min / max order quantity -------------------------------------------------


def test_round_down_to_pack():
    assert round_down_to_pack(40, 6) == 36
    assert round_down_to_pack(6, 6) == 6
    assert round_down_to_pack(5, 6) == 0
    assert round_down_to_pack(0, 6) == 0
    assert round_down_to_pack(-3, 6) == 0


def test_min_order_qty_rounds_up_to_pack():
    assert decide(base(min_order_qty=12)).order_qty == 54  # already above the minimum
    assert "minimum" not in decide(base(min_order_qty=12)).reason
    d = decide(base(min_order_qty=60))
    assert d.order_qty == 60
    assert d.reason.endswith("; raised to minimum order")
    assert decide(base(min_order_qty=61)).order_qty == 66  # 61 → next multiple of 6
    assert decide(base(min_order_qty=61)).order_qty % 6 == 0


def test_min_order_qty_never_creates_an_order():
    d = decide(base(on_hand=30, on_order=40, min_order_qty=12))
    assert d.order_qty == 0
    assert "minimum" not in d.reason


def test_max_order_qty_rounds_down_to_pack():
    d = decide(base(max_order_qty=40))
    assert d.order_qty == 36
    assert d.reason.endswith("; capped by maximum order")
    assert decide(base(max_order_qty=54)).order_qty == 54  # exactly the rounded order: no cap
    assert "maximum" not in decide(base(max_order_qty=54)).reason
    assert decide(base(max_order_qty=5)).order_qty == 0  # max below one case pack → nothing
    assert "maximum" in decide(base(max_order_qty=5)).reason
    assert decide(base(max_order_qty=0)).order_qty == 0


def test_min_max_order_validation():
    with pytest.raises(ValueError):
        decide(base(min_order_qty=-1))
    with pytest.raises(ValueError):
        decide(base(max_order_qty=-1))


def test_shelf_life_cap_wins_over_min_order_qty():
    d = decide(base(on_hand=0, shelf_life_days=3, case_pack=1, min_order_qty=100))
    assert d.order_qty == 12  # 3 days × 4 units sellable before expiry
    assert "capped by shelf life" in d.reason


def test_max_order_qty_interacts_with_shelf_life_cap():
    # max (8) tighter than the shelf-life cap (12): only the max is reported
    d = decide(base(on_hand=0, shelf_life_days=3, case_pack=1, max_order_qty=8))
    assert d.order_qty == 8
    assert "maximum order" in d.reason
    assert "shelf life" not in d.reason
    # max (30) applied first, then the tighter shelf-life cap (12): both steps are reported
    d2 = decide(base(on_hand=0, shelf_life_days=3, case_pack=1, max_order_qty=30))
    assert d2.order_qty == 12
    assert "maximum order" in d2.reason
    assert "shelf life" in d2.reason


def test_caps_win_over_floor_when_min_exceeds_max():
    d = decide(base(min_order_qty=60, max_order_qty=30))
    assert d.order_qty == 30
    assert "raised to minimum order" in d.reason
    assert "capped by maximum order" in d.reason
    assert d.order_qty % 6 == 0


# -- budget allocation ----------------------------------------------------------


def _items():
    return [
        BudgetItem((1, 1), 10, 5.0, 9.0),  # cost 50
        BudgetItem((1, 2), 4, 10.0, 7.0),  # cost 40 (priority tie with (2, 1))
        BudgetItem((2, 1), 6, 5.0, 7.0),  # cost 30
        BudgetItem((2, 2), 2, 20.0, 1.0),  # cost 40
    ]


def test_allocate_budget_greedy_first_fit_respects_budget():
    items = _items()
    approved = allocate_budget(items, 80.0)
    # (1,1) 50 → remaining 30; (1,2) 40 does not fit; (2,1) 30 fits → 0; (2,2) deferred
    assert approved == {(1, 1): 10, (1, 2): 0, (2, 1): 6, (2, 2): 0}
    cost = {it.key: it.unit_cost for it in items}
    assert sum(q * cost[k] for k, q in approved.items()) <= 80.0
    req = {it.key: it.order_qty for it in items}
    assert all(q in (0, req[k]) for k, q in approved.items())  # whole orders only


def test_allocate_budget_top_priority_affordable_item_always_approved():
    items = _items()
    approved = allocate_budget(items, 45.0)
    assert approved[(1, 1)] == 0  # top priority but 50 > 45
    assert approved[(1, 2)] == 4  # next priority, affordable → approved
    assert approved[(2, 1)] == 0 and approved[(2, 2)] == 0
    for budget in (0.0, 29.0, 30.0, 49.0, 50.0, 79.0, 80.0, 160.0):
        approved = allocate_budget(items, budget)
        affordable = [it for it in items if it.order_qty * it.unit_cost <= budget]
        if affordable:
            top = sorted(affordable, key=lambda it: (-it.priority, it.key))[0]
            assert approved[top.key] == top.order_qty
        assert sum(it.order_qty * it.unit_cost for it in items if approved[it.key]) <= budget


def test_allocate_budget_deterministic_with_equal_priorities():
    items = [BudgetItem((s, p), 2, 10.0, 5.0) for s in (3, 1, 2) for p in (2, 1)]  # 20 each
    approved = allocate_budget(items, 60.0)  # exactly three orders fit → smallest keys win
    assert {k for k, q in approved.items() if q} == {(1, 1), (1, 2), (2, 1)}
    assert list(approved) == [it.key for it in items]  # result keys follow the input order
    shuffled = list(items)
    random.Random(7).shuffle(shuffled)
    assert allocate_budget(shuffled, 60.0) == approved


def test_allocate_budget_none_approves_all_and_zero_approves_none():
    items = _items()
    assert allocate_budget(items, None) == {it.key: it.order_qty for it in items}
    assert allocate_budget(items, 0.0) == {it.key: 0 for it in items}
    assert allocate_budget([], 100.0) == {}
    with pytest.raises(ValueError):
        allocate_budget(items, -1.0)
    with pytest.raises(ValueError):
        allocate_budget([*items, BudgetItem((1, 1), 1, 1.0, 1.0)], 10.0)  # duplicate key


def test_allocate_budget_exact_fit_survives_float_round_off():
    items = [BudgetItem((1, 1), 1, 0.1, 2.0), BudgetItem((1, 2), 1, 0.2, 1.0)]
    assert allocate_budget(items, 0.3) == {(1, 1): 1, (1, 2): 1}  # 0.3 - 0.1 = 0.19999999999999998


def test_allocate_budget_randomised_invariants():
    rng = random.Random(42)
    for _ in range(40):
        items = [
            BudgetItem(
                (s, p), rng.randint(0, 12), round(rng.uniform(0.5, 30.0), 2), rng.uniform(0, 100)
            )
            for s in range(1, 4)
            for p in range(1, 6)
        ]
        budget = rng.uniform(0, 600)
        approved = allocate_budget(items, budget)
        assert set(approved) == {it.key for it in items}
        spent = sum(it.order_qty * it.unit_cost for it in items if approved[it.key])
        assert spent <= budget + 1e-6
        assert all(approved[it.key] in (0, it.order_qty) for it in items)
        # greedy first-fit replayed independently: deferred only if it did not fit at its turn
        remaining = budget
        for it in sorted(items, key=lambda it: (-it.priority, it.key)):
            cost = it.order_qty * it.unit_cost
            if cost <= remaining or math.isclose(cost, remaining, rel_tol=1e-9):
                assert approved[it.key] == it.order_qty
                remaining = max(0.0, remaining - cost)
            else:
                assert approved[it.key] == 0


# -- per-class service levels -----------------------------------------------------


def test_parse_service_levels_happy_paths():
    assert parse_service_levels("A=0.98,B=0.95,C=0.90") == {"A": 0.98, "B": 0.95, "C": 0.90}
    assert parse_service_levels(" a = 0.98 , b=0.95 ") == {"A": 0.98, "B": 0.95}
    assert parse_service_levels("C=0.5") == {"C": 0.5}  # lower bound inclusive
    assert parse_service_levels("A=0.98,") == {"A": 0.98}  # trailing comma tolerated
    assert parse_service_levels("") == {}


@pytest.mark.parametrize(
    ("spec", "token"),
    [
        ("A=0.98,D=0.9", "D=0.9"),  # unknown class
        ("A=1.0", "A=1.0"),  # upper bound exclusive
        ("B=0.49", "B=0.49"),  # below the lower bound
        ("A=0.9,B", "B"),  # missing '='
        ("A=", "A="),  # empty value
        ("A=high", "A=high"),  # not a number
        ("=0.9", "=0.9"),  # empty class
        ("A=0.9,a=0.95", "a=0.95"),  # duplicate class (case-insensitive)
        ("A=nan", "A=nan"),  # not a finite level
    ],
)
def test_parse_service_levels_errors_name_the_offending_token(spec, token):
    with pytest.raises(ValueError, match=re.escape(token)):
        parse_service_levels(spec)


def test_service_level_for_lookup_rules():
    overrides = parse_service_levels("A=0.98,C=0.90")
    assert service_level_for("A", 0.95, overrides) == 0.98
    assert service_level_for("a", 0.95, overrides) == 0.98  # case-insensitive
    assert service_level_for("B", 0.95, overrides) == 0.95  # no override → default
    assert service_level_for(None, 0.95, overrides) == 0.95
    assert service_level_for("A", 0.95, None) == 0.95
    assert service_level_for("A", 0.95, {}) == 0.95
    assert service_level_for(" c ", 0.95, {"c": 0.9}) == 0.9  # keys normalised too
    # the per-class level feeds straight into the policy
    hi = decide(base(service_level=service_level_for("A", 0.95, overrides)))
    lo = decide(base(service_level=service_level_for("C", 0.95, overrides)))
    assert hi.safety_stock > lo.safety_stock


# -- explain() ---------------------------------------------------------------------


def test_explain_contains_key_numbers():
    text = decide(base()).explain()
    assert "\n" not in text
    assert "IP 10" in text and "31.4" in text and "order 54" in text and "59.4" in text
    assert "risk 1.00" in text
    no_order = decide(base(on_hand=30, on_order=40)).explain()
    assert "IP 70" in no_order and "no order" in no_order and "risk 0.00" in no_order
    capped = decide(base(on_hand=0, shelf_life_days=3, case_pack=1)).explain()
    assert "order 12" in capped and "shelf life" in capped
