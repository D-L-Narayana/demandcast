"""Inventory replenishment policy: periodic-review (R, s, S) with forecast-driven safety stock.

For each store × product:

    lead_time_demand  = Σ yhat over the lead time (L days)
    safety_stock      = z(service_level) · σ_residual · √(L + R)
    reorder_point (s) = lead_time_demand + safety_stock
    order_up_to   (S) = Σ yhat over (L + R) days + safety_stock
    inventory_position = on_hand + on_order
    order_qty         = round_up_to_case_pack(S − IP)   if IP < s else 0

σ_residual is the backtest residual standard deviation of the *selected* model, so series
that are harder to forecast automatically carry more buffer stock. R is the review period
(how often we place orders), L is the supplier lead time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class ReplenishmentInput:
    on_hand: int
    on_order: int
    lead_time_days: int
    review_period_days: int
    case_pack: int
    forecast: list[float]  # daily point forecast, day 1 .. day H (H >= L + R)
    residual_std: float
    service_level: float = 0.95
    shelf_life_days: int | None = None


@dataclass(frozen=True)
class ReplenishmentDecision:
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    order_up_to: float
    inventory_position: int
    order_qty: int
    reason: str


def z_score(service_level: float) -> float:
    if not 0.5 <= service_level < 1.0:
        raise ValueError("service_level must be in [0.5, 1.0)")
    return NormalDist().inv_cdf(service_level)


def round_up_to_pack(qty: float, case_pack: int) -> int:
    if qty <= 0:
        return 0
    return int(math.ceil(qty / case_pack) * case_pack)


def decide(inp: ReplenishmentInput) -> ReplenishmentDecision:
    L, R = inp.lead_time_days, inp.review_period_days
    if len(inp.forecast) < L + R:
        raise ValueError(f"forecast horizon {len(inp.forecast)} shorter than L+R={L + R}")
    if inp.case_pack < 1:
        raise ValueError("case_pack must be >= 1")

    ltd = float(sum(inp.forecast[:L]))
    protection_demand = float(sum(inp.forecast[: L + R]))
    ss = z_score(inp.service_level) * inp.residual_std * math.sqrt(L + R)
    s = ltd + ss
    S = protection_demand + ss
    ip = inp.on_hand + inp.on_order

    if ip >= s:
        return ReplenishmentDecision(ltd, ss, s, S, ip, 0, "inventory position above reorder point")

    raw_qty = S - ip
    qty = round_up_to_pack(raw_qty, inp.case_pack)
    reason = "inventory position below reorder point"

    # Perishables: never order more than can sell before it expires.
    if inp.shelf_life_days is not None:
        sellable = float(sum(inp.forecast[: min(inp.shelf_life_days, len(inp.forecast))]))
        cap = max(0, int(math.floor(sellable)) - ip)
        cap = round_up_to_pack(cap, inp.case_pack) if cap > 0 else 0
        if qty > cap:
            qty = cap
            reason += "; capped by shelf life"

    return ReplenishmentDecision(ltd, ss, s, S, ip, qty, reason)
