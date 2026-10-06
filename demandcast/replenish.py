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

Risk view. Demand over the protection period L + R is treated as normal, D ~ N(μ_LR, σ_LR²),
with μ_LR = Σ yhat[:L+R], σ_LR = σ_residual · √(L + R) and k = (IP − μ_LR) / σ_LR:

    stockout_risk      = P(D > IP) = 1 − Φ(k)
    expected_shortfall = E[max(D − IP, 0)] = σ_LR · L(k),  L(k) = φ(k) − k · (1 − Φ(k))
    priority           = expected_shortfall · unit_cost          (cost-weighted shortfall)
    days_of_cover      = IP / (μ_LR / (L + R))                   (None when μ_LR = 0)

L(k) is the standard normal loss function. When σ_LR = 0 demand is deterministic: the risk is
1 if IP < μ_LR else 0 and the shortfall is max(0, μ_LR − IP). The risk numbers are reported for
every series, ordered or not, so the order book can be prioritised.

Order-quantity constraints apply after case-pack rounding, in this order: minimum order
(rounded UP to the pack), maximum order (rounded DOWN to the pack; a maximum below one pack
means nothing can be ordered), shelf-life cap (always wins). Spreading a limited order budget
across series is a separate greedy step, `allocate_budget`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

_STD_NORMAL = NormalDist()
_ABC_CLASSES = ("A", "B", "C")


@dataclass(frozen=True)
class ReplenishmentInput:
    """Everything the (R, s, S) policy needs for one store × product.

    `unit_cost` only weights the priority; `min_order_qty` / `max_order_qty` constrain the
    order after case-pack rounding (see `decide`). All three are optional.
    """

    on_hand: int
    on_order: int
    lead_time_days: int
    review_period_days: int
    case_pack: int
    forecast: list[float]  # daily point forecast, day 1 .. day H (H >= L + R)
    residual_std: float
    service_level: float = 0.95
    shelf_life_days: int | None = None
    unit_cost: float | None = None
    min_order_qty: int = 0
    max_order_qty: int | None = None


@dataclass(frozen=True)
class ReplenishmentDecision:
    """Policy outputs plus the risk view over the protection period L + R.

    With μ_LR = Σ forecast[:L+R], σ_LR = residual_std · √(L+R) and k = (IP − μ_LR) / σ_LR:

        stockout_risk      = 1 − Φ(k)
        expected_shortfall = σ_LR · [φ(k) − k · (1 − Φ(k))]
        priority           = expected_shortfall · (unit_cost or 1.0)
        days_of_cover      = IP / mean daily forecast over L + R   (None if that mean is 0)

    `reason` lists every constraint that changed the order quantity; `explain()` renders the
    whole decision as one line.
    """

    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    order_up_to: float
    inventory_position: int
    order_qty: int
    reason: str
    stockout_risk: float = 0.0
    expected_shortfall: float = 0.0
    priority: float = 0.0
    days_of_cover: float | None = None

    def explain(self) -> str:
        """One-line human rationale, e.g. ``IP 10 < s 31.4 → order 54 (S 59.4), risk 1.00``.

        Constraints that changed the quantity (minimum/maximum order, shelf life) are listed in
        square brackets at the end.
        """
        below = self.inventory_position < self.reorder_point
        action = f"order {self.order_qty}" if self.order_qty > 0 else "no order"
        text = (
            f"IP {self.inventory_position} {'<' if below else '>='} s {self.reorder_point:.1f}"
            f" → {action} (S {self.order_up_to:.1f}), risk {self.stockout_risk:.2f}"
        )
        notes = self.reason.split("; ")[1:]
        if notes:
            text += " [" + "; ".join(notes) + "]"
        return text


def z_score(service_level: float) -> float:
    if not 0.5 <= service_level < 1.0:
        raise ValueError("service_level must be in [0.5, 1.0)")
    return NormalDist().inv_cdf(service_level)


def round_up_to_pack(qty: float, case_pack: int) -> int:
    if qty <= 0:
        return 0
    return int(math.ceil(qty / case_pack) * case_pack)


def round_down_to_pack(qty: float, case_pack: int) -> int:
    """Largest multiple of `case_pack` that is ≤ qty (0 for anything below one pack)."""
    if qty <= 0:
        return 0
    return math.floor(qty / case_pack) * case_pack


def stockout_probability(mu: float, sigma: float, ip: float) -> float:
    """P(demand over the protection period exceeds the inventory position).

        risk = P(D > ip) = 1 − Φ((ip − mu) / sigma) = Φ((mu − ip) / sigma)

    for D ~ N(mu, sigma²); the mirrored form keeps precision in the upper tail. With sigma == 0
    demand is deterministic: 1.0 when ip < mu, else 0.0.
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return 1.0 if ip < mu else 0.0
    return _STD_NORMAL.cdf((mu - ip) / sigma)


def expected_shortfall(mu: float, sigma: float, ip: float) -> float:
    """Expected units short over the protection period — the normal loss function.

        E[max(D − ip, 0)] = sigma · L(k),   L(k) = φ(k) − k · (1 − Φ(k)),   k = (ip − mu) / sigma

    L(0) = φ(0) ≈ 0.3989, L(k) → 0 for k ≫ 0 and L(k) → −k for k ≪ 0, so the shortfall tends to
    mu − ip when the position is far below the mean. With sigma == 0: max(0, mu − ip).
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return max(0.0, mu - ip)
    k = (ip - mu) / sigma
    loss = _STD_NORMAL.pdf(k) - k * _STD_NORMAL.cdf(-k)
    return max(0.0, sigma * loss)


def decide(inp: ReplenishmentInput) -> ReplenishmentDecision:
    """Apply the (R, s, S) policy to one series and attach the protection-period risk view.

    Order quantity: ``round_up_to_pack(S − IP)`` when IP < s, else 0. Then, in this order:

    1. ``min_order_qty`` (rounded UP to the pack) lifts a smaller order
       → reason ``"; raised to minimum order"``;
    2. ``max_order_qty`` (rounded DOWN to the pack; below one pack → 0) caps a larger order
       → reason ``"; capped by maximum order"``;
    3. the shelf-life cap always wins → reason ``"; capped by shelf life"``.

    ``priority = expected_shortfall · (unit_cost or 1.0)``: an unknown (None) or zero cost
    weights by 1.0, i.e. the priority is then expressed in units.
    """
    L, R = inp.lead_time_days, inp.review_period_days
    if len(inp.forecast) < L + R:
        raise ValueError(f"forecast horizon {len(inp.forecast)} shorter than L+R={L + R}")
    if inp.case_pack < 1:
        raise ValueError("case_pack must be >= 1")
    if inp.residual_std < 0:
        raise ValueError("residual_std must be >= 0")
    if inp.unit_cost is not None and inp.unit_cost < 0:
        raise ValueError("unit_cost must be >= 0")
    if inp.min_order_qty < 0:
        raise ValueError("min_order_qty must be >= 0")
    if inp.max_order_qty is not None and inp.max_order_qty < 0:
        raise ValueError("max_order_qty must be >= 0")

    ltd = float(sum(inp.forecast[:L]))
    protection_demand = float(sum(inp.forecast[: L + R]))
    sigma_lr = inp.residual_std * math.sqrt(L + R)
    ss = z_score(inp.service_level) * sigma_lr
    s = ltd + ss
    S = protection_demand + ss
    ip = inp.on_hand + inp.on_order

    # Risk view over the protection period (reported whether or not an order is placed).
    risk = stockout_probability(protection_demand, sigma_lr, ip)
    shortfall = expected_shortfall(protection_demand, sigma_lr, ip)
    priority = shortfall * (inp.unit_cost or 1.0)
    mean_daily = protection_demand / (L + R) if L + R > 0 else 0.0
    cover = ip / mean_daily if mean_daily > 0 else None

    if ip >= s:
        return ReplenishmentDecision(
            ltd,
            ss,
            s,
            S,
            ip,
            0,
            "inventory position above reorder point",
            stockout_risk=risk,
            expected_shortfall=shortfall,
            priority=priority,
            days_of_cover=cover,
        )

    raw_qty = S - ip
    qty = round_up_to_pack(raw_qty, inp.case_pack)
    reason = "inventory position below reorder point"

    # Supplier minimum: the floor itself is rounded UP to a whole number of packs.
    if inp.min_order_qty > 0:
        floor_qty = round_up_to_pack(inp.min_order_qty, inp.case_pack)
        if qty < floor_qty:
            qty = floor_qty
            reason += "; raised to minimum order"

    # Supplier / storage maximum: rounded DOWN to packs; below one pack nothing can be ordered.
    if inp.max_order_qty is not None:
        ceiling_qty = round_down_to_pack(inp.max_order_qty, inp.case_pack)
        if qty > ceiling_qty:
            qty = ceiling_qty
            reason += "; capped by maximum order"

    # Perishables: never order more than can sell before it expires (wins over everything).
    if inp.shelf_life_days is not None:
        sellable = float(sum(inp.forecast[: min(inp.shelf_life_days, len(inp.forecast))]))
        cap = max(0, math.floor(sellable) - ip)
        cap = round_up_to_pack(cap, inp.case_pack) if cap > 0 else 0
        if qty > cap:
            qty = cap
            reason += "; capped by shelf life"

    return ReplenishmentDecision(
        ltd,
        ss,
        s,
        S,
        ip,
        qty,
        reason,
        stockout_risk=risk,
        expected_shortfall=shortfall,
        priority=priority,
        days_of_cover=cover,
    )


@dataclass(frozen=True)
class BudgetItem:
    """One candidate order for `allocate_budget`: cost = order_qty · unit_cost."""

    key: tuple[int, int]
    order_qty: int
    unit_cost: float
    priority: float


def _fits(cost: float, remaining: float) -> bool:
    """cost ≤ remaining, accepting exact fits that differ only by floating-point round-off."""
    return cost <= remaining or math.isclose(cost, remaining, rel_tol=1e-9)


def allocate_budget(
    items: Sequence[BudgetItem], budget: float | None
) -> dict[tuple[int, int], int]:
    """Greedy first-fit allocation of whole orders under a total cost budget.

    Items are visited by priority descending (ties: key ascending). An order is approved in
    full when ``order_qty · unit_cost ≤ remaining`` and its cost is deducted; otherwise it is
    deferred (approved quantity 0) and the walk continues with cheaper lower-priority orders.
    Hence Σ approved cost ≤ budget; exact fits that differ only by floating-point round-off
    (0.1 + 0.2 against 0.3) are accepted within a relative tolerance of 1e-9. ``budget=None``
    approves everything. Returns ``{key: approved_qty}`` in input order; keys must be unique.
    """
    keys = [it.key for it in items]
    if len(set(keys)) != len(keys):
        raise ValueError("BudgetItem keys must be unique")
    if budget is None:
        return {it.key: it.order_qty for it in items}
    if budget < 0:
        raise ValueError("budget must be >= 0")
    approved = dict.fromkeys(keys, 0)
    remaining = float(budget)
    for it in sorted(items, key=lambda it: (-it.priority, it.key)):
        cost = it.order_qty * it.unit_cost
        if _fits(cost, remaining):
            approved[it.key] = it.order_qty
            remaining = max(0.0, remaining - cost)
    return approved


def service_level_for(
    abc_class: str | None, default: float, overrides: Mapping[str, float] | None
) -> float:
    """Service level for an ABC class: its override if present, else `default`.

    Classes are compared case-insensitively and whitespace-trimmed on both sides.
    """
    if not overrides or abc_class is None:
        return default
    wanted = abc_class.strip().upper()
    for cls, level in overrides.items():
        if cls.strip().upper() == wanted:
            return level
    return default


def parse_service_levels(spec: str) -> dict[str, float]:
    """Parse ``"A=0.98,B=0.95,C=0.90"`` into ``{"A": 0.98, "B": 0.95, "C": 0.90}``.

    Classes are limited to A/B/C (case-insensitive, whitespace tolerated, empty tokens skipped);
    levels must lie in [0.5, 1.0). Every ValueError names the offending token.
    """
    levels: dict[str, float] = {}
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        cls, sep, value = token.partition("=")
        cls, value = cls.strip().upper(), value.strip()
        if not sep or not cls or not value:
            raise ValueError(f"malformed service level {token!r}: expected CLASS=LEVEL")
        if cls not in _ABC_CLASSES:
            raise ValueError(f"unknown ABC class in {token!r}: expected one of A, B, C")
        try:
            level = float(value)
        except ValueError:
            raise ValueError(f"service level in {token!r} is not a number") from None
        if not 0.5 <= level < 1.0:  # also rejects nan/inf
            raise ValueError(f"service level in {token!r} must be in [0.5, 1.0)")
        if cls in levels:
            raise ValueError(f"duplicate ABC class in {token!r}")
        levels[cls] = level
    return levels
