"""Synthetic but realistic retail dataset generator.

Demand for each store x product series is built from multiplicative components::

    lam_t    = base[s,p] * weekly[dow_t] * yearly(t) * trend(t) * promo[s,p](t) * holiday(t) * q4(t)
    demand_t ~ Poisson(lam_t)

with

    base[s,p]  ~ Gamma(shape=4, scale=mean_base[cat]/4) * FORMAT_MULT[store format]
    weekly     = (0.90, 0.85, 0.88, 0.95, 1.05, 1.30, 1.25)[weekday]      (Mon..Sun)
    yearly(t)  = 1 + 0.12 * sin(2*pi * (doy_t - 60) / 365.25)
    trend(t)   = exp(delta * t),  delta ~ N(0, 0.0004)                    (gentle drift)
    promo(t)   = prod over promotions active on day t of (1 + 3.2 * discount_pct)
    holiday(t) = largest multiplier of the holidays whose pre-holiday ramp covers day t
    q4(t)      = 1 + 0.35 * [doy_t > 300] for Toys / Electronics, else 1

Observed sales are then *censored* by a simple (s, S) inventory simulation so stock-outs appear
in the data exactly the way they do in real POS feeds::

    sold_t          = min(demand_t, on_hand_t)
    stockout_flag_t = [sold_t < demand_t]
    revenue_t       = round(sold_t * unit_price * (1 - d_t), 2)

where ``d_t`` is the largest ``discount_pct`` among the promotions (chain-wide or store-specific)
active for that store/product/day, and 0 when none is active.

Reproducibility contract: every random draw up to and including the sales loop happens in a
fixed order, so a given ``SimConfig`` always yields a byte-identical dataset.  Opt-in features
(``future_promo_days``) draw *after* the sales loop; ``snapshot_every_days`` draws nothing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .db import insert_many, transaction

CATEGORIES = {
    "Grocery": (12.0, 1.6, 4, 21),  # (mean_base, price_mult, lead_time_days, shelf_life)
    "Household": (5.0, 1.9, 7, None),
    "Beauty": (3.5, 2.4, 10, 365),
    "Apparel": (2.5, 2.8, 14, None),
    "Electronics": (0.8, 1.35, 21, None),
    "Toys": (1.8, 2.1, 14, None),
}

CITIES = [
    ("BLR", "Bengaluru", "South", "flagship"),
    ("HYD", "Hyderabad", "South", "standard"),
    ("CHN", "Chennai", "South", "standard"),
    ("VSK", "Visakhapatnam", "South", "express"),
    ("MUM", "Mumbai", "West", "flagship"),
    ("PUN", "Pune", "West", "standard"),
    ("AMD", "Ahmedabad", "West", "express"),
    ("DEL", "New Delhi", "North", "flagship"),
    ("JAI", "Jaipur", "North", "express"),
    ("KOL", "Kolkata", "East", "standard"),
]

FORMAT_MULT = {"flagship": 1.6, "standard": 1.0, "express": 0.55}

# (month, day, name, demand multiplier, days of effect)
HOLIDAYS = [
    (1, 26, "Republic Day", 1.35, 1),
    (3, 14, "Holi", 1.30, 2),
    (8, 15, "Independence Day", 1.30, 1),
    (10, 2, "Gandhi Jayanti", 1.15, 1),
    (10, 20, "Dussehra", 1.55, 3),
    (11, 1, "Diwali", 1.90, 5),
    (12, 25, "Christmas", 1.60, 3),
    (12, 31, "New Year's Eve", 1.40, 2),
]

PROMO_DISCOUNTS = [0.1, 0.15, 0.2, 0.25, 0.3]
FUTURE_PROMO_PROBABILITY = 0.5  # share of products that get one promotion in the future window


@dataclass(frozen=True)
class SimConfig:
    """Generator settings.

    ``snapshot_every_days``: in addition to the final-day inventory snapshot, write a snapshot
    on every day ``i`` with ``(days - 1 - i) % snapshot_every_days == 0`` (counting back from
    the last day), so backdated runs (``run --cutoff``) find a snapshot as of their cutoff.
    ``future_promo_days``: also schedule promotions that start strictly after the last
    calendar day and end within ``future_promo_days`` days after it, so promo-aware forecasts
    have something to react to.  Both default to the v0.3.0 behaviour (off).
    """

    n_stores: int = 10
    n_products: int = 40
    start: date = date(2024, 1, 1)
    days: int = 730
    seed: int = 42
    stockout_prob_scale: float = 1.0
    snapshot_every_days: int | None = None
    future_promo_days: int = 0

    def __post_init__(self) -> None:
        if self.snapshot_every_days is not None and self.snapshot_every_days < 1:
            raise ValueError("snapshot_every_days must be >= 1 or None")
        if self.future_promo_days < 0:
            raise ValueError("future_promo_days must be >= 0")


def _holiday_lookup(start: date, days: int) -> dict[date, tuple[str, float]]:
    """Map day -> (holiday name, demand multiplier) for every calendar year the range touches.

    Each holiday lifts demand on its anchor day by ``mult`` and ramps up over the ``span``
    preceding days: ``1 + (mult - 1) * (1 - k / span)`` for ``k`` days before the anchor.
    """
    out: dict[date, tuple[str, float]] = {}
    last_year = (start + timedelta(days=days)).year
    for year in range(start.year, last_year + 1):
        for m, d, name, mult, span in HOLIDAYS:
            anchor = date(year, m, d)
            for k in range(span):
                day = anchor - timedelta(days=k)
                # Pre-holiday ramp: full multiplier on the day, decaying before it.
                decay = mult if k == 0 else 1 + (mult - 1) * (1 - k / span)
                out[day] = (name, max(out.get(day, ("", 1.0))[1], decay))
    return out


def build_calendar(start: date, days: int) -> list[tuple]:
    """Calendar rows (day, day_of_week, week_of_year, month, year, is_weekend, holiday_name)."""
    hol = _holiday_lookup(start, days)
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        iso = d.isocalendar()
        rows.append(
            (
                d.isoformat(),
                d.weekday(),
                iso[1],
                d.month,
                d.year,
                1 if d.weekday() >= 5 else 0,
                hol.get(d, (None, 1.0))[0],
            )
        )
    return rows


def _future_promotions(
    rng: np.random.Generator, cfg: SimConfig, first_promo_id: int
) -> list[tuple[int, int, int | None, str, str, float]]:
    """Promotions scheduled after the calendar ends (opt-in via ``future_promo_days``).

    Must be called after the sales loop so the historical draws stay untouched.  Each product
    independently receives one promotion with probability ``FUTURE_PROMO_PROBABILITY``; its
    start offset, length, scope (chain-wide 60 % / single store) and discount follow the same
    distributions as the historical promotions and the whole promotion lies inside
    ``[last_day + 1, last_day + future_promo_days]``.
    """
    window = cfg.future_promo_days
    first_future_day = cfg.start + timedelta(days=cfg.days)
    out: list[tuple[int, int, int | None, str, str, float]] = []
    promo_id = first_promo_id
    for pid in range(1, cfg.n_products + 1):
        if rng.random() >= FUTURE_PROMO_PROBABILITY:
            continue
        start_off = int(rng.integers(0, max(1, window - 14)))
        length = int(rng.integers(3, 15))
        end_off = min(start_off + length, window - 1)
        store_id = None if rng.random() < 0.6 else int(rng.integers(1, cfg.n_stores + 1))
        discount = float(np.round(rng.choice(PROMO_DISCOUNTS), 2))
        out.append(
            (
                promo_id,
                pid,
                store_id,
                (first_future_day + timedelta(days=start_off)).isoformat(),
                (first_future_day + timedelta(days=end_off)).isoformat(),
                discount,
            )
        )
        promo_id += 1
    return out


def generate(conn: sqlite3.Connection, cfg: SimConfig | None = None) -> dict[str, int]:
    """Populate an initialised schema with a full synthetic dataset. Returns row counts.

    See the module docstring for the demand equation.  The draw order is part of the public
    contract (byte-identical datasets per ``SimConfig``): master data, promotions, then one
    (base, trend, Poisson demand) draw group per store x product series; future promotions are
    drawn last.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    counts: dict[str, int] = {}

    # ---- master data -----------------------------------------------------------------------
    stores = []
    for i in range(cfg.n_stores):
        code, city, region, fmt = CITIES[i % len(CITIES)]
        if i >= len(CITIES):
            code = f"{code}{i // len(CITIES) + 1}"
        opened = cfg.start - timedelta(days=int(rng.integers(200, 3000)))
        stores.append((i + 1, code, city, region, fmt, opened.isoformat()))

    cat_names = list(CATEGORIES)
    products = []
    for j in range(cfg.n_products):
        cat = cat_names[j % len(cat_names)]
        mean_base, price_mult, lead, shelf = CATEGORIES[cat]
        cost = float(np.round(rng.lognormal(mean=np.log(120), sigma=0.8), 2))
        price = float(np.round(cost * price_mult * rng.uniform(0.9, 1.15), 2))
        case_pack = int(rng.choice([1, 6, 12, 24], p=[0.25, 0.35, 0.3, 0.1]))
        products.append(
            (
                j + 1,
                f"SKU-{cat[:3].upper()}-{j + 1:04d}",
                f"{cat} item {j + 1}",
                cat,
                cost,
                max(price, cost),
                case_pack,
                int(lead + rng.integers(-2, 3)),
                shelf,
            )
        )

    calendar = build_calendar(cfg.start, cfg.days)
    hol = _holiday_lookup(cfg.start, cfg.days)

    # ---- promotions ------------------------------------------------------------------------
    promos: list[tuple[int, int, int | None, str, str, float]] = []
    promo_id = 1
    for pid in range(1, cfg.n_products + 1):
        for _ in range(int(rng.integers(2, 6))):
            promo_start = cfg.start + timedelta(days=int(rng.integers(0, cfg.days - 14)))
            length = int(rng.integers(3, 15))
            store_id = None if rng.random() < 0.6 else int(rng.integers(1, cfg.n_stores + 1))
            promos.append(
                (
                    promo_id,
                    pid,
                    store_id,
                    promo_start.isoformat(),
                    (promo_start + timedelta(days=length)).isoformat(),
                    float(np.round(rng.choice(PROMO_DISCOUNTS), 2)),
                )
            )
            promo_id += 1

    # promo multiplier tensor [product, store, day] and the matching discount tensor
    # (largest discount_pct active per store/product/day -> drives promo-day revenue, D9)
    promo_mult = np.ones((cfg.n_products, cfg.n_stores, cfg.days))
    promo_disc = np.zeros((cfg.n_products, cfg.n_stores, cfg.days))
    for _, pid, sid, start_iso, end_iso, disc in promos:
        s_i = (date.fromisoformat(start_iso) - cfg.start).days
        e_i = (date.fromisoformat(end_iso) - cfg.start).days
        lift = 1 + 3.2 * disc  # 10% off -> 1.32x, 30% off -> ~1.96x
        if sid is None:
            promo_mult[pid - 1, :, s_i : e_i + 1] *= lift
            window = promo_disc[pid - 1, :, s_i : e_i + 1]
            np.maximum(window, disc, out=window)
        else:
            promo_mult[pid - 1, sid - 1, s_i : e_i + 1] *= lift
            window = promo_disc[pid - 1, sid - 1, s_i : e_i + 1]
            np.maximum(window, disc, out=window)

    # ---- demand components -----------------------------------------------------------------
    t = np.arange(cfg.days)
    dow = np.array([(cfg.start + timedelta(days=int(i))).weekday() for i in t])
    weekly = np.array([0.90, 0.85, 0.88, 0.95, 1.05, 1.30, 1.25])[dow]
    doy = np.array([(cfg.start + timedelta(days=int(i))).timetuple().tm_yday for i in t])
    yearly = 1 + 0.12 * np.sin(2 * np.pi * (doy - 60) / 365.25)
    holiday = np.array([hol.get(cfg.start + timedelta(days=int(i)), ("", 1.0))[1] for i in t])

    # inventory snapshot days: always the final day; with snapshot_every_days=k also every
    # k-th day counting back from it
    if cfg.snapshot_every_days is None:
        snapshot_days = {cfg.days - 1}
    else:
        k = cfg.snapshot_every_days
        snapshot_days = {i for i in range(cfg.days) if (cfg.days - 1 - i) % k == 0}

    sales_rows: list[tuple] = []
    snap_rows: list[tuple] = []
    price_by_pid = {p[0]: p[5] for p in products}

    for pid, prod in enumerate(products, start=1):
        cat = prod[3]
        mean_base = CATEGORIES[cat][0]
        for sid, store in enumerate(stores, start=1):
            base = rng.gamma(shape=4.0, scale=mean_base / 4.0) * FORMAT_MULT[store[4]]
            trend = np.exp(rng.normal(0, 0.0004) * t)  # gentle drift per series
            lam = base * weekly * yearly * trend * holiday * promo_mult[pid - 1, sid - 1]
            # Category-specific seasonality kicks (Toys/Electronics peak in Q4).
            if cat in ("Toys", "Electronics"):
                lam = lam * (1 + 0.35 * (doy > 300))
            demand = rng.poisson(lam)
            disc_by_day = promo_disc[pid - 1, sid - 1]

            # Inventory censoring: naive (s, S) policy with lead-time delays.
            on_hand = int(lam[:28].sum() * 1.2)
            pipeline: list[tuple[int, int]] = []
            lead = prod[7]
            reorder_point = lam[:14].sum() * 1.05 * cfg.stockout_prob_scale
            order_up_to = lam[:28].sum() * 1.15
            for i in range(cfg.days):
                arrived = [q for (arr, q) in pipeline if arr == i]
                on_hand += sum(arrived)
                pipeline = [(a, q) for (a, q) in pipeline if a != i]
                sold = min(demand[i], on_hand)
                stockout = 1 if sold < demand[i] else 0
                on_hand -= sold
                on_order = sum(q for _, q in pipeline)
                if on_hand + on_order < reorder_point:
                    qty = int(max(0, order_up_to - on_hand - on_order))
                    pipeline.append((i + lead, qty))
                    on_order += qty
                day_iso = calendar[i][0]
                price = price_by_pid[pid]
                promo_discount = float(disc_by_day[i])  # 0.0 off-promo -> units * price exactly
                sales_rows.append(
                    (
                        sid,
                        pid,
                        day_iso,
                        int(sold),
                        round(sold * price * (1 - promo_discount), 2),
                        stockout,
                    )
                )
                if i in snapshot_days:
                    snap_rows.append((sid, pid, day_iso, int(on_hand), int(on_order)))

    # ---- future promotions (opt-in; drawn only after every historical draw) ----------------
    if cfg.future_promo_days > 0:
        promos.extend(_future_promotions(rng, cfg, promo_id))

    # ---- bulk load ------------------------------------------------------------------------
    with transaction(conn):
        counts["stores"] = insert_many(
            conn,
            "stores",
            ["store_id", "store_code", "city", "region", "format", "opened_on"],
            stores,
        )
        counts["products"] = insert_many(
            conn,
            "products",
            [
                "product_id",
                "sku",
                "name",
                "category",
                "unit_cost",
                "unit_price",
                "case_pack",
                "lead_time_days",
                "shelf_life_days",
            ],
            products,
        )
        counts["calendar"] = insert_many(
            conn,
            "calendar",
            ["day", "day_of_week", "week_of_year", "month", "year", "is_weekend", "holiday_name"],
            calendar,
        )
        counts["promotions"] = insert_many(
            conn,
            "promotions",
            ["promo_id", "product_id", "store_id", "start_day", "end_day", "discount_pct"],
            promos,
        )
        counts["sales_daily"] = insert_many(
            conn,
            "sales_daily",
            ["store_id", "product_id", "day", "units_sold", "revenue", "stockout_flag"],
            sales_rows,
        )
        counts["inventory_snapshots"] = insert_many(
            conn,
            "inventory_snapshots",
            ["store_id", "product_id", "snapshot_day", "on_hand", "on_order"],
            snap_rows,
        )
    return counts
