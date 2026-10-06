"""Render a self-contained HTML dashboard (inline SVG, zero JavaScript) from SQL results.

Section ↔ data map (``*`` = introduced with analytics v2; each one degrades to a muted
"Not available in this run" note when its query is missing from ``db.QUERIES``, fails on an
older schema, or returns no rows, so a v0.3 database still renders):

====================  ==============================================================
section name          source
====================  ==============================================================
kpis                  forecast_accuracy_leaderboard · replenishment_summary · days_of_cover
featured              abc_classification · series_history · forecasts (+ promo_flag*)
leaderboard           forecast_accuracy_leaderboard
categories            weekly_sales_trend (last 8 weeks **per category**)
accuracy*             evaluation_history · evaluation_summary
health_distribution*  inventory_health_distribution
stockouts             stockout_rate_by_store
health                days_of_cover
risk*                 stockout_risk_top
orders                replenishment_summary
order_cost*           order_cost_by_category
rollup*               forecast_rollup
abc                   abc_classification
promo_lift            promo_lift
promo_upcoming*       promo_calendar_upcoming
run_config*           forecast_runs.config_json / interval_level / engine_version
====================  ==============================================================

Every query is executed at most once per render and its rows are shared between sections.
The document is policy-clean under :data:`CSP_META` (no scripts, inline handlers, external
resources, ``url(…)`` or ``@font-face``) and carries that policy in a ``<meta>`` tag.
"""

from __future__ import annotations

import html
import json
import logging
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import db
from .charts import RED, bar_chart, hbar_chart, line_chart
from .pipeline import latest_successful_run

log = logging.getLogger("demandcast")

GREY = "#6b7280"
CSP_META = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
    "base-uri 'none'; form-action 'none'"
)
NOT_AVAILABLE = "Not available in this run"
DEFAULT_INTERVAL_LEVEL = 0.8
SECTIONS: tuple[str, ...] = (
    "kpis",
    "featured",
    "leaderboard",
    "categories",
    "accuracy",
    "health_distribution",
    "stockouts",
    "health",
    "risk",
    "orders",
    "order_cost",
    "rollup",
    "abc",
    "promo_lift",
    "promo_upcoming",
    "run_config",
)
HEALTH_COLOURS = {
    "CRITICAL": RED,
    "LOW": "#d97706",
    "OK": "#059669",
    "OVERSTOCK": "#2563eb",
    "NO_DEMAND": "#9ca3af",
}
ORDER_COLUMNS = [
    "store_code",
    "sku",
    "category",
    "on_hand",
    "on_order",
    "lead_time_demand",
    "safety_stock",
    "reorder_point",
    "order_qty",
    "order_cost",
    "expected_arrival",
    "reason",
]
ORDER_EXTRA_COLUMNS = ("requested_qty", "stockout_risk", "priority")

Row = dict[str, Any]
Formatter = Callable[[Any], str]


# ---------------------------------------------------------------------------------------------
# data access: every named query runs at most once per render
# ---------------------------------------------------------------------------------------------
class _Source:
    """Fetches named queries once, remembers rows and the reason when a dataset is unavailable."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._rows: dict[str, list[Row]] = {}
        self._reasons: dict[str, str] = {}

    @staticmethod
    def _key(name: str, params: Mapping[str, Any] | None) -> str:
        return name if not params else name + "|" + repr(sorted(params.items()))

    def get(self, name: str, params: Mapping[str, Any] | None = None) -> list[Row]:
        key = self._key(name, params)
        if key in self._rows:
            return self._rows[key]
        rows: list[Row] = []
        if name not in db.QUERIES:
            self._reasons[key] = f"query '{name}' is not installed in analytics.sql"
        else:
            try:
                rows = db.run_query(self.conn, name, params)
            except sqlite3.Error as exc:
                log.warning("dashboard: query %s failed: %s", name, exc)
                self._reasons[key] = f"query '{name}' failed on this database ({exc})"
            else:
                if not rows:
                    self._reasons[key] = f"query '{name}' returned no rows"
        self._rows[key] = rows
        return rows

    def reason(self, name: str, params: Mapping[str, Any] | None = None) -> str:
        return self._reasons.get(self._key(name, params), "")


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def _interval_level(run: Mapping[str, Any]) -> float:
    """Nominal interval level of a run: ``forecast_runs.interval_level`` when present, else 0.8."""
    raw = run.get("interval_level")
    if raw is None:
        return DEFAULT_INTERVAL_LEVEL
    try:
        level = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_LEVEL
    return level if 0.0 < level < 1.0 else DEFAULT_INTERVAL_LEVEL


def _run_config(run: Mapping[str, Any]) -> dict[str, Any] | None:
    """Parsed ``forecast_runs.config_json`` (schema v2) or ``None`` when absent/invalid."""
    raw = run.get("config_json")
    if not raw:
        return None
    try:
        cfg = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return cfg if isinstance(cfg, dict) else None


# ---------------------------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------------------------
def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return html.escape(str(v))


def _pct(v: Any) -> str:
    """Fraction → percentage with one decimal (``0.314`` → ``31.4%``)."""
    try:
        return "—" if v is None else f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return _fmt(v)


def _pct0(v: Any) -> str:
    try:
        return "—" if v is None else f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return _fmt(v)


def _table(
    rows: Sequence[Row],
    columns: Sequence[str] | None = None,
    limit: int = 15,
    *,
    caption: str = "",
    fmt: Mapping[str, Formatter] | None = None,
) -> str:
    """Accessible table: ``<caption>`` (visually hidden), ``scope="col"`` headers, ≤ ``limit`` rows."""
    if not rows:
        return '<p class="muted">No rows.</p>'
    cols = list(columns or rows[0].keys())
    head = "".join(f'<th scope="col">{html.escape(c)}</th>' for c in cols)
    body: list[str] = []
    for r in rows[:limit]:
        cells: list[str] = []
        for c in cols:
            v = r.get(c)
            f = fmt.get(c) if fmt else None
            text = f(v) if f else _fmt(v)
            klass = (
                ' class="num"' if isinstance(v, (int, float)) and not isinstance(v, bool) else ""
            )
            cells.append(f"<td{klass}>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    note = (
        f'<p class="muted">Showing {min(limit, len(rows))} of {len(rows)} rows.</p>'
        if len(rows) > limit
        else ""
    )
    return (
        f'<table><caption class="vh">{html.escape(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{note}"
    )


def _kpi(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi"><div class="kpi-v">{html.escape(value)}</div>'
        f'<div class="kpi-l">{html.escape(label)}</div>'
        f'<div class="kpi-s">{html.escape(sub)}</div></div>'
    )


def _na(reason: str = "") -> str:
    tail = f" — {html.escape(reason)}" if reason else ""
    return f'<p class="na">{NOT_AVAILABLE}{tail}.</p>'


def _card(inner: str, *, scroll_label: str | None = None, extra_class: str = "") -> str:
    classes = (
        "card" + (" scroll" if scroll_label else "") + (f" {extra_class}" if extra_class else "")
    )
    attrs = (
        f' tabindex="0" role="region" aria-label="{html.escape(scroll_label)}"'
        if scroll_label
        else ""
    )
    return f'<div class="{classes}"{attrs}>{inner}</div>'


def _section(name: str, title_html: str, body: str) -> str:
    sid = f"sec-{name}"
    return (
        f'<section id="{sid}" aria-labelledby="{sid}-h"><h2 id="{sid}-h">{title_html}</h2>'
        f"{body}</section>"
    )


def _table_section(
    name: str,
    title: str,
    rows: Sequence[Row],
    reason: str,
    *,
    columns: Sequence[str] | None = None,
    limit: int = 15,
    fmt: Mapping[str, Formatter] | None = None,
    lead: str = "",
) -> str:
    if not rows:
        return _section(name, html.escape(title), _card(lead + _na(reason)))
    table = _table(rows, columns, limit, caption=title, fmt=fmt)
    return _section(name, html.escape(title), _card(lead + table, scroll_label=f"{title} table"))


def _grid(parts: Sequence[str]) -> str:
    present = [p for p in parts if p]
    if not present:
        return ""
    if len(present) == 1:
        return present[0]
    return '<div class="grid2">' + "".join(present) + "</div>"


# ---------------------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------------------
def _orders_delta(conn: sqlite3.Connection, run_id: int, placed: int) -> str:
    """Comparison badge text vs the previous successful run (``""`` when there is none)."""
    prev = conn.execute(
        "SELECT MAX(run_id) AS rid FROM forecast_runs WHERE status='succeeded' AND run_id < ?",
        (run_id,),
    ).fetchone()
    if prev is None or prev["rid"] is None:
        return ""
    before = conn.execute(
        "SELECT COUNT(*) AS n FROM replenishment_orders WHERE run_id=? AND order_qty > 0",
        (prev["rid"],),
    ).fetchone()["n"]
    return f" · {placed - int(before):+d} vs run #{prev['rid']}"


def _kpi_section(
    conn: sqlite3.Connection,
    src: _Source,
    run: Mapping[str, Any],
    run_id: int,
    config: Mapping[str, Any] | None,
) -> str:
    p = {"run_id": run_id}
    leaderboard = src.get("forecast_accuracy_leaderboard", p)
    orders = src.get("replenishment_summary", p)
    cover = src.get("days_of_cover", p)
    placed = [o for o in orders if (o.get("order_qty") or 0) > 0]
    order_cost = sum(float(o.get("order_cost") or 0.0) for o in placed)
    critical = sum(1 for c in cover if c.get("health") == "CRITICAL")
    won = sum(int(r.get("series_won") or 0) for r in leaderboard)
    avg_wape = sum(
        float(r["avg_wape"]) * int(r.get("series_won") or 0)
        for r in leaderboard
        if r.get("avg_wape") is not None
    ) / max(won, 1)
    folds = config.get("n_folds") if config else None
    wape_sub = (
        f"lower is better; rolling-origin, {folds} folds"
        if folds
        else "lower is better; rolling-origin backtest"
    )
    deferred = sum(
        1 for o in orders if (o.get("order_qty") or 0) == 0 and (o.get("requested_qty") or 0) > 0
    )
    orders_sub = f"₹{order_cost:,.0f} order cost" + _orders_delta(conn, run_id, len(placed))
    if deferred:
        orders_sub += f" · {deferred} deferred (budget)"
    kpis = [
        _kpi("Weighted WAPE (backtest)", f"{avg_wape * 100:.1f}%", wape_sub),
        _kpi("Replenishment orders", f"{len(placed)}", orders_sub),
        _kpi(
            "Critical cover (<0.5 wk)",
            f"{critical}",
            "store \N{MULTIPLICATION SIGN} SKU below half of 7-day demand",
        ),
        _kpi("Series forecast", f"{run.get('series_count')}", str(run.get("notes") or "")),
    ]
    return (
        f'<section id="sec-kpis" aria-label="Key figures"><div class="kpis">{"".join(kpis)}</div>'
        "</section>"
    )


def _featured_section(
    conn: sqlite3.Connection, src: _Source, run: Mapping[str, Any], run_id: int, level: float
) -> str:
    """Highest-revenue product at the first store: history, forecast, band, promo shading."""
    abc = src.get("abc_classification")
    store = conn.execute(
        "SELECT store_id, store_code FROM stores ORDER BY store_id LIMIT 1"
    ).fetchone()
    if not abc or store is None:
        return _section("featured", "Featured series", _card(_na("no products with sales history")))
    sku = str(abc[0].get("sku") or "")
    pid, sid, store_code = abc[0]["product_id"], store["store_id"], str(store["store_code"])
    hist = src.get("series_history", {"store_id": sid, "product_id": pid})[-84:]
    promo_col = ", promo_flag" if "promo_flag" in _columns(conn, "forecasts") else ""
    fc = [
        dict(r)
        for r in conn.execute(
            f"SELECT target_day, yhat, yhat_lower, yhat_upper, model_name{promo_col} FROM forecasts "
            "WHERE run_id=? AND store_id=? AND product_id=? ORDER BY target_day",
            (run_id, sid, pid),
        )
    ]
    title = f"Featured series — {html.escape(sku)} @ store {html.escape(store_code)}"
    if not hist and not fc:
        return _section("featured", title, _card(_na("no history or forecast for this series")))
    hist_vals = [float(r["units_sold"]) for r in hist]
    last: float | None = hist_vals[-1] if hist_vals else None
    pivot: list[float | None] = [last] if hist_vals else []
    actual: list[float | None] = [*hist_vals, *([None] * len(fc))]
    forecast: list[float | None] = [
        *([None] * max(len(hist) - 1, 0)),
        *pivot,
        *[float(r["yhat"]) for r in fc],
    ]
    band_lo: list[float | None] = [
        *([None] * max(len(hist) - 1, 0)),
        *pivot,
        *[float(r["yhat_lower"]) for r in fc],
    ]
    band_hi: list[float | None] = [
        *([None] * max(len(hist) - 1, 0)),
        *pivot,
        *[float(r["yhat_upper"]) for r in fc],
    ]
    x_labels = [str(r["day"]) for r in hist] + [str(r["target_day"]) for r in fc]
    shade: list[bool] | None = None
    hist_flags = [bool(r.get("on_promo")) for r in hist] if hist and "on_promo" in hist[0] else None
    fc_flags = [bool(r.get("promo_flag")) for r in fc] if promo_col else None
    if hist_flags is not None or fc_flags is not None:
        shade = (hist_flags or [False] * len(hist)) + (fc_flags or [False] * len(fc))
    chart = line_chart(
        [("Actual units", actual, "currentColor"), ("Forecast", forecast, RED)],
        x_labels=x_labels,
        band=(band_lo, band_hi),
        shade=shade,
        vline=len(hist) - 1 if hist else None,
        aria_label=f"Actual units and forecast for {sku} at store {store_code}",
    )
    model_used = str(fc[0]["model_name"]) if fc else "n/a"
    promo_note = " Shaded spans mark promotion days." if shade and any(shade) else ""
    caption = (
        f'<p class="muted">Last {len(hist)} days of actuals, {run.get("horizon_days")}-day forecast '
        f"from <b>{html.escape(model_used)}</b> with {level:.0%} interval; the dashed line is the "
        f"forecast cutoff.{promo_note}</p>"
    )
    return _section("featured", title, _card(caption + chart, extra_class="featured"))


def _categories_section(src: _Source) -> str:
    """Units per category over the last 8 week_start values *of each category* (D1 fix)."""
    weekly = src.get("weekly_sales_trend")
    title = "Units by category (last 8 weeks)"
    if not weekly:
        return _section("categories", title, _card(_na(src.reason("weekly_sales_trend"))))
    by_cat: dict[str, list[tuple[str, float]]] = {}
    for w in weekly:
        by_cat.setdefault(str(w["category"]), []).append(
            (str(w["week_start"]), float(w["units"] or 0))
        )
    labels = sorted(by_cat)
    values = [sum(units for _, units in sorted(by_cat[c])[-8:]) for c in labels]
    chart = bar_chart(labels, values, aria_label="Units sold by category over the last 8 weeks")
    return _section("categories", title, _card(chart))


def _accuracy_section(src: _Source, run_id: int, level: float) -> str:
    """Realised accuracy from ``evaluation_history`` (+ per-model table for this run)."""
    title = "Realised accuracy"
    history = sorted(
        src.get("evaluation_history"), key=lambda r: (str(r.get("cutoff_day")), int(r["run_id"]))
    )
    if not history:
        return _section("accuracy", title, _card(_na(src.reason("evaluation_history"))))
    current = next((r for r in history if int(r["run_id"]) == run_id), None)
    latest = current or history[-1]
    where = f"run #{latest['run_id']} · cutoff {latest.get('cutoff_day')} · {latest.get('n_series')} series"
    coverage_sub = "share of actuals inside the interval"
    if current is not None:
        coverage_sub += f" · nominal {level:.0%}"
    kpis = (
        _kpi("Realised WAPE", _pct(latest.get("wape")), f"lower is better · {where}")
        + _kpi(
            "Realised MAE",
            _fmt(latest.get("mae")),
            f"units/day · {latest.get('horizon_days')}-day horizon",
        )
        + _kpi("Bias", _fmt(latest.get("bias")), "forecast \N{MINUS SIGN} actual, mean per day")
        + _kpi("Interval coverage", _pct0(latest.get("coverage")), coverage_sub)
    )
    chart = line_chart(
        [
            (
                "Realised WAPE %",
                [None if r.get("wape") is None else float(r["wape"]) * 100 for r in history],
                RED,
            ),
            (
                "Interval coverage %",
                [
                    None if r.get("coverage") is None else float(r["coverage"]) * 100
                    for r in history
                ],
                "#2563eb",
            ),
        ],
        x_labels=[str(r.get("cutoff_day")) for r in history],
        height=200,
        aria_label="Realised WAPE and interval coverage per evaluated run",
    )
    body = f'<div class="kpis">{kpis}</div>' + _card(
        '<p class="muted">Backdated runs evaluated against the sales that arrived afterwards '
        "(cutoff on the x-axis).</p>" + chart
    )
    by_model = src.get("evaluation_summary", {"run_id": run_id})
    if by_model:
        fmt = {"wape": _pct, "coverage": _pct0}
        body += "<h3>By model (this run)</h3>" + _card(
            _table(by_model, caption="Realised accuracy by model", fmt=fmt),
            scroll_label="Realised accuracy by model table",
        )
    return _section("accuracy", title, body)


def _health_distribution_section(src: _Source, run_id: int) -> str:
    title = "Inventory health distribution"
    rows = src.get("inventory_health_distribution", {"run_id": run_id})
    if not rows:
        return _section(
            "health_distribution",
            title,
            _card(_na(src.reason("inventory_health_distribution", {"run_id": run_id}))),
        )
    labels = [str(r["health"]) for r in rows]
    shares = [
        f" ({float(r['share_pct']):.0f}%)" if r.get("share_pct") is not None else "" for r in rows
    ]
    chart = hbar_chart(
        [lab + share for lab, share in zip(labels, shares, strict=True)],
        [float(r["n_series"] or 0) for r in rows],
        colours=[HEALTH_COLOURS.get(lab, GREY) for lab in labels],
        aria_label="Number of store \N{MULTIPLICATION SIGN} SKU series per inventory health class",
    )
    lead = (
        '<p class="muted">Store \N{MULTIPLICATION SIGN} SKU series by days-of-cover class '
        "for this run.</p>"
    )
    return _section("health_distribution", title, _card(lead + chart))


def _order_cost_section(src: _Source, run_id: int) -> str:
    title = "Order cost by category"
    rows = src.get("order_cost_by_category", {"run_id": run_id})
    if not rows:
        return _section(
            "order_cost",
            title,
            _card(_na(src.reason("order_cost_by_category", {"run_id": run_id}))),
        )
    chart = bar_chart(
        [str(r["category"]) for r in rows],
        [float(r.get("order_cost") or 0.0) for r in rows],
        aria_label="Replenishment order cost per product category",
    )
    table = _table(
        rows, ["category", "n_orders", "units", "order_cost", "n_deferred"], caption=title
    )
    return _section("order_cost", title, _card(chart + table, scroll_label=f"{title} table"))


def _run_config_section(run: Mapping[str, Any], config: Mapping[str, Any] | None) -> str:
    title = "Run configuration"
    if config is None:
        return _section(
            "run_config",
            title,
            _card(_na("forecast_runs.config_json is empty for this run (made before schema v2)")),
        )
    rows: list[Row] = []
    for key in ("engine_version", "interval_level", "status", "started_at", "finished_at"):
        if run.get(key) is not None:
            rows.append({"parameter": key, "value": run[key]})
    rows.extend(
        {"parameter": k, "value": v if isinstance(v, str) else json.dumps(v)}
        for k, v in config.items()
    )
    fmt: dict[str, Formatter] = {"value": lambda v: html.escape(str(v))}
    table = _table(rows, ["parameter", "value"], limit=len(rows), caption=title, fmt=fmt)
    lead = '<p class="muted">Persisted <code>RunConfig</code> — the run is reproducible from these values.</p>'
    return _section("run_config", title, _card(lead + table, scroll_label=f"{title} table"))


# ---------------------------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------------------------
CSS = """
:root{--red:#CC0000;--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--bg:#fafafa;--card:#fff;--row:#f3f4f6;--shade:#fde68a}
@media (prefers-color-scheme: dark){:root{--ink:#e5e7eb;--muted:#9ca3af;--line:#374151;--bg:#0f172a;--card:#1f2937;--row:#273449;--shade:#854d0e}}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);margin:0;background:var(--bg)}
header{background:var(--card);border-bottom:3px solid var(--red);padding:18px 32px;display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}
h1{margin:0;font-size:22px}h1 span{color:var(--red)}
h2{font-size:16px;margin:28px 0 8px;padding-bottom:4px;border-bottom:2px solid var(--red);display:inline-block}
h3{font-size:12px;margin:14px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
main{max-width:1100px;margin:0 auto;padding:16px 32px 48px}.muted{color:var(--muted)}
.na{color:var(--muted);font-style:italic;margin:8px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}.kpi-v{font-size:26px;font-weight:700}.kpi-l{font-weight:600}.kpi-s{color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px}
.scroll{overflow-x:auto}.scroll:focus{outline:2px solid var(--red);outline-offset:2px}
table{border-collapse:collapse;width:100%;font-size:13px}
caption.vh,.vh{position:absolute;width:1px;height:1px;margin:-1px;padding:0;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0}
th{text-align:left;color:var(--muted);font-weight:600;border-bottom:1px solid var(--line);padding:6px 8px;white-space:nowrap}
td{padding:6px 8px;border-bottom:1px solid var(--row);white-space:nowrap}td.num{font-variant-numeric:tabular-nums}
code{color:var(--ink);font-size:12px;background:var(--row);padding:1px 4px;border-radius:3px}
.chart{width:100%;height:auto;display:block}.chart .tick{font-size:10px;fill:var(--muted)}.chart .grid{stroke:var(--line)}.chart .shade{fill:var(--shade)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}@media(max-width:800px){.grid2{grid-template-columns:1fr}}
footer{color:var(--muted);font-size:12px;margin-top:32px}
@media print{body{background:#fff;color:#000}header{border-bottom-color:#000}.card,.kpi,section{break-inside:avoid}.card{border-color:#999}.scroll{overflow:visible}h2{break-after:avoid}.grid2{grid-template-columns:1fr 1fr}footer{margin-top:16px}}
"""


def render(
    conn: sqlite3.Connection,
    out_path: str | Path,
    run_id: int | None = None,
    sections: Sequence[str] | None = None,
) -> Path:
    """Render the dashboard for ``run_id`` (default: latest successful run) to ``out_path``.

    ``sections`` restricts the page to a subset of :data:`SECTIONS` (page order is fixed; the
    header and footer are always present; unknown names raise ``ValueError``). Sections whose
    query is missing, fails, or returns no rows show a muted "Not available in this run" note,
    so a v0.3 database still renders.
    """
    run_id = run_id or latest_successful_run(conn)
    if run_id is None:
        raise RuntimeError("no successful forecast run found — run the pipeline first")
    if sections is None:
        wanted = set(SECTIONS)
    else:
        unknown = sorted(set(sections) - set(SECTIONS))
        if unknown:
            raise ValueError(
                f"unknown dashboard section(s) {unknown}; valid names: {list(SECTIONS)}"
            )
        wanted = set(sections)
    row = conn.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"forecast run #{run_id} does not exist")
    run: Row = dict(row)
    src = _Source(conn)
    level = _interval_level(run)
    config = _run_config(run)
    p = {"run_id": run_id}
    parts: dict[str, str] = {}

    if "kpis" in wanted:
        parts["kpis"] = _kpi_section(conn, src, run, run_id, config)
    if "featured" in wanted:
        parts["featured"] = _featured_section(conn, src, run, run_id, level)
    if "leaderboard" in wanted:
        rows = src.get("forecast_accuracy_leaderboard", p)
        parts["leaderboard"] = _table_section(
            "leaderboard", "Model leaderboard", rows, src.reason("forecast_accuracy_leaderboard", p)
        )
    if "categories" in wanted:
        parts["categories"] = _categories_section(src)
    if "accuracy" in wanted:
        parts["accuracy"] = _accuracy_section(src, run_id, level)
    if "health_distribution" in wanted:
        parts["health_distribution"] = _health_distribution_section(src, run_id)
    if "stockouts" in wanted:
        rows = src.get("stockout_rate_by_store")
        parts["stockouts"] = _table_section(
            "stockouts",
            "Stock-out rate by store (28d)",
            rows,
            src.reason("stockout_rate_by_store"),
            limit=12,
        )
    if "health" in wanted:
        rows = src.get("days_of_cover", p)
        parts["health"] = _table_section(
            "health",
            "Inventory health — lowest days of cover",
            rows,
            src.reason("days_of_cover", p),
            limit=12,
        )
    if "risk" in wanted:
        rows = src.get("stockout_risk_top", p)
        parts["risk"] = _table_section(
            "risk",
            "Stock-out risk — top 10",
            rows,
            src.reason("stockout_risk_top", p),
            limit=10,
            fmt={"stockout_risk": _pct},
            lead='<p class="muted">P(demand over lead time + review period exceeds the inventory position); '
            "priority = cost-weighted expected shortfall.</p>",
        )
    if "orders" in wanted:
        rows = src.get("replenishment_summary", p)
        columns = list(ORDER_COLUMNS)
        if rows:
            extras = [c for c in ORDER_EXTRA_COLUMNS if c in rows[0]]
            columns[columns.index("order_cost") + 1 : columns.index("order_cost") + 1] = extras
        parts["orders"] = _table_section(
            "orders",
            "Replenishment order book (top by cost)",
            rows,
            src.reason("replenishment_summary", p),
            columns=columns,
            fmt={"stockout_risk": _pct},
        )
    if "order_cost" in wanted:
        parts["order_cost"] = _order_cost_section(src, run_id)
    if "rollup" in wanted:
        rows = src.get("forecast_rollup", p)
        parts["rollup"] = _table_section(
            "rollup",
            "Bottom-up forecast rollup",
            rows,
            src.reason("forecast_rollup", p),
            limit=30,
            lead=(
                '<p class="muted">Horizon units summed from the store \N{MULTIPLICATION SIGN} '
                "SKU forecasts.</p>"
            ),
        )
    if "abc" in wanted:
        rows = src.get("abc_classification")
        parts["abc"] = _table_section(
            "abc",
            "ABC classification (90-day revenue)",
            rows,
            src.reason("abc_classification"),
            columns=["sku", "category", "revenue_90d", "cum_revenue_share", "abc_class"],
            limit=12,
        )
    if "promo_lift" in wanted:
        rows = src.get("promo_lift")
        parts["promo_lift"] = _table_section(
            "promo_lift", "Promotion lift", rows, src.reason("promo_lift"), limit=10
        )
    if "promo_upcoming" in wanted:
        rows = src.get("promo_calendar_upcoming", p)
        parts["promo_upcoming"] = _table_section(
            "promo_upcoming",
            "Upcoming promotions",
            rows,
            src.reason("promo_calendar_upcoming", p),
            limit=12,
            lead='<p class="muted">Promotions overlapping the forecast horizon (store "ALL" = chain-wide).</p>',
        )
    if "run_config" in wanted:
        parts["run_config"] = _run_config_section(run, config)

    body = "\n".join(
        b
        for b in (
            parts.get("kpis", ""),
            parts.get("featured", ""),
            _grid([parts.get("leaderboard", ""), parts.get("categories", "")]),
            parts.get("accuracy", ""),
            _grid([parts.get("health_distribution", ""), parts.get("stockouts", "")]),
            parts.get("health", ""),
            parts.get("risk", ""),
            parts.get("orders", ""),
            _grid([parts.get("order_cost", ""), parts.get("rollup", "")]),
            _grid([parts.get("abc", ""), parts.get("promo_lift", "")]),
            _grid([parts.get("promo_upcoming", ""), parts.get("run_config", "")]),
        )
        if b
    )
    engine = (
        f" · engine {html.escape(str(run['engine_version']))}" if run.get("engine_version") else ""
    )
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="{CSP_META}">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="data:,">
<title>DemandCast — run #{run_id}</title><style>{CSS}</style></head><body>
<header><h1>Demand<span>Cast</span> <small class="muted">store-level forecasting &amp; replenishment</small></h1>
<div class="muted">run #{run_id} · cutoff {html.escape(str(run.get("cutoff_day")))} · horizon {run.get("horizon_days")}d · {run.get("series_count")} series{engine}</div></header>
<main>
{body}
<footer>Generated by DemandCast · Python + SQLite · every table on this page is a single SQL query in <code>demandcast/sql/analytics.sql</code>.</footer>
</main></body></html>
"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
