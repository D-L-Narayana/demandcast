"""Render a self-contained HTML dashboard (inline SVG, no JS frameworks) from SQL results."""

from __future__ import annotations

import html
import sqlite3
from pathlib import Path

from .db import run_query
from .pipeline import latest_successful_run

RED = "#CC0000"
GREY = "#6b7280"


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return html.escape(str(v))


def _table(rows: list[dict], columns: list[str] | None = None, limit: int = 15) -> str:
    if not rows:
        return "<p class='muted'>No rows.</p>"
    columns = columns or list(rows[0].keys())
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(r.get(c))}</td>" for c in columns) + "</tr>"
        for r in rows[:limit]
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _line_chart(
    series: list[tuple[str, list[float], str]], width=880, height=260, band: tuple | None = None
) -> str:
    """series: [(label, values, colour)], all same length. band: (lower, upper) arrays."""
    if not series or not series[0][1]:
        return ""
    n = len(series[0][1])
    all_vals = [v for _, vals, _ in series for v in vals if v is not None]
    if band:
        all_vals += [v for v in list(band[0]) + list(band[1]) if v is not None]
    lo, hi = 0.0, max(all_vals) * 1.1 or 1.0
    pad_l, pad_b, pad_t = 40, 24, 10
    w = width - pad_l - 10
    h = height - pad_b - pad_t

    def x(i):
        return pad_l + i * w / max(n - 1, 1)

    def y(v):
        return pad_t + h - (v - lo) / (hi - lo) * h

    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart' role='img'>"]
    for k in range(5):
        gy = pad_t + k * h / 4
        val = hi - k * (hi - lo) / 4
        parts.append(
            f"<line x1='{pad_l}' y1='{gy:.1f}' x2='{width - 10}' y2='{gy:.1f}' stroke='#e5e7eb'/>"
            f"<text x='{pad_l - 6}' y='{gy + 4:.1f}' text-anchor='end' class='tick'>{val:.0f}</text>"
        )
    if band:
        lower, upper = band
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(upper))
        pts += " " + " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in reversed(list(enumerate(lower))))
        parts.append(f"<polygon points='{pts}' fill='{RED}' fill-opacity='0.12' stroke='none'/>")
    for label, vals, colour in series:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals) if v is not None)
        parts.append(
            f"<polyline points='{pts}' fill='none' stroke='{colour}' stroke-width='2'>"
            f"<title>{html.escape(label)}</title></polyline>"
        )
    parts.append("</svg>")
    legend = " ".join(
        f"<span class='lg'><i style='background:{c}'></i>{html.escape(lab)}</span>"
        for lab, _, c in series
    )
    return f"<div class='legend'>{legend}</div>" + "".join(parts)


def _bar_chart(labels: list[str], values: list[float], width=880, height=220) -> str:
    if not values:
        return ""
    hi = max(values) * 1.15 or 1.0
    pad_l, pad_b, pad_t = 40, 40, 10
    w = width - pad_l - 10
    h = height - pad_b - pad_t
    bw = w / len(values) * 0.7
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart' role='img'>"]
    for i, (lab, v) in enumerate(zip(labels, values, strict=True)):
        bx = pad_l + i * w / len(values) + (w / len(values) - bw) / 2
        bh = v / hi * h
        parts.append(
            f"<rect x='{bx:.1f}' y='{pad_t + h - bh:.1f}' width='{bw:.1f}' height='{bh:.1f}' "
            f"fill='{RED}' rx='2'><title>{html.escape(lab)}: {v:,.1f}</title></rect>"
            f"<text x='{bx + bw / 2:.1f}' y='{pad_t + h + 16}' text-anchor='middle' class='tick'>"
            f"{html.escape(lab[:14])}</text>"
            f"<text x='{bx + bw / 2:.1f}' y='{pad_t + h - bh - 4:.1f}' text-anchor='middle' "
            f"class='tick'>{v:,.0f}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _kpi(label: str, value: str, sub: str = "") -> str:
    return (
        f"<div class='kpi'><div class='kpi-v'>{value}</div><div class='kpi-l'>{html.escape(label)}"
        f"</div><div class='kpi-s'>{html.escape(sub)}</div></div>"
    )


def render(conn: sqlite3.Connection, out_path: str | Path, run_id: int | None = None) -> Path:
    run_id = run_id or latest_successful_run(conn)
    if run_id is None:
        raise RuntimeError("no successful forecast run found — run the pipeline first")
    run = dict(conn.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone())

    leaderboard = run_query(conn, "forecast_accuracy_leaderboard", {"run_id": run_id})
    orders = run_query(conn, "replenishment_summary", {"run_id": run_id})
    cover = run_query(conn, "days_of_cover", {"run_id": run_id})
    abc = run_query(conn, "abc_classification")
    stockouts = run_query(conn, "stockout_rate_by_store")
    promo = run_query(conn, "promo_lift")
    weekly = run_query(conn, "weekly_sales_trend")

    total_orders = sum(1 for o in orders if o["order_qty"] > 0)
    order_cost = sum(o["order_cost"] for o in orders if o["order_qty"] > 0)
    critical = sum(1 for c in cover if c["health"] == "CRITICAL")
    avg_wape = sum(
        r["avg_wape"] * r["series_won"] for r in leaderboard if r["avg_wape"] is not None
    ) / max(sum(r["series_won"] for r in leaderboard), 1)

    # Featured series: the highest-revenue A item at the first store.
    top_pid = abc[0]["product_id"] if abc else 1
    first_store = conn.execute("SELECT MIN(store_id) AS s FROM stores").fetchone()["s"]
    hist = run_query(conn, "series_history", {"store_id": first_store, "product_id": top_pid})[-84:]
    fc = [
        dict(r)
        for r in conn.execute(
            "SELECT target_day, yhat, yhat_lower, yhat_upper, model_name FROM forecasts "
            "WHERE run_id=? AND store_id=? AND product_id=? ORDER BY target_day",
            (run_id, first_store, top_pid),
        )
    ]
    hist_vals = [float(r["units_sold"]) for r in hist]
    fc_vals = [None] * (len(hist) - 1) + [hist_vals[-1]] + [r["yhat"] for r in fc]
    band_lo = [hist_vals[-1]] * len(hist) + [r["yhat_lower"] for r in fc]
    band_hi = [hist_vals[-1]] * len(hist) + [r["yhat_upper"] for r in fc]
    hist_padded = hist_vals + [None] * len(fc)
    featured_chart = _line_chart(
        [("Actual units", hist_padded, "#111827"), ("Forecast", fc_vals, RED)],
        band=(band_lo, band_hi),
    )
    model_used = fc[0]["model_name"] if fc else "n/a"

    cats: dict[str, float] = {}
    for w in weekly[-6 * 8 :]:
        cats[w["category"]] = cats.get(w["category"], 0) + (w["units"] or 0)
    cat_chart = _bar_chart(list(cats), list(cats.values()))

    css = """
    :root{--red:#CC0000;--ink:#111827;--muted:#6b7280;--line:#e5e7eb}
    *{box-sizing:border-box}body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);margin:0;background:#fafafa}
    header{background:#fff;border-bottom:3px solid var(--red);padding:18px 32px;display:flex;justify-content:space-between;align-items:baseline}
    h1{margin:0;font-size:22px}h1 span{color:var(--red)}h2{font-size:16px;margin:28px 0 8px;padding-bottom:4px;border-bottom:2px solid var(--red);display:inline-block}
    main{max-width:1100px;margin:0 auto;padding:16px 32px 48px}.muted{color:var(--muted)}
    .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:16px}
    .kpi{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 16px}.kpi-v{font-size:26px;font-weight:700}.kpi-l{font-weight:600}.kpi-s{color:var(--muted);font-size:12px}
    .card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 16px;overflow-x:auto}
    table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;color:var(--muted);font-weight:600;border-bottom:1px solid var(--line);padding:6px 8px;white-space:nowrap}td{padding:6px 8px;border-bottom:1px solid #f3f4f6;white-space:nowrap}
    .chart{width:100%;height:auto}.tick{font-size:10px;fill:var(--muted)}.legend{font-size:12px;color:var(--muted);margin-bottom:4px}.lg i{display:inline-block;width:10px;height:10px;margin:0 4px 0 10px;border-radius:2px;vertical-align:middle}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:800px){.grid2{grid-template-columns:1fr}}
    footer{color:var(--muted);font-size:12px;margin-top:32px}
    """
    doc = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>DemandCast — run #{run_id}</title><style>{css}</style></head><body>
<header><h1>Demand<span>Cast</span> <small class='muted'>store-level forecasting &amp; replenishment</small></h1>
<div class='muted'>run #{run_id} · cutoff {html.escape(str(run["cutoff_day"]))} · horizon {run["horizon_days"]}d · {run["series_count"]} series</div></header>
<main>
<div class='kpis'>
{_kpi("Weighted WAPE (backtest)", f"{avg_wape * 100:.1f}%", "lower is better; rolling-origin, 4 folds")}
{_kpi("Replenishment orders", f"{total_orders}", f"₹{order_cost:,.0f} order cost")}
{_kpi("Critical cover (<0.5 wk)", f"{critical}", "store × SKU below half of 7-day demand")}
{_kpi("Series forecast", f"{run['series_count']}", html.escape(run["notes"] or ""))}
</div>

<h2>Featured series — {html.escape(abc[0]["sku"] if abc else "")} @ store {first_store}</h2>
<div class='card'><p class='muted'>Last 84 days of actuals, {run["horizon_days"]}-day forecast from <b>{html.escape(model_used)}</b> with 80% interval.</p>{featured_chart}</div>

<div class='grid2'>
<div><h2>Model leaderboard</h2><div class='card'>{_table(leaderboard)}</div></div>
<div><h2>Units by category (last 8 weeks)</h2><div class='card'>{cat_chart}</div></div>
</div>

<h2>Inventory health — lowest days of cover</h2>
<div class='card'>{_table(cover, limit=12)}</div>

<h2>Replenishment order book (top by cost)</h2>
<div class='card'>{_table(orders, ["store_code", "sku", "category", "on_hand", "on_order", "lead_time_demand", "safety_stock", "reorder_point", "order_qty", "order_cost", "expected_arrival", "reason"], limit=15)}</div>

<div class='grid2'>
<div><h2>ABC classification (90-day revenue)</h2><div class='card'>{_table(abc, ["sku", "category", "revenue_90d", "cum_revenue_share", "abc_class"], limit=12)}</div></div>
<div><h2>Stock-out rate by store (28d)</h2><div class='card'>{_table(stockouts, limit=12)}</div></div>
</div>

<h2>Promotion lift</h2>
<div class='card'>{_table(promo, limit=10)}</div>

<footer>Generated by DemandCast · Python + SQLite · every table on this page is a single SQL query in <code>demandcast/sql/analytics.sql</code>.</footer>
</main></body></html>"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
