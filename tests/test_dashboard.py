"""Dashboard v2 + SVG chart library tests.

Fixtures are local to this file: a 3x8x300 synthetic dataset with one pipeline run
(horizon 14, 2 folds, 1 worker) rendered once per module. Variants of that database are
produced with ``sqlite3.Connection.backup`` so no test mutates the shared fixture.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import date, timedelta
from html.parser import HTMLParser

import pytest

from demandcast import charts, db, pipeline
from demandcast.dashboard import CSP_META, NOT_AVAILABLE, SECTIONS, render
from demandcast.simulate import SimConfig, generate

CSP_LITERAL = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
    "base-uri 'none'; form-action 'none'"
)
BASELINE_QUERIES = {
    "weekly_sales_trend",
    "abc_classification",
    "stockout_rate_by_store",
    "promo_lift",
    "forecast_accuracy_leaderboard",
    "replenishment_summary",
    "series_history",
    "days_of_cover",
}
V2_SECTION_TITLES = {
    "run_config": "Run configuration",
    "accuracy": "Realised accuracy",
    "health_distribution": "Inventory health distribution",
    "risk": "Stock-out risk",
    "order_cost": "Order cost by category",
    "rollup": "Bottom-up forecast rollup",
    "promo_upcoming": "Upcoming promotions",
}

# Hand-written SQL returning exactly the C9 column shapes on a baseline schema (+ the additive
# schema-v2 columns applied by the ``v2_like_db`` fixture). Used to exercise the v2 sections
# with real data before the analytics v2 queries land.
_DAYS_OF_COVER = db.QUERIES["days_of_cover"].rstrip().rstrip(";")
V2_SQL = {
    "evaluation_history": """
        SELECT e.run_id, r.cutoff_day, r.horizon_days, COUNT(*) AS n_series,
               AVG(e.mae) AS mae,
               SUM(e.abs_error_sum) / NULLIF(SUM(e.actual_sum), 0) AS wape,
               AVG(e.bias) AS bias,
               SUM(e.coverage * e.n_days) / SUM(e.n_days) AS coverage,
               MAX(e.evaluated_at) AS evaluated_at
        FROM forecast_evaluations e JOIN forecast_runs r ON r.run_id = e.run_id
        GROUP BY e.run_id ORDER BY e.run_id""",
    "evaluation_summary": """
        SELECT model_name, COUNT(*) AS n_series, SUM(n_days) AS total_days,
               AVG(mae) AS avg_mae,
               SUM(abs_error_sum) / NULLIF(SUM(actual_sum), 0) AS wape,
               AVG(bias) AS avg_bias,
               SUM(coverage * n_days) / SUM(n_days) AS coverage
        FROM forecast_evaluations WHERE run_id = :run_id GROUP BY model_name""",
    "inventory_health_distribution": f"""
        WITH doc AS ({_DAYS_OF_COVER})
        SELECT health, COUNT(*) AS n_series,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS share_pct
        FROM doc GROUP BY health
        ORDER BY CASE health WHEN 'CRITICAL' THEN 0 WHEN 'LOW' THEN 1 WHEN 'OK' THEN 2
                             WHEN 'OVERSTOCK' THEN 3 ELSE 4 END""",
    "order_cost_by_category": """
        SELECT p.category, SUM(r.order_qty > 0) AS n_orders, SUM(r.order_qty) AS units,
               ROUND(SUM(r.order_qty * p.unit_cost), 2) AS order_cost,
               SUM(COALESCE(r.requested_qty, 0) > 0 AND r.order_qty = 0) AS n_deferred
        FROM replenishment_orders r JOIN products p ON p.product_id = r.product_id
        WHERE r.run_id = :run_id GROUP BY p.category ORDER BY order_cost DESC""",
    "forecast_rollup": """
        SELECT 'chain' AS "level", 'ALL' AS "key", ROUND(SUM(yhat), 1) AS horizon_units
        FROM forecasts WHERE run_id = :run_id
        UNION ALL
        SELECT 'region', st.region, ROUND(SUM(f.yhat), 1)
        FROM forecasts f JOIN stores st ON st.store_id = f.store_id
        WHERE f.run_id = :run_id GROUP BY st.region
        UNION ALL
        SELECT 'category', p.category, ROUND(SUM(f.yhat), 1)
        FROM forecasts f JOIN products p ON p.product_id = f.product_id
        WHERE f.run_id = :run_id GROUP BY p.category""",
    "stockout_risk_top": """
        SELECT st.store_code, p.sku, p.name, p.category, r.on_hand, r.on_order,
               r.stockout_risk, r.priority, r.order_qty, r.expected_arrival
        FROM replenishment_orders r
        JOIN stores st ON st.store_id = r.store_id
        JOIN products p ON p.product_id = r.product_id
        WHERE r.run_id = :run_id ORDER BY r.priority DESC, r.stockout_risk DESC LIMIT 10""",
    "promo_calendar_upcoming": """
        SELECT pr.promo_id, p.sku, COALESCE(st.store_code, 'ALL') AS store_code,
               pr.start_day, pr.end_day, pr.discount_pct
        FROM promotions pr
        JOIN products p ON p.product_id = pr.product_id
        LEFT JOIN stores st ON st.store_id = pr.store_id
        JOIN forecast_runs fr ON fr.run_id = :run_id
        WHERE pr.end_day > fr.cutoff_day
          AND pr.start_day <= DATE(fr.cutoff_day, '+' || fr.horizon_days || ' days')
        ORDER BY pr.start_day""",
    # baseline order book + the schema-v2 columns (C8: export rows of replenishment_summary carry
    # stockout_risk, priority, requested_qty) -- "columns may be added" per invariant 4
    "replenishment_summary": """
        SELECT r.order_id, st.store_code, st.city, p.sku, p.name, p.category,
               r.on_hand, r.on_order,
               ROUND(r.lead_time_demand, 1) AS lead_time_demand,
               ROUND(r.safety_stock, 1) AS safety_stock,
               ROUND(r.reorder_point, 1) AS reorder_point,
               r.order_qty, ROUND(r.order_qty * p.unit_cost, 2) AS order_cost,
               r.expected_arrival, r.reason, r.stockout_risk, r.priority, r.requested_qty
        FROM replenishment_orders r
        JOIN stores st ON st.store_id = r.store_id
        JOIN products p ON p.product_id = r.product_id
        WHERE r.run_id = :run_id ORDER BY order_cost DESC""",
    "series_history": """
        SELECT day, units_sold, stockout_flag,
               ROUND(AVG(units_sold) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS ma7,
               EXISTS (SELECT 1 FROM promotions pr
                       WHERE pr.product_id = s.product_id
                         AND (pr.store_id IS NULL OR pr.store_id = s.store_id)
                         AND s.day BETWEEN pr.start_day AND pr.end_day) AS on_promo
        FROM sales_daily s WHERE store_id = :store_id AND product_id = :product_id ORDER BY day""",
}
_EVALUATIONS_DDL = """
CREATE TABLE IF NOT EXISTS forecast_evaluations (
    run_id INTEGER NOT NULL REFERENCES forecast_runs(run_id),
    store_id INTEGER NOT NULL REFERENCES stores(store_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    model_name TEXT NOT NULL,
    n_days INTEGER NOT NULL CHECK (n_days > 0),
    mae REAL NOT NULL, wape REAL, bias REAL NOT NULL,
    coverage REAL NOT NULL CHECK (coverage BETWEEN 0 AND 1),
    abs_error_sum REAL NOT NULL, actual_sum REAL NOT NULL,
    stockout_days INTEGER NOT NULL DEFAULT 0,
    evaluated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, store_id, product_id)
) WITHOUT ROWID"""


# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dash_db() -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.init_schema(conn)
    generate(conn, SimConfig(n_stores=3, n_products=8, start=date(2024, 1, 1), days=300, seed=7))
    pipeline.run(conn, pipeline.RunConfig(horizon_days=14, n_folds=2, workers=1))
    return conn


@pytest.fixture(scope="module")
def dash_html(dash_db, tmp_path_factory) -> str:
    out = render(dash_db, tmp_path_factory.mktemp("dash") / "index.html")
    return out.read_text(encoding="utf-8")


def _copy_db(src: sqlite3.Connection) -> sqlite3.Connection:
    dst = db.connect(":memory:")
    src.backup(dst)
    return dst


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))


@pytest.fixture
def baseline_like_db(dash_db, monkeypatch) -> sqlite3.Connection:
    """Copy of the run DB with only the 8 baseline queries installed and no v2 run metadata."""
    conn = _copy_db(dash_db)
    for col in ("config_json", "interval_level"):
        if _has_column(conn, "forecast_runs", col):
            conn.execute(f"UPDATE forecast_runs SET {col} = NULL")
    conn.commit()
    monkeypatch.setattr(
        db, "QUERIES", {k: v for k, v in db.QUERIES.items() if k in BASELINE_QUERIES}
    )
    return conn


@pytest.fixture
def v2_like_db(dash_db, monkeypatch) -> tuple[sqlite3.Connection, int, dict]:
    """Copy of the run DB upgraded with the additive schema-v2 columns (C5 DDL), contract-shaped
    v2 data and the hand-written C9 queries above."""
    conn = _copy_db(dash_db)
    run_id = pipeline.latest_successful_run(conn)
    assert run_id is not None
    ddl = [
        ("forecast_runs", "config_json", "TEXT"),
        ("forecast_runs", "interval_level", "REAL"),
        ("forecast_runs", "engine_version", "TEXT"),
        ("forecasts", "promo_flag", "INTEGER NOT NULL DEFAULT 0"),
        ("replenishment_orders", "stockout_risk", "REAL"),
        ("replenishment_orders", "priority", "REAL"),
        ("replenishment_orders", "requested_qty", "INTEGER"),
    ]
    for table, col, typ in ddl:
        if not _has_column(conn, table, col):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    conn.execute(_EVALUATIONS_DDL)
    cfg = {
        "horizon_days": 14,
        "n_folds": 2,
        "min_history_days": 84,
        "review_period_days": 7,
        "service_level": 0.95,
        "interval_level": 0.9,
        "interval_method": "empirical",
        "workers": 1,
        "cutoff_day": None,
        "models": None,
        "promo_aware": True,
        "criterion": "mae",
        "order_budget": None,
        "service_level_overrides": None,
    }
    conn.execute(
        "UPDATE forecast_runs SET config_json=?, interval_level=0.9, engine_version='0.4.0' "
        "WHERE run_id=?",
        (json.dumps(cfg), run_id),
    )
    conn.execute(
        "UPDATE replenishment_orders SET stockout_risk = (order_id % 10) / 10.0, "
        "priority = order_id * 1.5 WHERE run_id=?",
        (run_id,),
    )
    conn.execute(
        "UPDATE replenishment_orders SET requested_qty = order_qty, order_qty = 0, "
        "reason = reason || '; deferred: order budget exhausted' "
        "WHERE run_id=? AND order_id = (SELECT MIN(order_id) FROM replenishment_orders "
        "WHERE run_id=? AND order_qty > 0)",
        (run_id, run_id),
    )
    conn.execute("DELETE FROM forecast_evaluations WHERE run_id=?", (run_id,))
    conn.execute(
        "INSERT INTO forecast_evaluations SELECT run_id, store_id, product_id, model_name, 14, "
        "1.5 + 0.1 * store_id, 0.31, -0.2, 0.85, 21.0, 70.0, 0, '2024-10-27T00:00:00+00:00' "
        "FROM (SELECT DISTINCT run_id, store_id, product_id, model_name FROM forecasts "
        "WHERE run_id=?)",
        (run_id,),
    )
    cutoff = date.fromisoformat(
        conn.execute("SELECT cutoff_day FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()[0]
    )
    pid = db.run_query(conn, "abc_classification")[0]["product_id"]
    sid = conn.execute("SELECT MIN(store_id) AS s FROM stores").fetchone()["s"]
    next_id = conn.execute("SELECT COALESCE(MAX(promo_id), 0) + 1 FROM promotions").fetchone()[0]
    recent = (cutoff - timedelta(days=9), cutoff - timedelta(days=2))
    future = (cutoff + timedelta(days=3), cutoff + timedelta(days=10))
    conn.executemany(
        "INSERT INTO promotions (promo_id, product_id, store_id, start_day, end_day, discount_pct) "
        "VALUES (?, ?, NULL, ?, ?, 0.2)",
        [
            (next_id, pid, recent[0].isoformat(), recent[1].isoformat()),
            (next_id + 1, pid, future[0].isoformat(), future[1].isoformat()),
        ],
    )
    conn.execute(
        "UPDATE forecasts SET promo_flag = 1 WHERE run_id=? AND store_id=? AND product_id=? "
        "AND target_day BETWEEN ? AND ?",
        (run_id, sid, pid, future[0].isoformat(), future[1].isoformat()),
    )
    conn.commit()
    patched = dict(db.QUERIES)
    patched.update(V2_SQL)
    monkeypatch.setattr(db, "QUERIES", patched)
    meta = {"cutoff": cutoff, "product_id": pid, "store_id": sid, "future_promo": future}
    return conn, run_id, meta


def _section(text: str, name: str) -> str:
    m = re.search(rf'<section id="sec-{name}".*?</section>', text, re.S)
    assert m is not None, f"section {name!r} not found"
    return m.group(0)


def _svg_text_nodes(svg: str) -> str:
    return " | ".join(re.findall(r"<(?:text|title)[^>]*>([^<]*)<", svg))


def _first_svg(fragment: str) -> str:
    return fragment[fragment.index("<svg") : fragment.index("</svg>")]


# --------------------------------------------------------------------------------------------
# chart helpers (C11)
# --------------------------------------------------------------------------------------------
def test_chart_helpers_escape_text():
    line = charts.line_chart([("a<b&c", [1.0, 2.0], "#000")], aria_label='Q "x" & <y>')
    assert "a&lt;b&amp;c" in line
    assert "a<b&c" not in line
    assert 'aria-label="Q &quot;x&quot; &amp; &lt;y&gt;"' in line
    bar = charts.bar_chart(["<b>&", "ok"], [1.0, 2.0])
    assert "&lt;b&gt;&amp;" in bar
    assert "<b>&" not in bar
    hbar = charts.hbar_chart(["x<y", "z&w"], [3.0, 4.0])
    assert "x&lt;y" in hbar
    assert "z&amp;w" in hbar
    assert "x<y" not in hbar
    stacked = charts.stacked_bar(["<g>"], {"s&t": [1.0]}, colours=["#111"])
    assert "&lt;g&gt;" in stacked
    assert "s&amp;t" in stacked
    assert "<g>" not in stacked
    xl = charts.line_chart([("s", [1.0, 2.0], "#000")], x_labels=["<d1>", "d&2"])
    assert "&lt;d1&gt;" in xl
    assert "d&amp;2" in xl


def test_line_chart_none_gaps_split_polylines():
    out = charts.line_chart([("gappy", [1.0, 2.0, None, 3.0, 4.0], "#000")])
    assert out.count("<polyline") == 2
    # an isolated point between gaps is still drawn (as a marker) rather than dropped
    single = charts.line_chart([("dots", [None, 5.0, None, 1.0, 2.0], "#000")])
    assert single.count("<polyline") == 1
    assert "<circle" in single
    # a series that is entirely None draws nothing but the chart still renders
    none_only = charts.line_chart([("empty", [None, None], "#000"), ("v", [1.0, 2.0], "#111")])
    assert none_only.count("<polyline") == 1


def test_line_chart_thins_x_labels_to_at_most_8_ticks():
    n = 112
    days = [(date(2024, 7, 1) + timedelta(days=i)).isoformat() for i in range(n)]
    vals = [float(i % 7) for i in range(n)]
    out = charts.line_chart([("units", vals, "#000")], x_labels=days)
    ticks = re.findall(r'class="tick xt"[^>]*>([^<]*)<', out)
    assert 2 <= len(ticks) <= 8, ticks
    assert ticks[0] == days[0]
    assert ticks[-1] == days[-1]
    assert all(t in days for t in ticks)
    for n_pts in (1, 2, 7, 8, 9, 15, 16, 17, 100, 365):
        idx = charts.tick_indices(n_pts)
        assert 1 <= len(idx) <= 8
        assert idx == sorted(set(idx))
        assert idx[-1] == n_pts - 1


def test_line_chart_shade_spans_and_band():
    shade = [False, False, True, True, True, False, True, False]
    vals = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0]
    out = charts.line_chart([("s", vals, "#000")], shade=shade)
    assert out.count('class="shade"') == 2
    lower = [0.5, 1.5, None, 1.5, 0.5, 1.5, 2.5, 3.5]
    upper = [1.5, 2.5, None, 2.5, 1.5, 2.5, 3.5, 4.5]
    banded = charts.line_chart([("s", vals, "#000")], band=(lower, upper))
    assert banded.count('class="band"') == 2  # the None breaks the band into two polygons
    assert charts.line_chart([]) == ""
    assert charts.line_chart([("s", [], "#000")]) == ""
    assert charts.line_chart([("s", [None, None], "#000")]) != ""  # renders an empty frame


def test_bar_and_hbar_charts():
    out = charts.bar_chart(["Grocery", "Toys"], [120.0, 30.5], colour="#123456")
    assert out.count("<rect") == 2
    assert out.count('fill="#123456"') == 2
    assert ">Grocery<" in out
    assert ">Toys<" in out
    assert "120" in out
    assert charts.bar_chart([], []) == ""
    with pytest.raises(ValueError, match="same length"):
        charts.bar_chart(["a"], [1.0, 2.0])
    hb = charts.hbar_chart(["CRITICAL", "OK"], [3.0, 9.0], colours=["#CC0000", "#059669"])
    assert hb.count("<rect") == 2
    assert hb.index('fill="#CC0000"') < hb.index('fill="#059669"')
    assert ">CRITICAL<" in hb
    assert ">OK<" in hb
    assert charts.hbar_chart([], []) == ""
    by_label = charts.hbar_chart(["b", "a"], [1.0, 2.0], colours={"a": "#000001", "b": "#000002"})
    assert by_label.index('fill="#000002"') < by_label.index('fill="#000001"')


def test_stacked_bar_segments_sum_to_totals():
    labels = ["A", "B", "C"]
    series = {"placed": [10.0, 20.0, 0.0], "deferred": [30.0, 0.0, 5.0]}
    out = charts.stacked_bar(labels, series, colours=["#CC0000", "#6b7280"])
    rects = re.findall(
        r'<rect class="stack" x="([\d.]+)" y="[\d.]+" width="[\d.]+" height="([\d.]+)"', out
    )
    assert len(rects) == 6
    heights: dict[str, float] = {}
    for x, h in rects:
        heights[x] = heights.get(x, 0.0) + float(h)
    totals = [40.0, 20.0, 5.0]
    stacks = [heights[x] for x in sorted(heights, key=float)]
    assert len(stacks) == 3
    ratios = [h / t for h, t in zip(stacks, totals, strict=True)]
    assert all(r == pytest.approx(ratios[0], rel=0.02) for r in ratios), ratios
    assert ">placed<" in out
    assert ">deferred<" in out
    assert charts.stacked_bar([], {"s": []}, colours=None) == ""
    with pytest.raises(ValueError, match="must have 1 values"):
        charts.stacked_bar(["A"], {"s": [1.0, 2.0]}, colours=None)


def test_charts_are_accessible_and_deterministic():
    outs = [
        charts.line_chart([("Actual", [1.0, 2.0], "#000"), ("Forecast", [None, 2.5], "#CC0000")]),
        charts.bar_chart(["a"], [1.0]),
        charts.hbar_chart(["a"], [1.0]),
        charts.stacked_bar(["a"], {"s": [1.0]}, colours=None),
    ]
    for out in outs:
        head = re.match(r"<svg\b[^>]*>", out)
        assert head is not None, out[:80]
        assert 'role="img"' in head.group(0)
        label = re.search(r'aria-label="([^"]*)"', head.group(0))
        assert label is not None
        assert len(label.group(1)) >= 3
        assert out.endswith("</svg>")
        assert "<script" not in out
    assert "Actual" in outs[0]
    assert "Forecast" in outs[0]
    assert charts.bar_chart(["a", "b"], [1.0, 2.0]) == charts.bar_chart(["a", "b"], [1.0, 2.0])


# --------------------------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------------------------
def test_dashboard_all_categories_in_chart(dash_db, dash_html):
    """D1 regression: the category chart aggregates the last 8 weeks for EVERY category."""
    categories = [r[0] for r in dash_db.execute("SELECT DISTINCT category FROM products")]
    assert len(categories) >= 2
    pos = dash_html.index("Units by category")
    svg = _first_svg(dash_html[pos:])
    nodes = _svg_text_nodes(svg)
    missing = [c for c in categories if c not in nodes]
    assert not missing, f"categories missing from the chart: {missing}; nodes: {nodes}"
    # per-category sums over the last 8 week_start values, not the last 48 rows overall
    weekly = db.run_query(dash_db, "weekly_sales_trend")
    by_cat: dict[str, list[tuple[str, float]]] = {}
    for w in weekly:
        by_cat.setdefault(w["category"], []).append((w["week_start"], float(w["units"] or 0)))
    for cat, rows in by_cat.items():
        expected = sum(u for _, u in sorted(rows)[-8:])
        assert f"{cat}: {expected:,.1f}" in svg, (cat, expected)


def test_dashboard_has_no_script(dash_html):
    assert "<script" not in dash_html.lower()
    assert re.search(r"<[^>]*\son[a-z]+\s*=", dash_html, re.I) is None


def test_dashboard_csp_compliant(dash_html):
    """C12: the document itself is clean under the production Content-Security-Policy."""
    text = dash_html
    assert CSP_META == CSP_LITERAL
    assert re.search(
        r'<meta charset="utf-8">\s*<meta http-equiv="Content-Security-Policy" content="'
        + re.escape(CSP_META)
        + r'">',
        text,
    ), "CSP meta tag must directly follow <meta charset>"
    assert text.count('<meta http-equiv="Content-Security-Policy"') == 1
    assert '<link rel="icon" href="data:,">' in text
    assert "<script" not in text.lower()
    assert re.search(r"<[^>]*\son[a-z]+\s*=", text, re.I) is None
    assert "javascript:" not in text.lower()
    for m in re.finditer(r"url\(\s*['\"]?([^'\")\s]*)", text, re.I):
        assert m.group(1).startswith("data:"), m.group(0)
    assert "@font-face" not in text.lower()
    assert re.search(r"<base[\s>]", text, re.I) is None
    assert (
        re.search(
            r"<(link|img|iframe|object|embed|source|video|audio)\b[^>]*\b(src|href)\s*=\s*[\"']?\s*https?://",
            text,
            re.I,
        )
        is None
    )
    links = re.findall(r"<link\b[^>]*>", text)
    assert links == ['<link rel="icon" href="data:,">']
    assert "<style>" in text  # everything inline
    assert 'rel="stylesheet"' not in text
    assert re.search(r"<(img|iframe|object|embed|video|audio)\b", text, re.I) is None


def test_dashboard_balanced_tags(dash_html):
    for tag in ("table", "div", "svg", "section", "main", "header", "footer"):
        opened = len(re.findall(rf"<{tag}[\s>]", dash_html))
        closed = dash_html.count(f"</{tag}>")
        assert opened == closed, (tag, opened, closed)
        assert opened > 0, tag


def test_dashboard_accessibility_markup(dash_html):
    text = dash_html
    assert '<html lang="en"' in text
    assert '<meta name="viewport"' in text
    assert text.count("<table") == text.count("<caption") > 0
    assert len(re.findall(r"<th\b", text)) == len(re.findall(r'<th scope="col"', text)) > 0
    # axe svg-img-alt: every role="img" SVG carries a meaningful aria-label
    svgs = re.findall(r"<svg\b[^>]*>", text)
    assert len(svgs) >= 2
    for tag in svgs:
        assert 'role="img"' in tag, tag
        label = re.search(r'aria-label="([^"]*)"', tag)
        assert label is not None, tag
        assert len(label.group(1)) >= 3, tag
    # axe scrollable-region-focusable: horizontally scrollable table wrappers are keyboard
    # reachable; chart-only cards are not scroll containers at all
    style = re.search(r"<style>(.*?)</style>", text, re.S)
    assert style is not None
    css = style.group(1)
    assert re.search(r"\.card\{[^}]*overflow", css) is None
    assert re.search(r"\.scroll\{[^}]*overflow-x:auto", css)
    chunks = re.split(r'(?=<div class="card)', text)[1:]
    assert chunks
    seen_table = False
    for chunk in chunks:
        opening = re.match(r'<div class="card[^"]*"[^>]*>', chunk)
        assert opening is not None, chunk[:80]
        tag = opening.group(0)
        if "<table" in chunk:
            seen_table = True
            assert "scroll" in tag, tag
            assert 'tabindex="0"' in tag, tag
            assert 'role="region"' in tag, tag
            assert re.search(r'aria-label="[^"]{3,}"', tag), tag
        else:
            assert "scroll" not in tag, tag
            assert "tabindex" not in tag, tag
    assert seen_table
    # dark mode + print styles via CSS variables
    assert "@media (prefers-color-scheme: dark)" in css
    assert "@media print" in css
    assert "--red:#CC0000" in css
    assert "var(--ink)" in css


def _css_hex_vars(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{3,6})", block))


def _rel_luminance(colour: str) -> float:
    """WCAG 2.x relative luminance: sRGB channels -> linear, then 0.2126 R + 0.7152 G + 0.0722 B."""
    h = colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    linear = []
    for i in (0, 2, 4):
        c = int(h[i : i + 2], 16) / 255
        linear.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = linear
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(fg: str, bg: str) -> float:
    hi, lo = sorted((_rel_luminance(fg), _rel_luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _resolve_var(value: str, variables: dict[str, str]) -> str:
    m = re.fullmatch(r"var\(--([a-z-]+)\)", value.strip())
    return variables[m.group(1)] if m else value.strip()


def _colour_schemes(css: str) -> dict[str, dict[str, str]]:
    light_block = re.search(r":root\{([^}]*)\}", css)
    dark_block = re.search(r"@media \(prefers-color-scheme: dark\)\{:root\{([^}]*)\}", css)
    assert light_block is not None
    assert dark_block is not None
    light = _css_hex_vars(light_block.group(1))
    return {"light": light, "dark": {**light, **_css_hex_vars(dark_block.group(1))}}


def test_dashboard_css_colour_contrast(dash_html):
    """WCAG AA (>= 4.5:1) for muted/ink text on every surface it sits on, in both colour schemes.

    The footer and lead paragraphs are muted text; ``<code>`` inside them gets the ``--row``
    background, so the code foreground (explicit ``color`` on the ``code`` rule, otherwise the
    inherited muted colour) must pass against the code background too.
    """
    style = re.search(r"<style>(.*?)</style>", dash_html, re.S)
    assert style is not None
    css = style.group(1)
    code_rule = re.search(r"(?<![\w.-])code\{([^}]*)\}", css)
    assert code_rule is not None
    decls: dict[str, str] = {}
    for decl in code_rule.group(1).split(";"):
        if ":" in decl:
            key, _, val = decl.partition(":")
            decls[key.strip()] = val.strip()
    schemes = _colour_schemes(css)
    assert schemes["light"]["red"] == "#CC0000"
    assert schemes["dark"]["red"] == "#CC0000"
    for scheme, v in schemes.items():
        for name in ("ink", "muted", "bg", "card", "row"):
            assert name in v, (scheme, name)
        code_fg = _resolve_var(decls["color"], v) if "color" in decls else v["muted"]
        code_bg = _resolve_var(decls.get("background", decls.get("background-color", "")), v)
        pairs = {
            "muted on bg": (v["muted"], v["bg"]),
            "muted on card": (v["muted"], v["card"]),
            "ink on row": (v["ink"], v["row"]),
            "ink on card": (v["ink"], v["card"]),
            "code fg on code bg": (code_fg, code_bg or v["card"]),
        }
        for label, (fg, bg) in pairs.items():
            ratio = _contrast(fg, bg)
            assert ratio >= 4.5, f"{scheme}: {label} {fg} on {bg} = {ratio:.2f}:1 (< 4.5:1)"


def _split_media(css: str) -> tuple[str, dict[str, str]]:
    """Split a stylesheet into (base rules, {media query: inner rules}) by brace depth."""
    base: list[str] = []
    media: dict[str, str] = {}
    i, n = 0, len(css)
    while i < n:
        at = css.find("@media", i)
        if at == -1:
            base.append(css[i:])
            break
        base.append(css[i:at])
        open_brace = css.index("{", at)
        query = re.sub(r"\s+", "", css[at + len("@media") : open_brace])
        depth, j = 1, open_brace + 1
        while j < n and depth:
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            j += 1
        media[query] = media.get(query, "") + css[open_brace + 1 : j - 1]
        i = j
    return "".join(base), media


def _css_rules(css: str) -> list[tuple[list[str], dict[str, str]]]:
    """``[(selectors, {property: value})]`` of a media-free stylesheet, whitespace normalised."""
    rules: list[tuple[list[str], dict[str, str]]] = []
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        selectors = [re.sub(r"\s+", "", s) for s in sel.split(",") if s.strip()]
        decls: dict[str, str] = {}
        for decl in body.split(";"):
            if ":" in decl:
                key, _, val = decl.partition(":")
                decls[key.strip().lower()] = re.sub(r"\s+", "", val).lower()
        rules.append((selectors, decls))
    return rules


def _by_selector(rules: list[tuple[list[str], dict[str, str]]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for selectors, decls in rules:
        for s in selectors:
            out.setdefault(s, {}).update(decls)
    return out


_VOID_TAGS = {"meta", "link", "br", "hr", "img", "input", "source", "wbr", "col", "base", "embed"}


class _GridChildren(HTMLParser):
    """Records ``(tag, attrs)`` of every direct child of a ``.grid2`` element."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, bool]] = []  # (tag, is a .grid2 element)
        self.children: list[tuple[str, dict[str, str | None]]] = []
        self.grids = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.stack and self.stack[-1][1]:
            self.children.append((tag, a))
        is_grid = "grid2" in (a.get("class") or "").split()
        self.grids += int(is_grid)
        if tag not in _VOID_TAGS:
            self.stack.append((tag, is_grid))

    def handle_startendtag(self, tag, attrs):
        if self.stack and self.stack[-1][1]:
            self.children.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def test_dashboard_grid_items_do_not_force_document_overflow(dash_html):
    """Regression for the live 390-px overflow (host review #2, blocker B).

    CSS-grid items default to ``min-width:auto``, so a ``<section>`` holding a nowrap table grows
    to the table's min-content width and widens the DOCUMENT instead of letting its ``.scroll``
    wrapper scroll. The generated CSS must shrink grid children (``min-width:0``), nothing may undo
    it in a media block, the in-table scroll mechanism must stay, and the rule must reach every
    ``.grid2`` child (they are all ``<section>`` elements).
    """
    style = re.search(r"<style>(.*?)</style>", dash_html, re.S)
    assert style is not None
    base_css, media = _split_media(style.group(1))
    rules = _css_rules(base_css)
    grid_selectors = {".grid2>section", ".grid2>*", "section"}
    shrink = [
        decls
        for selectors, decls in rules
        if grid_selectors & set(selectors) and decls.get("min-width") in {"0", "0px"}
    ]
    assert shrink, "grid children need min-width:0 so wide tables scroll inside .scroll"
    for query, inner in media.items():
        for selectors, decls in _css_rules(inner):
            if grid_selectors & set(selectors):
                assert decls.get("min-width", "0") in {"0", "0px"}, (query, selectors, decls)
    by_sel = _by_selector(rules)
    assert by_sel[".scroll"].get("overflow-x") == "auto"
    assert by_sel[".chart"].get("width") == "100%"
    assert by_sel[".grid2"].get("display") == "grid"
    assert by_sel["main"].get("max-width") == "1100px"
    for sel in ("html", "body", "main"):  # no blanket clipping of the document
        assert by_sel.get(sel, {}).get("overflow") != "hidden", sel
        assert by_sel.get(sel, {}).get("overflow-x") != "hidden", sel
    narrow = next((inner for q, inner in media.items() if "max-width" in q), "")
    assert _by_selector(_css_rules(narrow)).get(".grid2", {}).get("grid-template-columns") == "1fr"
    # print keeps tables fully visible (overflow:visible); with shrunk grid items that is only safe
    # in a single column (a wide table must not paint over a neighbouring column) and with cells
    # allowed to wrap
    print_by = _by_selector(_css_rules(media.get("print", "")))
    assert print_by.get(".scroll", {}).get("overflow") == "visible"
    assert print_by.get(".grid2", {}).get("grid-template-columns") == "1fr"
    assert print_by.get("td", {}).get("white-space") == "normal"
    assert print_by.get("th", {}).get("white-space") == "normal"
    parser = _GridChildren()
    parser.feed(dash_html)
    assert parser.grids >= 2
    assert parser.children, "no .grid2 children found"
    assert all(tag == "section" for tag, _ in parser.children), parser.children
    assert len(parser.children) == 2 * parser.grids


def test_dashboard_footer_and_titles_unchanged(dash_html):
    assert "Generated by DemandCast · Python + SQLite ·" in dash_html
    # the footer is the only "generated by" statement on the page (no added attribution)
    assert len(re.findall(r"generated by", dash_html, re.I)) == 1
    assert "Model leaderboard" in dash_html
    assert "Replenishment order book" in dash_html
    assert "#CC0000" in dash_html


def test_dashboard_featured_chart(dash_html):
    sec = _section(dash_html, "featured")
    assert "80% interval" in sec  # no forecast_runs.interval_level on this run -> default
    svg = _first_svg(sec)
    assert svg.count('class="band"') >= 1
    assert svg.count("<polyline") >= 2  # actual history + forecast
    assert 'stroke="#CC0000"' in svg
    ticks = re.findall(r'class="tick xt"[^>]*>([^<]*)<', svg)
    assert 2 <= len(ticks) <= 8
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", t) for t in ticks)
    assert "Actual units" in svg
    assert "Forecast" in svg


def test_dashboard_degrades_on_baseline_schema(baseline_like_db, tmp_path):
    text = render(baseline_like_db, tmp_path / "baseline.html").read_text(encoding="utf-8")
    assert "<svg" in text
    assert "Model leaderboard" in text
    assert "Replenishment order book" in text
    for name, title in V2_SECTION_TITLES.items():
        sec = _section(text, name)
        assert title in sec, name
        assert NOT_AVAILABLE in sec, f"section {name} should say it is not available"
    assert "80% interval" in _section(text, "featured")
    for name in ("leaderboard", "categories", "health", "orders", "abc", "stockouts", "promo_lift"):
        assert NOT_AVAILABLE not in _section(text, name), name


def test_dashboard_v2_sections_with_contract_shaped_data(v2_like_db, tmp_path):
    conn, run_id, meta = v2_like_db
    text = render(conn, tmp_path / "v2.html", run_id=run_id).read_text(encoding="utf-8")
    for name in V2_SECTION_TITLES:
        assert NOT_AVAILABLE not in _section(text, name), name
    cfg = _section(text, "run_config")
    assert "interval_method" in cfg
    assert "empirical" in cfg
    assert "0.4.0" in cfg
    acc = _section(text, "accuracy")
    assert "Realised WAPE" in acc
    assert "Interval coverage" in acc
    assert "30.0%" in acc  # aggregate WAPE = sum|e| / sum|y| = 21 / 70 from the fixture rows
    assert "85%" in acc  # coverage KPI
    assert "nominal 90%" in acc  # nominal level of this run
    assert "<svg" in acc  # history chart
    assert "avg_mae" in acc  # per-model table
    dist_svg = _first_svg(_section(text, "health_distribution"))
    health_labels = {r["health"] for r in db.run_query(conn, "days_of_cover", {"run_id": run_id})}
    assert health_labels
    dist_nodes = _svg_text_nodes(dist_svg)
    assert all(h in dist_nodes for h in health_labels), (health_labels, dist_nodes)
    risk = _section(text, "risk")
    assert "stockout_risk" in risk
    assert risk.count("<tr>") <= 11
    assert "%" in risk
    cost = _section(text, "order_cost")
    assert "<svg" in cost
    assert "n_deferred" in cost
    cats = {r["category"] for r in db.run_query(conn, "order_cost_by_category", {"run_id": run_id})}
    assert cats
    assert all(c in cost for c in cats), cats
    rollup = _section(text, "rollup")
    assert "chain" in rollup
    assert "region" in rollup
    assert "category" in rollup
    promo = _section(text, "promo_upcoming")
    assert "ALL" in promo
    assert meta["future_promo"][0].isoformat() in promo
    feat = _section(text, "featured")
    assert "90% interval" in feat
    assert _first_svg(feat).count('class="shade"') >= 2  # recent promo + future promo spans
    assert "promotion" in feat.lower()
    # order book picks up the added columns; the deferred order (cost 0, so never "top by cost")
    # is surfaced through the KPI badge
    orders = _section(text, "orders")
    assert "stockout_risk" in orders
    assert "requested_qty" in orders
    assert "priority" in orders
    assert "1 deferred (budget)" in _section(text, "kpis")


def test_dashboard_sections_selection(dash_db, tmp_path):
    out = render(dash_db, tmp_path / "lb.html", sections=["leaderboard"])
    text = out.read_text(encoding="utf-8")
    assert "Model leaderboard" in text
    assert "Replenishment order book" not in text
    assert "Featured series" not in text
    assert "Generated by DemandCast" in text
    assert "<svg" not in text
    two = render(dash_db, tmp_path / "two.html", sections=("featured", "orders")).read_text()
    assert "Featured series" in two
    assert "Replenishment order book" in two
    assert "Model leaderboard" not in two
    with pytest.raises(ValueError, match="unknown dashboard section"):
        render(dash_db, tmp_path / "bad.html", sections=["nope"])
    assert "kpis" in SECTIONS
    assert "leaderboard" in SECTIONS
    assert len(SECTIONS) >= 15


def test_dashboard_output_size_and_speed(dash_db, dash_html, tmp_path):
    assert len(dash_html.encode("utf-8")) < 400_000
    assert dash_html.count("<svg") >= 2
    t0 = time.perf_counter()
    render(dash_db, tmp_path / "timed.html")
    assert time.perf_counter() - t0 < 2.0


def test_dashboard_orders_delta_badge(v2_like_db, tmp_path):
    conn, run_id, _ = v2_like_db
    # a synthetic earlier successful run with three placed orders -> KPI shows the delta
    conn.execute(
        "INSERT INTO forecast_runs (run_id, started_at, finished_at, cutoff_day, horizon_days, "
        "series_count, status, notes) VALUES (0, 't', 't', '2024-10-01', 14, 3, 'succeeded', '')"
    )
    rows = conn.execute(
        "SELECT store_id, product_id FROM replenishment_orders WHERE run_id=? LIMIT 3", (run_id,)
    ).fetchall()
    conn.executemany(
        "INSERT INTO replenishment_orders (run_id, store_id, product_id, order_day, "
        "expected_arrival, on_hand, on_order, lead_time_demand, safety_stock, reorder_point, "
        "order_up_to, order_qty, service_level, reason) VALUES (0, ?, ?, '2024-10-02', "
        "'2024-10-09', 1, 0, 1.0, 1.0, 1.0, 1.0, 6, 0.95, 'x')",
        [(r["store_id"], r["product_id"]) for r in rows],
    )
    conn.commit()
    placed = conn.execute(
        "SELECT COUNT(*) FROM replenishment_orders WHERE run_id=? AND order_qty > 0", (run_id,)
    ).fetchone()[0]
    text = render(conn, tmp_path / "delta.html", run_id=run_id).read_text(encoding="utf-8")
    assert f"{placed - 3:+d} vs run #0" in text


def test_dashboard_v2_sections_render_with_real_queries(tmp_path):
    """Full v2 stack: schema v2 + analytics v2 + backdated run + evaluate + current run."""
    from demandcast import evaluate

    conn = db.connect(":memory:")
    db.init_schema(conn)
    generate(
        conn,
        SimConfig(
            n_stores=3, n_products=8, start=date(2024, 1, 1), days=300, seed=7, future_promo_days=14
        ),
    )
    last = date.fromisoformat(conn.execute("SELECT MAX(day) FROM sales_daily").fetchone()[0])
    pipeline.run(
        conn,
        pipeline.RunConfig(
            horizon_days=14, n_folds=2, workers=1, cutoff_day=last - timedelta(days=14)
        ),
    )
    evaluate.evaluate_run(conn)
    run_id = pipeline.run(conn, pipeline.RunConfig(horizon_days=14, n_folds=2, workers=1))
    text = render(conn, tmp_path / "full.html", run_id=run_id).read_text(encoding="utf-8")
    for name in V2_SECTION_TITLES:
        assert NOT_AVAILABLE not in _section(text, name), name
    assert "Realised WAPE" in text
    assert "interval_method" in text
    assert "<script" not in text.lower()
    assert len(text.encode()) < 400_000
    categories = [r[0] for r in conn.execute("SELECT DISTINCT category FROM products")]
    nodes = _svg_text_nodes(_section(text, "categories"))
    assert all(c in nodes for c in categories), (categories, nodes)
