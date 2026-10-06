"""Packaging, deployment-header and dashboard-verifier guard tests.

These tests pin the facts that CI, the static Vercel deployment and the packaging metadata
rely on:

* ``pyproject.toml`` version == ``demandcast.__version__``; the runtime dependency is NumPy
  only; package data ships the SQL files and ``py.typed``; the console script points at the
  CLI; mypy is part of the dev extras and configured.
* every ``-- name:`` marker in ``analytics.sql`` is unique and loaded.
* ``vercel.json`` is a headers-only configuration whose Content-Security-Policy equals the
  canonical ``CSP_HEADER`` literal (PLAN C12), with exactly the agreed header set.
* ``scripts/verify_dashboard.py`` flags every CSP obligation violation on synthetic documents,
  serves a directory with the exact production headers (checked with a real HTTP request) and
  never reports a pass when the browser check cannot run.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

import demandcast

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VERCEL_JSON = ROOT / "vercel.json"
VERIFIER = ROOT / "scripts" / "verify_dashboard.py"

CSP_META = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
    "base-uri 'none'; form-action 'none'"
)
CSP_HEADER = CSP_META + "; frame-ancestors 'none'"
EXPECTED_HEADERS = {
    "Content-Security-Policy": CSP_HEADER,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}
HTML_CONTENT_TYPE = "text/html; charset=utf-8"


# ---- pyproject.toml (regex based: tomllib is 3.11+) ---------------------------------------
def _pyproject_text() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _toml_section(text: str, header: str) -> str:
    """Return the body of one TOML table, up to the next table header."""
    m = re.search(rf"^\[{re.escape(header)}\]\n(.*?)(?=^\[|\Z)", text, flags=re.M | re.S)
    assert m is not None, f"[{header}] missing from pyproject.toml"
    return m.group(1)


def _toml_string_list(body: str, key: str) -> list[str]:
    m = re.search(rf"^{re.escape(key)}\s*=\s*\[(.*?)\]", body, flags=re.M | re.S)
    assert m is not None, f"{key} missing"
    return re.findall(r'"([^"]*)"', m.group(1))


def test_version_matches_package():
    body = _toml_section(_pyproject_text(), "project")
    m = re.search(r'^version\s*=\s*"([^"]+)"', body, flags=re.M)
    assert m is not None
    assert m.group(1) == demandcast.__version__


def test_runtime_dependencies_are_numpy_only():
    deps = _toml_string_list(_toml_section(_pyproject_text(), "project"), "dependencies")
    assert deps == ["numpy>=1.24"]


def test_dev_extras_include_test_lint_and_type_tools():
    body = _toml_section(_pyproject_text(), "project.optional-dependencies")
    names = {re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0] for d in _toml_string_list(body, "dev")}
    assert {"pytest", "ruff", "mypy"} <= names


def test_mypy_configuration_present():
    body = _toml_section(_pyproject_text(), "tool.mypy")
    assert re.search(r'^python_version\s*=\s*"3\.10"', body, flags=re.M)
    assert re.search(r"^check_untyped_defs\s*=\s*true", body, flags=re.M)
    assert re.search(r"^no_implicit_optional\s*=\s*true", body, flags=re.M)


def test_package_data_ships_sql_and_py_typed():
    body = _toml_section(_pyproject_text(), "tool.setuptools.package-data")
    data = _toml_string_list(body, "demandcast")
    assert "sql/*.sql" in data
    assert "py.typed" in data
    assert (ROOT / "demandcast" / "py.typed").is_file()


def test_console_script_points_to_cli_main():
    body = _toml_section(_pyproject_text(), "project.scripts")
    assert re.search(r'^demandcast\s*=\s*"demandcast\.cli:main"', body, flags=re.M)


def test_analytics_query_names_are_unique_and_loaded():
    from demandcast import db

    sql = (ROOT / "demandcast" / "sql" / "analytics.sql").read_text(encoding="utf-8")
    names = re.findall(r"^--\s*name:\s*(\w+)\s*$", sql, flags=re.M)
    assert len(names) >= 8
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert duplicates == []
    assert set(names) == set(db.QUERIES)


# ---- vercel.json (C12) --------------------------------------------------------------------
def _vercel_cfg() -> dict:
    return json.loads(VERCEL_JSON.read_text(encoding="utf-8"))


def test_vercel_json_is_headers_only_with_canonical_csp():
    cfg = _vercel_cfg()
    assert sorted(cfg) == ["headers"]
    rules = cfg["headers"]
    assert len(rules) == 1
    assert sorted(rules[0]) == ["headers", "source"]
    assert rules[0]["source"] == "/(.*)"
    configured = {h["key"]: h["value"] for h in rules[0]["headers"]}
    assert configured["Content-Security-Policy"] == CSP_HEADER
    assert configured == EXPECTED_HEADERS
    assert [h["key"] for h in rules[0]["headers"]] == list(EXPECTED_HEADERS)


def test_dashboard_csp_meta_constant_matches_contract():
    from demandcast import dashboard

    assert dashboard.CSP_META == CSP_META


# ---- scripts/verify_dashboard.py ----------------------------------------------------------
@pytest.fixture(scope="module")
def verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_dashboard", VERIFIER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def make_html(
    *,
    csp: str | None = CSP_META,
    favicon: bool = True,
    order: str = "charset_first",
    head_extra: str = "",
    css_extra: str = "",
    body_extra: str = "",
    pad: bool = True,
) -> str:
    """Build a small dashboard-like document; the defaults satisfy every C12 obligation."""
    csp_tag = f'<meta http-equiv="Content-Security-Policy" content="{csp}">' if csp else ""
    viewport = "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    icon = "<link rel='icon' href='data:,'>" if favicon else ""
    if order == "viewport_between":
        head = f"<meta charset='utf-8'>{viewport}{csp_tag}{icon}"
    else:
        head = f"<meta charset='utf-8'>\n{csp_tag}\n{viewport}{icon}"
    padding = "<p class='muted'>" + "x" * 1200 + "</p>" if pad else ""
    return (
        "<!doctype html><html lang='en'><head>"
        f"{head}<title>DemandCast - run #1</title>"
        f"<style>:root{{--red:#CC0000}}body{{color:#111827}}{css_extra}</style>{head_extra}"
        "</head><body><header><h1>Demand<span>Cast</span></h1></header><main>"
        "<h2>Model leaderboard</h2><div class='card'><table><tr><td>moving_average</td></tr>"
        "</table></div><h2>Replenishment order book (top by cost)</h2>"
        "<svg viewBox='0 0 10 10' role='img'><rect x='1' y='1' width='2' height='2' "
        f"fill='#CC0000'><title>Toys: 1</title></rect></svg>{body_extra}{padding}"
        "<footer>Generated by DemandCast</footer></main></body></html>"
    )


def _failed(findings) -> set[str]:
    return {f.check for f in findings if not f.ok}


def test_check_static_passes_on_compliant_document(verifier):
    findings = verifier.check_static(make_html(), _vercel_cfg())
    assert _failed(findings) == set()
    checks = {f.check for f in findings}
    assert {
        "html.size",
        "html.meta_csp",
        "html.meta_csp_position",
        "html.favicon",
        "html.script",
        "html.inline_handler",
        "html.javascript_url",
        "html.css_url",
        "html.font_face",
        "html.base",
        "html.external_resource",
        "vercel.keys",
        "vercel.csp",
        "vercel.headers",
    } <= checks
    for key, value in EXPECTED_HEADERS.items():  # every configured header is echoed
        echoed = [f.detail for f in findings if f.check == f"vercel.header.{key}"]
        assert echoed == [value]


@pytest.mark.parametrize(
    ("kwargs", "expected_failures"),
    [
        ({"csp": None}, {"html.meta_csp", "html.meta_csp_position"}),
        ({"csp": "default-src 'self'"}, {"html.meta_csp"}),
        ({"order": "viewport_between"}, {"html.meta_csp_position"}),
        ({"favicon": False}, {"html.favicon"}),
        ({"body_extra": "<script>alert(1)</script>"}, {"html.script"}),
        ({"body_extra": "<button onclick='alert(1)'>x</button>"}, {"html.inline_handler"}),
        ({"body_extra": "<svg onload='alert(1)'></svg>"}, {"html.inline_handler"}),
        ({"body_extra": "<a href='javascript:alert(1)'>x</a>"}, {"html.javascript_url"}),
        ({"css_extra": "body{background:url(https://example.com/bg.png)}"}, {"html.css_url"}),
        ({"body_extra": "<div style=\"background:url('/x.png')\"></div>"}, {"html.css_url"}),
        (
            {"css_extra": "@font-face{font-family:X;src:url(data:font/woff2;base64,AAAA)}"},
            {"html.font_face"},
        ),
        ({"head_extra": "<base href='https://example.com/'>"}, {"html.base"}),
        (
            {"body_extra": "<img src='https://example.com/x.png' alt=''>"},
            {"html.external_resource"},
        ),
        ({"body_extra": "<img src='//example.com/x.png' alt=''>"}, {"html.external_resource"}),
        (
            {"head_extra": "<link rel='stylesheet' href='https://example.com/a.css'>"},
            {"html.external_resource"},
        ),
        (
            {"body_extra": "<iframe src='https://example.com/'></iframe>"},
            {"html.external_resource"},
        ),
        ({"pad": False}, {"html.size"}),
    ],
    ids=[
        "missing_meta",
        "wrong_meta_policy",
        "meta_not_after_charset",
        "missing_favicon",
        "script",
        "inline_handler",
        "svg_onload",
        "javascript_url",
        "css_url",
        "inline_style_url",
        "font_face",
        "base",
        "external_img",
        "protocol_relative_img",
        "external_stylesheet",
        "iframe",
        "too_small",
    ],
)
def test_check_static_flags_each_violation(verifier, kwargs, expected_failures):
    findings = verifier.check_static(make_html(**kwargs), _vercel_cfg())
    assert _failed(findings) == expected_failures


def test_check_static_accepts_double_quoted_head_markup(verifier):
    """The renderer uses double quotes throughout; quoting style must not matter."""
    html = make_html().replace("<meta charset='utf-8'>", '<meta charset="utf-8">')
    html = html.replace("<link rel='icon' href='data:,'>", '<link rel="icon" href="data:,">')
    assert '<meta charset="utf-8">' in html
    assert _failed(verifier.check_static(html, _vercel_cfg())) == set()


def _mutated_cfg(kind: str) -> dict:
    cfg = copy.deepcopy(_vercel_cfg())
    rule = cfg["headers"][0]
    if kind == "extra_key":
        cfg["routes"] = []
    elif kind == "wrong_csp":
        for h in rule["headers"]:
            if h["key"] == "Content-Security-Policy":
                h["value"] = "default-src 'self'"
    elif kind == "missing_header":
        rule["headers"] = [h for h in rule["headers"] if h["key"] != "X-Frame-Options"]
    elif kind == "source_does_not_match":
        rule["source"] = "/other.html"
    return cfg


@pytest.mark.parametrize(
    ("kind", "expected_failures"),
    [
        ("extra_key", {"vercel.keys"}),
        ("wrong_csp", {"vercel.csp", "vercel.headers"}),
        ("missing_header", {"vercel.headers"}),
        ("source_does_not_match", {"vercel.csp", "vercel.headers"}),
    ],
)
def test_check_static_flags_vercel_config_problems(verifier, kind, expected_failures):
    findings = verifier.check_static(make_html(), _mutated_cfg(kind))
    assert _failed(findings) == expected_failures


def test_source_matching_supports_vercel_catch_all_and_literal_paths(verifier):
    assert verifier.source_matches("/(.*)", "/") is True
    assert verifier.source_matches("/(.*)", "/index.html") is True
    assert verifier.source_matches("/(.*)", "/nested/path.html") is True
    assert verifier.source_matches("/index.html", "/index.html") is True
    assert verifier.source_matches("/index.html", "/") is False
    assert verifier.source_matches("/index.html", "/index.htmlx") is False
    assert verifier.source_matches("/:slug", "/report.html") is True
    assert verifier.source_matches("/:slug", "/a/b") is False


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def test_served_mode_applies_every_header_end_to_end(verifier, tmp_path):
    (tmp_path / "index.html").write_text(make_html(), encoding="utf-8")
    cfg = _vercel_cfg()
    opener = _no_proxy_opener()
    with verifier.serve_with_headers(tmp_path, cfg) as server:
        assert server.base_url.startswith("http://127.0.0.1:")
        for path in ("/", "/index.html"):
            with opener.open(server.base_url + path, timeout=10) as resp:
                body = resp.read()
                assert resp.status == 200
                assert resp.headers["Content-Type"] == HTML_CONTENT_TYPE
                for key, value in EXPECTED_HEADERS.items():
                    assert resp.headers[key] == value, key
            assert b"Model leaderboard" in body
        findings = verifier.check_served(server.base_url, cfg)
        request_log = list(server.request_log)
    assert _failed(findings) == set()
    checked = {f.check for f in findings}
    for path in ("/", "/index.html"):
        assert f"served.{path}.status" in checked
        assert f"served.{path}.content_type" in checked
        for key in EXPECTED_HEADERS:
            assert f"served.{path}.header.{key}" in checked
    assert any(line.startswith("GET / HTTP/1.1") for line in request_log)
    assert any(line.startswith("GET /index.html HTTP/1.1") for line in request_log)


def test_check_served_reports_missing_and_wrong_headers(verifier, tmp_path):
    (tmp_path / "index.html").write_text(make_html(), encoding="utf-8")
    served_cfg = _mutated_cfg("missing_header")
    with verifier.serve_with_headers(tmp_path, served_cfg) as server:
        findings = verifier.check_served(server.base_url, _vercel_cfg(), ("/index.html",))
    assert _failed(findings) == {"served./index.html.header.X-Frame-Options"}

    wrong_cfg = _mutated_cfg("wrong_csp")
    with verifier.serve_with_headers(tmp_path, wrong_cfg) as server:
        findings = verifier.check_served(server.base_url, _vercel_cfg(), ("/index.html",))
    assert _failed(findings) == {"served./index.html.header.Content-Security-Policy"}


def test_check_served_reports_missing_file(verifier, tmp_path):
    with verifier.serve_with_headers(tmp_path, _vercel_cfg()) as server:
        findings = verifier.check_served(server.base_url, _vercel_cfg(), ("/index.html",))
    assert "served./index.html.status" in _failed(findings)


def test_main_json_summary_pass_and_fail(verifier, tmp_path, capsys):
    html = tmp_path / "index.html"
    html.write_text(make_html(), encoding="utf-8")
    args = ["--html", str(html), "--vercel", str(VERCEL_JSON), "--json"]

    assert verifier.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["passed"] is True
    assert summary["headers"] == EXPECTED_HEADERS
    assert all(c["ok"] for c in summary["checks"])
    assert summary["served"]["base_url"].startswith("http://127.0.0.1:")
    assert len(summary["served"]["requests"]) == 2
    assert summary["browser"] is None

    html.write_text(make_html(body_extra="<script>1</script>"), encoding="utf-8")
    assert verifier.main(args) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["passed"] is False
    assert [c["check"] for c in summary["checks"] if not c["ok"]] == ["html.script"]


def test_main_rejects_missing_inputs(verifier, tmp_path, capsys):
    rc = verifier.main(["--html", str(tmp_path / "nope.html"), "--vercel", str(VERCEL_JSON)])
    assert rc == 2
    assert "nope.html" in capsys.readouterr().out
    html = tmp_path / "index.html"
    html.write_text(make_html(), encoding="utf-8")
    rc = verifier.main(["--html", str(html), "--vercel", str(tmp_path / "missing.json")])
    assert rc == 2


def test_browser_mode_without_playwright_exits_3_and_never_passes(
    verifier, tmp_path, capsys, monkeypatch
):
    html = tmp_path / "index.html"
    html.write_text(make_html(), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(verifier.BrowserUnavailable):
        verifier.check_browser("http://127.0.0.1:9/")
    rc = verifier.main(["--html", str(html), "--vercel", str(VERCEL_JSON), "--browser", "--json"])
    captured = capsys.readouterr()
    assert rc == 3
    assert "browser check unavailable: playwright not installed" in captured.out + captured.err
    summary = json.loads(captured.out)
    assert summary["passed"] is False
    assert summary["browser"]["available"] is False


def test_rendered_dashboard_passes_static_and_served_checks(verifier, tmp_path):
    from demandcast import dashboard, db, pipeline
    from demandcast.simulate import SimConfig, generate

    assert hasattr(dashboard, "CSP_META")
    conn = db.connect(":memory:")
    db.init_schema(conn)
    generate(conn, SimConfig(2, 4, date(2024, 1, 1), 200, 5))
    pipeline.run(conn, pipeline.RunConfig(horizon_days=14, n_folds=2, workers=1))
    out = dashboard.render(conn, tmp_path / "index.html")
    cfg = _vercel_cfg()
    assert _failed(verifier.check_static(out.read_text(encoding="utf-8"), cfg)) == set()
    with verifier.serve_with_headers(tmp_path, cfg) as server:
        served = verifier.check_served(server.base_url, cfg)
    assert _failed(served) == set()


def test_committed_dashboard_artifact_is_policy_compliant(verifier):
    """The published artifact must satisfy the policy it is deployed with (gates 9 and 11)."""
    from demandcast import dashboard

    assert hasattr(dashboard, "CSP_META")
    assert demandcast.__version__ != "0.3.0"
    html = ROOT / "dashboard" / "index.html"
    cfg = _vercel_cfg()
    assert _failed(verifier.check_static(html.read_text(encoding="utf-8"), cfg)) == set()
    with verifier.serve_with_headers(html.parent, cfg) as server:
        served = verifier.check_served(server.base_url, cfg)
    assert _failed(served) == set()
