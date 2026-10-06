#!/usr/bin/env python3
"""Verify the static dashboard artifact against the production security policy.

DemandCast publishes ``dashboard/index.html`` as a static site; ``vercel.json`` adds the
response security headers. This script checks that the artifact and the header configuration
are consistent with the policy both *at rest* and *as served*. It uses the standard library
only; Playwright is optional and only touched in ``--browser`` mode.

Checks
------
1. **static** - the HTML file exists and is larger than 1 KB; the document honours every
   dashboard obligation (``<meta http-equiv="Content-Security-Policy">`` equal to
   ``CSP_META`` directly after ``<meta charset>``, a ``data:`` favicon, no ``<script>``, no
   ``on*=`` handlers, no ``javascript:`` URLs, no CSS ``url()`` other than ``data:``, no
   ``@font-face``, no ``<base>``, no external ``src``/``href`` on resource tags);
   ``vercel.json`` contains only the ``headers`` key and its Content-Security-Policy header
   equals ``CSP_HEADER``; every configured header is echoed in the report.
2. **served** - the HTML directory is served by ``http.server`` on ``127.0.0.1`` with the
   headers from ``vercel.json`` applied the way Vercel applies them (``source`` patterns);
   ``/`` and ``/index.html`` must answer 200 with ``text/html; charset=utf-8`` and every
   configured header with its exact value. The server's request lines are recorded.
3. **browser** (``--browser``) - headless Chromium (Playwright) loads the served page under
   the enforced policy: zero ``securitypolicyviolation`` events, zero console
   errors/warnings, zero page errors, no request leaving the served origin,
   ``document.scripts.length == 0`` and the key headings visible. ``--screenshot PATH`` saves
   a full-page capture. Without Playwright the check is reported as unavailable (exit 3),
   never as a pass.

Exit codes: 0 every performed check passed - 1 at least one check failed -
2 missing input file / usage error - 3 all performed checks passed but the requested
browser check could not run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

CSP_META = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
    "base-uri 'none'; form-action 'none'"
)
CSP_HEADER = CSP_META + "; frame-ancestors 'none'"

#: The production header set (PLAN C12). ``vercel.json`` must apply exactly these values.
EXPECTED_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CSP_HEADER,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}

MIN_HTML_BYTES = 1024
HTML_CONTENT_TYPE = "text/html; charset=utf-8"
RESOURCE_TAGS = frozenset(
    {"link", "img", "iframe", "frame", "object", "embed", "source", "video", "audio", "track"}
)
URL_ATTRIBUTES = ("src", "href", "data", "poster", "srcset", "xlink:href")
REQUIRED_HEADINGS = ("Model leaderboard", "Replenishment order book")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_BROWSER_UNAVAILABLE = 3
BROWSER_UNAVAILABLE_MESSAGE = "browser check unavailable: playwright not installed"

_CSP_LISTENER_JS = """
(() => {
  window.__cspViolations = [];
  document.addEventListener('securitypolicyviolation', (e) => {
    window.__cspViolations.push({
      blockedURI: e.blockedURI,
      violatedDirective: e.violatedDirective,
      sourceFile: e.sourceFile,
    });
  }, true);
})();
"""


@dataclass(frozen=True)
class Finding:
    """One performed check: a stable identifier, pass/fail and a human-readable detail."""

    check: str
    ok: bool
    detail: str = ""


class BrowserUnavailable(RuntimeError):
    """Raised when ``--browser`` cannot run (Playwright missing or Chromium failed to start)."""


# ---- vercel.json helpers --------------------------------------------------------------------
def source_matches(source: str, path: str) -> bool:
    """Return True when a Vercel ``source`` pattern matches ``path``.

    Supports literal paths, the catch-all ``/(.*)``, embedded regex groups ``(...)``,
    ``:param`` segments and ``*`` wildcards (the path-to-regexp subset Vercel documents).
    """
    if source == path:
        return True
    pattern = ""
    i = 0
    while i < len(source):
        ch = source[i]
        if ch == "(":
            depth, j = 1, i + 1
            while j < len(source) and depth:
                depth += {"(": 1, ")": -1}.get(source[j], 0)
                j += 1
            pattern += source[i:j]
            i = j
        elif ch == ":":
            j = i + 1
            while j < len(source) and (source[j].isalnum() or source[j] == "_"):
                j += 1
            pattern += "([^/]+)"
            i = j
        elif ch == "*":
            pattern += ".*"
            i += 1
        else:
            pattern += re.escape(ch)
            i += 1
    try:
        return re.fullmatch(pattern, path) is not None
    except re.error:
        return False


def header_rules(vercel_cfg: Any) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return ``[(source, [(key, value), ...]), ...]`` from a parsed ``vercel.json``."""
    rules: list[tuple[str, list[tuple[str, str]]]] = []
    if not isinstance(vercel_cfg, dict):
        return rules
    for rule in vercel_cfg.get("headers") or []:
        if not isinstance(rule, dict):
            continue
        headers = [
            (str(h["key"]), str(h["value"]))
            for h in rule.get("headers") or []
            if isinstance(h, dict) and "key" in h and "value" in h
        ]
        rules.append((str(rule.get("source", "")), headers))
    return rules


def headers_for(vercel_cfg: Any, path: str) -> list[tuple[str, str]]:
    """Every configured header whose ``source`` matches ``path``, in configuration order."""
    out: list[tuple[str, str]] = []
    for source, headers in header_rules(vercel_cfg):
        if source_matches(source, path):
            out.extend(headers)
    return out


def check_vercel_config(vercel_cfg: Any, path: str = "/index.html") -> list[Finding]:
    """Static checks on the parsed ``vercel.json`` (headers-only file, canonical CSP)."""
    if not isinstance(vercel_cfg, dict):
        return [Finding("vercel.keys", False, "vercel.json must contain a JSON object")]
    keys = sorted(vercel_cfg)
    findings = [
        Finding(
            "vercel.keys",
            keys == ["headers"],
            f"top-level keys {keys} (only 'headers' is allowed: no routes/builds/rewrites)",
        )
    ]
    applied = headers_for(vercel_cfg, path)
    for key, value in applied:  # echo every configured header
        findings.append(Finding(f"vercel.header.{key}", True, value))
    applied_map = {k.lower(): v for k, v in applied}
    csp = applied_map.get("content-security-policy")
    findings.append(
        Finding(
            "vercel.csp",
            csp == CSP_HEADER,
            "Content-Security-Policy header equals CSP_HEADER"
            if csp == CSP_HEADER
            else f"Content-Security-Policy header {csp!r} != CSP_HEADER {CSP_HEADER!r}",
        )
    )
    wrong = [k for k, v in EXPECTED_HEADERS.items() if applied_map.get(k.lower()) != v]
    findings.append(
        Finding(
            "vercel.headers",
            not wrong,
            f"all {len(EXPECTED_HEADERS)} production headers configured for {path}"
            if not wrong
            else f"missing or different for {path}: {wrong}",
        )
    )
    return findings


# ---- static HTML checks ---------------------------------------------------------------------
class _DocumentScanner(HTMLParser):
    """Collect start tags (with attributes and positions), style blocks and inline styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str], int]] = []
        self.style_blocks: list[str] = []
        self.inline_styles: list[str] = []
        self._in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        self.tags.append((name, attr_map, self.getpos()[0]))
        if name == "style":
            self._in_style = True
        if "style" in attr_map:
            self.inline_styles.append(attr_map["style"])

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.style_blocks.append(data)


def _is_inline_url(value: str) -> bool:
    v = value.strip().lower()
    return v == "" or v.startswith("data:") or v.startswith("#")


def check_static(html_text: str, vercel_cfg: Any) -> list[Finding]:
    """Check one HTML document and the parsed ``vercel.json`` against the C12 obligations."""
    findings: list[Finding] = []
    size = len(html_text.encode("utf-8"))
    findings.append(
        Finding("html.size", size > MIN_HTML_BYTES, f"{size} bytes (must exceed {MIN_HTML_BYTES})")
    )

    scanner = _DocumentScanner()
    scanner.feed(html_text)
    scanner.close()
    tags = scanner.tags

    def _is_csp_meta(tag: str, attrs: dict[str, str]) -> bool:
        return tag == "meta" and attrs.get("http-equiv", "").strip().lower() == (
            "content-security-policy"
        )

    csp_metas = [a for t, a, _ in tags if _is_csp_meta(t, a)]
    if not csp_metas:
        findings.append(Finding("html.meta_csp", False, "no Content-Security-Policy meta tag"))
    else:
        content = csp_metas[0].get("content", "")
        ok = len(csp_metas) == 1 and content == CSP_META
        findings.append(
            Finding(
                "html.meta_csp",
                ok,
                "meta policy equals CSP_META"
                if ok
                else f"meta policy {content!r} != CSP_META {CSP_META!r} "
                f"({len(csp_metas)} CSP meta tag(s))",
            )
        )
    charset_idx = next(
        (i for i, (t, a, _) in enumerate(tags) if t == "meta" and "charset" in a), None
    )
    if charset_idx is None:
        findings.append(Finding("html.meta_csp_position", False, "no <meta charset> tag"))
    else:
        following = tags[charset_idx + 1] if charset_idx + 1 < len(tags) else None
        ok = following is not None and _is_csp_meta(following[0], following[1])
        findings.append(
            Finding(
                "html.meta_csp_position",
                ok,
                "CSP meta tag directly follows <meta charset>"
                if ok
                else "CSP meta tag must be the first tag after <meta charset>",
            )
        )

    icons = [a for t, a, _ in tags if t == "link" and "icon" in a.get("rel", "").lower().split()]
    icon_ok = any(a.get("href", "").strip().lower().startswith("data:") for a in icons)
    findings.append(
        Finding(
            "html.favicon",
            icon_ok,
            "data: favicon link present"
            if icon_ok
            else 'missing <link rel="icon" href="data:,"> (browsers would request /favicon.ico)',
        )
    )

    script_lines = [line for t, _, line in tags if t == "script"]
    raw_scripts = len(re.findall(r"<script\b", html_text, flags=re.IGNORECASE))
    scripts_ok = not script_lines and raw_scripts == 0
    findings.append(
        Finding(
            "html.script",
            scripts_ok,
            "no <script> elements"
            if scripts_ok
            else f"{max(len(script_lines), raw_scripts)} <script> occurrence(s)"
            + (f", first at line {script_lines[0]}" if script_lines else ""),
        )
    )

    handlers = [
        f"<{t} {k}> line {line}"
        for t, a, line in tags
        for k in a
        if re.match(r"^on[a-z]", k) is not None
    ]
    findings.append(
        Finding(
            "html.inline_handler",
            not handlers,
            "no inline event handler attributes"
            if not handlers
            else "inline event handlers: " + ", ".join(handlers[:5]),
        )
    )

    js_urls = [
        f"<{t} {k}> line {line}"
        for t, a, line in tags
        for k, v in a.items()
        if re.match(r"^\s*(javascript|vbscript):", v, flags=re.IGNORECASE) is not None
    ]
    findings.append(
        Finding(
            "html.javascript_url",
            not js_urls,
            "no javascript: URLs" if not js_urls else "javascript: URLs: " + ", ".join(js_urls[:5]),
        )
    )

    css = "\n".join([*scanner.style_blocks, *scanner.inline_styles])
    css_urls = [
        m.group(1)
        for m in re.finditer(r"url\(\s*[\"']?\s*([^\"')\s]*)", css, flags=re.IGNORECASE)
        if not m.group(1).lower().startswith("data:")
    ]
    findings.append(
        Finding(
            "html.css_url",
            not css_urls,
            "no CSS url() other than data:"
            if not css_urls
            else "CSS url() references: " + ", ".join(css_urls[:5]),
        )
    )
    font_face = "@font-face" in css.lower()
    findings.append(
        Finding(
            "html.font_face",
            not font_face,
            "no @font-face" if not font_face else "@font-face rule present (fonts must be system)",
        )
    )

    base_lines = [line for t, _, line in tags if t == "base"]
    findings.append(
        Finding(
            "html.base",
            not base_lines,
            "no <base> element" if not base_lines else f"<base> element at line {base_lines[0]}",
        )
    )

    externals = [
        f"<{t} {k}={v.strip()[:60]!r}> line {line}"
        for t, a, line in tags
        if t in RESOURCE_TAGS
        for k in URL_ATTRIBUTES
        for v in [a.get(k, "")]
        if not _is_inline_url(v)
    ]
    findings.append(
        Finding(
            "html.external_resource",
            not externals,
            "no external resources on link/img/iframe/object/embed/source/video/audio"
            if not externals
            else "non-inline resources: " + ", ".join(externals[:5]),
        )
    )

    findings.extend(check_vercel_config(vercel_cfg))
    return findings


# ---- served checks --------------------------------------------------------------------------
class _HeaderHandler(SimpleHTTPRequestHandler):
    """``SimpleHTTPRequestHandler`` that applies the matching ``vercel.json`` headers."""

    server_version = "DemandCastVerifier/1.0"

    def end_headers(self) -> None:
        server = self.server
        if isinstance(server, HeaderServer):
            path = self.path.split("?", 1)[0].split("#", 1)[0]
            for key, value in headers_for(server.vercel_cfg, path):
                self.send_header(key, value)
        super().end_headers()

    def guess_type(self, path: str | os.PathLike[str]) -> str:
        ctype = super().guess_type(path)
        return HTML_CONTENT_TYPE if ctype == "text/html" else ctype

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        server = self.server
        if isinstance(server, HeaderServer):
            server.request_log.append(f"{self.requestline} -> {code}")

    def log_message(self, format: str, *args: Any) -> None:  # silence stderr
        return


class HeaderServer(ThreadingHTTPServer):
    """Threaded static file server for one directory with ``vercel.json`` headers applied."""

    daemon_threads = True

    def __init__(
        self,
        directory: str | Path,
        vercel_cfg: Any,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.directory = str(directory)
        self.vercel_cfg = vercel_cfg
        self.request_log: list[str] = []
        self._thread: threading.Thread | None = None
        super().__init__((host, port), partial(_HeaderHandler, directory=self.directory))

    @property
    def base_url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        host_text = host.decode() if isinstance(host, (bytes, bytearray)) else str(host)
        return f"http://{host_text}:{port}"

    def start(self) -> HeaderServer:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self.serve_forever, name="verify-dashboard-http", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> HeaderServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def serve_with_headers(
    directory: str | Path, vercel_cfg: Any, host: str = "127.0.0.1", port: int = 0
) -> HeaderServer:
    """Start serving ``directory`` with the ``vercel.json`` headers; use as a context manager."""
    return HeaderServer(directory, vercel_cfg, host, port).start()


def check_served(
    base_url: str,
    vercel_cfg: Any,
    paths: Sequence[str] = ("/", "/index.html"),
    timeout: float = 10.0,
) -> list[Finding]:
    """GET every path and compare status, Content-Type and each configured header exactly."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    findings: list[Finding] = []
    for path in paths:
        prefix = f"served.{path}"
        try:
            with opener.open(base_url + path, timeout=timeout) as resp:
                status = int(resp.status)
                headers = {k.lower(): v for k, v in resp.getheaders()}
                body = resp.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            headers = {k.lower(): v for k, v in exc.headers.items()}
            body = exc.read()
        except (urllib.error.URLError, OSError) as exc:
            findings.append(Finding(f"{prefix}.status", False, f"request failed: {exc}"))
            continue
        findings.append(Finding(f"{prefix}.status", status == 200, f"HTTP {status}"))
        ctype = headers.get("content-type")
        findings.append(
            Finding(
                f"{prefix}.content_type",
                ctype == HTML_CONTENT_TYPE,
                f"Content-Type: {ctype!r}"
                + ("" if ctype == HTML_CONTENT_TYPE else f" (expected {HTML_CONTENT_TYPE!r})"),
            )
        )
        findings.append(Finding(f"{prefix}.body", len(body) > MIN_HTML_BYTES, f"{len(body)} bytes"))
        for key, value in headers_for(vercel_cfg, path):
            got = headers.get(key.lower())
            findings.append(
                Finding(
                    f"{prefix}.header.{key}",
                    got == value,
                    f"{key}: {got!r}" + ("" if got == value else f" (expected {value!r})"),
                )
            )
    return findings


# ---- browser check (optional Playwright) ----------------------------------------------------
def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def check_browser(
    url: str,
    *,
    screenshot: str | Path | None = None,
    executable_path: str | None = None,
    color_scheme: str | None = None,
    settle_ms: int = 300,
) -> tuple[list[Finding], dict[str, Any]]:
    """Load ``url`` in headless Chromium under the enforced policy and collect the evidence.

    Returns ``(findings, info)`` where ``info`` holds the raw console messages, request URLs,
    CSP violations and browser version. Raises :class:`BrowserUnavailable` when Playwright is
    not importable or Chromium cannot be launched.
    """
    try:
        sync_api = importlib.import_module("playwright.sync_api")
    except ImportError as exc:
        raise BrowserUnavailable(BROWSER_UNAVAILABLE_MESSAGE) from exc

    origin = _origin(url)
    console: list[dict[str, str]] = []
    page_errors: list[str] = []
    requests: list[str] = []
    info: dict[str, Any] = {
        "available": True,
        "url": url,
        "color_scheme": color_scheme or "default",
        "console": console,
        "page_errors": page_errors,
        "requests": requests,
    }
    with sync_api.sync_playwright() as playwright:
        launch_kwargs: dict[str, Any] = {"headless": True}
        if executable_path:
            launch_kwargs["executable_path"] = executable_path
        try:
            browser = playwright.chromium.launch(**launch_kwargs)
        except Exception as exc:
            raise BrowserUnavailable(
                f"browser check unavailable: chromium launch failed: {exc}"
            ) from exc
        try:
            info["browser_version"] = browser.version
            context_kwargs: dict[str, Any] = {}
            if color_scheme:
                context_kwargs["color_scheme"] = color_scheme
            context = browser.new_context(**context_kwargs)
            context.add_init_script(_CSP_LISTENER_JS)
            page = context.new_page()
            page.on("console", lambda m: console.append({"type": m.type, "text": m.text}))
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.on("request", lambda r: requests.append(r.url))
            response = page.goto(url, wait_until="load")
            page.wait_for_timeout(settle_ms)
            violations = page.evaluate("() => window.__cspViolations || []")
            n_scripts = page.evaluate("() => document.scripts.length")
            headings = {
                h: bool(page.locator("h1, h2, h3").filter(has_text=h).first.is_visible())
                for h in REQUIRED_HEADINGS
            }
            status = response.status if response is not None else None
            response_headers = dict(response.headers) if response is not None else {}
            if screenshot:
                Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot), full_page=True)
        finally:
            browser.close()

    info.update(
        {"status": status, "response_headers": response_headers, "csp_violations": violations}
    )
    noisy = [m for m in console if m["type"] in ("error", "warning")]
    foreign = [u for u in requests if not (u.startswith(origin) or u.startswith("data:"))]
    csp_header = response_headers.get("content-security-policy")
    findings = [
        Finding("browser.status", status == 200, f"HTTP {status}"),
        Finding(
            "browser.csp_header",
            csp_header == CSP_HEADER,
            "enforced Content-Security-Policy header equals CSP_HEADER"
            if csp_header == CSP_HEADER
            else f"Content-Security-Policy header {csp_header!r} != CSP_HEADER",
        ),
        Finding(
            "browser.csp_violations",
            not violations,
            "no securitypolicyviolation events"
            if not violations
            else f"{len(violations)} violation(s): {json.dumps(violations[:5])}",
        ),
        Finding(
            "browser.console",
            not noisy,
            f"no console errors/warnings ({len(console)} console message(s))"
            if not noisy
            else f"{len(noisy)} console error(s)/warning(s): {json.dumps(noisy[:5])}",
        ),
        Finding(
            "browser.page_errors",
            not page_errors,
            "no page errors" if not page_errors else "; ".join(page_errors[:5]),
        ),
        Finding(
            "browser.foreign_requests",
            not foreign,
            f"all {len(requests)} request(s) stay on {origin}"
            if not foreign
            else f"{len(foreign)} request(s) left {origin}: {foreign[:5]}",
        ),
        Finding("browser.scripts", n_scripts == 0, f"document.scripts.length == {n_scripts}"),
    ]
    for heading, visible in headings.items():
        findings.append(
            Finding(
                f"browser.heading.{heading}",
                visible,
                f"heading {heading!r} " + ("visible" if visible else "not visible"),
            )
        )
    if screenshot:
        saved = Path(screenshot).is_file()
        findings.append(
            Finding(
                "browser.screenshot",
                saved,
                f"full-page screenshot {'saved to' if saved else 'missing at'} {screenshot}",
            )
        )
        info["screenshot"] = str(screenshot)
    return findings, info


# ---- CLI ------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_dashboard.py",
        description="Check dashboard/index.html and vercel.json against the production policy.",
    )
    p.add_argument("--html", default="dashboard/index.html", help="dashboard artifact to check")
    p.add_argument("--vercel", default="vercel.json", help="headers configuration to check")
    p.add_argument("--browser", action="store_true", help="also load the page in headless Chromium")
    p.add_argument("--screenshot", help="save a full-page screenshot (browser mode)")
    p.add_argument("--executable-path", help="Chromium binary for Playwright (browser mode)")
    p.add_argument(
        "--color-scheme",
        choices=("light", "dark"),
        help="emulate prefers-color-scheme in browser mode",
    )
    p.add_argument("--json", action="store_true", help="print a JSON summary instead of text")
    return p


def _print_text_report(findings: list[Finding], report: dict[str, Any]) -> None:
    for f in findings:
        print(f"{'PASS' if f.ok else 'FAIL'}  {f.check:<48} {f.detail}")
    if report.get("headers"):
        print("configured response headers:")
        for key, value in report["headers"].items():
            print(f"  {key}: {value}")
    served = report.get("served")
    if served:
        print(f"served from {served['base_url']}:")
        for line in served["requests"]:
            print(f"  {line}")
    failed = [f for f in findings if not f.ok]
    print(f"{len(findings)} check(s), {len(failed)} failed")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    html_path = Path(args.html)
    vercel_path = Path(args.vercel)
    report: dict[str, Any] = {
        "html": str(html_path),
        "vercel": str(vercel_path),
        "passed": False,
        "checks": [],
        "headers": {},
        "served": None,
        "browser": None,
    }
    findings: list[Finding] = []

    if not html_path.is_file():
        print(f"error: html file not found: {html_path}")
        return EXIT_USAGE
    try:
        vercel_cfg: Any = json.loads(vercel_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {vercel_path}: {exc}")
        return EXIT_USAGE

    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    findings.append(Finding("html.exists", True, f"{html_path} ({len(html_text)} characters)"))
    findings.extend(check_static(html_text, vercel_cfg))
    paths = ["/", "/index.html"] if html_path.name == "index.html" else ["/" + html_path.name]
    report["headers"] = dict(headers_for(vercel_cfg, paths[-1]))

    browser_message: str | None = None
    with serve_with_headers(html_path.parent, vercel_cfg) as server:
        findings.extend(check_served(server.base_url, vercel_cfg, paths))
        if args.browser:
            try:
                browser_findings, browser_info = check_browser(
                    server.base_url + paths[-1],
                    screenshot=args.screenshot,
                    executable_path=args.executable_path,
                    color_scheme=args.color_scheme,
                )
                findings.extend(browser_findings)
                report["browser"] = browser_info
            except BrowserUnavailable as exc:
                browser_message = str(exc)
                report["browser"] = {"available": False, "message": browser_message}
        report["served"] = {"base_url": server.base_url, "requests": list(server.request_log)}

    all_ok = all(f.ok for f in findings)
    report["checks"] = [asdict(f) for f in findings]
    report["passed"] = all_ok and browser_message is None
    if browser_message:
        print(browser_message, file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text_report(findings, report)
    if not all_ok:
        return EXIT_FAIL
    return EXIT_BROWSER_UNAVAILABLE if browser_message else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
