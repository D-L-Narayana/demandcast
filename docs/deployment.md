# Deployment: the static dashboard on Vercel

The live dashboard (<https://demandcast-bay.vercel.app>) is nothing more than the committed file
`dashboard/index.html`, served as a static site by Vercel. There is no build step, no server
code, no secrets and no JavaScript: `demandcast dashboard` renders one self-contained HTML
document (inline CSS, inline SVG) from the SQL queries, the file is committed, and Vercel
redeploys it.

## Response headers (`vercel.json`)

`vercel.json` contains **only** a `headers` section (no routes, builds, rewrites, redirects or
functions), applied to every path (`"source": "/(.*)"`):

| Header | Value | Why |
|---|---|---|
| `Content-Security-Policy` | `default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'` | the page needs nothing but its own inline styles and `data:` images; everything else (scripts, fonts, frames, remote images, form posts) is refused by the browser |
| `X-Content-Type-Options` | `nosniff` | never reinterpret the document as another type |
| `X-Frame-Options` | `DENY` | legacy equivalent of `frame-ancestors 'none'` |
| `Referrer-Policy` | `no-referrer` | outbound links do not leak the dashboard URL |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=()` | no powerful features |
| `Cross-Origin-Opener-Policy` | `same-origin` | isolate the browsing context |
| `Cross-Origin-Resource-Policy` | `same-origin` | the document is not embeddable elsewhere |

The same policy, minus `frame-ancestors` (which a `<meta>` tag cannot express), is embedded in
the document itself as `<meta http-equiv="Content-Security-Policy" content="...">`, directly
after `<meta charset>`, so the file is protected even when opened from disk or served by a
host that strips headers. The two literals are frozen in code: `dashboard.CSP_META` and the
verifier's `CSP_META` / `CSP_HEADER`; tests fail if they drift apart.

## What the dashboard must satisfy

Because `default-src 'none'`, the renderer guarantees - and the verifier checks - that the page

* has the CSP meta tag with the exact policy, immediately after `<meta charset>`;
* declares `<link rel="icon" href="data:,">` (otherwise browsers request `/favicon.ico`, which
  shows up as a 404 in the console);
* contains no `<script>`, no `on*=` attributes, no `javascript:` URLs, no `<base>`;
* uses no CSS `url()` other than `data:` and no `@font-face` (system font stack only);
* loads no external resource: no `http(s)://` or `//` in `src`/`href` of `link`, `img`,
  `iframe`, `object`, `embed`, `source`, `video`, `audio`. Plain `<a href>` links are fine.

## Verifying locally

```bash
python scripts/verify_dashboard.py                                   # dashboard/index.html + vercel.json
python scripts/verify_dashboard.py --html out/index.html --json      # a freshly rendered file, JSON report
make verify-dashboard HTML=out/index.html
```

The script needs only the standard library and performs two groups of checks:

1. **static** - file exists and is larger than 1 KB, every obligation above, `vercel.json` is
   headers-only, its CSP equals the canonical literal; every configured header is echoed.
2. **served** - a local `http.server` on `127.0.0.1` serves the file's directory and applies the
   `vercel.json` headers exactly as Vercel does (`source` patterns, including `/(.*)` and literal
   paths). `GET /` and `GET /index.html` must return `200`, `Content-Type: text/html;
   charset=utf-8` and every header with its exact value. The request lines are reported.

### Browser mode

```bash
python scripts/verify_dashboard.py --browser --screenshot out/dashboard.png [--color-scheme dark]
```

With `--browser` the served page is loaded in headless Chromium through Playwright (which is
**not** a project dependency - install it separately when you want this check). The script
registers a `securitypolicyviolation` listener before the document loads and collects console
messages, page errors and every request URL. It asserts: zero CSP violations, zero console
errors or warnings, zero page errors, every request stays on the served origin,
`document.scripts.length == 0`, the response carried the canonical CSP header, and the headings
"Model leaderboard" and "Replenishment order book" are visible. `--screenshot PATH` saves a
full-page capture; `--executable-path` selects a Chromium binary; `--color-scheme light|dark`
exercises the dark theme.

If Playwright cannot be imported (or Chromium cannot start) the script prints
`browser check unavailable: playwright not installed` and exits with **3** - an unavailable
check is never reported as a pass.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | every performed check passed |
| 1 | at least one check failed (see the `FAIL` lines or `"ok": false` entries) |
| 2 | the HTML or `vercel.json` input is missing / unreadable, or a usage error |
| 3 | all performed checks passed but the requested browser check could not run |

## Release procedure for the dashboard

1. Run the pipeline on the dataset you want to publish (for the README numbers: the default
   10 x 40 x 730 dataset, `demandcast run --workers 2`).
2. `demandcast dashboard --out dashboard/index.html`.
3. `python scripts/verify_dashboard.py` (and `--browser` where Playwright is available) must
   exit 0.
4. Commit `dashboard/index.html` (and `vercel.json` if it changed). Vercel deploys the commit;
   the headers take effect immediately because they are part of the project configuration, not
   of a build.

CI performs steps 2-3 on a small synthetic dataset for every push (`out/index.html` is uploaded
as a workflow artifact), so a regression in the policy or the renderer is caught before it
reaches the live site.
