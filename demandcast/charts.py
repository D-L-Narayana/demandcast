"""Pure, deterministic SVG chart helpers for the zero-JavaScript dashboard.

Every helper returns one ``<svg role="img" aria-label="…">`` string (``""`` for empty input),
escapes all text with :func:`html.escape`, and never emits scripts, external references or
``url(…)``. Colours are plain presentation attributes; a few stable CSS classes (``chart``,
``tick``, ``grid``, ``shade``, ``band``, ``vline``, ``s<i>``) let the embedding page re-colour
charts (e.g. dark mode) because CSS rules always override presentation attributes.

Coordinate mapping for ``n`` points inside the plot area ``[pad_l, width - pad_r]`` by
``[pad_t, pad_t + h]``::

    x(i) = pad_l + i · (width - pad_l - pad_r) / max(n - 1, 1)
    y(v) = pad_t + h - (v - lo) / (hi - lo) · h

with ``lo = 1.1 · min(0, min v)`` and ``hi = 1.1 · max(0, max v)`` (``hi = lo + 1`` when flat),
so the zero line is always inside the frame and the top has 10 % head-room.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence

RED = "#CC0000"
MAX_X_TICKS = 8
PALETTE = (RED, "#2563eb", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#6b7280")
_GRID = "#e5e7eb"
_SHADE = "#fde68a"
_MAX_ARIA_ITEMS = 12

Series = tuple[str, Sequence["float | None"], str]
Band = tuple[Sequence["float | None"], Sequence["float | None"]]


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _finite(value: object) -> float | None:
    """Coerce to float; ``None``/non-numeric/NaN/±inf become ``None`` (a gap)."""
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _num(v: float) -> str:
    """Compact deterministic number label: ``1,234`` · ``12.5`` · ``0.75``."""
    a = abs(v)
    if a >= 100 or v == int(v):
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:,.1f}"
    return f"{v:,.2f}"


def _scale(values: Sequence[float]) -> tuple[float, float]:
    """Axis bounds ``(lo, hi)``: ``lo = 1.1·min(0, min v)``, ``hi = 1.1·max(0, max v)``."""
    if not values:
        return 0.0, 1.0
    lo = min(0.0, min(values)) * 1.1
    hi = max(0.0, max(values)) * 1.1
    if hi - lo <= 0:
        hi = lo + 1.0
    return lo, hi


def _runs(flags: Sequence[bool]) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` index runs where ``flags`` are truthy."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _truncate(label: str, n: int) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def _open(width: int, height: float, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height:.0f}" class="chart" role="img" '
        f'aria-label="{_esc(label)}">'
    )


def _grid_lines(lo: float, hi: float, pad_l: int, pad_t: int, w: float, h: float) -> str:
    parts = []
    for k in range(5):
        gy = pad_t + k * h / 4
        val = hi - k * (hi - lo) / 4
        parts.append(
            f'<line class="grid" x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + w:.1f}" y2="{gy:.1f}" '
            f'stroke="{_GRID}"/>'
            f'<text class="tick" x="{pad_l - 6}" y="{gy + 4:.1f}" text-anchor="end">'
            f"{_num(val)}</text>"
        )
    return "".join(parts)


def _legend(items: Sequence[tuple[str, str]], x0: float, y0: float) -> str:
    parts = []
    cx = x0
    for label, colour in items:
        parts.append(
            f'<rect class="lg" x="{cx:.1f}" y="{y0 - 9}" width="10" height="10" rx="2" '
            f'fill="{_esc(colour)}"/>'
            f'<text class="tick" x="{cx + 14:.1f}" y="{y0}">{_esc(label)}</text>'
        )
        cx += 14 + 6.2 * len(label) + 16
    return "".join(parts)


def _aria_items(labels: Sequence[str], values: Sequence[float]) -> str:
    pairs = [f"{lab}: {_num(v)}" for lab, v in zip(labels, values, strict=True)]
    if len(pairs) > _MAX_ARIA_ITEMS:
        pairs = [*pairs[:_MAX_ARIA_ITEMS], "…"]
    return "; ".join(pairs)


def _colour_for(
    colours: Sequence[str] | Mapping[str, str] | None, index: int, key: str, default: str
) -> str:
    if colours is None:
        return default
    if isinstance(colours, Mapping):
        return str(colours.get(key, default))
    return str(colours[index]) if index < len(colours) else default


def tick_indices(n: int, max_ticks: int = MAX_X_TICKS) -> list[int]:
    """Indices of at most ``max_ticks`` evenly spaced x-axis labels for ``n`` points.

    Always includes the first and the last point::

        idx_k = round(k · (n - 1) / (m - 1)),  k = 0 … m - 1,  m = min(max_ticks, n)
    """
    if n <= 0:
        return []
    m = max(1, min(max_ticks, n))
    if m == 1:
        return [0]
    out: list[int] = []
    for k in range(m):
        i = round(k * (n - 1) / (m - 1))
        if not out or i != out[-1]:
            out.append(i)
    return out


def line_chart(
    series: Sequence[Series],
    *,
    x_labels: Sequence[str] | None = None,
    band: Band | None = None,
    shade: Sequence[bool] | None = None,
    vline: int | None = None,
    width: int = 880,
    height: int = 260,
    aria_label: str = "",
) -> str:
    """Multi-series line chart.

    ``series`` is ``[(label, values, colour)]``; a ``None`` value is a gap, so each series is
    drawn as one ``<polyline>`` per contiguous run (isolated points become small circles).
    ``band=(lower, upper)`` adds a translucent polygon over the runs where both are known.
    ``shade`` (one flag per point, e.g. promotion days) adds background spans. ``x_labels`` are
    thinned to at most :data:`MAX_X_TICKS` ticks (first and last always kept). ``vline`` draws a
    dashed separator at that point index (e.g. the forecast cutoff).
    """
    clean: list[tuple[str, list[float | None], str]] = [
        (str(lab), [_finite(v) for v in vals], str(col)) for lab, vals, col in series
    ]
    n = max((len(vals) for _, vals, _ in clean), default=0)
    if n == 0:
        return ""
    numbers = [v for _, vals, _ in clean for v in vals if v is not None]
    lower: list[float | None] = []
    upper: list[float | None] = []
    if band is not None:
        lower = [_finite(v) for v in band[0]]
        upper = [_finite(v) for v in band[1]]
        numbers += [v for v in lower + upper if v is not None]
    lo, hi = _scale(numbers)
    pad_l, pad_r, pad_t = 46, 12, 26
    pad_b = 30 if x_labels else 14
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    step = w / max(n - 1, 1)

    def x(i: int) -> float:
        return pad_l + i * step

    def y(v: float) -> float:
        return pad_t + h - (v - lo) / (hi - lo) * h

    label = aria_label or (
        "Line chart of " + " and ".join(lab for lab, _, _ in clean) + f", {n} points"
    )
    out = [_open(width, height, label)]
    if shade:
        for a, b in _runs([bool(f) for f in list(shade)[:n]]):
            x0 = max(float(pad_l), x(a) - step / 2)
            x1 = min(float(pad_l + w), x(b) + step / 2)
            out.append(
                f'<rect class="shade" x="{x0:.1f}" y="{pad_t}" width="{x1 - x0:.1f}" '
                f'height="{h:.1f}" fill="{_SHADE}" fill-opacity="0.45"/>'
            )
    out.append(_grid_lines(lo, hi, pad_l, pad_t, w, h))
    if band is not None:
        known = [
            i < len(lower) and i < len(upper) and lower[i] is not None and upper[i] is not None
            for i in range(n)
        ]
        for a, b in _runs(known):
            if a == b:
                continue
            top = " ".join(f"{x(i):.1f},{y(upper[i] or 0.0):.1f}" for i in range(a, b + 1))
            bottom = " ".join(f"{x(i):.1f},{y(lower[i] or 0.0):.1f}" for i in range(b, a - 1, -1))
            out.append(
                f'<polygon class="band" points="{top} {bottom}" fill="{RED}" '
                f'fill-opacity="0.12" stroke="none"/>'
            )
    for si, (lab, vals, col) in enumerate(clean):
        parts = [f'<g class="s{si}"><title>{_esc(lab)}</title>']
        segment: list[tuple[int, float]] = []
        segments: list[list[tuple[int, float]]] = []
        for i, v in enumerate(vals):
            if v is None:
                if segment:
                    segments.append(segment)
                    segment = []
            else:
                segment.append((i, v))
        if segment:
            segments.append(segment)
        for seg in segments:
            if len(seg) == 1:
                i, v = seg[0]
                parts.append(
                    f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.5" fill="{_esc(col)}"/>'
                )
            else:
                pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in seg)
                parts.append(
                    f'<polyline points="{pts}" fill="none" stroke="{_esc(col)}" stroke-width="2"/>'
                )
        parts.append("</g>")
        out.append("".join(parts))
    if vline is not None and 0 <= vline < n:
        out.append(
            f'<line class="vline" x1="{x(vline):.1f}" y1="{pad_t}" x2="{x(vline):.1f}" '
            f'y2="{pad_t + h:.1f}" stroke="#9ca3af" stroke-dasharray="4 3"/>'
        )
    if x_labels:
        labels = [str(t) for t in list(x_labels)[:n]]
        idx = tick_indices(len(labels))
        for j, i in enumerate(idx):
            if len(idx) == 1:
                anchor = "middle"
            elif j == 0:
                anchor = "start"
            elif j == len(idx) - 1:
                anchor = "end"
            else:
                anchor = "middle"
            out.append(
                f'<line class="grid" x1="{x(i):.1f}" y1="{pad_t + h:.1f}" x2="{x(i):.1f}" '
                f'y2="{pad_t + h + 4:.1f}" stroke="{_GRID}"/>'
                f'<text class="tick xt" x="{x(i):.1f}" y="{pad_t + h + 16:.1f}" '
                f'text-anchor="{anchor}">{_esc(labels[i])}</text>'
            )
    out.append(_legend([(lab, col) for lab, _, col in clean], pad_l, 14))
    out.append("</svg>")
    return "".join(out)


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    colour: str = RED,
    width: int = 880,
    height: int = 220,
    aria_label: str = "",
) -> str:
    """Vertical bars (one colour) with value labels; negative values hang below the zero line."""
    labs = [str(lab) for lab in labels]
    vals = [_finite(v) or 0.0 for v in values]
    if len(labs) != len(vals):
        raise ValueError("labels and values must have the same length")
    n = len(vals)
    if n == 0:
        return ""
    lo, hi = _scale(vals)
    pad_l, pad_r, pad_t, pad_b = 46, 12, 18, 36
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    slot = w / n
    bw = slot * 0.7

    def y(v: float) -> float:
        return pad_t + h - (v - lo) / (hi - lo) * h

    label = aria_label or ("Bar chart — " + _aria_items(labs, vals))
    out = [_open(width, height, label), _grid_lines(lo, hi, pad_l, pad_t, w, h)]
    for i, (lab, v) in enumerate(zip(labs, vals, strict=True)):
        bx = pad_l + i * slot + (slot - bw) / 2
        top, bottom = y(max(v, 0.0)), y(min(v, 0.0))
        text_y = top - 4 if v >= 0 else bottom + 11
        out.append(
            f'<rect class="bar" x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" '
            f'height="{bottom - top:.1f}" fill="{_esc(colour)}" rx="2">'
            f"<title>{_esc(lab)}: {v:,.1f}</title></rect>"
            f'<text class="tick" x="{bx + bw / 2:.1f}" y="{pad_t + h + 16}" '
            f'text-anchor="middle">{_esc(_truncate(lab, 14))}</text>'
            f'<text class="tick" x="{bx + bw / 2:.1f}" y="{text_y:.1f}" text-anchor="middle">'
            f"{_num(v)}</text>"
        )
    out.append("</svg>")
    return "".join(out)


def hbar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    colours: Sequence[str] | Mapping[str, str] | None = None,
    width: int = 880,
    aria_label: str = "",
) -> str:
    """Horizontal bars, one row per label; ``colours`` is per bar (sequence) or by label (map).

    The height grows with the number of rows (26 px each); negative values draw as zero-width.
    """
    labs = [str(lab) for lab in labels]
    vals = [_finite(v) or 0.0 for v in values]
    if len(labs) != len(vals):
        raise ValueError("labels and values must have the same length")
    n = len(vals)
    if n == 0:
        return ""
    row, pad_l, pad_r, pad_t, pad_b = 26, 130, 70, 8, 8
    height = pad_t + n * row + pad_b
    w = width - pad_l - pad_r
    vmax = max(max(vals), 0.0) or 1.0
    label = aria_label or ("Horizontal bar chart — " + _aria_items(labs, vals))
    out = [_open(width, height, label)]
    for i, (lab, v) in enumerate(zip(labs, vals, strict=True)):
        by = pad_t + i * row
        bwidth = max(v, 0.0) / vmax * w
        colour = _colour_for(colours, i, lab, RED)
        out.append(
            f'<text class="tick" x="{pad_l - 8}" y="{by + 17:.1f}" text-anchor="end">'
            f"{_esc(_truncate(lab, 20))}</text>"
            f'<rect class="bar" x="{pad_l}" y="{by + 4:.1f}" width="{bwidth:.1f}" '
            f'height="{row - 8}" fill="{_esc(colour)}" rx="2">'
            f"<title>{_esc(lab)}: {v:,.1f}</title></rect>"
            f'<text class="tick" x="{pad_l + bwidth + 6:.1f}" y="{by + 17:.1f}">{_num(v)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def stacked_bar(
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    colours: Sequence[str] | Mapping[str, str] | None,
    width: int = 880,
    height: int = 220,
    aria_label: str = "",
) -> str:
    """Stacked vertical bars: one stack per label, one segment per series (negatives clamp to 0).

    Segment height is ``v / hi · h`` where ``hi = 1.1 · max stack total``, so the segments of a
    stack sum to the total's height. ``colours`` is per series (sequence, in ``series`` order) or
    by name (map); ``None`` cycles :data:`PALETTE`.
    """
    labs = [str(lab) for lab in labels]
    names = [str(k) for k in series]
    n = len(labs)
    if n == 0 or not names:
        return ""
    matrix: dict[str, list[float]] = {}
    for name, vals in series.items():
        column = [max(_finite(v) or 0.0, 0.0) for v in vals]
        if len(column) != n:
            raise ValueError(f"series {name!r} must have {n} values")
        matrix[str(name)] = column
    totals = [sum(matrix[name][i] for name in names) for i in range(n)]
    hi = (max(totals) * 1.1) or 1.0
    pad_l, pad_r, pad_t, pad_b = 46, 12, 26, 36
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    slot = w / n
    bw = slot * 0.7
    palette = [
        _colour_for(colours, i, name, PALETTE[i % len(PALETTE)]) for i, name in enumerate(names)
    ]
    label = aria_label or (
        f"Stacked bar chart of {', '.join(names)} by {', '.join(labs[:_MAX_ARIA_ITEMS])}"
    )
    out = [_open(width, height, label), _grid_lines(0.0, hi, pad_l, pad_t, w, h)]
    for i, lab in enumerate(labs):
        bx = pad_l + i * slot + (slot - bw) / 2
        cum = 0.0
        for name, colour in zip(names, palette, strict=True):
            v = matrix[name][i]
            seg_h = v / hi * h
            top = pad_t + h - (cum + v) / hi * h
            out.append(
                f'<rect class="stack" x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                f'height="{seg_h:.1f}" fill="{_esc(colour)}">'
                f"<title>{_esc(name)} — {_esc(lab)}: {v:,.1f}</title></rect>"
            )
            cum += v
        total_y = pad_t + h - totals[i] / hi * h - 4
        out.append(
            f'<text class="tick" x="{bx + bw / 2:.1f}" y="{pad_t + h + 16}" text-anchor="middle">'
            f"{_esc(_truncate(lab, 14))}</text>"
            f'<text class="tick" x="{bx + bw / 2:.1f}" y="{total_y:.1f}" text-anchor="middle">'
            f"{_num(totals[i])}</text>"
        )
    out.append(_legend(list(zip(names, palette, strict=True)), pad_l, 14))
    out.append("</svg>")
    return "".join(out)
