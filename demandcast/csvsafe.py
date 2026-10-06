"""Spreadsheet-safe cell policy for the *reporting* CSV exports.

Spreadsheet applications evaluate CSV text cells that begin with ``=``, ``+``, ``-`` or ``@``
(and some also strip leading tabs, carriage returns or invisible whitespace first), so a store
name or SKU such as ``=1+1`` or ``@SUM(A1:A9)`` that entered the database through CSV ingest
would turn into a live formula when a reporting export is opened in a spreadsheet. CSV quoting
does not prevent that.

The policy applied by ``demandcast export orders|forecasts|metrics`` (CSV) and by
``demandcast query --format csv`` when ``cells="safe"`` (the default):

* a **string** cell is unsafe when its first character is a tab / CR / LF, or when its first
  *visible* character — skipping whitespace and Unicode control (Cc), format (Cf) and separator
  (Zs/Zl/Zp) characters — is one of ``=``, ``+``, ``-``, ``@``;
* an unsafe string is written as ``'`` + original text (the conventional spreadsheet "treat as
  text" marker; some applications show the apostrophe);
* every other value is written unchanged — numbers stay numeric (``-0.14`` is a negative
  number, not a formula), dates start with a digit, ``None`` stays an empty field.

``cells="raw"`` writes every cell verbatim for machine consumers. ``export dataset`` is always
raw: it is the lossless machine-data round trip for ``demandcast load``. Database values are
never changed by any of this.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

#: Cell policies accepted by the reporting exports and ``query --format csv``.
CELL_POLICIES = ("safe", "raw")
#: First visible characters that spreadsheets interpret as the start of a formula.
TRIGGER_CHARS = frozenset("=+-@")
#: Leading characters that are a problem on their own (DDE / cell-break tricks).
LEADING_TRIGGERS = frozenset("\t\r\n")
#: Marker prepended to unsafe text cells.
PREFIX = "'"

_INVISIBLE_CATEGORIES = ("Cc", "Cf", "Zs", "Zl", "Zp")


def _is_invisible(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch) in _INVISIBLE_CATEGORIES


def is_unsafe_cell(text: str) -> bool:
    """True when a spreadsheet could interpret ``text`` as a formula (see module docstring)."""
    if not text:
        return False
    if text[0] in LEADING_TRIGGERS:
        return True
    for ch in text:
        if _is_invisible(ch):
            continue
        return ch in TRIGGER_CHARS
    return False


def safe_cell(value: Any) -> Any:
    """Return ``value`` made inert for spreadsheets: unsafe strings get :data:`PREFIX`.

    Non-string values (numbers, booleans, ``None``) are returned as they are; a string that is
    already prefixed is unsafe no longer (its first visible character is ``'``), so the function
    is idempotent.
    """
    if isinstance(value, str) and is_unsafe_cell(value):
        return PREFIX + value
    return value


def safe_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """New row dicts with every string cell passed through :func:`safe_cell` (key order kept)."""
    return [{k: safe_cell(v) for k, v in row.items()} for row in rows]


def check_policy(cells: str) -> str:
    """Validate a cell-policy name (``"safe"`` or ``"raw"``); raises ``ValueError`` otherwise."""
    if cells not in CELL_POLICIES:
        raise ValueError(f"cells must be one of {', '.join(CELL_POLICIES)} (got '{cells}')")
    return cells
