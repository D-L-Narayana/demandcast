"""Spreadsheet-safe cell policy (demandcast.csvsafe) used by the reporting CSV exports."""

from __future__ import annotations

import csv
import io

import pytest

from demandcast import csvsafe

HOSTILE = [
    "=1+1",
    '=HYPERLINK("http://example.invalid","click")',
    "=cmd|' /C calc'!A0",
    "+1",
    "-1",
    "-2+3",
    "@SUM(A1:A9)",
    "\t=1+1",  # leading tab (DDE)
    "\r=1+1",  # leading carriage return
    "\n@x",  # leading newline
    "\t",  # a lone leading tab is a trigger on its own (cell-break / DDE tricks)
    " =1+1",  # leading space
    "   -5",  # several spaces
    " -5",  # no-break space
    "​=1+1",  # zero-width space (format char)
    "﻿=1+1",  # byte-order mark
    " +1",  # line separator
]

BENIGN = [
    "BLR",
    "Grocery item 7",
    "SKU-GRO-0007",
    "2026-01-16",
    "inventory position below reorder point; capped by shelf life",
    "1-2",
    "a=b",
    "x+y",
    "me@example.invalid",
    "ok, with comma",
    "multi\nline",
    'quote " inside',
    "",
    "   ",  # only whitespace: nothing to trigger
    "'already prefixed",
    "'=1+1",
]


@pytest.mark.parametrize("text", HOSTILE)
def test_hostile_string_cells_are_detected_and_prefixed(text):
    assert csvsafe.is_unsafe_cell(text) is True
    assert csvsafe.safe_cell(text) == csvsafe.PREFIX + text


@pytest.mark.parametrize("text", BENIGN)
def test_benign_string_cells_are_untouched(text):
    assert csvsafe.is_unsafe_cell(text) is False
    assert csvsafe.safe_cell(text) == text


@pytest.mark.parametrize("value", [-0.14, -5, 0, 1.5, 42, True, False, None])
def test_non_string_values_keep_their_type_and_value(value):
    out = csvsafe.safe_cell(value)
    assert out is value or out == value
    assert type(out) is type(value)


def test_safe_cell_is_idempotent():
    once = csvsafe.safe_cell("=1+1")
    assert once == "'=1+1"
    assert csvsafe.safe_cell(once) == once


def test_safe_rows_converts_only_string_cells_and_does_not_mutate_input():
    rows = [
        {"sku": "=1+1", "bias": -0.14, "order_qty": -5, "reason": "ok", "note": None},
        {"sku": "@x", "bias": 0.5, "order_qty": 3, "reason": "+late", "note": "\t=2"},
    ]
    snapshot = [dict(r) for r in rows]
    out = csvsafe.safe_rows(rows)
    assert out == [
        {"sku": "'=1+1", "bias": -0.14, "order_qty": -5, "reason": "ok", "note": None},
        {"sku": "'@x", "bias": 0.5, "order_qty": 3, "reason": "'+late", "note": "'\t=2"},
    ]
    assert rows == snapshot  # input untouched
    assert [list(r) for r in out] == [list(r) for r in rows]  # key order preserved


def test_policy_names_and_constants():
    assert csvsafe.CELL_POLICIES == ("safe", "raw")
    assert set("=+-@") == set(csvsafe.TRIGGER_CHARS)
    assert csvsafe.PREFIX == "'"


def test_csv_round_trip_keeps_numbers_numeric_and_quotes_special_characters():
    """End-to-end through the csv module: numbers stay bare, text keeps commas/newlines."""
    rows = csvsafe.safe_rows(
        [
            {
                "sku": "=1+1",
                "name": "ok, with comma",
                "note": "multi\nline",
                "bias": -0.14,
                "qty": -5,
            },
        ]
    )
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["sku", "name", "note", "bias", "qty"])
    writer.writeheader()
    writer.writerows(rows)
    text = buf.getvalue()
    lines = text.splitlines()
    assert lines[0] == "sku,name,note,bias,qty"
    assert lines[1].startswith("'=1+1,")
    assert ",-0.14,-5" in text  # legitimate negative numbers are not prefixed
    back = list(csv.DictReader(io.StringIO(text)))
    assert back == [
        {
            "sku": "'=1+1",
            "name": "ok, with comma",
            "note": "multi\nline",
            "bias": "-0.14",
            "qty": "-5",
        }
    ]
