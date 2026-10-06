"""Compatibility and portability guards.

* ``demandcast/`` imports only the standard library and NumPy (the runtime-dependency promise).
* Nothing in the repository's Python code relies on Python 3.11+ syntax or names (CI runs 3.10).
* No tracked text file carries private harness paths, virtualenv paths, session identifiers or
  creator/model attribution lines (public-repository hygiene).

The scanner helpers are exercised on synthetic snippets as well, so a regression in the guards
themselves is caught - not only a regression in the code they guard.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "demandcast"
ALLOWED_THIRD_PARTY = {"numpy"}
FIRST_PARTY = {"demandcast"}

# Modules and attributes that only exist from Python 3.11 (or later) onwards.
PY311_MODULES = {"tomllib", "wsgiref.types", "string.templatelib"}
PY311_NAMES: dict[str, set[str]] = {
    "typing": {
        "Self",
        "override",
        "LiteralString",
        "Never",
        "TypeVarTuple",
        "Unpack",
        "Required",
        "NotRequired",
        "assert_never",
        "assert_type",
        "reveal_type",
        "dataclass_transform",
        "TypeAliasType",
        "ReadOnly",
        "TypeIs",
        "NoDefault",
        "get_protocol_members",
        "is_protocol",
    },
    "datetime": {"UTC"},
    "enum": {
        "StrEnum",
        "ReprEnum",
        "verify",
        "member",
        "nonmember",
        "global_enum",
        "EnumCheck",
        "FlagBoundary",
        "show_flag_values",
    },
    "asyncio": {"TaskGroup", "timeout", "timeout_at", "Runner", "Barrier"},
    "contextlib": {"chdir"},
    "hashlib": {"file_digest"},
    "itertools": {"batched"},
    "math": {"cbrt", "exp2", "sumprod", "fma"},
    "operator": {"call"},
    "inspect": {"getmembers_static", "markcoroutinefunction", "BufferFlags"},
    "logging": {"getLevelNamesMapping", "getHandlerByName", "getHandlerNames"},
    "statistics": {"kde", "kde_random"},
    "warnings": {"deprecated"},
    "os": {"process_cpu_count"},
    "pathlib": {"UnsupportedOperation"},
    "sqlite3": {"Blob"},
    "sys": {"exception", "monitoring", "activate_stack_trampoline"},
}
BUILTIN_PY311_NAMES = {"ExceptionGroup", "BaseExceptionGroup", "PythonFinalizationError"}

# Fragments are assembled from parts so this file does not trip its own scan.
FORBIDDEN_FRAGMENTS: dict[str, str] = {
    "private harness path": "/home/" + "user",
    "virtualenv binary path": ".venv" + "/bin",
    "session identifier": "session" + "_id",
    "commit attribution trailer": "Co-Authored" + "-By",
    "model/creator attribution (vendor)": "anthr" + "opic",
    "model/creator attribution (assistant)": "cla" + "ude",
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".sql",
    ".html",
    ".csv",
    ".json",
    ".txt",
    ".cfg",
    ".ini",
}
TEXT_NAMES = {"Makefile", "LICENSE", "CONTRIBUTING", ".gitignore", "py.typed"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "out",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".vercel",
}
SKIP_RELATIVE_DIRS = {"benchmarks/results"}


# ---- scanner helpers ------------------------------------------------------------------------
def third_party_imports(tree: ast.AST) -> set[str]:
    """Top-level names of absolute imports that are neither standard library nor first-party."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    return {name for name in found if name not in stdlib and name not in FIRST_PARTY}


def python311_findings(source: str, filename: str = "<snippet>") -> list[str]:
    """Human-readable findings for every Python 3.11+-only construct in ``source``.

    Syntax is checked by parsing with ``feature_version=(3, 10)`` (rejects ``except*``, the
    ``type`` statement and PEP 695 type parameters); names are checked against
    ``PY311_MODULES`` / ``PY311_NAMES`` / ``BUILTIN_PY311_NAMES`` through imports, module
    aliases (``import datetime as dt`` -> ``dt.UTC``) and bare names.
    """
    try:
        tree = ast.parse(source, filename=filename, feature_version=(3, 10))
    except SyntaxError as exc:
        return [f"{filename}:{exc.lineno}: not valid Python 3.10 syntax ({exc.msg})"]
    findings: list[str] = []
    aliases: dict[str, str] = {}  # local name -> module it refers to
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if alias.name in PY311_MODULES or top in PY311_MODULES:
                    findings.append(f"{filename}:{node.lineno}: module {alias.name} is 3.11+")
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    aliases[top] = top
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            module = node.module
            if module in PY311_MODULES or module.split(".")[0] in PY311_MODULES:
                findings.append(f"{filename}:{node.lineno}: module {module} is 3.11+")
            for alias in node.names:
                if alias.name in PY311_NAMES.get(module, set()):
                    findings.append(f"{filename}:{node.lineno}: {module}.{alias.name} is 3.11+")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = aliases.get(node.value.id, "")
            if node.attr in PY311_NAMES.get(module, set()):
                findings.append(f"{filename}:{node.lineno}: {module}.{node.attr} is 3.11+")
        elif isinstance(node, ast.Name) and node.id in BUILTIN_PY311_NAMES:
            findings.append(f"{filename}:{node.lineno}: {node.id} is 3.11+")
    return findings


def portability_hits(text: str) -> list[tuple[int, str, str]]:
    """``(line number, label, line text)`` for every forbidden fragment found in ``text``."""
    hits: list[tuple[int, str, str]] = []
    lowered = {label: fragment.lower() for label, fragment in FORBIDDEN_FRAGMENTS.items()}
    for lineno, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        hits.extend(
            (lineno, label, line.strip()[:120])
            for label, fragment in lowered.items()
            if fragment in low
        )
    return hits


def tracked_text_files(root: Path) -> list[Path]:
    """Text files under ``root`` that would be committed (symlinks and build dirs excluded)."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in SKIP_DIRS
            and not d.endswith(".egg-info")
            and str(rel_dir / d) not in SKIP_RELATIVE_DIRS
            and not (Path(dirpath) / d).is_symlink()
        )
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            if path.suffix in TEXT_SUFFIXES or name in TEXT_NAMES:
                files.append(path)
    return files


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


PACKAGE_FILES = sorted(PACKAGE.rglob("*.py"))
OTHER_PYTHON_FILES = sorted(
    [
        *(ROOT / "tests").glob("*.py"),
        *(ROOT / "scripts").glob("*.py"),
        *(ROOT / "benchmarks").glob("*.py"),
    ]
)


# ---- guards over the real tree --------------------------------------------------------------
@pytest.mark.parametrize("path", PACKAGE_FILES, ids=_rel)
def test_package_imports_only_stdlib_and_numpy(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert third_party_imports(tree) <= ALLOWED_THIRD_PARTY


@pytest.mark.parametrize("path", PACKAGE_FILES, ids=_rel)
def test_package_has_no_python311_only_constructs(path):
    assert python311_findings(path.read_text(encoding="utf-8"), _rel(path)) == []


@pytest.mark.parametrize("path", OTHER_PYTHON_FILES, ids=_rel)
def test_tests_scripts_and_benchmarks_are_python310_compatible(path):
    assert python311_findings(path.read_text(encoding="utf-8"), _rel(path)) == []


def test_no_private_paths_or_attribution_in_tracked_text_files():
    files = tracked_text_files(ROOT)
    assert any(f.name == "ci.yml" for f in files)  # the walk reaches .github/workflows
    assert any(f.name == "vercel.json" for f in files)
    hits: list[tuple[str, int, str, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits.extend((_rel(path), *hit) for hit in portability_hits(text))
    assert hits == []


# ---- the scanners themselves (synthetic inputs) --------------------------------------------
def test_third_party_scanner_detects_non_numpy_imports():
    tree = ast.parse(
        "import numpy as np\n"
        "import pandas\n"
        "from scipy import stats\n"
        "from . import db\n"
        "from demandcast.db import connect\n"
        "import os.path\n"
        "from collections.abc import Mapping\n"
    )
    assert third_party_imports(tree) == {"numpy", "pandas", "scipy"}
    assert third_party_imports(tree) - ALLOWED_THIRD_PARTY == {"pandas", "scipy"}


@pytest.mark.parametrize(
    ("snippet", "fragment"),
    [
        ("import tomllib\n", "tomllib"),
        ("from typing import Self\n", "typing.Self"),
        ("import typing\n\nx = typing.override\n", "typing.override"),
        ("import datetime as dt\n\nnow = dt.datetime.now(dt.UTC)\n", "datetime.UTC"),
        ("from enum import StrEnum\n", "enum.StrEnum"),
        ("try:\n    pass\nexcept* ValueError:\n    pass\n", "3.10"),
        ("raise ExceptionGroup('x', [])\n", "ExceptionGroup"),
        ("type Alias = int\n", "3.10"),
        ("def f[T](x: T) -> T:\n    return x\n", "3.10"),
        ("from itertools import batched\n", "itertools.batched"),
    ],
    ids=[
        "tomllib",
        "typing_self",
        "typing_override_attr",
        "datetime_utc_alias",
        "str_enum",
        "except_star",
        "exception_group",
        "type_statement",
        "type_params",
        "itertools_batched",
    ],
)
def test_python311_scanner_flags_each_construct(snippet, fragment):
    findings = python311_findings(snippet)
    assert findings != []
    assert any(fragment in f for f in findings)


def test_python311_scanner_accepts_python310_code():
    snippet = (
        "from __future__ import annotations\n"
        "import datetime\n"
        "from typing import Any\n"
        "\n"
        "match 1:\n"
        "    case 1:\n"
        "        pass\n"
        "now = datetime.datetime.now(datetime.timezone.utc)\n"
        "x: int | None = None\n"
    )
    assert python311_findings(snippet) == []


def test_portability_scanner_detects_each_marker_case_insensitively():
    for label, fragment in FORBIDDEN_FRAGMENTS.items():
        hits = portability_hits(f"line one\nsee {fragment.upper()} here\n")
        assert [(h[0], h[1]) for h in hits] == [(2, label)]
    assert portability_hits("clean text\nnothing to see\n") == []
