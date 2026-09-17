"""Docs-snippet guard: hands-on walkthrough Python must stay syntactically
valid and reference real ``haptix`` attributes.

Fails if ``docs/hands-on.md`` python fences rot (wrong top-level imports /
``haptix.NAME`` names) or if ``examples/walkthrough.py`` drifts from the
public API. Locate files via ``Path(__file__)``, not the process cwd.
Does not touch the network.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import haptix

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HANDS_ON = _REPO_ROOT / "docs" / "hands-on.md"
_WALKTHROUGH = _REPO_ROOT / "examples" / "walkthrough.py"

_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_HEREDOC_RE = re.compile(
    r"^\$PY\b[^\n]*<<['\"]?EOF['\"]?\s*\n(.*)\nEOF\s*$",
    re.DOTALL,
)
_FROM_HAPTIX_RE = re.compile(r"^\s*from\s+haptix\s+import\s+(.+)$", re.MULTILINE)
_HAPTIX_ATTR_RE = re.compile(r"\bhaptix\.([A-Za-z_][A-Za-z0-9_]*)")


def _unwrap_python_fence(block: str) -> str:
    """Return pure Python from a fence, stripping an optional ``$PY`` heredoc."""
    text = block.strip("\n")
    match = _HEREDOC_RE.match(text.strip())
    if match:
        return match.group(1)
    return text


def _extract_python_blocks(md: str) -> list[str]:
    return [_unwrap_python_fence(m.group(1)) for m in _FENCE_RE.finditer(md)]


def _names_from_import_clause(clause: str) -> list[str]:
    """Parse ``A, B as C, (D, E)`` style import tails into exported names."""
    names: list[str] = []
    # Drop parenthesized newlines for a flat split on commas.
    flat = " ".join(clause.replace("(", " ").replace(")", " ").split())
    for part in flat.split(","):
        part = part.strip()
        if not part or part == "*":
            continue
        if " as " in part:
            part = part.split(" as ", 1)[0].strip()
        names.append(part)
    return names


def _top_level_haptix_names(source: str) -> set[str]:
    names: set[str] = set()
    for match in _FROM_HAPTIX_RE.finditer(source):
        names.update(_names_from_import_clause(match.group(1)))
    names.update(_HAPTIX_ATTR_RE.findall(source))
    return names


def test_hands_on_python_blocks_parse_and_resolve():
    assert _HANDS_ON.is_file(), f"missing {_HANDS_ON}"
    blocks = _extract_python_blocks(_HANDS_ON.read_text(encoding="utf-8"))
    assert blocks, "docs/hands-on.md must contain fenced python blocks"

    missing: list[str] = []
    for i, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            raise AssertionError(f"hands-on python block {i} failed ast.parse: {exc}") from exc
        for name in _top_level_haptix_names(block):
            if not hasattr(haptix, name):
                missing.append(f"block {i}: haptix.{name}")

    assert not missing, "hands-on.md references missing haptix attributes:\n  - " + "\n  - ".join(
        missing
    )


def test_walkthrough_parses_and_resolves():
    assert _WALKTHROUGH.is_file(), f"missing {_WALKTHROUGH}"
    source = _WALKTHROUGH.read_text(encoding="utf-8")
    ast.parse(source)
    missing = sorted(name for name in _top_level_haptix_names(source) if not hasattr(haptix, name))
    assert (
        not missing
    ), "walkthrough.py references missing haptix attributes:\n  - " + "\n  - ".join(missing)
