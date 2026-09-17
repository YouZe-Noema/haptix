"""Docs-coverage guard: public API names must appear in ``docs/api.md``.

Fails if ``haptix.__all__`` or flagship-class public methods/properties stop
being mentioned, so the API reference cannot silently rot. Also walks every
``haptix/**/*.py`` top-level public class/function via ``ast`` and requires
each name in the docs (or an explicit allowlist reason). Locate the docs
file via ``Path(__file__)``, not the process cwd.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import haptix
from haptix.core import HaptData, Provenance, RawData, SensorMeta, Source
from haptix.encoders import SensorEncoder
from haptix.recorder import HaptRecorder
from haptix.streaming import HaptArchive
from haptix.torch_dataset import WindowedDataset

# Intentionally not required in docs/api.md. Each entry needs a one-line reason.
_ALLOWLIST: dict[str, str] = {
    "page_gallery": "Streamlit page entrypoint in haptix.browser.app, not a library API",
    "page_compare": "Streamlit page entrypoint in haptix.browser.app, not a library API",
    "main": "CLI/app entrypoints (haptix.browser.cli / .app), not a library API",
    "is_hapt_path": "internal browser path predicate; find_hapt_files is the documented surface",
}

_FLAGSHIP_CLASSES = (
    HaptData,
    RawData,
    SensorMeta,
    Provenance,
    Source,
    HaptArchive,
    WindowedDataset,
    HaptRecorder,
    SensorEncoder,
)

_DOCS_PATH = Path(__file__).resolve().parents[1] / "docs" / "api.md"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "haptix"


def _public_members(cls) -> list[str]:
    """Public methods and properties (names not starting with ``_``)."""
    names: list[str] = []
    for name, obj in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if name in _ALLOWLIST:
            continue
        static = inspect.getattr_static(cls, name, None)
        if inspect.isroutine(obj) or isinstance(static, (property, staticmethod, classmethod)):
            names.append(name)
    return sorted(names)


def _top_level_public_names(package_root: Path) -> set[str]:
    """Collect top-level public ClassDef / FunctionDef names under *package_root*."""
    names: set[str] = set()
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("_"):
                names.add(node.name)
    return names


@pytest.fixture(scope="module")
def api_docs() -> str:
    assert _DOCS_PATH.is_file(), f"missing API docs at {_DOCS_PATH}"
    return _DOCS_PATH.read_text(encoding="utf-8")


def test_allowlist_entries_have_reasons():
    """Bare allowlist keys are forbidden — every skip needs a judgement note."""
    for name, reason in _ALLOWLIST.items():
        assert isinstance(reason, str) and reason.strip(), f"allowlist[{name!r}] needs a reason"


def test_all_exports_mentioned_in_api_docs(api_docs: str):
    missing = [name for name in haptix.__all__ if name not in api_docs and name not in _ALLOWLIST]
    assert not missing, "Names in haptix.__all__ missing from docs/api.md:\n  - " + "\n  - ".join(
        missing
    )


@pytest.mark.parametrize("cls", _FLAGSHIP_CLASSES, ids=lambda c: c.__name__)
def test_flagship_public_members_mentioned_in_api_docs(cls, api_docs: str):
    missing = [name for name in _public_members(cls) if name not in api_docs]
    assert (
        not missing
    ), f"{cls.__name__} public members missing from docs/api.md:\n  - " + "\n  - ".join(missing)


def test_package_public_names_mentioned_in_api_docs(api_docs: str):
    """Every top-level public class/function under haptix/ must appear in api.md."""
    names = _top_level_public_names(_PACKAGE_ROOT)
    missing = sorted(n for n in names if n not in api_docs and n not in _ALLOWLIST)
    assert (
        not missing
    ), "Public top-level names under haptix/ missing from docs/api.md:\n  - " + "\n  - ".join(
        missing
    )
