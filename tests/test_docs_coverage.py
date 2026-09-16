"""Docs-coverage guard: public API names must appear in ``docs/api.md``.

Fails if ``haptix.__all__`` or flagship-class public methods/properties stop
being mentioned, so the API reference cannot silently rot. Locate the docs
file via ``Path(__file__)``, not the process cwd.
"""

from __future__ import annotations

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
    # Example (none currently): "SomeHelper": "internal helper re-exported for tests only",
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
