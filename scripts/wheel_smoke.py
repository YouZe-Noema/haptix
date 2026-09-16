"""Wheel install smoke test for a core-only haptix install (no extras).

Guards against shipping a wheel that passes ``twine check`` but is missing
``py.typed``, a subpackage, an unresolvable ``__all__`` export, a broken
``haptix-browser`` console script registration, or a save/load round-trip that
fails without optional extras. Intended to run against an installed wheel in a
clean venv (see the ``package`` job in ``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys
import tempfile

import numpy as np

import haptix
from haptix.core import InteractionMeta, Labels, RawData, SensorMeta


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def check_version() -> None:
    version = haptix.__version__
    if not isinstance(version, str) or not version:
        raise AssertionError(f"haptix.__version__ must be a non-empty str, got {version!r}")
    dist_version = importlib.metadata.version("haptix")
    if version != dist_version:
        raise AssertionError(
            f"haptix.__version__={version!r} != importlib.metadata.version={dist_version!r}"
        )
    _ok(f"version={version!r} matches installed distribution")


def check_py_typed() -> None:
    marker = pathlib.Path(haptix.__file__).parent / "py.typed"
    if not marker.is_file():
        raise AssertionError(f"PEP 561 marker missing: {marker}")
    _ok(f"py.typed present at {marker}")


def check_all_exports() -> None:
    missing = [name for name in haptix.__all__ if not hasattr(haptix, name)]
    if missing:
        raise AssertionError(f"__all__ names missing on haptix: {missing}")
    for name in haptix.__all__:
        getattr(haptix, name)
    _ok(f"all {len(haptix.__all__)} names in haptix.__all__ resolve")


def check_console_script() -> None:
    eps = importlib.metadata.entry_points(group="console_scripts")
    names = {ep.name for ep in eps}
    if "haptix-browser" not in names:
        raise AssertionError(
            f"console_scripts entry point 'haptix-browser' not registered; found {sorted(names)}"
        )
    _ok("console_scripts entry point 'haptix-browser' is registered")


def check_roundtrip() -> None:
    frames = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)
    original = haptix.HaptData(
        raw=RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype="uint8",
            shape=frames.shape,
        ),
        sensor=SensorMeta(type="DIGIT_v2"),
        modality="imaging",
        sampling_rate_hz=60.0,
        interaction=InteractionMeta(type="sliding", speed_mm_s=50.0, normal_force_N=2.0),
        labels=Labels(material="smoke_test", task="sliding"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "smoke.hapt"
        saved = haptix.save(original, path)
        loaded = haptix.load(saved)
        if loaded.raw.checksum != original.raw.checksum:
            raise AssertionError(
                f"checksum mismatch after round-trip: "
                f"{original.raw.checksum!r} -> {loaded.raw.checksum!r}"
            )
        if not np.array_equal(loaded.raw.array, original.raw.array):
            raise AssertionError("raw array mismatch after round-trip")
    _ok("save/load round-trip preserves raw checksum (core deps only)")


def main() -> int:
    checks = (
        check_version,
        check_py_typed,
        check_all_exports,
        check_console_script,
        check_roundtrip,
    )
    for check in checks:
        check()
    print(f"OK: wheel smoke passed ({len(checks)} checks)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
