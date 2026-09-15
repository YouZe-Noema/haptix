"""PEP 561 typing marker and packaging declarations."""

from pathlib import Path

import haptix


def _load_pyproject() -> dict:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = Path(haptix.__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        return tomllib.load(f)


def test_py_typed_exists_and_is_empty():
    """haptix/py.typed must exist as an empty PEP 561 marker file."""
    marker = Path(haptix.__file__).resolve().parent / "py.typed"
    assert marker.is_file()
    assert marker.stat().st_size == 0


def test_pyproject_declares_py_typed_package_data():
    """Wheels must include py.typed via setuptools package-data."""
    data = _load_pyproject()
    package_data = data.get("tool", {}).get("setuptools", {}).get("package-data", {})
    assert package_data.get("haptix") == ["py.typed"]
