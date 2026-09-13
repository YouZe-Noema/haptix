"""
Tests for ``haptix.browser.cli`` (``haptix-browser`` entry point).

Does not launch Streamlit — stubs ``streamlit.web.cli.main`` so we can assert
argv rewriting, ``HAPTIX_BROWSER_ROOT`` export, and the ImportError hint path.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

import haptix.browser.cli as browser_cli


def _install_fake_stcli(monkeypatch: pytest.MonkeyPatch, fake_main):
    """Make ``from streamlit.web import cli`` resolve to a stub module."""
    st = types.ModuleType("streamlit")
    web = types.ModuleType("streamlit.web")
    cli_mod = types.ModuleType("streamlit.web.cli")
    cli_mod.main = fake_main
    web.cli = cli_mod
    st.web = web
    monkeypatch.setitem(sys.modules, "streamlit", st)
    monkeypatch.setitem(sys.modules, "streamlit.web", web)
    monkeypatch.setitem(sys.modules, "streamlit.web.cli", cli_mod)


def test_positional_data_dir_exported_and_remaining_args_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """First positional arg becomes HAPTIX_BROWSER_ROOT; rest go to streamlit."""
    data = tmp_path / "data"
    data.mkdir()
    recorded: dict = {}

    def fake_main():
        recorded["argv"] = list(sys.argv)
        return 0

    _install_fake_stcli(monkeypatch, fake_main)
    monkeypatch.delenv("HAPTIX_BROWSER_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    rc = browser_cli.main(["data", "--server.port", "8601"])
    assert rc == 0
    root = Path(os.environ["HAPTIX_BROWSER_ROOT"])
    assert root.is_absolute()
    assert root == data.resolve()
    assert recorded["argv"] == [
        "streamlit",
        "run",
        str(browser_cli.APP_FILE),
        "--server.port",
        "8601",
    ]


def test_tilde_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``~`` in DATA_DIR is expanded via Path.expanduser()."""
    home = tmp_path / "home"
    data = home / "episodes"
    data.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    recorded: dict = {}

    def fake_main():
        recorded["argv"] = list(sys.argv)
        return 0

    _install_fake_stcli(monkeypatch, fake_main)
    monkeypatch.delenv("HAPTIX_BROWSER_ROOT", raising=False)

    rc = browser_cli.main(["~/episodes"])
    assert rc == 0
    assert os.environ["HAPTIX_BROWSER_ROOT"] == str(data.resolve())
    assert recorded["argv"] == ["streamlit", "run", str(browser_cli.APP_FILE)]


def test_flag_first_leaves_env_untouched(monkeypatch: pytest.MonkeyPatch):
    """When first arg starts with ``-``, HAPTIX_BROWSER_ROOT is not set."""
    monkeypatch.delenv("HAPTIX_BROWSER_ROOT", raising=False)
    recorded: dict = {}

    def fake_main():
        recorded["argv"] = list(sys.argv)
        return 0

    _install_fake_stcli(monkeypatch, fake_main)

    rc = browser_cli.main(["--server.headless", "true"])
    assert rc == 0
    assert "HAPTIX_BROWSER_ROOT" not in os.environ
    assert recorded["argv"] == [
        "streamlit",
        "run",
        str(browser_cli.APP_FILE),
        "--server.headless",
        "true",
    ]


def test_no_args_uses_default_streamlit_argv(monkeypatch: pytest.MonkeyPatch):
    """Bare ``main([])`` → ``streamlit run <APP_FILE>`` only."""
    recorded: dict = {}

    def fake_main():
        recorded["argv"] = list(sys.argv)
        return 0

    _install_fake_stcli(monkeypatch, fake_main)
    monkeypatch.delenv("HAPTIX_BROWSER_ROOT", raising=False)

    rc = browser_cli.main([])
    assert rc == 0
    assert recorded["argv"] == ["streamlit", "run", str(browser_cli.APP_FILE)]


def test_streamlit_missing_returns_1_and_prints_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """ImportError on ``streamlit.web.cli`` → exit 1 + install hint on stderr."""
    import builtins

    real_import = builtins.__import__

    def boom(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError("no streamlit")
        return real_import(name, globals, locals, fromlist, level)

    for key in list(sys.modules):
        if key == "streamlit" or key.startswith("streamlit."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    monkeypatch.setattr(builtins, "__import__", boom)
    rc = browser_cli.main([])
    assert rc == 1
    err = capsys.readouterr().err
    assert "pip install 'haptix[browser]'" in err


def test_main_return_value_is_int(monkeypatch: pytest.MonkeyPatch):
    """``main()`` coerces ``stcli.main()`` result via ``int(...)``."""

    def fake_main():
        return 3

    _install_fake_stcli(monkeypatch, fake_main)
    assert browser_cli.main([]) == 3


def test_main_reads_sys_argv_when_argv_is_none(monkeypatch: pytest.MonkeyPatch):
    """``argv is None`` uses ``sys.argv[1:]``."""
    recorded: dict = {}

    def fake_main():
        recorded["argv"] = list(sys.argv)
        return 0

    _install_fake_stcli(monkeypatch, fake_main)
    monkeypatch.setattr(sys, "argv", ["haptix-browser", "--browser.gatherUsageStats", "false"])
    monkeypatch.delenv("HAPTIX_BROWSER_ROOT", raising=False)

    rc = browser_cli.main(None)
    assert rc == 0
    assert recorded["argv"] == [
        "streamlit",
        "run",
        str(browser_cli.APP_FILE),
        "--browser.gatherUsageStats",
        "false",
    ]
