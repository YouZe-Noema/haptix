"""
``haptix-browser`` entry point — launch the Streamlit tactile data browser.

Usage::

    haptix-browser [DATA_DIR] [streamlit args...]

The optional first positional argument is the directory to browse (defaults to
``HAPTIX_BROWSER_ROOT`` env var, then ``~/.haptix/cache/datasets`` if it
exists, then the current directory). It is handed to the app via the
``HAPTIX_BROWSER_ROOT`` environment variable. Remaining arguments are passed
through to ``streamlit run`` (e.g. ``--server.port 8601``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_FILE = Path(__file__).resolve().parent / "app.py"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    root: str | None = None
    if args and not args[0].startswith("-"):
        root = args.pop(0)
        os.environ["HAPTIX_BROWSER_ROOT"] = str(Path(root).expanduser().resolve())

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print(
            "haptix-browser requires streamlit. Install with:\n"
            "    pip install 'haptix[browser]'",
            file=sys.stderr,
        )
        return 1

    sys.argv = ["streamlit", "run", str(APP_FILE), *args]
    return int(stcli.main())


if __name__ == "__main__":
    raise SystemExit(main())
