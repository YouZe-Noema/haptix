"""
Tactile data browser — interactive episode gallery for ``.hapt`` recordings.

The browser is the roadmap "Beyond" item "Tactile data browser / visualization
tool" (decision 2026-08-10: Streamlit web app, ``haptix[browser]`` extra).

Two layers:

- :mod:`haptix.browser.core` — pure numpy/Pillow library functions (discovery,
  summaries, frame/trace access). No streamlit dependency; testable in CI and
  reusable from notebooks/scripts.
- :mod:`haptix.browser.app` — the Streamlit UI (launch with ``haptix-browser``
  or ``python -m streamlit run haptix/browser/app.py``).

Public API (re-exported here and at the ``haptix`` top level):
:func:`find_hapt_files`, :func:`scan_directory`, :func:`episode_summary`,
:func:`make_gallery_dataframe`, :func:`frame_array`, :func:`frame_signals`,
:func:`frame_image`, :func:`signal_trace`, :func:`unified_trace`.

Guide: ``docs/browser.md``.
"""

from haptix.browser.core import (
    episode_summary,
    find_hapt_files,
    frame_array,
    frame_image,
    frame_signals,
    make_gallery_dataframe,
    scan_directory,
    signal_trace,
    unified_trace,
)

__all__ = [
    "episode_summary",
    "find_hapt_files",
    "frame_array",
    "frame_image",
    "frame_signals",
    "make_gallery_dataframe",
    "scan_directory",
    "signal_trace",
    "unified_trace",
]
