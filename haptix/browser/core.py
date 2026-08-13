"""
Core library functions for the tactile data browser.

Pure numpy / Pillow — deliberately **no streamlit / plotly import** — so the
browser logic is testable in CI (core deps only), reusable from notebooks and
scripts, and keeps ``import haptix`` lean. The Streamlit UI
(:mod:`haptix.browser.app`) is a thin layer over these functions.

The browser treats each ``.hapt`` recording as an *episode*:

- :func:`find_hapt_files` / :func:`scan_directory` — discover episodes under
  a directory (``.hapt`` dirs, ``.hapt.zarr``, ``.hapt.zip``) and summarize
  them from metadata only (no raw array materialization).
- :func:`episode_summary` — one episode's metadata dict (sensor, modality,
  labels, interaction, provenance, unified, size, ...).
- :func:`frame_array` / :func:`frame_image` / :func:`frame_signals` — access
  a single frame, rendered as numpy / PIL image / 1-D signal.
- :func:`signal_trace` / :func:`unified_trace` — full-length traces over time
  for dynamic signals and the ``unified/`` cross-sensor embedding.
- :func:`make_gallery_dataframe` — turn a scan result into a pandas table for
  display / filtering.

All functions accept either a path (``str`` / ``Path``) or an already-open
:class:`~haptix.streaming.HaptArchive`; paths are opened lazily and closed
before returning, so callers never leak file handles.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from haptix.streaming import HaptArchive, open_archive

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


# ── Discovery ─────────────────────────────────────────────────────────────


def is_hapt_path(path: str | os.PathLike) -> bool:
    """Return True if *path* looks like a haptix recording.

    Accepts ``.hapt`` directories and ``.hapt.zip`` / ``.hapt.zarr`` files
    (a ``.zip``/``.zarr`` file only counts when its name carries ``.hapt``,
    so random archives are not picked up).
    """
    p = Path(path)
    try:
        if p.is_dir():
            return p.suffix == ".hapt"
        if p.is_file():
            return p.suffix in (".zip", ".zarr") and ".hapt" in p.name
    except OSError:
        return False
    return False


def find_hapt_files(root: str | os.PathLike, *, recursive: bool = True) -> list[Path]:
    """Discover haptix recordings under *root*.

    Returns a sorted list of episode paths: ``.hapt`` directories,
    ``.hapt.zarr`` files and ``.hapt.zip`` files. Hidden directories
    (dot-prefixed, e.g. ``.git``) are skipped. Episodes found inside a
    ``.hapt`` directory are not descended into (a recording is a leaf).

    Parameters
    ----------
    root : str or Path
        Directory to scan.
    recursive : bool, default True
        Descend into subdirectories.

    Returns
    -------
    list[Path]
        Episode paths, sorted by name.

    Raises
    ------
    FileNotFoundError
        If *root* does not exist.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    hits: list[Path] = []

    def walk(d: Path) -> None:
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for p in entries:
            if not is_hapt_path(p):
                if p.is_dir() and not p.name.startswith(".") and recursive:
                    walk(p)
                continue
            hits.append(p)

    walk(root)
    return hits


# ── Summaries ─────────────────────────────────────────────────────────────


def _size_bytes(path: Path) -> int:
    """Total on-disk size of a recording (recursive for directory format)."""
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


def _shape_str(shape: tuple[int, ...]) -> str:
    return "×".join(str(int(s)) for s in shape)


def episode_summary(path: str | os.PathLike, *, verify: bool = False) -> dict:
    """Metadata summary for one haptix recording (no raw materialization).

    Opens the recording lazily (:func:`~haptix.streaming.open_archive`),
    reads metadata only, and returns a flat JSON-compatible dict with pure
    Python types (safe for pandas, streamlit caching, and pickling):

    - identity: ``path``, ``name``, ``format`` (``dir``/``zarr``/``zip``),
      ``size_bytes``
    - sensor: ``sensor``, ``serial``, ``modality``
    - array: ``n_frames``, ``shape`` (tuple), ``shape_str``, ``dtype``,
      ``sampling_rate_hz``, ``duration_s``, ``timestamps`` (bool),
      ``coordinate_frame``, ``version``
    - labels: ``material``, ``material_category``, ``object_name``,
      ``object_category``, ``task``, ``custom_tags``
    - interaction: ``interaction_type``, ``speed_mm_s``, ``normal_force_N``,
      ``approach_angle_deg``, ``temperature_C``, ``humidity_pct``
    - provenance: ``file_hash``, ``derived_from``, ``is_lossy``, ``created``,
      ``created_by``, ``source`` (dict), ``processing`` (list[dict])
    - unified: ``unified`` (bool), ``unified_method``, ``unified_shape``

    Parameters
    ----------
    path : str or Path
        Path to a ``.hapt`` directory, ``.hapt.zarr`` or ``.hapt.zip``.
    verify : bool, default False
        If True, also stream-verify the raw array checksum (memory-bounded;
        raises :class:`~haptix.io.ChecksumError` on mismatch).

    Returns
    -------
    dict
        Flat metadata summary (JSON-compatible values only).

    Raises
    ------
    FileNotFoundError, HaptFormatError, ChecksumError
        Propagated from :func:`~haptix.streaming.open_archive` /
        :meth:`HaptArchive.verify` for unreadable / corrupt recordings.
    """
    p = Path(path)
    with open_archive(p) as arc:
        if verify:
            arc.verify()
        shape = tuple(int(s) for s in arc.shape)
        ts = arc.timestamps_s
        prov = arc.provenance
        labels = arc.labels
        inter = arc.interaction
        unified_shape = arc.unified_shape
        unified_shape = tuple(int(s) for s in unified_shape) if unified_shape else None
        duration_s: float | None
        if ts is not None and len(ts) > 1:
            duration_s = float(ts[-1] - ts[0])
        elif arc.n_frames and arc.sampling_rate_hz:
            duration_s = float(arc.n_frames / arc.sampling_rate_hz)
        else:
            duration_s = None
        fmt = "zarr" if p.suffix == ".zarr" else "zip" if p.suffix == ".zip" else "dir"
        return {
            "path": str(p),
            "name": p.name,
            "format": fmt,
            "size_bytes": _size_bytes(p),
            "sensor": arc.sensor.type,
            "serial": arc.sensor.serial,
            "modality": arc.modality,
            "n_frames": int(arc.n_frames),
            "shape": shape,
            "shape_str": _shape_str(shape),
            "dtype": str(arc.dtype),
            "sampling_rate_hz": float(arc.sampling_rate_hz),
            "duration_s": duration_s,
            "timestamps": ts is not None,
            "coordinate_frame": arc.coordinate_frame,
            "version": arc.version,
            "material": labels.material,
            "material_category": labels.material_category,
            "object_name": labels.object_name,
            "object_category": labels.object_category,
            "task": labels.task,
            "custom_tags": list(labels.custom_tags),
            "interaction_type": inter.type,
            "speed_mm_s": inter.speed_mm_s,
            "normal_force_N": inter.normal_force_N,
            "approach_angle_deg": inter.approach_angle_deg,
            "temperature_C": inter.temperature_C,
            "humidity_pct": inter.humidity_pct,
            "file_hash": prov.file_hash if prov is not None else None,
            "derived_from": prov.derived_from if prov is not None else None,
            "is_lossy": prov.is_lossy if prov is not None else None,
            "created": prov.created if prov is not None else None,
            "created_by": prov.created_by if prov is not None else None,
            "source": prov.source.to_dict() if prov is not None else {},
            "processing": (
                [s.to_dict() if hasattr(s, "to_dict") else s for s in prov.processing]
                if prov is not None
                else []
            ),
            "unified": unified_shape is not None,
            "unified_method": arc.unified_method,
            "unified_shape": unified_shape,
        }


def scan_directory(
    root: str | os.PathLike,
    *,
    recursive: bool = True,
    max_episodes: int | None = None,
    verify: bool = False,
) -> dict:
    """Scan a directory and summarize every haptix recording under it.

    Tolerant by design: a corrupt or unreadable recording is reported in the
    ``errors`` list instead of aborting the scan, so a mixed directory still
    yields its healthy episodes.

    Parameters
    ----------
    root : str or Path
        Directory to scan.
    recursive : bool, default True
        Descend into subdirectories.
    max_episodes : int, optional
        Cap the number of episodes scanned (useful for very large trees).
    verify : bool, default False
        Passed to :func:`episode_summary` (stream-verify each raw array).

    Returns
    -------
    dict
        ``{"root": str, "episodes": [summary, ...], "errors": [{"path", "error"}, ...]}``
    """
    paths = find_hapt_files(root, recursive=recursive)
    if max_episodes is not None:
        paths = paths[: max(0, int(max_episodes))]
    episodes: list[dict] = []
    errors: list[dict] = []
    for p in paths:
        try:
            episodes.append(episode_summary(p, verify=verify))
        except Exception as exc:  # noqa: BLE001 - gallery must survive one bad file
            errors.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
    return {"root": str(Path(root)), "episodes": episodes, "errors": errors}


def make_gallery_dataframe(scan: dict) -> pd.DataFrame:
    """Turn a :func:`scan_directory` result into a pandas table.

    Returns a DataFrame with display-friendly columns (name, sensor,
    modality, frames, shape, sampling rate, material, object, task, unified,
    size MB, format, path) for the episode gallery. Empty scans produce an
    empty DataFrame with the same columns.
    """
    rows = []
    for ep in scan.get("episodes", []):
        rows.append(
            {
                "name": ep["name"],
                "sensor": ep["sensor"],
                "modality": ep["modality"],
                "frames": ep["n_frames"],
                "shape": ep["shape_str"],
                "sampling_rate_hz": ep["sampling_rate_hz"],
                "material": ep["material"],
                "object": ep["object_name"],
                "task": ep["task"],
                "unified": ep["unified"],
                "size_mb": round(ep["size_bytes"] / 1e6, 2),
                "format": ep["format"],
                "path": ep["path"],
            }
        )
    columns = [
        "name",
        "sensor",
        "modality",
        "frames",
        "shape",
        "sampling_rate_hz",
        "material",
        "object",
        "task",
        "unified",
        "size_mb",
        "format",
        "path",
    ]
    return pd.DataFrame(rows, columns=columns)


# ── Frame / trace access ──────────────────────────────────────────────────


@contextlib.contextmanager
def _as_archive(source: str | os.PathLike | HaptArchive) -> Iterator[HaptArchive]:
    """Yield an archive from a path (opened + closed) or an existing handle."""
    if isinstance(source, HaptArchive):
        yield source
    elif isinstance(source, (str, os.PathLike)):
        with open_archive(Path(source)) as arc:
            yield arc
    else:
        raise TypeError(f"Expected a .hapt path or HaptArchive, got {type(source).__name__}")


def frame_array(source: str | os.PathLike | HaptArchive, frame: int) -> np.ndarray:
    """Return a single frame as numpy.

    Imaging recordings yield ``[H, W, C]`` (or ``[H, W]`` for grayscale);
    dynamic recordings yield the flattened channel vector ``[F]``.

    Parameters
    ----------
    source : str, Path or HaptArchive
        Recording to read from.
    frame : int
        Frame index (``0 <= frame < n_frames``).

    Returns
    -------
    np.ndarray
        The frame (a copy — safe to mutate).

    Raises
    ------
    IndexError
        If *frame* is out of range.
    """
    with _as_archive(source) as arc:
        n = arc.n_frames
        if not 0 <= int(frame) < n:
            raise IndexError(f"frame {frame} out of range [0, {n})")
        win = arc.window(int(frame), int(frame) + 1)
        return np.asarray(win.raw.numpy())[0]


def frame_signals(source: str | os.PathLike | HaptArchive, frame: int) -> np.ndarray:
    """Return one frame's channel values as a 1-D array (dynamic recordings).

    Equivalent to :func:`frame_array` with the frame flattened to 1-D —
    the shape used by the signal explorer charts.
    """
    arr = frame_array(source, frame)
    return np.asarray(arr).reshape(-1)


def frame_image(
    source: str | os.PathLike | HaptArchive,
    frame: int,
    *,
    max_size: int | None = None,
):
    """Render one imaging frame as a PIL image.

    Handles uint8 RGB/RGBA, grayscale, single-channel and float arrays
    (float data is normalized per-frame to 0–255).

    Parameters
    ----------
    source : str, Path or HaptArchive
        Recording to read from.
    frame : int
        Frame index.
    max_size : int, optional
        If given, downscale the image (keeping aspect ratio) so its largest
        side is at most *max_size* pixels.

    Returns
    -------
    PIL.Image.Image
        The frame as an image.

    Raises
    ------
    ValueError
        If the frame is not image-shaped (use :func:`frame_signals` for
        dynamic recordings).
    """
    from PIL import Image  # core dependency, imported lazily

    arr = frame_array(source, frame)
    a = np.asarray(arr)
    if a.ndim == 2:
        img = Image.fromarray(_to_uint8(a), mode="L")
    elif a.ndim == 3 and a.shape[-1] == 1:
        img = Image.fromarray(_to_uint8(a[..., 0]), mode="L")
    elif a.ndim == 3 and a.shape[-1] in (3, 4):
        img = Image.fromarray(_to_uint8(a))
    else:
        raise ValueError(
            f"frame {frame} has non-image shape {a.shape} — use frame_signals() "
            "for dynamic recordings"
        )
    if max_size is not None:
        img.thumbnail((int(max_size), int(max_size)))
    return img


def _to_uint8(a: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to uint8 (per-frame min-max)."""
    if a.dtype == np.uint8:
        return np.asarray(a)
    f = np.asarray(a, dtype=np.float64)
    lo, hi = float(f.min()), float(f.max())
    if hi - lo < 1e-9:
        return np.zeros(f.shape, dtype=np.uint8)
    out = (f - lo) / (hi - lo) * 255.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def signal_trace(
    source: str | os.PathLike | HaptArchive,
    channels: list[int] | None = None,
    *,
    max_frames: int | None = None,
) -> dict:
    """Full-length signal traces for a dynamic recording.

    Parameters
    ----------
    source : str, Path or HaptArchive
        Recording to read from.
    channels : list[int], optional
        Channel indices to return (default: all channels).
    max_frames : int, optional
        Cap the number of frames read from the start of the recording.

    Returns
    -------
    dict
        ``{"t": np.ndarray, "y": np.ndarray, "channels": list[int]}`` where
        ``t`` is time in seconds (per-frame timestamps when present, else
        frame indices) and ``y`` is ``[T, n_channels]``.
    """
    with _as_archive(source) as arc:
        n = arc.n_frames
        stop = min(n, int(max_frames)) if max_frames is not None else n
        if stop <= 0:
            return {"t": np.array([], dtype=np.float64), "y": np.empty((0, 0)), "channels": []}
        ts = arc.timestamps_s
        t = ts[:stop] if ts is not None else np.arange(stop, dtype=np.float64)
        arr = np.asarray(arc.window(0, stop).raw.numpy())
        flat = arr.reshape(stop, -1)
        if channels is None:
            return {"t": t, "y": flat, "channels": list(range(flat.shape[1]))}
        chans = [int(c) for c in channels]
        return {"t": t, "y": flat[:, chans], "channels": chans}


def unified_trace(
    source: str | os.PathLike | HaptArchive, *, max_frames: int | None = None
) -> np.ndarray | None:
    """The ``unified/`` cross-sensor embedding ``[T, D]``, or None if absent.

    Only the time axis is bounded by *max_frames*; the embedding dimension is
    returned in full.
    """
    with _as_archive(source) as arc:
        if arc.unified_shape is None:
            return None
        stop = min(arc.n_frames, int(max_frames)) if max_frames is not None else arc.n_frames
        if stop <= 0:
            return None
        win = arc.window(0, stop)
        if win.unified is None:
            return None
        return np.asarray(win.unified.numpy())
