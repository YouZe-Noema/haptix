"""
Streaming / temporal windowing for long recordings.

``haptix.load()`` reads an entire recording into memory — fine for a demo,
prohibitive for hours of 30 Hz tactile data (100K+ frames of 480x640 RGB is
~92 GB). :func:`open_archive` is the lazy counterpart: it opens a ``.hapt``
directory, ``.hapt.zarr``, or ``.hapt.zip`` and reads *metadata only*. The
raw array stays on disk (memory-mapped for directory format, chunked for
zarr, decompressed-once for zip) until a window is requested.

Public API:

- :func:`open_archive` — lazy archive handle (context manager).
- :class:`HaptArchive` — metadata accessors, :meth:`HaptArchive.window`,
  :meth:`HaptArchive.iter_windows`, :meth:`HaptArchive.verify`,
  :meth:`HaptArchive.frame_index_at`.
- :func:`load` — unchanged eager loader (round-trip guarantee, full verify).

Windows are ordinary :class:`~haptix.core.HaptData` objects: each carries
its own checksum (SHA-256 of the window bytes), sliced timestamps when
present, and the recording's sensor/interaction/labels metadata, so a
window can be saved, re-loaded, and fed to ``.to_torch()`` / ``.to_jax()``
exactly like a full recording.

Format notes (memory behavior):

- Directory (``.hapt/``): ``np.load(..., mmap_mode='r')`` — windows are
  views into the memory-mapped file; O(window) memory.
- Zarr (``.hapt.zarr``): chunked array handle — each window reads only the
  chunks it touches; O(window) memory (plus decompression).
- ZIP (``.hapt.zip``): DEFLATE is not random-access, so the raw member is
  decompressed once into memory at open; windows slice that array. O(full
  array) memory — prefer directory/zarr for very long recordings.

Spec impact: none — this is an API addition on top of the unchanged
``.hapt`` v0.2 container.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    Modality,
    Provenance,
    RawData,
    SensorMeta,
    UnifiedData,
)
from haptix.io import ChecksumError, HaptFormatError

# Memory-bounded verification / windowing chunk (elements per pass).
_VERIFY_CHUNK = 1 << 22  # 4 Mi elements


def _sha256_chunked(arr: np.ndarray, chunk: int = _VERIFY_CHUNK) -> str:
    """SHA-256 of an ndarray without materializing a flat copy.

    Iterates over C-contiguous chunks of the (possibly memory-mapped)
    array so peak memory stays O(chunk). Slices that are not contiguous
    are copied per chunk, never as a whole.
    """
    h = hashlib.sha256()
    flat = arr.reshape(-1)
    for i in range(0, flat.size, chunk):
        h.update(np.ascontiguousarray(flat[i : i + chunk]).tobytes())
    return h.hexdigest()


class HaptArchive:
    """Lazy handle to a ``.hapt`` recording (directory / zarr / zip).

    Reads metadata eagerly (manifest, labels, provenance) but defers raw
    array materialization to :meth:`window` / :meth:`iter_windows` /
    :meth:`verify`. Use as a context manager to guarantee the underlying
    file handle / memory map is released::

        with haptix.open_archive("long.hapt") as arc:
            for win in arc.iter_windows(window_size=256):
                ...  # win is a HaptData [256, ...]

    Attributes mirror :class:`~haptix.core.HaptData` where available:
    ``shape``, ``dtype``, ``n_frames``, ``sampling_rate_hz``,
    ``timestamps_s``, ``sensor``, ``modality``, ``interaction``,
    ``labels``, ``provenance``, ``coordinate_frame``, ``version``.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._format: str = "dir"  # "dir" | "zarr" | "zip"
        self._raw: np.ndarray | None = None  # full array (zip mode only)
        self._memmap: np.ndarray | None = None  # dir mode mmap
        self._zarr_root = None  # zarr mode group
        self._zarr_raw = None  # zarr mode raw array handle
        self._zarr_unified = None  # zarr mode unified handle (optional)
        self._zip_file: zipfile.ZipFile | None = None
        self._closed = False
        self._manifest: dict = {}
        self._labels_dict: dict = {}
        self._provenance_dict: dict = {}
        self._unified_arr: np.ndarray | None = None  # dir/zip unified (optional)
        self._unified_method: str | None = None
        self._unified_source: str | None = None
        self._unified_target: str | None = None
        self._unified_is_lossy: bool = True

        self.sensor = SensorMeta(type="unknown")
        self.modality: Modality = "dynamic"
        self.sampling_rate_hz = 30.0
        self.interaction = InteractionMeta(type="pressing")
        self.labels = Labels()
        self.provenance: Provenance | None = None
        self.coordinate_frame: str | None = None
        self.version = "0.1.0"
        self.shape: tuple[int, ...] = ()
        self.dtype = "float32"

        self._open()

    # ── Opening ─────────────────────────────────────────────────────────

    def _open(self) -> None:
        p = self.path
        if p.suffix == ".zarr" and p.is_file():
            self._open_zarr(p)
        elif p.suffix == ".zip" and p.is_file():
            self._open_zip(p)
        elif p.is_dir():
            self._open_dir(p)
        elif p.suffix == ".hapt" and p.is_file():
            raise HaptFormatError(
                "Flat .hapt files are deprecated and no longer supported. "
                "Use directory format, .hapt.zarr, or .hapt.zip."
            )
        else:
            raise FileNotFoundError(f"Not a valid .hapt path: {p}")

    def _parse_manifest(self, manifest: dict, labels_dict: dict) -> None:
        self._manifest = manifest
        self._labels_dict = labels_dict
        self.sensor = SensorMeta.from_dict(manifest["sensor"])
        modality = manifest["modality"]
        assert modality in ("imaging", "dynamic", "force", "multimodal")
        self.modality = modality
        self.sampling_rate_hz = float(manifest["sampling"]["rate_hz"])
        self.interaction = InteractionMeta.from_dict(manifest["interaction"])
        self.labels = Labels.from_dict(labels_dict)
        self.coordinate_frame = manifest.get("coordinate_frame")
        self.version = manifest.get("version", "0.1.0")

    def _open_dir(self, p: Path) -> None:
        self._format = "dir"
        manifest_path = p / "manifest.json"
        raw_path = p / "raw" / "data.npy"
        labels_path = p / "labels.json"
        for path, msg in (
            (manifest_path, "Missing manifest.json"),
            (raw_path, "Missing raw/data.npy"),
            (labels_path, "Missing labels.json"),
        ):
            if not path.exists():
                raise HaptFormatError(f"{msg} in {p}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        with open(labels_path) as f:
            labels_dict = json.load(f)
        self._parse_manifest(manifest, labels_dict)

        self._memmap = np.load(raw_path, mmap_mode="r")
        self.shape = tuple(self._memmap.shape)
        self.dtype = str(self._memmap.dtype)

        prov_path = p / "provenance.json"
        if prov_path.exists():
            with open(prov_path) as f:
                self._provenance_dict = json.load(f)
            self.provenance = Provenance.from_dict(self._provenance_dict)

        unified_dir = p / "unified"
        unified_data = unified_dir / "data.npy"
        transform_path = unified_dir / "transform.json"
        if unified_data.exists() and transform_path.exists():
            self._unified_arr = np.load(unified_data)
            with open(transform_path) as f:
                transform = json.load(f)
            self._unified_method = transform["method"]
            self._unified_source = transform.get("source_modality")
            self._unified_target = transform.get("target_modality", "unified")
            self._unified_is_lossy = transform.get("is_lossy", True)

    def _open_zip(self, p: Path) -> None:
        self._format = "zip"
        try:
            zf = zipfile.ZipFile(p, "r")
        except zipfile.BadZipFile as e:
            raise HaptFormatError(f"Not a valid .hapt.zip archive: {p}") from e
        self._zip_file = zf
        names = set(zf.namelist())
        for member, msg in (
            ("manifest.json", "Missing manifest.json"),
            ("raw/data.npy", "Missing raw/data.npy"),
            ("labels.json", "Missing labels.json"),
        ):
            if member not in names:
                raise HaptFormatError(f"{msg} in .hapt.zip")
        manifest = json.loads(zf.read("manifest.json"))
        labels_dict = json.loads(zf.read("labels.json"))
        self._parse_manifest(manifest, labels_dict)

        # DEFLATE is not random-access: decompress the raw member once.
        self._raw = np.load(io.BytesIO(zf.read("raw/data.npy")))
        self.shape = tuple(self._raw.shape)
        self.dtype = str(self._raw.dtype)

        if "provenance.json" in names:
            self._provenance_dict = json.loads(zf.read("provenance.json"))
            self.provenance = Provenance.from_dict(self._provenance_dict)

        if "unified/data.npy" in names and "unified/transform.json" in names:
            self._unified_arr = np.load(io.BytesIO(zf.read("unified/data.npy")))
            transform = json.loads(zf.read("unified/transform.json"))
            self._unified_method = transform.get("method", "unknown")
            self._unified_source = transform.get(
                "source_modality", manifest.get("modality", "unknown")
            )
            self._unified_target = transform.get("target_modality", "unified")
            self._unified_is_lossy = transform.get("is_lossy", True)

    def _open_zarr(self, p: Path) -> None:
        self._format = "zarr"
        from haptix.io import _ensure_zarr

        zarr, _ = _ensure_zarr()
        store = zarr.storage.ZipStore(str(p), mode="r")
        try:
            root = zarr.open_group(store=store, mode="r")
        except (zarr.errors.GroupNotFoundError, FileNotFoundError):
            store.close()
            raise FileNotFoundError(f"Not a valid .hapt path: {p}") from None
        except (zipfile.BadZipFile, ValueError, KeyError):
            store.close()
            raise HaptFormatError(f"Not a valid .hapt.zarr archive: {p}") from None
        self._zarr_root = root
        if "raw/data" not in root:
            raise HaptFormatError("Missing raw/data array in .hapt.zarr")
        if "manifest" not in root.attrs or "labels" not in root.attrs:
            raise HaptFormatError("Missing manifest/labels attributes in .hapt.zarr")

        manifest = root.attrs["manifest"]
        labels_dict = root.attrs["labels"]
        self._parse_manifest(manifest, labels_dict)

        self._zarr_raw = root["raw/data"]
        self.shape = tuple(self._zarr_raw.shape)
        self.dtype = str(self._zarr_raw.dtype)

        if "provenance" in root.attrs:
            self._provenance_dict = root.attrs["provenance"]
            self.provenance = Provenance.from_dict(self._provenance_dict)

        if "unified/data" in root:
            self._zarr_unified = root["unified/data"]
            if "transform" in self._zarr_unified.attrs:
                transform = self._zarr_unified.attrs["transform"]
                self._unified_method = transform.get("method", "unknown")
                self._unified_source = transform.get(
                    "source_modality", manifest.get("modality", "unknown")
                )
                self._unified_target = transform.get("target_modality", "unified")
                self._unified_is_lossy = transform.get("is_lossy", True)

    # ── Metadata helpers ────────────────────────────────────────────────

    @property
    def n_frames(self) -> int:
        """Number of frames (time steps) in the recording."""
        if not self.shape:
            return 0
        return int(self.shape[0])

    @property
    def timestamps_s(self) -> np.ndarray | None:
        """Per-frame timestamps (seconds) or None if equally spaced."""
        ts = self._manifest.get("sampling", {}).get("timestamps_s")
        if ts is None:
            return None
        return np.asarray(ts, dtype=np.float64)

    @property
    def unified_method(self) -> str | None:
        """Version tag of the unified representation, if present."""
        return self._unified_method

    # ── Raw access ──────────────────────────────────────────────────────

    def _raw_slice(self, start: int, stop: int) -> np.ndarray:
        """Return a (copied) window of the raw array along the time axis."""
        if self._format == "zarr":
            assert self._zarr_raw is not None
            return np.asarray(self._zarr_raw[start:stop])
        if self._memmap is not None:
            return np.array(self._memmap[start:stop], copy=True)
        assert self._raw is not None
        return np.array(self._raw[start:stop], copy=True)

    def _unified_slice(self, start: int, stop: int) -> UnifiedData | None:
        if self._zarr_unified is not None:
            arr = np.asarray(self._zarr_unified[start:stop])
            return UnifiedData(
                array=arr,
                method=self._unified_method or "unknown",
                source_modality=self._unified_source or self.modality,
                target_modality=self._unified_target or "unified",
                is_lossy=self._unified_is_lossy,
                checksum=_sha256_chunked(arr),
            )
        if self._unified_arr is not None:
            arr = np.array(self._unified_arr[start:stop], copy=True)
            return UnifiedData(
                array=arr,
                method=self._unified_method or "unknown",
                source_modality=self._unified_source or self.modality,
                target_modality=self._unified_target or "unified",
                is_lossy=self._unified_is_lossy,
                checksum=_sha256_chunked(arr),
            )
        return None

    # ── Windowing ───────────────────────────────────────────────────────

    def window(self, start: int, stop: int) -> HaptData:
        """Materialize frames ``[start, stop)`` as a :class:`HaptData`.

        Parameters
        ----------
        start : int
            First frame index (inclusive).
        stop : int
            Last frame index (exclusive). Clamped to the recording length.

        Returns
        -------
        HaptData
            A standalone window with its own checksum and (sliced)
            timestamps. Saving it produces a valid, independently
            verifiable ``.hapt`` file.
        """
        self._ensure_open()
        n = self.n_frames
        start = max(0, int(start))
        stop = min(n, int(stop))
        if stop <= start:
            raise ValueError(f"window({start}, {stop}): stop must be > start")
        arr = self._raw_slice(start, stop)
        ts = self.timestamps_s
        return HaptData(
            raw=RawData(
                array=arr,
                checksum=_sha256_chunked(arr),
                dtype=str(arr.dtype),
                shape=arr.shape,
            ),
            sensor=self.sensor,
            modality=self.modality,
            sampling_rate_hz=self.sampling_rate_hz,
            interaction=self.interaction,
            labels=self.labels,
            unified=self._unified_slice(start, stop),
            provenance=self.provenance,
            coordinate_frame=self.coordinate_frame,
            timestamps_s=ts[start:stop].tolist() if ts is not None else None,
            version=self.version,
        )

    def iter_windows(
        self,
        window_size: int,
        stride: int | None = None,
        *,
        start: int = 0,
        stop: int | None = None,
        drop_last: bool = False,
    ) -> Iterator[HaptData]:
        """Iterate over temporal windows of the recording.

        Windows are yielded as :class:`HaptData` objects along the time
        axis. The final window may be shorter than *window_size* unless
        ``drop_last=True``.

        Parameters
        ----------
        window_size : int
            Number of frames per window (>= 1).
        stride : int, optional
            Frames between window starts. Defaults to *window_size*
            (non-overlapping windows). ``stride < window_size`` yields
            overlapping windows.
        start, stop : int
            Frame bounds to iterate over (defaults: full recording).
        drop_last : bool, default False
            Drop the final partial window (shorter than *window_size*).

        Yields
        ------
        HaptData
            Each window with its own checksum, sliced timestamps and the
            recording's metadata.
        """
        self._ensure_open()
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride is None:
            stride = window_size
        if stride < 1:
            raise ValueError("stride must be >= 1")
        n = self.n_frames
        start = max(0, int(start))
        stop = min(n, int(stop)) if stop is not None else n
        i = start
        while i < stop:
            j = min(i + window_size, stop)
            if drop_last and j - i < window_size:
                break
            yield self.window(i, j)
            i += stride

    def window_count(
        self,
        window_size: int,
        stride: int | None = None,
        *,
        start: int = 0,
        stop: int | None = None,
        drop_last: bool = False,
    ) -> int:
        """Number of windows :meth:`iter_windows` would yield (no I/O)."""
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride is None:
            stride = window_size
        if stride < 1:
            raise ValueError("stride must be >= 1")
        n = self.n_frames
        start = max(0, int(start))
        stop = min(n, int(stop)) if stop is not None else n
        if stop <= start:
            return 0
        count = 0
        i = start
        while i < stop:
            j = min(i + window_size, stop)
            if drop_last and j - i < window_size:
                break
            count += 1
            i += stride
        return count

    def frame_index_at(self, time_s: float) -> int:
        """Frame index nearest to a recording time (seconds).

        Uses per-frame timestamps when present, else assumes equal
        spacing at ``sampling_rate_hz``. Clamped to [0, n_frames - 1].
        """
        n = self.n_frames
        if n == 0:
            return 0
        ts = self.timestamps_s
        if ts is not None:
            idx = int(np.argmin(np.abs(ts - time_s)))
            return int(np.clip(idx, 0, n - 1))
        idx = round(time_s * self.sampling_rate_hz)
        return int(np.clip(idx, 0, n - 1))

    # ── Verification ────────────────────────────────────────────────────

    def verify(self) -> bool:
        """Stream SHA-256 verification of the raw array (memory-bounded).

        Compares against the checksum stored in the container. Returns
        True on match, raises :class:`~haptix.io.ChecksumError` on
        mismatch (mirrors ``haptix.load`` semantics). Does NOT load the
        whole array into memory.
        """
        self._ensure_open()
        stored = self._stored_checksum()
        computed = self._compute_checksum()
        if computed != stored:
            raise ChecksumError(
                f"Checksum mismatch! Stored: {stored[:16]}..., " f"Computed: {computed[:16]}..."
            )
        return True

    def _stored_checksum(self) -> str:
        if self._format == "zarr":
            assert self._zarr_raw is not None
            return str(self._zarr_raw.attrs.get("checksum", ""))
        if self._format == "zip":
            assert self._zip_file is not None
            return self._zip_file.read("raw/checksum.sha256").decode().strip()
        checksum_path = self.path / "raw" / "checksum.sha256"
        if checksum_path.exists():
            return checksum_path.read_text().strip()
        assert self._memmap is not None
        return _sha256_chunked(self._memmap)

    def _compute_checksum(self) -> str:
        if self._format == "zarr":
            # Iterate time-chunked to stay memory-bounded.
            h = hashlib.sha256()
            tail = int(np.prod(self.shape[1:])) if len(self.shape) > 1 else 1
            chunk = max(1, _VERIFY_CHUNK // max(1, tail))
            assert self._zarr_raw is not None
            for i in range(0, self.n_frames, chunk):
                h.update(np.ascontiguousarray(self._zarr_raw[i : i + chunk]).tobytes())
            return h.hexdigest()
        arr = self._memmap if self._memmap is not None else self._raw
        assert arr is not None
        return _sha256_chunked(arr)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Release file handles / memory maps. Idempotent."""
        if self._closed:
            return
        self._memmap = None
        self._raw = None
        self._unified_arr = None
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None
        if self._zarr_root is not None:
            from contextlib import suppress

            store = getattr(self._zarr_root, "store", None)
            with suppress(Exception):  # best-effort cleanup on close
                if store is not None:
                    store.close()
            self._zarr_root = None
            self._zarr_raw = None
            self._zarr_unified = None
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("HaptArchive is closed")

    def __enter__(self) -> HaptArchive:  # noqa: PYI034 - typing.Self is 3.11+; targets 3.10
        self._ensure_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"HaptArchive({str(self.path)!r}, {self.modality}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )


def open_archive(path: str | Path) -> HaptArchive:
    """Open a ``.hapt`` recording lazily for streaming / windowing.

    Reads metadata only; the raw array stays on disk until a window is
    requested or :meth:`HaptArchive.verify` runs. Supports directory,
    ``.hapt.zarr`` and ``.hapt.zip`` formats (auto-detected).

    Use as a context manager::

        with haptix.open_archive("long.hapt") as arc:
            print(arc.n_frames, arc.shape)
            for win in arc.iter_windows(window_size=256):
                ...
    """
    return HaptArchive(path)
