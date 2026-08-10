"""
Real-time data collection toolkit: incremental .hapt recording.

``haptix.save()`` needs the complete array up front — fine for offline
conversion, impossible for live capture (a camera produces frames one at a
time, forever, until you stop it). :class:`HaptRecorder` is the streaming
writer counterpart: append frames incrementally, and on :meth:`close` it
produces a fully valid ``.hapt`` directory (identical to what ``save()``
writes) that ``haptix.load()`` and ``haptix.open_archive()`` can read.

Design:

- Frames are buffered and flushed to ``raw/chunks/000000.npy``,
  ``000001.npy``, ... during capture (O(buffer) memory, crash-safe: a
  partial capture leaves recoverable chunk files).
- :meth:`close` streams the chunks into ``raw/data.npy`` (memory-bounded
  concatenation — never holds the full array), writes the SHA-256
  checksum, ``manifest.json``, ``labels.json`` and ``provenance.json``,
  then removes the chunk files. The result is a standard ``.hapt`` dir.
- ``timestamps_s`` (when provided) are accumulated per frame and written
  into the manifest for exact time indexing after the fact.

Usage::

    rec = haptix.HaptRecorder(
        "live.hapt",
        sensor=SensorMeta(type="DIGIT_v2", serial="cam-001"),
        modality="imaging",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="pressing"),
        labels=Labels(task="live_demo"),
    )
    try:
        for frame in camera_stream():      # generator of [H, W, C] arrays
            rec.write_frame(frame)
    finally:
        rec.close()                        # finalize into a valid .hapt

    data = haptix.load("live.hapt")        # round-trip verified

Hardware adapters (DIGIT camera, GelSight device, ...) can be layered on
top: any source that yields one frame array per step is a valid input.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from haptix.core import InteractionMeta, Labels, Modality, SensorMeta
from haptix.io import _default_provenance

# Frames per chunk file. Keep small enough for crash-safe granularity.
_DEFAULT_BUFFER_FRAMES = 64


class HaptRecorder:
    """Incrementally record frames into a valid ``.hapt`` directory.

    Parameters mirror :class:`~haptix.core.HaptData` construction, minus the
    raw array (which arrives frame-by-frame via :meth:`write_frame`).

    Parameters
    ----------
    path : str | Path
        Destination directory (``.hapt`` extension by convention). Must not
        already exist.
    sensor : SensorMeta
        Sensor metadata.
    modality : Modality
        ``"imaging"`` | ``"dynamic"`` | ``"force"`` | ``"multimodal"``.
    sampling_rate_hz : float
        Nominal rate used for equal-spacing time indexing when no
        per-frame timestamps are supplied.
    interaction : InteractionMeta
        Interaction metadata (required by the format).
    labels : Labels
        Annotations for the recording.
    coordinate_frame : str, optional
        Reference frame (spec v0.2).
    timestamps : Iterable[float] | None, optional
        Optional per-frame timestamps (seconds). If given, exactly one
        value per frame must be supplied to :meth:`write_frame` via the
        ``timestamp`` argument (or by passing a tuple ``(frame, ts)``).
    buffer_frames : int, default 64
        Frames buffered before flushing a chunk file.
    version : str, default "0.2.0"
        Manifest version tag.
    """

    def __init__(
        self,
        path: str | Path,
        sensor: SensorMeta,
        modality: Modality,
        sampling_rate_hz: float,
        interaction: InteractionMeta,
        labels: Labels,
        *,
        coordinate_frame: str | None = None,
        buffer_frames: int = _DEFAULT_BUFFER_FRAMES,
        version: str = "0.2.0",
    ):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"recording path already exists: {self.path}")
        self.sensor = sensor
        self.modality = modality
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.interaction = interaction
        self.labels = labels
        self.coordinate_frame = coordinate_frame
        self.version = version
        self.buffer_frames = max(1, int(buffer_frames))

        self._chunks_dir = self.path / "raw" / "chunks"
        self._chunks_dir.mkdir(parents=True, exist_ok=False)
        self._dtype: np.dtype | None = None
        self._frame_shape: tuple[int, ...] | None = None
        self._buffer: list[np.ndarray] = []
        self._n_frames = 0
        self._chunk_index = 0
        self._timestamps: list[float] = []
        self._closed = False
        self._started_at = time.time()

    # ── Recording ───────────────────────────────────────────────────────

    def write_frame(self, frame: np.ndarray | tuple, timestamp: float | None = None) -> int:
        """Append one frame and return the running frame count.

        Parameters
        ----------
        frame : np.ndarray or (array, timestamp) tuple
            Frame array (any dtype/shape; the first frame fixes the
            recording dtype and shape — later frames must match). A tuple
            ``(array, timestamp)`` supplies the per-frame timestamp
            positionally.
        timestamp : float, optional
            Per-frame timestamp in seconds (also accepted inside *frame*).

        Returns
        -------
        int
            Total frames recorded so far.
        """
        self._ensure_open()
        if isinstance(frame, tuple):
            arr, ts = frame
            if timestamp is not None:
                raise ValueError("timestamp given both positionally and in the tuple")
            timestamp = ts
        else:
            arr = frame
        arr = np.asarray(arr)

        if self._dtype is None:
            self._dtype = arr.dtype
            self._frame_shape = arr.shape
        else:
            if arr.dtype != self._dtype or arr.shape != self._frame_shape:
                raise ValueError(
                    f"frame {self._n_frames} has shape/dtype {arr.shape}/{arr.dtype}, "
                    f"expected {self._frame_shape}/{self._dtype}"
                )

        self._buffer.append(arr)
        if timestamp is not None:
            self._timestamps.append(float(timestamp))
        self._n_frames += 1
        if len(self._buffer) >= self.buffer_frames:
            self._flush()
        return self._n_frames

    def write(self, frames, timestamps=None) -> int:
        """Append a batch of frames (array or iterable), return frame count."""
        if isinstance(frames, np.ndarray):
            for i, arr in enumerate(frames):
                ts = None if timestamps is None else timestamps[i]
                self.write_frame(arr, ts)
            return self._n_frames
        for i, frame in enumerate(frames):
            ts = None if timestamps is None else timestamps[i]
            self.write_frame(frame, ts)
        return self._n_frames

    @property
    def n_frames(self) -> int:
        """Frames recorded so far."""
        return self._n_frames

    @property
    def is_open(self) -> bool:
        """Whether the recorder is still accepting frames."""
        return not self._closed

    # ── Finalization ────────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush buffered frames to a chunk file (no-op on close)."""
        if self._closed:
            return
        self._flush()

    def close(self) -> Path:
        """Finalize the recording into a valid ``.hapt`` directory.

        Streams chunk files into ``raw/data.npy``, writes the checksum,
        manifest, labels and provenance, then removes the chunk files.
        Idempotent: returns the final path. Raises ``ValueError`` if no
        frames were recorded.
        """
        if self._closed:
            return self.path
        if self._n_frames == 0:
            raise ValueError("cannot finalize an empty recording (no frames written)")

        self._flush()
        assert self._dtype is not None and self._frame_shape is not None
        final_shape = (self._n_frames,) + self._frame_shape

        # Stream chunks into raw/data.npy (memory-bounded concatenation).
        raw_dir = self.path / "raw"
        data_path = raw_dir / "data.npy"
        h = hashlib.sha256()
        dtype_obj = np.dtype(self._dtype)
        with open(data_path, "wb") as f:
            # Header first (matches np.save layout), then chunk data.
            header = {
                "shape": final_shape,
                "fortran_order": False,
                "descr": np.lib.format.dtype_to_descr(dtype_obj),
            }
            np.lib.format.write_array_header_1_0(f, header)
            for chunk_path in sorted(self._chunks_dir.glob("*.npy")):
                chunk = np.load(chunk_path)
                data = np.ascontiguousarray(chunk).tobytes()
                f.write(data)
                h.update(data)

        # Checksum, manifest, labels, provenance.
        checksum = h.hexdigest()
        (raw_dir / "checksum.sha256").write_text(checksum + "\n")

        # Build the manifest without allocating the full array.
        manifest = {
            "version": self.version,
            "sensor": self.sensor.to_dict(),
            "modality": self.modality,
            "coordinate_frame": self.coordinate_frame,
            "sampling": {
                "rate_hz": self.sampling_rate_hz,
                "num_frames": self._n_frames,
                "timestamps_s": self._timestamps or None,
            },
            "raw_shape": list(final_shape),
            "raw_dtype": str(np.dtype(self._dtype)),
            "interaction": self.interaction.to_dict(),
        }
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (self.path / "labels.json").write_text(json.dumps(self.labels.to_dict(), indent=2))
        provenance = _default_provenance()
        (self.path / "provenance.json").write_text(json.dumps(provenance.to_dict(), indent=2))

        shutil.rmtree(self._chunks_dir)
        self._closed = True
        return self.path

    def _flush(self) -> None:
        if not self._buffer:
            return
        arr = np.stack(self._buffer)
        chunk_path = self._chunks_dir / f"{self._chunk_index:06d}.npy"
        np.save(chunk_path, arr)
        self._chunk_index += 1
        self._buffer = []

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("HaptRecorder is closed")

    def __enter__(self) -> HaptRecorder:  # noqa: PYI034 - typing.Self is 3.11+; targets 3.10
        self._ensure_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "open" if not self._closed else "closed"
        return (
            f"HaptRecorder({str(self.path)!r}, {self.modality}, "
            f"{self._n_frames} frames, {state})"
        )
