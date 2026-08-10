"""
PyTorch-native episode / windowed dataset (``haptix.torch_dataset``).

Bridges the lazy streaming layer (:class:`~haptix.streaming.HaptArchive`) to
``torch.utils.data.Dataset`` so PyTorch-based robot-learning frameworks
(Diffusion Policy, ACT, LeRobot, ...) can consume ``.hapt`` recordings
directly. Each recording is an *episode*; each dataset item is a temporal
*window* of frames, and a window never crosses an episode boundary.

Key design decisions:

- **Lazy raw reads.** Metadata (including per-source label values) is read
  eagerly at construction; raw arrays stay on disk until ``__getitem__``.
  Windows are read with the archive's raw/unified slice helpers, skipping
  per-window SHA-256 — checksumming is a verification concern, not a
  training-loop concern. Call ``archive.verify()`` once per file to
  validate integrity instead.
- **Global label encoding.** String-valued labels are encoded to class
  indices with a single dataset-wide mapping (sorted, deterministic), so
  the same material maps to the same class across every episode. Numeric
  label fields are returned as float tensors (regression).
- **Worker-safe.** ``__getstate__`` / ``__setstate__`` re-open archive
  handles from their recorded paths, so the dataset pickles cleanly for
  ``DataLoader(num_workers>0)`` on platforms that default to ``spawn``
  (macOS, Windows).

Spec impact: none — this is an API addition on top of the unchanged
``.hapt`` v0.2 container and the ``haptix.streaming`` layer.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import numpy as np

from haptix.core import HaptData
from haptix.streaming import HaptArchive, open_archive

try:
    import torch as _torch
except ImportError:
    _torch = None

# Label fields resolvable from public attributes of both HaptData and
# HaptArchive (mirrors HaptData.to_torch's label vocabulary).
_LABEL_FIELDS = (
    "material",
    "material_category",
    "object_name",
    "object_category",
    "task",
)
_INTERACTION_FIELDS = (
    "speed_mm_s",
    "normal_force_N",
    "approach_angle_deg",
    "temperature_C",
    "humidity_pct",
    "interaction_type",
)
_LABEL_FIELD_NAMES = _LABEL_FIELDS + _INTERACTION_FIELDS + ("sensor_type",)


def _resolve_label_value(obj, field: str):
    """Resolve a label field name to its value (public-attribute lookup)."""
    if field in _LABEL_FIELDS:
        return getattr(obj.labels, field)
    if field == "interaction_type":
        return obj.interaction.type
    if field in _INTERACTION_FIELDS:
        return getattr(obj.interaction, field)
    if field == "sensor_type":
        return obj.sensor.type
    raise ValueError(
        f"Unknown label field: '{field}'. Available fields: {list(_LABEL_FIELD_NAMES)}"
    )


class _Source:
    """Uniform windowing + label access over one recording.

    Wraps either a lazily-opened :class:`HaptArchive` (from a path or
    passed in) or an in-memory :class:`HaptData`. Raw windows are read
    directly from the archive slice helpers — no per-window checksum.
    """

    __slots__ = ("_archive", "_data", "_owned", "_use_unified", "path")

    def __init__(self, spec, use_unified: bool):
        self._archive: HaptArchive | None = None
        self._data: HaptData | None = None
        self._use_unified = bool(use_unified)
        self.path: Path | None = None
        self._owned = False

        if isinstance(spec, HaptArchive):
            self._archive = spec
            self.path = spec.path
        elif isinstance(spec, HaptData):
            self._data = spec
        elif isinstance(spec, (str, Path)):
            self.path = Path(spec)
            self._archive = open_archive(self.path)
            self._owned = True
        else:
            raise TypeError(
                "sources must be HaptArchive, HaptData, path-like, or an iterable "
                f"of those; got {type(spec).__name__}"
            )

        if use_unified and not self.has_unified:
            raise ValueError(
                "source has no unified representation (use_unified=True): "
                f"{self.path or '<in-memory HaptData>'}"
            )

    @classmethod
    def from_spec(cls, spec, use_unified: bool) -> _Source:
        """Reconstruct from a picklable spec (``to_spec`` output)."""
        if isinstance(spec, tuple) and len(spec) == 2 and spec[0] in ("path", "data"):
            kind, value = spec
            return cls(Path(value) if kind == "path" else value, use_unified)
        return cls(spec, use_unified)

    # ── Introspection ───────────────────────────────────────────────────

    @property
    def n_frames(self) -> int:
        if self._archive is not None:
            return self._archive.n_frames
        assert self._data is not None
        return int(self._data.raw.array.shape[0])

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """Per-frame shape (everything after the time axis)."""
        if self._archive is not None:
            return tuple(self._archive.shape[1:])
        assert self._data is not None
        return tuple(self._data.raw.array.shape[1:])

    @property
    def unified_shape(self) -> tuple[int, ...] | None:
        """Per-frame unified shape, or None if no unified representation."""
        if self._archive is not None:
            us = self._archive.unified_shape
            return tuple(us[1:]) if us is not None else None
        assert self._data is not None
        u = self._data.unified
        return tuple(u.array.shape[1:]) if u is not None else None

    @property
    def has_unified(self) -> bool:
        return self.unified_shape is not None

    # ── Data access ─────────────────────────────────────────────────────

    def window_array(self, start: int, stop: int) -> np.ndarray:
        """Materialize frames ``[start, stop)`` as a numpy array."""
        if self._archive is not None:
            if self._use_unified:
                u = self._archive._unified_slice(start, stop)
                assert u is not None, "unified availability checked at construction"
                return u.array
            return self._archive._raw_slice(start, stop)
        assert self._data is not None
        if self._use_unified:
            u = self._data.unified
            assert u is not None, "unified availability checked at construction"
            return np.array(u.array[start:stop], copy=True)
        return np.array(self._data.raw.array[start:stop], copy=True)

    def window_count(self, window_size: int, stride: int, *, drop_last: bool) -> int:
        if self._archive is not None:
            return self._archive.window_count(window_size, stride, drop_last=drop_last)
        n = self.n_frames
        count = 0
        i = 0
        while i < n:
            j = min(i + window_size, n)
            if drop_last and j - i < window_size:
                break
            count += 1
            i += stride
        return count

    def label_value(self, field: str):
        obj = self._archive if self._archive is not None else self._data
        return _resolve_label_value(obj, field)

    # ── Lifecycle / pickling ────────────────────────────────────────────

    def close(self) -> None:
        if self._archive is not None and self._owned:
            self._archive.close()

    def to_spec(self):
        """Picklable representation (archives become their paths)."""
        if self.path is not None:
            return ("path", str(self.path))
        return ("data", self._data)


class WindowedDataset:
    """PyTorch Dataset over temporal windows of one or more ``.hapt`` episodes.

    Each recording (or in-memory :class:`HaptData`) is an episode; each
    item is a window of ``window_size`` frames along the time axis,
    converted to a ``torch.Tensor`` of shape ``[T, *frame_shape]``. With
    ``label`` set, items are ``(X, y)`` tuples.

    Parameters
    ----------
    sources : HaptArchive | HaptData | str | Path | Iterable of those
        One or more recordings. Strings/Paths are opened lazily as
        archives; archives the dataset opens itself are closed by
        :meth:`close` (or the context manager). Passed-in ``HaptArchive``
        / ``HaptData`` objects keep their owner's lifecycle.
    window_size : int
        Number of frames per window (>= 1). The final window of an episode
        may be shorter unless ``drop_last=True``.
    stride : int, optional
        Frames between window starts. Defaults to ``window_size``
        (non-overlapping windows); ``stride < window_size`` yields
        overlapping windows.
    label : str, optional
        Label field resolved per window. String-valued fields are encoded
        to class indices with one dataset-wide mapping (same value maps to
        the same class across every source); numeric fields are returned
        as float tensors (regression). Available fields: material,
        material_category, object_name, object_category, task,
        speed_mm_s, normal_force_N, approach_angle_deg, temperature_C,
        humidity_pct, interaction_type, sensor_type.
    transform : callable, optional
        Applied to each window tensor (X).
    target_transform : callable, optional
        Applied to each label tensor (y). Only used when ``label`` is set.
    dtype : str or torch.dtype, default 'float32'
        Target dtype for window tensors.
    drop_last : bool, default False
        Drop each source's final partial window (shorter than
        ``window_size``). Recommended for fixed-length DataLoader batches.
    use_unified : bool, default False
        Window the ``unified/`` cross-sensor representation instead of raw
        frames. All sources must carry a unified array (else
        ``ValueError`` at construction).

    Examples
    --------
    ::

        import haptix
        from haptix.torch_dataset import WindowedDataset

        ds = WindowedDataset(
            ["ep1.hapt", "ep2.hapt", "ep3.hapt"],   # one episode per file
            window_size=32,
            stride=16,
            label="material",                        # classification
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True,
                                             num_workers=2, drop_last=True)
        for X, y in loader:                          # X: [B, 32, ...], y: [B]
            ...
    """

    def __init__(
        self,
        sources,
        window_size: int,
        stride: int | None = None,
        *,
        label: str | None = None,
        transform=None,
        target_transform=None,
        dtype="float32",
        drop_last: bool = False,
        use_unified: bool = False,
    ):
        if _torch is None:
            raise ImportError(
                "torch is required for WindowedDataset(). "
                "Install with: pip install 'haptix[torch]'"
            )
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride is None:
            stride = window_size
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if isinstance(sources, (HaptArchive, HaptData, str, Path)):
            sources = [sources]
        source_list = list(sources)
        if not source_list:
            raise ValueError("sources must contain at least one recording")
        if label is not None and label not in _LABEL_FIELD_NAMES:
            raise ValueError(
                f"Unknown label field: '{label}'. " f"Available fields: {list(_LABEL_FIELD_NAMES)}"
            )

        self._sources = [_Source(s, use_unified) for s in source_list]
        self._window_size = int(window_size)
        self._stride = int(stride)
        self._label = label
        self._transform = transform
        self._target_transform = target_transform
        self._drop_last = bool(drop_last)
        self._use_unified = bool(use_unified)
        self._dtype = getattr(_torch, dtype) if isinstance(dtype, str) else dtype
        self._label_encoding: dict | None = None
        self._label_kind: str | None = None  # "classification" | "regression"
        self._closed = False

        if label is not None:
            self._build_label_encoding()

        # Fail fast on shape inconsistency: DataLoader stacking would fail
        # at runtime anyway, but with a much less helpful error.
        if use_unified:
            shapes = {s.unified_shape for s in self._sources}
            if len(shapes) > 1:
                raise ValueError(f"sources have inconsistent unified shapes: {sorted(shapes)}")
        else:
            shapes = {s.frame_shape for s in self._sources}
            if len(shapes) > 1:
                raise ValueError(f"sources have inconsistent frame shapes: {sorted(shapes)}")

    # ── Label encoding ──────────────────────────────────────────────────

    def _build_label_encoding(self) -> None:
        values = [s.label_value(self._label) for s in self._sources]
        has_str = any(isinstance(v, str) for v in values)
        has_num = any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        has_none = any(v is None for v in values)
        if has_str and has_num:
            raise ValueError(
                f"label field '{self._label}' mixes string and numeric values across "
                "sources; use a consistent field (e.g. material for classification)"
            )
        if has_str or (has_none and not has_num):
            uniq = sorted({v for v in values if isinstance(v, str)})
            # None maps to class 0 (matches HaptData.to_torch semantics).
            self._label_encoding = {v: i + 1 for i, v in enumerate(uniq)}
            self._label_encoding[None] = 0
            self._label_kind = "classification"
        else:
            if has_none:
                raise ValueError(
                    f"label field '{self._label}' is None in some sources; numeric "
                    "label fields must be present in every source"
                )
            self._label_encoding = None
            self._label_kind = "regression"

    def _encode_label(self, value):
        assert self._label_kind is not None
        if self._label_kind == "classification":
            assert self._label_encoding is not None
            return _torch.tensor(self._label_encoding[value], dtype=_torch.long)
        return _torch.tensor(float(value), dtype=_torch.float32)

    # ── torch.utils.data.Dataset interface ──────────────────────────────

    def __len__(self) -> int:
        if self._closed:
            raise RuntimeError("WindowedDataset is closed")
        return sum(
            s.window_count(self._window_size, self._stride, drop_last=self._drop_last)
            for s in self._sources
        )

    def __getitem__(self, index: int):
        if self._closed:
            raise RuntimeError("WindowedDataset is closed")
        src_idx, local = self._locate(index)
        src = self._sources[src_idx]
        start = local * self._stride
        stop = min(start + self._window_size, src.n_frames)
        X = self._to_tensor(src.window_array(start, stop))
        if self._transform is not None:
            X = self._transform(X)
        if self._label is None:
            return X
        y = self._encode_label(src.label_value(self._label))
        if self._target_transform is not None:
            y = self._target_transform(y)
        return X, y

    def _locate(self, index: int) -> tuple[int, int]:
        """Map a flat index to (source index, window index within source)."""
        total = len(self)
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError(f"index {index} out of range for dataset of length {total}")
        remaining = index
        for i, src in enumerate(self._sources):
            cnt = src.window_count(self._window_size, self._stride, drop_last=self._drop_last)
            if remaining < cnt:
                return i, remaining
            remaining -= cnt
        raise IndexError(f"index {index} out of range")  # pragma: no cover

    def _to_tensor(self, arr: np.ndarray, dtype=None):
        dtype = dtype or self._dtype
        if arr.dtype.kind in ("i", "u"):
            # Integer types: convert to float first, then cast (matches
            # HaptData.to_torch's safe type handling).
            return _torch.from_numpy(arr.astype(np.float64)).to(dtype)
        return _torch.from_numpy(arr).to(dtype)

    # ── Introspection ───────────────────────────────────────────────────

    @property
    def n_sources(self) -> int:
        """Number of episodes in the dataset."""
        return len(self._sources)

    @property
    def n_windows(self) -> int:
        """Total number of windows (same as ``len(dataset)``)."""
        return len(self)

    @property
    def source_paths(self) -> list[Path | None]:
        """Paths of the underlying recordings (None for in-memory HaptData)."""
        return [s.path for s in self._sources]

    @property
    def label_classes(self) -> dict | None:
        """Classification mapping ``label value -> class index``, or None.

        None when ``label`` is unset or the label field is numeric
        (regression).
        """
        if self._label_encoding is None:
            return None
        return dict(self._label_encoding)

    # ── Lifecycle / pickling ────────────────────────────────────────────

    def close(self) -> None:
        """Close archives the dataset opened itself. Idempotent."""
        for s in self._sources:
            s.close()
        self._closed = True

    def __enter__(self) -> WindowedDataset:  # noqa: PYI034 - typing.Self is 3.11+; targets 3.10
        if self._closed:
            raise RuntimeError("WindowedDataset is closed")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        with suppress(Exception):  # best-effort cleanup on GC
            self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Archive handles are not picklable (zip/zarr stores); replace them
        # with their paths so workers re-open their own handles.
        state["_sources"] = [s.to_spec() for s in self._sources]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sources = [_Source.from_spec(spec, self._use_unified) for spec in self._sources]

    def __repr__(self) -> str:
        return (
            f"WindowedDataset(n_sources={self.n_sources}, n_windows={len(self)}, "
            f"window_size={self._window_size}, stride={self._stride}, "
            f"label={self._label!r})"
        )


# Episodic / temporal naming alias (see the robot-learning-framework
# roadmap item: Diffusion Policy, ACT, LeRobot all consume windowed
# episode datasets).
TemporalDataset = WindowedDataset
