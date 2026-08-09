"""
Per-sensor encoder base: protocol, embedding-dim conventions, and the
shared untrained encoder machinery.

The per-sensor encoder is the *front-end* of the encoding stack: raw
sensor data (``HaptData``) -> fixed-dimensional embedding ``[T, D]``.
The alignment layer (``haptix.unified.CrossModalEncoder``) owns the
shared-space semantics on top of these embeddings, so a per-sensor
encoder never needs to know about other sensors.

Conventions (docs/encoder-registry.md §3.1)
-------------------------------------------
- ``embedding_dim`` is 256 for imaging sensors, 128 for dynamic sensors.
- Once an encoder's dim is published it is stable for that sensor type —
  never changes without a version bump.
- ``encode()`` is deterministic: same input -> same embedding.
- Encoders without trained weights are valid registry entries: they
  document the architecture and emit shape-correct deterministic
  embeddings until trained weights (or a ``fit()`` step) replace the
  projection. ``trained`` is ``False`` so callers can tell.
"""

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from haptix.core import HaptData

# Default embedding dims per modality (docs/encoder-registry.md §3.1).
_IMAGING_DIM = 256
_DYNAMIC_DIM = 128

# Modality of every sensor type in the adapter catalog. Adapters do not
# expose a modality attribute (it lives on HaptData), so the encoder
# registry keeps its own map — decoupled from the adapter registry.
_KNOWN_MODALITIES: dict[str, str] = {
    "GelSight": "imaging",
    "GelSight_Mini": "imaging",
    "GelSight_Wedge": "imaging",
    "DIGIT": "imaging",
    "DIGIT_v2": "imaging",
    "CoroCapacitive": "dynamic",
    "BioTac": "dynamic",
    "BioTac_SP": "dynamic",
    "TacTip": "dynamic",
}


def _modality_for(sensor_type: str) -> str:
    """Modality of a sensor type, defaulting to ``"dynamic"`` for unknown types."""
    return _KNOWN_MODALITIES.get(sensor_type, "dynamic")


def _default_dim_for(sensor_type: str) -> int:
    """Default embedding dim for a sensor type: 256 imaging / 128 dynamic.

    Unknown sensor types fall back to the dynamic dim — shape-correct,
    never raises.
    """
    return _IMAGING_DIM if _modality_for(sensor_type) == "imaging" else _DYNAMIC_DIM


@runtime_checkable
class SensorEncoder(Protocol):
    """Protocol for per-sensor encoders.

    Mirrors :class:`haptix.sensors.SensorAdapter`: one class per sensor
    family, registered by ``sensor_type``. The registry contract is the
    fixed ``embedding_dim``: once published, an encoder's dim is stable
    for that sensor type and never changes without a version bump. The
    alignment layer handles heterogeneous dims natively (CCA supports
    D_a != D_b), so mixed 256/128 encoders compose cleanly.
    """

    sensor_type: str  # "GelSight", "DIGIT", "CoroCapacitive", ...
    modality: str  # "imaging" | "dynamic" | "force" | "multimodal"
    embedding_dim: int  # fixed output dim; 256 imaging / 128 dynamic
    version: str  # "encoders/gelsight/v0.1"

    def encode(self, data: HaptData) -> np.ndarray:
        """[T, ...] -> [T, embedding_dim]. Deterministic."""
        ...

    def save(self, path: Path) -> None:
        """Serialize weights (+ config) to a single .npz file."""
        ...

    @classmethod
    def load(cls, path: Path) -> "SensorEncoder":
        """Load weights from .npz, returning a ready-to-encode instance."""
        ...


def _encode_imaging(arr: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Encode imaging frames [T, H, W, C] -> [T, embedding_dim].

    Deterministic untrained projection: grayscale-convert each frame,
    Lanczos-resize to a ``floor(sqrt(embedding_dim))`` spatial grid, then
    flatten to exactly ``embedding_dim`` floats per timestep (padded or
    truncated if the grid is not a perfect square). Grayscale keeps the
    output dimension exact regardless of channel count — RGB images would
    otherwise flatten to 3x the declared dim.
    """
    from PIL import Image

    if arr.ndim == 4:
        frames = arr  # [T, H, W, C]
    elif arr.ndim == 3:
        frames = arr[..., np.newaxis]  # [T, H, W] -> [T, H, W, 1]
    else:
        raise ValueError(f"imaging encode() expects [T, H, W, C] or [T, H, W], got {arr.shape}")

    spatial = int(np.floor(np.sqrt(embedding_dim)))
    if spatial < 1:
        raise ValueError(f"embedding_dim too small for imaging: {embedding_dim}")

    T = frames.shape[0]
    out = np.zeros((T, spatial * spatial), dtype=np.float32)
    for t in range(T):
        frame = frames[t].astype(np.float32)
        if frame.ndim == 3 and frame.shape[2] > 1:
            frame = frame.mean(axis=2)  # RGB -> grayscale
        else:
            frame = frame[..., 0]
        frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
        img = Image.fromarray(frame_u8).resize((spatial, spatial), Image.LANCZOS)
        out[t] = np.array(img, dtype=np.float32).reshape(-1) / 255.0

    return _pad_to_dim(out, embedding_dim)


def _encode_dynamic(arr: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Encode dynamic frames [T, F] -> [T, embedding_dim].

    Deterministic untrained projection: pad with zeros or truncate the
    feature columns to exactly ``embedding_dim`` per timestep. Higher-
    dimensional arrays are flattened over the non-time axes first.
    """
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return _pad_to_dim(arr, embedding_dim)


def _pad_to_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate dynamic features to ``target_dim`` columns.

    Fewer columns -> zero-pad; more columns -> keep the first
    ``target_dim``. Always returns float32 [T, target_dim].
    """
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    T, F = arr.shape
    if F == target_dim:
        return arr
    if F < target_dim:
        padded = np.zeros((T, target_dim), dtype=np.float32)
        padded[:, :F] = arr
        return padded
    return arr[:, :target_dim]


class _BaseSensorEncoder:
    """Shared implementation for deterministic (untrained) per-sensor encoders.

    Subclasses declare ``sensor_type``, ``modality``, ``embedding_dim`` and
    ``version`` and are registered via ``@register_encoder``. ``encode()``
    is a deterministic untrained projection:

    - imaging sensors: grayscale Lanczos resize to a
      ``sqrt(embedding_dim)`` grid, flattened to exactly [T, D];
    - dynamic sensors: pad/truncate features to exactly [T, D].

    Trained weights replace this projection in a later version; the fixed
    output dim is the registry contract. ``save()``/``load()`` serialize
    the config (no learned weights yet), so a loaded encoder is an
    equivalent untrained instance.
    """

    sensor_type: str = ""
    modality: str = "dynamic"
    embedding_dim: int = _DYNAMIC_DIM
    version: str = ""
    trained: ClassVar[bool] = False

    def __init__(self, embedding_dim: int | None = None):
        """Create an encoder instance.

        Parameters
        ----------
        embedding_dim : int, optional
            Override the class-declared output dim. Defaults to the
            class attribute (the registry-validated contract).
        """
        if embedding_dim is not None:
            self.embedding_dim = int(embedding_dim)

    def encode(self, data: HaptData) -> np.ndarray:
        """Encode HaptData into a fixed-dimensional embedding [T, D].

        Deterministic: same input -> same output. The output dim is
        ``self.embedding_dim`` regardless of the input shape.
        """
        arr = data.raw.array.astype(np.float32)
        if self.modality == "imaging":
            return _encode_imaging(arr, self.embedding_dim)
        return _encode_dynamic(arr, self.embedding_dim)

    def save(self, path: Path) -> None:
        """Serialize config to a single .npz file.

        Untrained encoders have no learned weights, so this writes the
        config (sensor_type, modality, embedding_dim, version). A future
        trained version appends weight arrays to the same file.
        """
        np.savez(
            path,
            sensor_type=self.sensor_type,
            modality=self.modality,
            embedding_dim=self.embedding_dim,
            version=self.version,
            trained=self.trained,
        )

    @classmethod
    def load(cls, path: Path) -> "_BaseSensorEncoder":
        """Load config from .npz, returning a ready-to-encode instance.

        Raises
        ------
        ValueError
            If the file was saved by a different encoder class
            (``sensor_type`` mismatch).
        """
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            if "embedding_dim" not in files:
                raise ValueError(f"{path} is not a haptix encoder file")
            saved_type = str(z["sensor_type"]) if "sensor_type" in files else ""
            if saved_type and saved_type != cls.sensor_type:
                raise ValueError(
                    f"{path} was saved by encoder '{saved_type}', not '{cls.sensor_type}'"
                )
            obj = cls(embedding_dim=int(z["embedding_dim"]))
            if "version" in files:
                obj.version = str(z["version"])
        return obj

    def benchmark(self, dataset: str = "unavailable") -> dict:
        """Return a structured benchmark report (contributor contract).

        Untrained encoders cannot be meaningfully evaluated, so the score
        is ``None`` until trained weights (or a ``fit()`` step) exist.
        Trained contributions must return dataset, metric, score, split
        (docs/encoder-registry.md §4).
        """
        return {
            "dataset": dataset,
            "metric": "unavailable",
            "score": None,
            "split": "unavailable",
            "note": "untrained encoder — train weights or fit() before benchmarking",
        }


class SurrogateEncoder:
    """Deterministic surrogate served by ``get_encoder()`` as fallback.

    Returned for sensor types without a registered encoder. Replicates
    the deterministic semantics of ``haptix.unified.SharedForceEncoder``
    with a dim-exact imaging path (grayscale resize) so every embedding
    honors the fixed ``embedding_dim`` contract. The version tag carries
    ``/surrogate`` so placeholder embeddings are never mistaken for
    learned ones.
    """

    trained: ClassVar[bool] = False

    def __init__(self, sensor_type: str, embedding_dim: int, modality: str = "dynamic"):
        self.sensor_type = sensor_type
        self.modality = modality
        self.embedding_dim = int(embedding_dim)
        self.version = "unified/shared-force/v0.1/surrogate"

    def encode(self, data: HaptData) -> np.ndarray:
        """Encode HaptData into a deterministic [T, embedding_dim] embedding."""
        arr = data.raw.array.astype(np.float32)
        if self.modality == "imaging":
            return _encode_imaging(arr, self.embedding_dim)
        return _encode_dynamic(arr, self.embedding_dim)

    def save(self, path: Path) -> None:
        """Serialize config to a single .npz file."""
        np.savez(
            path,
            sensor_type=self.sensor_type,
            modality=self.modality,
            embedding_dim=self.embedding_dim,
            version=self.version,
            trained=False,
        )

    @classmethod
    def load(cls, path: Path) -> "SurrogateEncoder":
        """Load config from .npz, returning a ready-to-encode instance."""
        with np.load(path, allow_pickle=False) as z:
            if "embedding_dim" not in set(z.files):
                raise ValueError(f"{path} is not a haptix encoder file")
            obj = cls(
                sensor_type=str(z["sensor_type"]),
                embedding_dim=int(z["embedding_dim"]),
                modality=str(z["modality"]),
            )
            if "version" in set(z.files):
                obj.version = str(z["version"])
        return obj

    def benchmark(self, dataset: str = "unavailable") -> dict:
        """Return a structured benchmark report (score None: surrogate)."""
        return {
            "dataset": dataset,
            "metric": "unavailable",
            "score": None,
            "split": "unavailable",
            "note": "surrogate encoder — no learned weights to benchmark",
        }
