"""
Cross-sensor unified representation encoders.

Inspired by UniForce (arXiv 2602.01153): learns a shared latent force space
across diverse tactile sensors by jointly modeling inverse dynamics
(image-to-force) and forward dynamics (force-to-image).

This module provides a clean encoder interface and a prototype
SharedForceEncoder that maps different sensor modalities into a common
embedding space. Encoders are versioned so any .hapt file carrying
unified data can trace its transform back to a specific encoder.

Design
------
- Imaging sensors (GelSight, DIGIT): [T, H, W, C] → spatial encoder → [T, D]
- Dynamic sensors (Coro, BioTac, TacTip): [T, F] → temporal encoder → [T, D]
- Output: fixed-dimensional shared embedding vectors per timestep

The encoder uses SURROGATE (untrained) projections for the prototype.
In production, encoders should be trained via cross-sensor alignment
losses (reconstruction + force equilibrium) per the UniForce paper.
The interface stays the same — only the weights change.
"""

from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from haptix.core import HaptData, UnifiedData
from haptix.sensors import list_sensors

# Shared embedding dimension — all modalities map to this size.
_DEFAULT_EMBEDDING_DIM = 128

# Encoder version — bump when projection logic changes.
_ENCODER_VERSION = "unified/shared-force/v0.1"


@runtime_checkable
class UnifiedEncoder(Protocol):
    """Protocol for cross-sensor unified encoders.

    An encoder takes :class:`HaptData` from any sensor modality and
    produces a :class:`UnifiedData` object containing fixed-dimensional
    embeddings in a shared latent space.

    Implementations must be deterministic: same input → same output.
    """

    version: str

    def encode(self, data: HaptData) -> UnifiedData:
        """Encode HaptData into unified representation."""
        ...


def _resize_image_embedding(
    arr: np.ndarray, target_dim: int = _DEFAULT_EMBEDDING_DIM
) -> np.ndarray:
    """Downsample imaging modality frames to a fixed spatial embedding.

    Strategy: high-quality Lanczos resize each frame to target_dim × target_dim,
    then flatten to [T, target_dim²]. This preserves spatial structure
    while producing uniform-size vectors.
    """
    from PIL import Image

    T, C = arr.shape[0], arr.shape[3]

    # Convert to [T, target_dim, target_dim, C] via PIL bicubic resize
    resized = np.zeros((T, target_dim, target_dim, C), dtype=np.float32)

    for t in range(T):
        frame = arr[t].astype(np.uint8)
        if C == 1:
            frame = frame.squeeze(-1)
        pil_img = Image.fromarray(frame).resize((target_dim, target_dim), Image.LANCZOS)
        np_frame = np.array(pil_img, dtype=np.float32)
        if np_frame.ndim == 2:
            np_frame = np_frame[..., np.newaxis]
        resized[t] = np_frame

    # Normalize to [0, 1] and flatten
    resized /= 255.0
    return resized.reshape(T, -1)


def _pad_to_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate dynamic features to target_dim.

    If arr has fewer columns than target_dim: pad with zeros.
    If arr has more columns: use first target_dim columns.
    Handles 1D arrays by reshaping to (1, -1) first.
    """
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


class SharedForceEncoder:
    """Prototype encoder mapping tactile data into a shared latent space.

    Inspired by the UniForce framework, this encoder uses surrogate
    (untrained) projections to produce fixed-dimensional embeddings.
    The same interface will work when trained weights are plugged in.

    Modality routing
    ----------------
    - ``imaging`` (GelSight, DIGIT): spatial resize → flat → [T, D²]
    - ``dynamic`` (Coro, BioTac, TacTip): pad/trunc → [T, D²]
    - ``force``, ``multimodal``: dynamic path as default

    Parameters
    ----------
    embedding_dim : int, default 128
        Target dimension for the shared latent space. Image frames are
        resized to sqrt(embedding_dim) × sqrt(embedding_dim), so use
        a perfect square for clean spatial mapping (128 → not square;
        prefer 64, 144, 256 for image modality).
    seed : int, optional
        RNG seed for deterministic random projection (if used).
    """

    version = _ENCODER_VERSION

    # Sensors the encoder supports (ClassVar: shared across instances)
    _IMAGING_SENSORS: ClassVar[set[str]] = {
        "GelSight",
        "GelSight_Mini",
        "GelSight_Wedge",
        "DIGIT",
        "DIGIT_v2",
    }
    _DYNAMIC_SENSORS: ClassVar[set[str]] = {"CoroCapacitive", "BioTac_SP", "TacTip"}

    def __init__(self, embedding_dim: int = _DEFAULT_EMBEDDING_DIM, seed: int = 42):
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed)

    def encode(self, data: HaptData) -> UnifiedData:
        """Encode HaptData into a shared latent representation.

        Parameters
        ----------
        data : HaptData
            Loaded tactile data from any supported sensor.

        Returns
        -------
        UnifiedData
            Cross-sensor embedding with shape [T, D] where D is the
            shared embedding dimension. ``method`` records the encoder
            version and transform parameters.
        """
        arr = data.raw.array.astype(np.float32)
        modality = data.modality

        # Route by sensor type for modality-specific encoding
        sensor_type = data.sensor.type

        if sensor_type in self._IMAGING_SENSORS or modality == "imaging":
            embedding = self._encode_imaging(arr)
        elif sensor_type in self._DYNAMIC_SENSORS or modality in ("dynamic", "force"):
            embedding = self._encode_dynamic(arr)
        else:
            # Unknown sensor — treat as dynamic (vector) path
            embedding = self._encode_dynamic(arr)

        return UnifiedData(
            array=embedding,
            method=self._method_tag(sensor_type),
            source_modality=modality,
            target_modality="shared_force_" + str(self.embedding_dim) + "d",
            is_lossy=True,
            checksum=(
                UnifiedData.compute_checksum(embedding)
                if hasattr(UnifiedData, "compute_checksum")
                else self._sha256(embedding)
            ),
        )

    def _encode_imaging(self, arr: np.ndarray) -> np.ndarray:
        """Encode imaging modality [T, H, W, C] → [T, D].

        Uses bicubic resize to a fixed spatial grid, then flattens.
        For a 64-dim embedding, frames become 8×8 spatial embeddings.
        """
        # Use the largest perfect square ≤ embedding_dim for spatial grid
        spatial_dim = int(np.floor(np.sqrt(self.embedding_dim)))
        return _resize_image_embedding(arr, target_dim=spatial_dim)

    def _encode_dynamic(self, arr: np.ndarray) -> np.ndarray:
        """Encode dynamic modality [T, F] → [T, D].

        Pads or truncates feature columns to embedding_dim.
        Production encoders would use a learned projection here.
        """
        # Flatten if ndim > 2 (unlikely but safe)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return _pad_to_dim(arr, self.embedding_dim)

    def _method_tag(self, sensor_type: str) -> str:
        """Produce a versioned method tag for provenance."""
        return f"{_ENCODER_VERSION}/{sensor_type}"

    @staticmethod
    def _sha256(arr: np.ndarray) -> str:
        import hashlib

        return hashlib.sha256(arr.tobytes()).hexdigest()

    @classmethod
    def supported_sensors(cls) -> list[str]:
        """List sensors this encoder can handle."""
        all_registered = list_sensors()
        return [s for s in all_registered if s in cls._IMAGING_SENSORS or s in cls._DYNAMIC_SENSORS]
