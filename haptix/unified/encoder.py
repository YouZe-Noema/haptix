"""
Cross-sensor unified representation encoders.

Inspired by UniForce (arXiv 2602.01153): learns a shared latent force space
across diverse tactile sensors by jointly modeling inverse dynamics
(image-to-force) and forward dynamics (force-to-image).

This module provides a clean encoder interface and two implementations:

- :class:`SharedForceEncoder` — deterministic surrogate (untrained)
  projections that map different sensor modalities into a common embedding
  space. Zero dependencies, always available, useful for shape checks and
  quick experiments.
- :class:`CrossModalEncoder` — a TRAINED encoder that learns per-modality
  linear projections into a shared space via canonical correlation analysis
  (CCA) over class-aligned centroids. Weights are serializable to a single
  ``.npz`` file, so encoders can be pre-trained, versioned, and shipped.

Encoders are versioned so any .hapt file carrying unified data can trace
its transform back to a specific encoder.

Design
------
- Imaging sensors (GelSight, DIGIT): [T, H, W, C] → spatial encoder → [T, D]
- Dynamic sensors (Coro, BioTac, TacTip): [T, F] → temporal encoder → [T, D]
- Output: fixed-dimensional shared embedding vectors per timestep
"""

import hashlib
import json
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from haptix.core import HaptData, UnifiedData
from haptix.sensors import list_sensors

# Shared embedding dimension — all modalities map to this size.
_DEFAULT_EMBEDDING_DIM = 128

# Encoder version — bump when projection logic changes.
_ENCODER_VERSION = "unified/shared-force/v0.1"

# Cross-modal (trained) encoder version.
_CROSS_MODAL_VERSION = "unified/cross-modal/v0.2"


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


# ═══════════════════════════════════════════════════════════════════════════
# Trained cross-modal encoder
# ═══════════════════════════════════════════════════════════════════════════


def _safe_svd_components(matrix: np.ndarray, n_components: int) -> np.ndarray:
    """Top-n_components right singular vectors, zero-padded to requested size.

    Handles the common case where n_samples < n_components (fewer classes
    than embedding dims): SVD returns at most min(shape) vectors, the rest
    are zero columns so the shared space keeps a fixed [T, D] shape.
    """
    _, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    rank = int((S > 1e-12).sum())
    k = min(n_components, rank, Vt.shape[0])
    out = np.zeros((matrix.shape[1], n_components), dtype=np.float64)
    if k > 0:
        out[:, :k] = Vt[:k].T
    return out


def _inv_sqrt_psd(matrix: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Matrix^{-1/2} for a symmetric PSD matrix via eigendecomposition.

    Adds ``reg`` to the diagonal before inverting so near-singular
    covariance matrices (few classes → low rank) stay numerically stable.
    Eigenvalues are clamped to a positive floor.
    """
    n = matrix.shape[0]
    mat = (matrix + matrix.T) / 2.0 + reg * np.eye(n)
    w, V = np.linalg.eigh(mat)
    w = np.clip(w, 1e-10, None)
    return (V / np.sqrt(w)) @ V.T


def _cca_projections(
    A: np.ndarray, B: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical correlation analysis between two paired sample matrices.

    Parameters
    ----------
    A : (K, D_a) centered samples (e.g. imaging class centroids)
    B : (K, D_b) centered samples (e.g. dynamic class centroids)
    n_components : int
        Maximum number of canonical directions to return.

    Returns
    -------
    W_a : (D_a, d) projection for A
    W_b : (D_b, d) projection for B
    corrs : (d,) canonical correlations, in decreasing order

    Both projections map into a shared space where paired samples are
    maximally correlated; directions are weighted by their canonical
    correlation so weakly-aligned directions contribute less.
    """
    D_a, D_b = A.shape[1], B.shape[1]
    S_aa = A.T @ A
    S_bb = B.T @ B
    S_ab = A.T @ B

    Wa_inv = _inv_sqrt_psd(S_aa)
    Wb_inv = _inv_sqrt_psd(S_bb)
    K = Wa_inv @ S_ab @ Wb_inv
    U, corrs, Vt = np.linalg.svd(K)

    d = min(n_components, len(corrs))
    W_a = (Wa_inv @ U[:, :d]).T  # (d, D_a)
    W_b = (Wb_inv @ Vt[:d].T).T  # (d, D_b)
    # Weight each canonical direction by its correlation.
    W_a = W_a * corrs[:d, None]
    W_b = W_b * corrs[:d, None]

    # Zero-pad to exactly n_components rows so output stays [T, D].
    if d < n_components:
        pad_a = np.zeros((n_components - d, D_a), dtype=np.float64)
        pad_b = np.zeros((n_components - d, D_b), dtype=np.float64)
        W_a = np.concatenate([W_a, pad_a], axis=0)
        W_b = np.concatenate([W_b, pad_b], axis=0)

    return W_a, W_b, corrs[:d]


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2D array in place of a new array.

    Zero rows are left as-is (norm set to 1 to avoid division by zero).
    """
    out = arr.astype(np.float64, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


class CrossModalEncoder:
    """Trained cross-sensor encoder mapping modalities into a shared space.

    Learns per-modality linear projections from labeled records so that
    samples with the same label (e.g. material) land close together in a
    common embedding space — the "cross-sensor latent force space" idea
    from UniForce, implemented with classical linear algebra.

    Training procedure (pure numpy, no sklearn):
      1. Per modality, extract feature vectors (imaging → spatial resize,
         dynamic → padded columns), mean-pooled over time per record.
      2. Per modality, compute class centroids in raw feature space for
         labels present in BOTH modalities.
      3. Run canonical correlation analysis (CCA) on the two centered
         centroid matrices: the resulting projections map each modality
         into a shared space where same-label centroids are maximally
         correlated. Directions are weighted by their canonical
         correlation, so weakly-aligned structure contributes less.
      4. Store mean, CCA projections as "weights".

    The resulting encoder is deterministic and serializable via
    :meth:`save` / :meth:`load`, so it can be pre-trained on one dataset
    and applied to any other — the ``unified/`` transform is versioned by
    the encoder version string.

    Parameters
    ----------
    embedding_dim : int, default 64
        Shared embedding dimension. With few classes the effective rank is
        lower; unused columns are zero (fixed [T, D] output shape).
    seed : int, optional
        Unused (deterministic), kept for API parity with
        :class:`SharedForceEncoder`.
    """

    version = _CROSS_MODAL_VERSION

    _IMAGING_SENSORS: ClassVar[set[str]] = {
        "GelSight",
        "GelSight_Mini",
        "GelSight_Wedge",
        "DIGIT",
        "DIGIT_v2",
    }
    _DYNAMIC_SENSORS: ClassVar[set[str]] = {"CoroCapacitive", "BioTac_SP", "TacTip"}

    def __init__(self, embedding_dim: int = 64, seed: int | None = None):
        self.embedding_dim = int(embedding_dim)
        # Fitted weights (None until fit()/load()):
        self._fitted: bool = False
        self._mean_img: np.ndarray | None = None
        self._mean_dyn: np.ndarray | None = None
        self._W_img: np.ndarray | None = None  # (D_img, D)
        self._W_dyn: np.ndarray | None = None  # (D_dyn, D)
        self._R: np.ndarray | None = None  # (D, D) Procrustes rotation
        self._classes: list[str] = []
        self._n_records: int = 0

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        """Whether fit()/load() has populated trained weights."""
        return self._fitted

    @property
    def classes(self) -> list[str]:
        """Labels present in BOTH modalities used for centroid alignment."""
        return list(self._classes)

    @property
    def n_records(self) -> int:
        """Number of HaptData records the encoder was trained on."""
        return self._n_records

    # ── Training ─────────────────────────────────────────────────────────

    def fit(self, records: list[HaptData], label_key: str = "material") -> "CrossModalEncoder":
        """Fit per-modality projections from labeled records.

        Parameters
        ----------
        records : list[HaptData]
            Records from ANY mixture of imaging and dynamic sensors. Every
            record must carry a label (``labels.material`` by default) so
            class centroids can be aligned across modalities.
        label_key : str, default "material"
            Which :class:`Labels` field to group by. Options: ``material``,
            ``material_category``, ``object_name``, ``object_category``.

        Returns
        -------
        CrossModalEncoder
            Self, fitted. ``encode()`` is now meaningful.
        """
        if not records:
            raise ValueError("fit() requires at least one HaptData record")

        img_feats: list[np.ndarray] = []
        dyn_feats: list[np.ndarray] = []
        img_labels: list[str] = []
        dyn_labels: list[str] = []

        for rec in records:
            label = self._record_label(rec, label_key)
            arr = rec.raw.array.astype(np.float64)
            if self._is_imaging(rec):
                img_feats.append(self._extract_imaging_flat(arr).mean(axis=0))
                img_labels.append(label)
            else:
                dyn_feats.append(self._extract_dynamic_flat(arr).mean(axis=0))
                dyn_labels.append(label)

        if not img_feats or not dyn_feats:
            raise ValueError(
                "fit() needs at least one imaging record AND one dynamic record "
                "to align modalities across a shared label space"
            )

        X_img = np.stack(img_feats)  # (n_img, D_img)
        X_dyn = np.stack(dyn_feats)  # (n_dyn, D_dyn)

        # 1. Per-modality means (centering).
        self._mean_img = X_img.mean(axis=0)
        self._mean_dyn = X_dyn.mean(axis=0)

        # 2. Class centroids in RAW feature space, for labels in BOTH
        #    modalities. CCA aligns the two centroid sets directly — no
        #    unsupervised whitening that could discard class structure.
        img_centroids: dict[str, np.ndarray] = {}
        dyn_centroids: dict[str, np.ndarray] = {}
        for label in set(img_labels) & set(dyn_labels):
            idx_img = [i for i, lab in enumerate(img_labels) if lab == label]
            idx_dyn = [i for i, lab in enumerate(dyn_labels) if lab == label]
            img_centroids[label] = X_img[idx_img].mean(axis=0) - self._mean_img
            dyn_centroids[label] = X_dyn[idx_dyn].mean(axis=0) - self._mean_dyn

        shared_labels = sorted(img_centroids.keys())
        self._classes = shared_labels

        if len(shared_labels) >= 2:
            # 3. CCA between centered class-centroid matrices: finds the
            #    directions in each modality's feature space that are
            #    maximally correlated across modalities, so same-label
            #    records land close in the shared space.
            C_img = np.stack([img_centroids[lab] for lab in shared_labels])
            C_dyn = np.stack([dyn_centroids[lab] for lab in shared_labels])
            W_img, W_dyn, _ = _cca_projections(C_img, C_dyn, self.embedding_dim)
            # Store as (D, n_components) so encode() uses centered @ W.
            self._W_img = W_img.T  # (D_img, D)
            self._W_dyn = W_dyn.T  # (D_dyn, D)
        else:
            # Fewer than 2 shared classes — nothing to align. Fall back to
            # per-modality PCA whitening so encode() still yields a
            # fixed-shape embedding.
            self._W_img = _safe_svd_components(X_img - self._mean_img, self.embedding_dim)
            self._W_dyn = _safe_svd_components(X_dyn - self._mean_dyn, self.embedding_dim)

        # Rotation is folded into the CCA projections; identity kept for
        # serialization compatibility.
        self._R = np.eye(self.embedding_dim, dtype=np.float64)

        self._n_records = len(records)
        self._fitted = True
        return self

    def encode(self, data: HaptData) -> UnifiedData:
        """Encode HaptData into the trained shared latent space.

        Routes by sensor modality (imaging vs dynamic), applies the learned
        CCA projection, then L2-normalizes each timestep so both modalities
        live on the same unit sphere (comparable cosine/Euclidean distances).

        Returns
        -------
        UnifiedData
            Cross-sensor embedding [T, D] with versioned method tag.
        """
        if not self._fitted:
            raise ValueError(
                "CrossModalEncoder is not fitted. Call fit(records) with labeled "
                "HaptData records, or load() a saved encoder."
            )
        assert self._mean_img is not None and self._mean_dyn is not None
        assert self._W_img is not None and self._W_dyn is not None and self._R is not None
        arr = data.raw.array.astype(np.float64)
        if self._is_imaging(data):
            feat = self._extract_imaging_flat(arr)  # (T, D_img)
            if feat.shape[1] != self._mean_img.shape[0]:
                raise ValueError(
                    f"Imaging feature dim mismatch: data has {feat.shape[1]} "
                    f"(spatial {int(np.sqrt(feat.shape[1] // arr.shape[-1]))}² × "
                    f"{arr.shape[-1]} ch), encoder was trained with "
                    f"{self._mean_img.shape[0]} (spatial "
                    f"{int(np.sqrt(self._mean_img.shape[0] // 3))}² × 3 ch). "
                    "Train or reload an encoder on matching image dimensions."
                )
            centered = feat - self._mean_img
            emb = centered @ self._W_img  # (T, D)
        else:
            feat = self._extract_dynamic_flat(arr)  # (T, D_dyn)
            if feat.shape[1] != self._mean_dyn.shape[0]:
                raise ValueError(
                    f"Dynamic feature dim mismatch: data has {feat.shape[1]} "
                    f"columns, encoder was trained with {self._mean_dyn.shape[0]}. "
                    "Train or reload an encoder on matching feature counts."
                )
            centered = feat - self._mean_dyn
            emb = centered @ self._W_dyn @ self._R  # (T, D)

        # L2-normalize each timestep: makes imaging and dynamic embeddings
        # directly comparable in the shared space regardless of source scale.
        emb = _l2_normalize_rows(emb)

        emb = emb.astype(np.float32)
        return UnifiedData(
            array=emb,
            method=self._method_tag(data.sensor.type),
            source_modality=data.modality,
            target_modality="shared_force_" + str(self.embedding_dim) + "d",
            is_lossy=True,
            checksum=hashlib.sha256(emb.tobytes()).hexdigest(),
        )

    # ── Serialization ────────────────────────────────────────────────────

    def save(self, path) -> None:
        """Serialize trained weights to a single .npz file.

        The file embeds the encoder version and training summary, so any
        ``unified/`` transform that references it stays traceable.
        """
        if not self._fitted:
            raise ValueError("Cannot save an unfitted encoder — call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self._mean_img is not None and self._W_img is not None
        assert self._mean_dyn is not None and self._W_dyn is not None
        assert self._R is not None
        metadata = {
            "version": self.version,
            "embedding_dim": self.embedding_dim,
            "classes": self._classes,
            "n_records": self._n_records,
            "mean_img_shape": list(self._mean_img.shape),
            "mean_dyn_shape": list(self._mean_dyn.shape),
            "W_img_shape": list(self._W_img.shape),
            "W_dyn_shape": list(self._W_dyn.shape),
        }
        np.savez(
            path,
            mean_img=self._mean_img,
            mean_dyn=self._mean_dyn,
            W_img=self._W_img,
            W_dyn=self._W_dyn,
            R=self._R,
            metadata=np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8),
        )

    @classmethod
    def load(cls, path) -> "CrossModalEncoder":
        """Load a trained encoder from a .npz file written by :meth:`save`."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(z["metadata"].tobytes().decode("utf-8"))
            enc = cls(embedding_dim=int(metadata["embedding_dim"]))
            enc._mean_img = z["mean_img"].astype(np.float64)
            enc._mean_dyn = z["mean_dyn"].astype(np.float64)
            enc._W_img = z["W_img"].astype(np.float64)
            enc._W_dyn = z["W_dyn"].astype(np.float64)
            enc._R = z["R"].astype(np.float64)
            enc._classes = list(metadata["classes"])
            enc._n_records = int(metadata["n_records"])
            enc._fitted = True
        return enc

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _record_label(data: HaptData, label_key: str) -> str:
        """Extract the grouping label for a record, with fallbacks.

        Prefers ``labels.<label_key>``, then ``object_name``, then
        ``"unknown"`` so unlabeled records still participate in the
        per-modality whitening step.
        """
        labels = data.labels
        if labels is not None:
            primary = getattr(labels, label_key)
            if primary:
                return primary
            if labels.object_name:
                return labels.object_name
        return "unknown"

    def _is_imaging(self, data: HaptData) -> bool:
        sensor_type = data.sensor.type
        return sensor_type in self._IMAGING_SENSORS or data.modality == "imaging"

    def _extract_imaging_flat(self, arr: np.ndarray) -> np.ndarray:
        """Imaging [T, H, W, C] → [T, spatial² · C] float64 features."""
        # Match SharedForceEncoder: resize to sqrt(embedding_dim) grid.
        spatial_dim = max(1, int(np.floor(np.sqrt(self.embedding_dim))))
        return _resize_image_embedding(arr, target_dim=spatial_dim).astype(np.float64)

    def _extract_dynamic_flat(self, arr: np.ndarray) -> np.ndarray:
        """Dynamic [T, F] → [T, D_dyn] float64 features (pad/truncate)."""
        return _pad_to_dim(arr.astype(np.float32), self.embedding_dim).astype(np.float64)

    def _method_tag(self, sensor_type: str) -> str:
        return f"{_CROSS_MODAL_VERSION}/{sensor_type}"
