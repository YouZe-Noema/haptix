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
    trained: bool  # False until fit()/load() supplies learned weights

    def encode(self, data: HaptData) -> np.ndarray:
        """[T, ...] -> [T, embedding_dim]. Deterministic."""
        ...

    def fit(self, records: list[HaptData], label_key: str = "material") -> "SensorEncoder":
        """Fit a learned linear projection from labeled records (optional)."""
        ...

    def save(self, path: Path) -> None:
        """Serialize weights (+ config) to a single .npz file."""
        ...

    @classmethod
    def load(cls, path: Path) -> "SensorEncoder":
        """Load weights from .npz, returning a ready-to-encode instance."""
        ...

    def benchmark(self, dataset: str = "unavailable") -> dict:
        """Structured report: dataset, metric, score, split (contributor contract)."""
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


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2D array.

    Zero rows are left as-is (norm set to 1 to avoid division by zero),
    matching ``haptix.unified._l2_normalize_rows``.
    """
    out = arr.astype(np.float64, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def _whitening_projection(features: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """PCA whitening matrix for centered feature rows.

    Returns ``W_w`` (D, D) such that ``(X - mean) @ W_w`` has identity
    covariance. Computed from the SVD of the centered sample matrix, so it
    is deterministic and handles n_samples < D (fewer records than dims)
    gracefully: null directions stay null (regularized).
    """
    Xc = features - features.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    inv_s = 1.0 / (s + reg)
    return (Vt.T * inv_s) @ Vt


def _class_rotation(whitened: np.ndarray, labels: list[str], reg: float = 1e-9) -> np.ndarray:
    """Orthonormal rotation that concentrates class structure (LDA-style).

    Computes the between-class scatter of the whitened samples and returns
    its eigenvector basis (D, D), with columns sorted by **descending**
    eigenvalue so class-discriminative directions come first. Columns
    beyond the effective rank (``min(C-1, D)``) are zeroed: the projection
    ``(X - mean) @ W_w @ R`` therefore lives almost entirely in the
    class-aligned subspace, and cosine/Euclidean distance on the full
    embedding reflects class structure instead of whitening noise.

    With fewer than 2 classes (or all-same labels) returns the identity:
    pure whitening, no class rotation.
    """
    unique = sorted(set(labels))
    if len(unique) < 2:
        return np.eye(whitened.shape[1], dtype=np.float64)
    # Between-class scatter in whitened space.
    S_B = np.zeros((whitened.shape[1], whitened.shape[1]), dtype=np.float64)
    for lab in unique:
        idx = [i for i, lab_i in enumerate(labels) if lab_i == lab]
        centroid = whitened[idx].mean(axis=0)
        S_B += len(idx) * np.outer(centroid, centroid)
    S_B /= len(labels)
    w, V = np.linalg.eigh((S_B + S_B.T) / 2.0 + reg * np.eye(S_B.shape[0]))
    # eigh returns ascending eigenvalues; flip so class dirs come first.
    V = V[:, ::-1]
    rank = min(len(unique) - 1, whitened.shape[1])
    if rank < whitened.shape[1]:
        V[:, rank:] = 0.0
    return V  # orthonormal basis, class-aligned leading columns (rest zeroed)


class _BaseSensorEncoder:
    """Shared implementation for deterministic per-sensor encoders.

    Subclasses declare ``sensor_type``, ``modality``, ``embedding_dim`` and
    ``version`` and are registered via ``@register_encoder``.

    Two modes:

    - **Untrained** (default): ``encode()`` is a deterministic projection —
      imaging sensors grayscale-Lanczos-resize to a ``sqrt(embedding_dim)``
      grid flattened to exactly [T, D]; dynamic sensors pad/truncate
      features to exactly [T, D].
    - **Trained**: after ``fit(records)`` (or ``load()`` of a saved weight
      file), ``encode()`` applies a learned linear projection
      ``(features - mean) @ W`` — PCA whitening followed by an
      LDA-style class-aligned rotation (docs/encoder-registry.md §6). The
      projection is pure numpy, deterministic, and serializable to a single
      ``.npz`` (``save()``/``load()``). ``trained`` is ``True`` only when
      learned weights are present, so callers can always distinguish
      placeholder embeddings from learned ones.
    """

    sensor_type: str = ""
    modality: str = "dynamic"
    embedding_dim: int = _DYNAMIC_DIM
    version: str = ""
    # Plain class attribute (not ClassVar): fit()/load() set it per-instance.
    trained: bool = False

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
        # Learned weights (None until fit()/load()).
        self._mean: np.ndarray | None = None  # (D,) centering vector
        self._W: np.ndarray | None = None  # (D, D) whitening @ rotation
        self._benchmark_report: dict | None = None  # populated by fit()

    # ── Feature extraction (shared by encode/fit) ───────────────────────

    def _extract_features(self, data: HaptData) -> np.ndarray:
        """[T, ...] -> [T, D] raw features (pre-projection)."""
        arr = data.raw.array.astype(np.float32)
        if self.modality == "imaging":
            return _encode_imaging(arr, self.embedding_dim)
        return _encode_dynamic(arr, self.embedding_dim)

    # ── Training ────────────────────────────────────────────────────────

    def fit(
        self,
        records: list[HaptData],
        label_key: str = "material",
        seed: int = 0,
    ) -> "_BaseSensorEncoder":
        """Fit a learned linear projection from labeled records.

        Learns ``mean`` + ``W`` (PCA whitening followed by an LDA-style
        class-aligned rotation, docs/encoder-registry.md §6) so that
        same-label records land close together in the embedding space.
        Pure numpy, deterministic.

        Features are extracted **per frame** (matching ``encode()``), so a
        record with 80 frames contributes 80 samples. The benchmark report
        uses leave-one-record-out nearest-centroid accuracy — each record
        is scored against centroids fit on the remaining records, so
        held-out generalization is honest even with few records.

        Parameters
        ----------
        records : list[HaptData]
            Labeled records from this sensor family. Records without the
            label fall back to ``object_name`` then ``"unknown"`` (they
            still contribute to whitening).
        label_key : str, default "material"
            Which :class:`Labels` field to group by (``material``,
            ``material_category``, ``object_name``, ``object_category``).
        seed : int, default 0
            Kept for API parity — the training path is fully deterministic.

        Returns
        -------
        _BaseSensorEncoder
            Self, fitted.
        """
        if not records:
            raise ValueError("fit() requires at least one HaptData record")

        feats: list[np.ndarray] = []
        rec_labels: list[str] = []
        rec_sizes: list[int] = []
        for rec in records:
            label = self._record_label(rec, label_key)
            feat = self._extract_features(rec)  # (T, D) per-frame features
            feats.append(feat)
            rec_labels.append(label)
            rec_sizes.append(feat.shape[0])
        X = np.concatenate(feats, axis=0).astype(np.float64)  # (n, D)
        # Per-frame label array (one label per timestep of each record).
        labels: list[str] = []
        for lab, size in zip(rec_labels, rec_sizes):
            labels.extend([lab] * size)

        self._mean = X.mean(axis=0).astype(np.float64)

        # PCA whitening: (X - mean) @ W_w has identity covariance.
        W_w = _whitening_projection(X - self._mean)

        # Class-aligned rotation in whitened space.
        whitened = (X - self._mean) @ W_w
        R = _class_rotation(whitened, labels)
        self._W = (W_w @ R).astype(np.float64)  # (D, D)

        # Benchmark: honest leave-one-record-out nearest-centroid accuracy.
        # Each fold REFITS the projection on the remaining records only
        # (whitening + class rotation), so the eval record never leaks into
        # the projection. The shipped ``self._W`` is fit on ALL records (the
        # usual "final model" practice) — the benchmark reports the honest
        # generalization estimate, which is strictly harder.
        unique = sorted(set(rec_labels))
        if len(unique) < 2:
            # Unsupervised: mean absolute off-diagonal correlation before vs
            # after whitening. Lower is better (0 = fully decorrelated).
            # Constant dimensions (e.g. zero-padded dynamic features) are
            # excluded, and NaN (constant feature) is treated as 0 corr.
            def _mean_offdiag_corr(M: np.ndarray) -> float:
                std = M.std(axis=1)
                keep = std > 1e-12
                if keep.sum() < 2:
                    return 0.0
                C = np.corrcoef(M[keep])
                off = np.abs(C - np.eye(C.shape[0]))
                return float(np.nanmean(off))

            raw_off = _mean_offdiag_corr(X.T)
            whitened_all = (X - self._mean) @ self._W
            wh_off = _mean_offdiag_corr(whitened_all.T)
            self._benchmark_report = {
                "dataset": f"fit({len(records)} records, {len(unique)} classes, {len(X)} frames)",
                "metric": "whitening_decorrelation",
                "score": round(float(wh_off), 4),
                "split": "full corpus (unsupervised — no labels)",
                "note": (
                    f"mean |off-diagonal corr| {raw_off:.3f} -> {wh_off:.3f} "
                    "(lower is better; 0 = fully decorrelated)"
                ),
            }
        else:
            correct = 0
            total = 0
            for i, lab in enumerate(rec_labels):
                lo = sum(rec_sizes[:i])
                hi = lo + rec_sizes[i]
                mask = np.ones(len(X), dtype=bool)
                mask[lo:hi] = False
                Xtr, ytr = X[mask], np.asarray(labels)[mask]
                # Refit projection on train records only (honest fold).
                mean_fold = Xtr.mean(axis=0)
                Ww_fold = _whitening_projection(Xtr - mean_fold)
                R_fold = _class_rotation((Xtr - mean_fold) @ Ww_fold, list(ytr))
                W_fold = Ww_fold @ R_fold
                # Centroids from the same train records.
                centroids: dict[str, np.ndarray] = {}
                for c in unique:
                    idx = np.where(ytr == c)[0]
                    if len(idx):
                        emb = _l2_normalize_rows((Xtr[idx] - mean_fold) @ W_fold)
                        centroids[c] = emb.mean(axis=0)
                if not centroids:
                    continue
                emb = _l2_normalize_rows((X[lo:hi] - mean_fold) @ W_fold)
                for row in emb:
                    total += 1
                    best_sim = -np.inf
                    best_label = ""
                    for c, cent in centroids.items():
                        sim = float(row @ cent / (np.linalg.norm(cent) + 1e-12))
                        if sim > best_sim:
                            best_sim = sim
                            best_label = c
                    if best_label == lab:
                        correct += 1
            acc = correct / max(1, total)
            self._benchmark_report = {
                "dataset": f"fit({len(records)} records, {len(unique)} classes, {len(X)} frames)",
                "metric": "nearest_centroid_accuracy",
                "score": round(float(acc), 4),
                "split": "leave-one-record-out, per-fold refit "
                "(projection refit on train records each fold)",
                "note": "honest generalization on the fit corpus — real benchmarks "
                "come from public datasets (docs/encoder-registry.md §4)",
            }

        # Trained encoders bump the version (e.g. encoders/gelsight/v1.0).
        base = self.version.rsplit("/", 1)[0]
        self.version = f"{base}/v1.0"
        self.trained = True  # instance attr shadows the ClassVar
        return self

    @staticmethod
    def _record_label(data: HaptData, label_key: str) -> str:
        """Extract the grouping label for a record, with fallbacks."""
        labels = data.labels
        if labels is not None:
            primary = getattr(labels, label_key, None)
            if primary:
                return str(primary)
            if labels.object_name:
                return labels.object_name
        return "unknown"

    # ── Encoding ────────────────────────────────────────────────────────

    def encode(self, data: HaptData) -> np.ndarray:
        """Encode HaptData into a fixed-dimensional embedding [T, D].

        Deterministic: same input -> same output. The output dim is
        ``self.embedding_dim`` regardless of the input shape. When trained
        weights are present the raw features pass through the learned
        projection; otherwise the untrained deterministic projection is
        used.
        """
        feat = self._extract_features(data)
        if self._W is not None and self._mean is not None:
            emb = (feat - self._mean) @ self._W
            return _l2_normalize_rows(emb).astype(np.float32)
        return feat

    def save(self, path: Path) -> None:
        """Serialize config (+ learned weights when trained) to one .npz."""
        kwargs: dict = {
            "sensor_type": self.sensor_type,
            "modality": self.modality,
            "embedding_dim": self.embedding_dim,
            "version": self.version,
            "trained": self.trained,
        }
        if self._mean is not None and self._W is not None:
            kwargs["mean"] = self._mean
            kwargs["W"] = self._W
        if self._benchmark_report is not None:
            import json

            kwargs["benchmark"] = np.frombuffer(
                json.dumps(self._benchmark_report).encode("utf-8"), dtype=np.uint8
            )
        np.savez(path, **kwargs)

    @classmethod
    def load(cls, path: Path) -> "_BaseSensorEncoder":
        """Load config (+ learned weights) from .npz.

        Raises
        ------
        ValueError
            If the file was saved by a different encoder class
            (``sensor_type`` mismatch) or is not a haptix encoder file.
        """
        import json

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
            if "mean" in files and "W" in files:
                obj._mean = z["mean"].astype(np.float64)
                obj._W = z["W"].astype(np.float64)
                obj.trained = True  # instance attr shadows the ClassVar
            elif "trained" in files and bool(z["trained"]):
                obj.trained = True
            if "benchmark" in files:
                obj._benchmark_report = json.loads(z["benchmark"].tobytes().decode("utf-8"))
        return obj

    def benchmark(self, dataset: str = "unavailable") -> dict:
        """Return a structured benchmark report (contributor contract).

        Trained encoders return the report computed at fit time (dataset,
        metric, score, split). Untrained encoders report ``score=None``
        until trained weights (or a ``fit()`` step) exist.
        """
        if self._benchmark_report is not None:
            return dict(self._benchmark_report)
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

    def fit(self, records: list[HaptData], label_key: str = "material") -> "SurrogateEncoder":
        """Not supported: the surrogate is a placeholder, not a trainable encoder.

        Raises
        ------
        NotImplementedError
            Always. Train a registered per-sensor encoder (``fit()`` on
            ``get_encoder(sensor_type)``) and save/load its weights instead.
        """
        raise NotImplementedError(
            "SurrogateEncoder is a deterministic placeholder and cannot be trained. "
            "Use a registered per-sensor encoder (get_encoder(sensor_type).fit(records))."
        )

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
