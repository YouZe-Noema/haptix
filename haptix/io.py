"""
Load and save .hapt files.

.hapt files are directories (or single-file archives) with:
  manifest.json  — sensor meta + interaction params
  raw/           — native sensor data + checksum
  unified/       — optional cross-sensor rep
  labels.json    — annotations

Compressed single-file modes:
  .hapt.zarr — ZipStore with Zstd-compressed chunks (needs zarr + numcodecs)
  .hapt.zip  — plain ZIP archive (stdlib only, DEFLATE compression)

Both maintain the same logical structure as the directory format.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    Provenance,
    RawData,
    SensorMeta,
    Source,
    UnifiedData,
)

# Optional zarr import for compressed .hapt.zarr format
_zarr = None
_numcodecs = None


def _ensure_zarr():
    """Lazy-import zarr and numcodecs. Raises ImportError if unavailable."""
    global _zarr, _numcodecs
    if _zarr is None:
        try:
            import numcodecs as _nc_mod
            import zarr as _zarr_mod

            _zarr = _zarr_mod
            _numcodecs = _nc_mod
        except ImportError:
            raise ImportError(
                "zarr and numcodecs are required for .hapt.zarr format. "
                "Install with: pip install 'haptix[all]' or pip install zarr numcodecs"
            ) from None
    return _zarr, _numcodecs


def _zarr_major_version(zarr) -> int:
    """Return the major version of the installed zarr package (2, 3, ...)."""
    return int(zarr.__version__.split(".")[0])


def _zarr_group(zarr, store):
    """Create a group, compatible with both zarr 2.x and 3.x.

    zarr 2.x only knows format 2 and rejects the zarr_format kwarg;
    zarr 3.x defaults to format 3, so we pin zarr_format=2 there for
    compressor compatibility (Blosc/Zstd works with format 2)."""
    if _zarr_major_version(zarr) >= 3:
        return zarr.group(store=store, zarr_format=2)
    return zarr.group(store=store)


def _zarr_create_array(root, name, shape, dtype, chunks, compressor):
    """Create a chunked array, compatible with both zarr 2.x and 3.x.

    zarr 2.x rejects the zarr_format kwarg; 3.x needs it to stay on
    format 2 (matching the group created by _zarr_group)."""
    if _zarr_major_version(_zarr) >= 3:
        return root.zeros(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compressor=compressor,
            zarr_format=2,
        )
    return root.zeros(
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=compressor,
    )


class ChecksumError(ValueError):
    """Raised when checksum verification fails."""


class HaptFormatError(ValueError):
    """Raised when .hapt file structure is invalid."""


def load(path: str | Path) -> HaptData:
    """Load a .hapt file from disk.

    Supports four formats:
    - Directory: traditional .hapt directory with raw/data.npy
    - .hapt.zarr: single-file ZipStore with Zstd compression
    - .hapt.zip: single-file ZIP archive (stdlib DEFLATE)
    - .hapt (flat file): unsupported, raises HaptFormatError

    Automatically verifies checksum. Raises ChecksumError on mismatch."""
    path = Path(path)

    # Support .hapt.zarr compressed format (single file)
    if path.suffix == ".zarr" and path.is_file():
        return _load_zarr(path)

    # Support .hapt.zip single-file archive
    if path.suffix == ".zip" and path.is_file():
        return _load_zip(path)

    # Support both directory and legacy flat file
    if path.is_dir():
        return _load_dir(path)
    if path.suffix == ".hapt" and path.is_file():
        raise HaptFormatError(
            "Flat .hapt files are not supported. Use directory format, " ".hapt.zarr, or .hapt.zip."
        )
    raise FileNotFoundError(f"Not a valid .hapt path: {path}")


def _load_dir(path: Path) -> HaptData:
    manifest_path = path / "manifest.json"
    raw_dir = path / "raw"
    labels_path = path / "labels.json"
    provenance_path = path / "provenance.json"
    unified_dir = path / "unified"

    if not manifest_path.exists():
        raise HaptFormatError(f"Missing manifest.json in {path}")
    if not raw_dir.exists():
        raise HaptFormatError(f"Missing raw/ directory in {path}")
    if not labels_path.exists():
        raise HaptFormatError(f"Missing labels.json in {path}")

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load raw data
    raw_data_path = raw_dir / "data.npy"
    checksum_path = raw_dir / "checksum.sha256"

    if not raw_data_path.exists():
        raise HaptFormatError("Missing raw/data.npy")

    raw_array = np.load(raw_data_path)
    computed_checksum = hashlib.sha256(raw_array.tobytes()).hexdigest()

    # Verify checksum
    if checksum_path.exists():
        with open(checksum_path) as f:
            stored_checksum = f.read().strip()
    else:
        stored_checksum = computed_checksum

    if computed_checksum != stored_checksum:
        raise ChecksumError(
            f"Checksum mismatch! Stored: {stored_checksum[:16]}..., "
            f"Computed: {computed_checksum[:16]}..."
        )

    raw = RawData(
        array=raw_array,
        checksum=computed_checksum,
        dtype=str(raw_array.dtype),
        shape=raw_array.shape,
    )

    # Validate required manifest fields
    required_manifest_fields = {
        "sensor": "Missing 'sensor' in manifest.json",
        "modality": "Missing 'modality' in manifest.json",
        "sampling": "Missing 'sampling' in manifest.json",
        "interaction": "Missing 'interaction' in manifest.json",
    }
    for field, msg in required_manifest_fields.items():
        if field not in manifest:
            raise HaptFormatError(msg)

    # Validate nested fields
    if "rate_hz" not in manifest.get("sampling", {}):
        raise HaptFormatError("Missing 'sampling.rate_hz' in manifest.json")

    # Parse metadata
    sensor = SensorMeta.from_dict(manifest["sensor"])

    interaction = InteractionMeta.from_dict(manifest["interaction"])

    # Load labels
    with open(labels_path) as f:
        labels_dict = json.load(f)
    labels = Labels.from_dict(labels_dict)

    # Optional unified
    unified = None
    if unified_dir.exists():
        unified_data_path = unified_dir / "data.npy"
        transform_path = unified_dir / "transform.json"
        if unified_data_path.exists() and transform_path.exists():
            unified_array = np.load(unified_data_path)
            with open(transform_path) as f:
                transform = json.load(f)
            unified = UnifiedData(
                array=unified_array,
                method=transform["method"],
                source_modality=transform["source_modality"],
                target_modality=transform["target_modality"],
                is_lossy=transform.get("is_lossy", True),
                checksum=hashlib.sha256(unified_array.tobytes()).hexdigest(),
            )

    # Optional provenance (v0.2+)
    provenance = None
    if provenance_path.exists():
        with open(provenance_path) as f:
            provenance_dict = json.load(f)
        provenance = Provenance.from_dict(provenance_dict)

    # New v0.2 manifest fields (backward-compatible: default to None)
    coordinate_frame = manifest.get("coordinate_frame")
    timestamps_s = manifest.get("sampling", {}).get("timestamps_s")

    return HaptData(
        raw=raw,
        sensor=sensor,
        modality=manifest["modality"],
        sampling_rate_hz=manifest["sampling"]["rate_hz"],
        interaction=interaction,
        labels=labels,
        unified=unified,
        provenance=provenance,
        coordinate_frame=coordinate_frame,
        timestamps_s=timestamps_s,
        version=manifest.get("version", "0.1.0"),
    )


def save(data: HaptData, path: str | Path) -> Path:
    """Save HaptData to a .hapt file.

    Format is determined by file extension:
    - .hapt or directory path → traditional directory format
    - .hapt.zarr → single-file ZipStore with Zstd compression
    - .hapt.zip → single-file ZIP archive (stdlib DEFLATE)

    Writes manifest.json, raw/data.npy + checksum, labels.json,
    and optionally unified/ data."""
    path = Path(path)

    if path.suffix == ".zarr":
        return _save_zarr(data, path)
    if path.suffix == ".zip":
        return _save_zip(data, path)

    path.mkdir(parents=True, exist_ok=True)
    return _save_dir(data, path)


def _build_manifest(data: HaptData) -> dict:
    """Build the manifest dict shared by all storage backends.

    Centralizes the v0.2 manifest schema so directory, .hapt.zarr, and
    .hapt.zip backends can never drift apart."""
    return {
        "version": data.version,
        "sensor": data.sensor.to_dict(),
        "modality": data.modality,
        "coordinate_frame": data.coordinate_frame,
        "sampling": {
            "rate_hz": data.sampling_rate_hz,
            "num_frames": data.raw.shape[0],
            "timestamps_s": data.timestamps_s,
        },
        "raw_shape": list(data.raw.shape),
        "raw_dtype": str(data.raw.dtype),
        "interaction": data.interaction.to_dict(),
        "created": "2026-07-24T00:00:00Z",  # TODO: use actual timestamp
        "created_by": "haptix/0.2.0",
    }


def _default_provenance() -> Provenance:
    """Auto-generate minimal provenance for data that has none (v0.1 origin)."""
    import datetime

    return Provenance(
        file_hash="",  # Computed on save
        source=Source(),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        created_by="haptix/0.2.0",
    )


def _save_dir(data: HaptData, path: Path) -> Path:
    """Save HaptData to a .hapt directory (traditional format)."""
    # Write raw data
    raw_dir = path / "raw"
    raw_dir.mkdir(exist_ok=True)

    np.save(raw_dir / "data.npy", data.raw.array)
    checksum = data.raw.checksum or hashlib.sha256(data.raw.array.tobytes()).hexdigest()
    with open(raw_dir / "checksum.sha256", "w") as f:
        f.write(checksum + "\n")

    # Write manifest
    manifest = _build_manifest(data)
    with open(path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Write labels
    with open(path / "labels.json", "w") as f:
        json.dump(data.labels.to_dict(), f, indent=2)

    # Write provenance (v0.2+)
    if data.provenance is not None:
        with open(path / "provenance.json", "w") as f:
            json.dump(data.provenance.to_dict(), f, indent=2)
    else:
        provenance = _default_provenance()
        with open(path / "provenance.json", "w") as f:
            json.dump(provenance.to_dict(), f, indent=2)

    # Optional unified
    if data.unified is not None:
        unified_dir = path / "unified"
        unified_dir.mkdir(exist_ok=True)
        np.save(unified_dir / "data.npy", data.unified.array)
        with open(unified_dir / "checksum.sha256", "w") as f:
            f.write(data.unified.checksum + "\n")
        transform = {
            "method": data.unified.method,
            "source_modality": data.unified.source_modality,
            "target_modality": data.unified.target_modality,
            "is_lossy": data.unified.is_lossy,
        }
        with open(unified_dir / "transform.json", "w") as f:
            json.dump(transform, f, indent=2)

    return path


def _chunk_shape(shape: tuple, max_chunk_time: int = 32) -> tuple:
    """Compute sensible Zarr chunks: time-axis chunked, others full.

    For tactile data, time is the natural chunking dimension —
    frames within a short time window are accessed together.
    Spatial/feature dimensions are kept whole per chunk."""
    if not shape:
        return shape
    time_chunks = min(shape[0], max_chunk_time)
    return (time_chunks,) + shape[1:]


def _save_zarr(data: HaptData, path: Path) -> Path:
    """Save HaptData to a single-file .hapt.zarr with Zstd compression.

    Structure:
      /raw/data          — compressed numpy array (Zstd, clevel=3)
      /raw/.zattrs       — checksum
      /unified/data      — optional unified representation
      /.zattrs           — manifest, labels, provenance as JSON
    """
    zarr, numcodecs = _ensure_zarr()

    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    store = zarr.storage.ZipStore(str(path), mode="w")
    root = _zarr_group(zarr, store)

    # Store raw data
    raw_chunks = _chunk_shape(data.raw.array.shape)
    raw_arr = _zarr_create_array(
        root, "raw/data", data.raw.array.shape, data.raw.array.dtype, raw_chunks, compressor
    )
    raw_arr[:] = data.raw.array

    checksum = data.raw.checksum or hashlib.sha256(data.raw.array.tobytes()).hexdigest()
    raw_arr.attrs["checksum"] = checksum

    # Build manifest
    manifest = _build_manifest(data)

    # Store manifest, labels, provenance as root-level attributes
    root.attrs["manifest"] = manifest
    root.attrs["labels"] = data.labels.to_dict()

    if data.provenance is not None:
        root.attrs["provenance"] = data.provenance.to_dict()
    else:
        provenance = _default_provenance()
        root.attrs["provenance"] = provenance.to_dict()

    # Optional unified
    if data.unified is not None:
        unified_chunks = _chunk_shape(data.unified.array.shape)
        unified_arr = _zarr_create_array(
            root,
            "unified/data",
            data.unified.array.shape,
            data.unified.array.dtype,
            unified_chunks,
            compressor,
        )
        unified_arr[:] = data.unified.array
        unified_arr.attrs["checksum"] = data.unified.checksum
        unified_arr.attrs["transform"] = {
            "method": data.unified.method,
            "source_modality": data.unified.source_modality,
            "target_modality": data.unified.target_modality,
            "is_lossy": data.unified.is_lossy,
        }

    store.close()
    return path


def _load_zarr(path: Path) -> HaptData:
    """Load a .hapt.zarr file.

    Reads all data and verifies checksum. Raises ChecksumError on mismatch."""
    zarr, _ = _ensure_zarr()

    # zarr 3.x: read-only ZipStore needs open_group(mode="r")
    store = zarr.storage.ZipStore(str(path), mode="r")
    try:
        root = zarr.open_group(store=store, mode="r")
    except zarr.errors.GroupNotFoundError:
        # zarr 2.x raises GroupNotFoundError for an empty store;
        # zarr 3.x raises FileNotFoundError. Normalize to FileNotFoundError.
        store.close()
        raise FileNotFoundError(f"Not a valid .hapt path: {path}") from None

    # Validate required structures
    if "raw/data" not in root:
        store.close()
        raise HaptFormatError("Missing raw/data array in .hapt.zarr")

    if "manifest" not in root.attrs:
        store.close()
        raise HaptFormatError("Missing manifest attributes in .hapt.zarr")

    if "labels" not in root.attrs:
        store.close()
        raise HaptFormatError("Missing labels attributes in .hapt.zarr")

    # Load manifest and labels
    manifest = root.attrs["manifest"]
    labels_dict = root.attrs["labels"]

    # Load raw data
    raw_arr = root["raw/data"]
    raw_array = raw_arr[:]
    stored_checksum = raw_arr.attrs.get("checksum", "")
    computed_checksum = hashlib.sha256(raw_array.tobytes()).hexdigest()

    if stored_checksum and computed_checksum != stored_checksum:
        store.close()
        raise ChecksumError(
            f"Checksum mismatch! Stored: {stored_checksum[:16]}..., "
            f"Computed: {computed_checksum[:16]}..."
        )

    raw = RawData(
        array=raw_array,
        checksum=computed_checksum,
        dtype=str(raw_array.dtype),
        shape=raw_array.shape,
    )

    # Parse metadata
    sensor = SensorMeta.from_dict(manifest["sensor"])
    interaction = InteractionMeta.from_dict(manifest["interaction"])
    labels = Labels.from_dict(labels_dict)

    # Optional provenance
    provenance = None
    if "provenance" in root.attrs:
        provenance = Provenance.from_dict(root.attrs["provenance"])

    # Optional unified
    unified = None
    if "unified/data" in root:
        unified_arr = root["unified/data"]
        unified_array = unified_arr[:]
        transform = unified_arr.attrs.get("transform", {})
        unified = UnifiedData(
            array=unified_array,
            method=transform.get("method", "unknown"),
            source_modality=transform.get("source_modality", manifest.get("modality", "unknown")),
            target_modality=transform.get("target_modality", "unified"),
            is_lossy=transform.get("is_lossy", True),
            checksum=hashlib.sha256(unified_array.tobytes()).hexdigest(),
        )

    coordinate_frame = manifest.get("coordinate_frame")
    timestamps_s = manifest.get("sampling", {}).get("timestamps_s")

    store.close()

    return HaptData(
        raw=raw,
        sensor=sensor,
        modality=manifest["modality"],
        sampling_rate_hz=manifest["sampling"]["rate_hz"],
        interaction=interaction,
        labels=labels,
        unified=unified,
        provenance=provenance,
        coordinate_frame=coordinate_frame,
        timestamps_s=timestamps_s,
        version=manifest.get("version", "0.1.0"),
    )


def _save_zip(data: HaptData, path: Path) -> Path:
    """Save HaptData to a single-file .hapt.zip archive.

    Mirrors the directory layout inside the archive:
      manifest.json
      raw/data.npy + raw/checksum.sha256
      labels.json
      provenance.json          (auto-generated if absent)
      unified/data.npy + unified/transform.json   (optional)

    Uses stdlib zipfile with DEFLATE — no extra dependencies. The archive
    is a plain ZIP, so any tool (unzip, file managers) can inspect it.
    """
    checksum = data.raw.checksum or hashlib.sha256(data.raw.array.tobytes()).hexdigest()

    def _npy_bytes(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(_build_manifest(data), indent=2))
        zf.writestr("labels.json", json.dumps(data.labels.to_dict(), indent=2))
        zf.writestr("raw/data.npy", _npy_bytes(data.raw.array))
        zf.writestr("raw/checksum.sha256", checksum + "\n")

        if data.provenance is not None:
            zf.writestr("provenance.json", json.dumps(data.provenance.to_dict(), indent=2))
        else:
            zf.writestr("provenance.json", json.dumps(_default_provenance().to_dict(), indent=2))

        if data.unified is not None:
            zf.writestr("unified/data.npy", _npy_bytes(data.unified.array))
            zf.writestr("unified/checksum.sha256", data.unified.checksum + "\n")
            zf.writestr(
                "unified/transform.json",
                json.dumps(
                    {
                        "method": data.unified.method,
                        "source_modality": data.unified.source_modality,
                        "target_modality": data.unified.target_modality,
                        "is_lossy": data.unified.is_lossy,
                    },
                    indent=2,
                ),
            )

    return path


def _load_zip(path: Path) -> HaptData:
    """Load a .hapt.zip single-file archive.

    Reads all members, verifies the SHA-256 checksum, and reconstructs a
    HaptData. Raises ChecksumError on mismatch, HaptFormatError on
    malformed archives."""
    try:
        zf = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as e:
        raise HaptFormatError(f"Not a valid .hapt.zip archive: {path}") from e

    try:
        names = set(zf.namelist())

        required = {
            "manifest.json": "Missing manifest.json in .hapt.zip",
            "raw/data.npy": "Missing raw/data.npy in .hapt.zip",
            "raw/checksum.sha256": "Missing raw/checksum.sha256 in .hapt.zip",
            "labels.json": "Missing labels.json in .hapt.zip",
        }
        for member, msg in required.items():
            if member not in names:
                raise HaptFormatError(msg)

        manifest = json.loads(zf.read("manifest.json"))
        labels_dict = json.loads(zf.read("labels.json"))

        # Load raw data and verify checksum
        raw_array = np.load(io.BytesIO(zf.read("raw/data.npy")))
        stored_checksum = zf.read("raw/checksum.sha256").decode().strip()
        computed_checksum = hashlib.sha256(raw_array.tobytes()).hexdigest()

        if computed_checksum != stored_checksum:
            raise ChecksumError(
                f"Checksum mismatch! Stored: {stored_checksum[:16]}..., "
                f"Computed: {computed_checksum[:16]}..."
            )

        raw = RawData(
            array=raw_array,
            checksum=computed_checksum,
            dtype=str(raw_array.dtype),
            shape=raw_array.shape,
        )

        sensor = SensorMeta.from_dict(manifest["sensor"])
        interaction = InteractionMeta.from_dict(manifest["interaction"])
        labels = Labels.from_dict(labels_dict)

        provenance = None
        if "provenance.json" in names:
            provenance = Provenance.from_dict(json.loads(zf.read("provenance.json")))

        unified = None
        if "unified/data.npy" in names and "unified/transform.json" in names:
            unified_array = np.load(io.BytesIO(zf.read("unified/data.npy")))
            transform = json.loads(zf.read("unified/transform.json"))
            unified = UnifiedData(
                array=unified_array,
                method=transform.get("method", "unknown"),
                source_modality=transform.get(
                    "source_modality", manifest.get("modality", "unknown")
                ),
                target_modality=transform.get("target_modality", "unified"),
                is_lossy=transform.get("is_lossy", True),
                checksum=hashlib.sha256(unified_array.tobytes()).hexdigest(),
            )

        coordinate_frame = manifest.get("coordinate_frame")
        timestamps_s = manifest.get("sampling", {}).get("timestamps_s")

        return HaptData(
            raw=raw,
            sensor=sensor,
            modality=manifest["modality"],
            sampling_rate_hz=manifest["sampling"]["rate_hz"],
            interaction=interaction,
            labels=labels,
            unified=unified,
            provenance=provenance,
            coordinate_frame=coordinate_frame,
            timestamps_s=timestamps_s,
            version=manifest.get("version", "0.1.0"),
        )
    finally:
        zf.close()
