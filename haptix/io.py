"""
Load and save .hapt files.

.hapt files are directories (or ZIP archives) with:
  manifest.json  — sensor meta + interaction params
  raw/           — native sensor data + checksum
  unified/       — optional cross-sensor rep
  labels.json    — annotations
"""

import hashlib
import json
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


class ChecksumError(ValueError):
    """Raised when checksum verification fails."""


class HaptFormatError(ValueError):
    """Raised when .hapt file structure is invalid."""


def load(path: str | Path) -> HaptData:
    """Load a .hapt file from disk.

    Automatically verifies checksum. Raises ChecksumError on mismatch."""
    path = Path(path)

    # Support both directory and legacy flat file
    if path.is_dir():
        return _load_dir(path)
    if path.suffix == ".hapt" and path.is_file():
        # Future: ZIP archive
        raise HaptFormatError("Compressed .hapt files not yet supported. Use directory format.")
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
    """Save HaptData to a .hapt directory.

    Writes manifest.json, raw/data.npy + checksum, labels.json,
    and optionally unified/ data."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Write raw data
    raw_dir = path / "raw"
    raw_dir.mkdir(exist_ok=True)

    np.save(raw_dir / "data.npy", data.raw.array)
    checksum = data.raw.checksum or hashlib.sha256(data.raw.array.tobytes()).hexdigest()
    with open(raw_dir / "checksum.sha256", "w") as f:
        f.write(checksum + "\n")

    # Write manifest
    manifest = {
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
        # Auto-generate minimal provenance for v0.1-origin data
        import datetime

        provenance = Provenance(
            file_hash="",  # Computed on save
            source=Source(),
            created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            created_by="haptix/0.2.0",
        )
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
