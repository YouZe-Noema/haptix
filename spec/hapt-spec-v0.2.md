# Haptic Data Format (.hapt) Specification v0.2

## Abstract

`.hapt` is a container format for tactile sensor data designed for machine learning workflows. It provides lossless encapsulation of raw sensor data, standardized interaction metadata, and optional cross-sensor unified representations.

## Design Principles

1. **Raw is sacred** — native sensor data is stored verbatim with checksum verification. `.hapt` is a container, not a transcoder.
2. **Interaction metadata is mandatory** — no tactile data is meaningful without contact parameters (speed, force, angle).
3. **Modality-aware** — different sensors produce different signal types (imaging, dynamic, force). The format declares modality rather than forcing unification.
4. **ML-first** — designed for direct consumption by PyTorch/JAX dataloaders.
5. **Content-addressable** — every .hapt file has a deterministic SHA-256 identity. Derived files trace to their parent. Files are portable across storage locations.

## File Structure

A `.hapt` file is a directory (or ZIP archive or Zarr store) with the following structure:

```
example.hapt/
├── manifest.json       # Required: sensor metadata + interaction params
├── provenance.json     # Required: origin, lineage, and processing history
├── raw/                # Required: native sensor data
│   ├── data.npy        # NumPy array of raw sensor data
│   ├── checksum.sha256 # SHA-256 of data.npy
│   └── attrs.json      # Optional: sensor-specific attributes
├── unified/            # Optional: cross-sensor representation
│   ├── data.npy        # NumPy array of unified data
│   ├── transform.json  # Transform metadata (method, version, parameters)
│   └── checksum.sha256
└── labels.json         # Required: annotations
```

### Compression Modes

| Extension | Storage | Compression | Use Case |
|-----------|---------|-------------|----------|
| `.hapt/` | Directory | None | Development, debugging, random access |
| `.hapt.zip` | ZIP archive | Deflate | Sharing, archiving, single-file transfer |
| `.hapt.zarr/` | Zarr store (v3) | Zstd (configurable) | ML training, chunked I/O, cloud storage |

All modes contain the same logical structure. `haptix.load()` auto-detects the format.

## manifest.json Schema (v0.2)

```json
{
  "version": "0.2.0",
  "sensor": {
    "type": "DIGIT_v2",
    "serial": "optional-serial-number",
    "calibration_date": "2026-01-15",
    "calibration_params": {}
  },
  "modality": "imaging",
  "coordinate_frame": "sensor_local",
  "sampling": {
    "rate_hz": 60,
    "duration_s": 5.0,
    "num_frames": 300,
    "timestamps_s": null
  },
  "raw_shape": [300, 480, 640, 3],
  "raw_dtype": "uint8",
  "interaction": {
    "type": "sliding",
    "speed_mm_s": 50,
    "normal_force_N": 2.0,
    "approach_angle_deg": 90,
    "temperature_C": 23.0,
    "humidity_pct": 45
  },
  "environment": {
    "surface_material": "sandpaper_grit_80",
    "surface_temperature_C": 23.0
  },
  "created": "2026-07-24T15:00:00Z",
  "created_by": "haptix/0.2.0"
}
```

### New in v0.2

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `coordinate_frame` | `string \| null` | Yes | Sensor reference frame. Always present; `null` if unknown. Values: `"world"`, `"sensor_local"`, `"robot_base"`, `"object"`, or custom string. The actual transformation matrices (T_world_sensor, etc.) are a v0.3 feature. |
| `sampling.timestamps_s` | `[float, ...] \| null` | Yes | Per-frame timestamps in seconds relative to recording start. Always present; `null` if frames are equally spaced (use `rate_hz`). When non-null, `len(timestamps_s)` MUST equal `num_frames`. This field is a contract: downstream dataloaders can always check it exists. |

### provenance.json Schema (v0.2)

Every `.hapt` file carries a `provenance.json` that records its origin and transformation history. This is separate from `manifest.json` — provenance tracks lineage, manifest describes content.

```json
{
  "file_hash": "sha256:abc123def456...",
  "derived_from": null,
  "processing": [],
  "is_lossy": false,
  "source": {
    "dataset": null,
    "url": null,
    "citation": null,
    "license": null,
    "collection_date": null,
    "sensor_calibration": null
  },
  "created": "2026-07-24T15:00:00Z",
  "created_by": "haptix/0.2.0"
}
```

#### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `file_hash` | `string` | SHA-256 of the entire `.hapt` directory contents (deterministic, order-independent). This is the file's permanent identity — it does not change when the file is moved or renamed. |
| `derived_from` | `string \| null` | `file_hash` of the parent `.hapt` file this was derived from. `null` for original (sensor-origin) files. Forms a DAG of data lineage. |
| `processing` | `[{name, params, tool}]` | Ordered list of processing steps applied. Each step records the operation name, parameters, and tool used (e.g. `{"name": "gaussian_filter", "params": {"sigma": 2}, "tool": "scipy.ndimage.gaussian_filter"}`). |
| `is_lossy` | `boolean` | Whether the raw data has been lossily modified from the parent. If `derived_from` is non-null, this flag is critical for downstream consumers. |
| `source` | `object` | Origin metadata. For published datasets: `dataset`, `url`, `citation`, `license`. For lab captures: `collection_date`, `sensor_calibration`. All fields optional but the object is always present. |

#### Example: Derived File (denoised)

```json
{
  "file_hash": "sha256:789xyz...",
  "derived_from": "sha256:abc123def456...",
  "processing": [
    {
      "name": "gaussian_filter",
      "params": {"sigma": 2},
      "tool": "scipy.ndimage.gaussian_filter"
    },
    {
      "name": "force_calibration_v2",
      "params": {"calib_date": "2026-01-15"},
      "tool": "lab_toolkit/calib.py:apply_calibration"
    }
  ],
  "is_lossy": true,
  "source": {
    "dataset": null,
    "url": null,
    "citation": null,
    "license": null,
    "collection_date": null,
    "sensor_calibration": null
  },
  "created": "2026-07-25T10:00:00Z",
  "created_by": "haptix/0.2.0"
}
```

### Modality Types

| Modality | Description | Data Shape | Example Sensors |
|----------|-------------|------------|-----------------|
| `imaging` | 2D or 3D image sequence | [T, H, W, C] or [T, H, W] | DIGIT, GelSight, TacTip |
| `dynamic` | 1D time-series vector | [T, D] | BioTac electrodes, accelerometer |
| `force` | Force/torque vectors | [T, 6] or [T, 3] | ATI Nano17, load cell |
| `multimodal` | Multiple synchronized streams | mapping of streams | Any combination above |

## labels.json Schema

```json
{
  "material": "sandpaper_grit_80",
  "material_category": "abrasive",
  "object": null,
  "object_category": null,
  "task": "sliding",
  "grasp_type": null,
  "success": null,
  "custom_tags": ["rough", "high_friction"]
}
```

## Checksum Verification

```
sha256sum raw/data.npy → stored in raw/checksum.sha256
```

On load, verify checksum matches. If not, raise `ChecksumError`.

The `file_hash` in `provenance.json` is computed over the entire `.hapt` directory contents (excluding `provenance.json` itself) using a deterministic order-independent algorithm — filenames are sorted, file contents are hashed individually, then the list of `(filename, file_hash)` pairs is hashed to produce the final `file_hash`.

## Unified Representation (Optional)

When present, `unified/transform.json` MUST declare:

```json
{
  "method": "uniforce_v1",
  "source_modality": "imaging",
  "target_modality": "force",
  "parameters": {
    "model": "uniforce-resnet18",
    "checkpoint": "sha256:abc123..."
  },
  "is_lossy": true,
  "loss_description": "6D force estimation from tactile image, ~5% MAE"
}
```

The `is_lossy` flag is critical — downstream consumers must know whether they can round-trip to original data.

## Supported Sensors (v0.1)

- DIGIT / DIGIT v2
- GelSight (via conversion)
- Lab-CORO Capacitive Tactile Sensor
- Future: BioTac, TacTip, Custom adapters

## Versioning

Format version follows semver. Breaking changes increment major version. v0.x is pre-stable.

### v0.1 → v0.2 Migration

v0.2 is backward-compatible with v0.1 `.hapt` files. New fields added to `manifest.json` (`coordinate_frame`, `sampling.timestamps_s`) are always present but default to `null` when loading v0.1 files. `provenance.json` is new in v0.2 — v0.1 files loaded by a v0.2 reader auto-generate a minimal provenance record with `file_hash` computed from the directory contents.

## References

### Foundational

1. **Luo, S., Bimbo, J., Dahiya, R., & Liu, H. (2017).** "Robotic tactile perception of object properties: A review." *Mechatronics*, 48, 54-67.
   - Comprehensive survey of tactile sensing technologies, material/shape recognition, and sensor fusion.
   - Establishes the three-tier sensor taxonomy (single-point / high-res array / large-area skin) that maps to `.hapt` modality types.
   - Key insight: "interpretation of tactile sensors readings has not yet been fully taken into consideration" — the gap haptix addresses still exists 8+ years later.

### Reading List

See `docs/reading-list.md` for ongoing literature review tracking datasets, methods, and potential collaborators.

### Future Directions (Design Notes)

**Temporal modes (v0.2):** Current v0.2 uses `timestamps_s` for per-frame time alignment and `coordinate_frame` for spatial reference. Full temporal complexity — multi-phase interaction segments with per-segment metadata (`approach → contact → slide → release`) and 6-DOF sensor pose trajectories — is deferred to v0.3 when real multi-phase datasets demand it.

**Coordinate transforms (v0.3):** v0.2 establishes `coordinate_frame` as a reserved string enum. v0.3 will add `T_world_sensor` transformation matrices (4×4 homogeneous transform per frame or per segment) for multi-sensor fusion and robot-world alignment.
