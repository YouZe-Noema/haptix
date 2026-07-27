# Haptic Data Format (.hapt) Specification v0.1

## Abstract

`.hapt` is a container format for tactile sensor data designed for machine learning workflows. It provides lossless encapsulation of raw sensor data, standardized interaction metadata, and optional cross-sensor unified representations.

## Design Principles

1. **Raw is sacred** — native sensor data is stored verbatim with checksum verification. `.hapt` is a container, not a transcoder.
2. **Interaction metadata is mandatory** — no tactile data is meaningful without contact parameters (speed, force, angle).
3. **Modality-aware** — different sensors produce different signal types (imaging, dynamic, force). The format declares modality rather than forcing unification.
4. **ML-first** — designed for direct consumption by PyTorch/JAX dataloaders.

## File Structure

A `.hapt` file is a directory (or ZIP archive in compressed mode) with the following structure:

```
example.hapt/
├── manifest.json       # Required: sensor metadata + interaction params
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

## manifest.json Schema

```json
{
  "version": "0.1.0",
  "sensor": {
    "type": "DIGIT_v2",
    "serial": "optional-serial-number",
    "calibration_date": "2026-01-15",
    "calibration_params": {}
  },
  "modality": "imaging",
  "sampling": {
    "rate_hz": 60,
    "duration_s": 5.0,
    "num_frames": 300
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
  "created_by": "haptix/0.1.0"
}
```

### Modality Types

| Modality | Description | Data Shape | Example Sensors |
|---|---|---|---|
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
- Future: BioTac, TacTip, Custom adapters

## Versioning

Format version follows semver. Breaking changes increment major version. v0.x is pre-stable.

## References

### Foundational

1. **Luo, S., Bimbo, J., Dahiya, R., & Liu, H. (2017).** "Robotic tactile perception of object properties: A review." *Mechatronics*, 48, 54-67.
   - Comprehensive survey of tactile sensing technologies, material/shape recognition, and sensor fusion.
   - Establishes the three-tier sensor taxonomy (single-point / high-res array / large-area skin) that maps to `.hapt` modality types.
   - Key insight: "interpretation of tactile sensors readings has not yet been fully taken into consideration" — the gap haptix addresses still exists 8+ years later.

### Reading List

See `docs/reading-list.md` for ongoing literature review tracking datasets, methods, and potential collaborators.

### Future Directions (Design Notes)

**Temporal modes (v0.2):** Current v0.1 uses global interaction parameters — one speed, one force for the entire recording. Real tactile exploration has phases (approach → contact → slide → release) with varying parameters. v0.2 should introduce:

- `mode`: `"snapshot"` (T small, constant params) vs `"trajectory"` (T large, varying params)
- `temporal_segments`: per-phase interaction metadata with frame ranges
- `trajectory`: optional 6-DOF sensor pose synchronized with tactile frames

This maps to the two core application domains: snapshot for material classification/grasp assessment, trajectory for teleoperation and in-hand manipulation.

Design principle: introduce temporal complexity only when real data sources demand it. No speculation-driven engineering.
