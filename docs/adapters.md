# Sensor Adapter Authoring Guide

> How to add support for a new tactile sensor to haptix.

## Overview

A sensor adapter is a Python class that implements the `SensorAdapter` protocol.
It knows how to detect its native file format and convert it into the `.hapt`
representation.

Adapters live in `haptix/sensors/<name>.py` and are registered via a decorator.
Once registered, they appear in `haptix.list_sensors()` and are callable via
`haptix.get_sensor()`.

## Protocol

```python
class SensorAdapter(Protocol):
    """Protocol for sensor format adapters."""

    sensor_type: str

    def can_load(self, path: Path) -> bool:
        """Check if this adapter can handle the given file."""
        ...

    def load(
        self,
        path: Path,
        interaction: InteractionMeta,
        labels: Labels,
    ) -> HaptData:
        """Load native sensor file and return HaptData."""
        ...
```

That's it. Two methods and a class variable.

## Step-by-step walkthrough

### 1. Identify the sensor modality

Refer to the `.hapt` spec for modality types:

| Modality | Shape convention | Example Sensors |
|---|---|---|
| `"imaging"` | `[T, H, W, C]` or `[T, H, W]` | DIGIT, GelSight, TacTip |
| `"dynamic"` | `[T, D]` (time x channels) | BioTac electrodes |
| `"force"` | `[T, 6]` or `[T, 3]` | ATI Nano17, load cell |
| `"multimodal"` | multiple streams | Any combination |

The data array must be `[T, ...]` — time-first — so frames can be indexed
by time step for ML dataloaders.

### 2. Detect your format in `can_load()`

Return `True` when the path matches your sensor's native format. Be specific
enough to avoid false positives when multiple adapters are present.

```python
def can_load(self, path: Path) -> bool:
    if path.is_dir():
        # Check for sensor-specific marker files
        if (path / "digit_config.json").exists():
            return True
        images = sorted(path.glob("*.png")) + sorted(path.glob("*.jpg"))
        return len(images) > 0
    return path.suffix in self.SUPPORTED_EXTENSIONS
```

### 3. Implement `load()`

Parse the native format, create the core data objects, and return a `HaptData`.

```python
def load(
    self,
    path: Path,
    interaction: InteractionMeta,
    labels: Labels,
    sensor_meta: SensorMeta | None = None,
) -> HaptData:
    # 1. Parse native format into NumPy array [T, H, W, C]
    frames = self._parse_frames(path)

    # 2. Create RawData with checksum
    raw = RawData(
        array=frames,
        checksum=RawData.compute_checksum(frames),
        dtype=str(frames.dtype),
        shape=frames.shape,
    )

    # 3. Sensor metadata (provide defaults, allow override via sensor_meta)
    if sensor_meta is None:
        sensor_meta = SensorMeta(type="MySensor")

    # 4. Return assembled HaptData
    return HaptData(
        raw=raw,
        sensor=sensor_meta,
        modality="imaging",
        sampling_rate_hz=self._detect_framerate(path),
        interaction=interaction,
        labels=labels,
    )
```

### 4. Register the adapter

Use the `@register` decorator with your sensor type name.

```python
from haptix.sensors import register

@register("MySensor")
class MySensorAdapter:
    sensor_type = "MySensor"
    ...
```

You can register multiple type names for the same adapter:

```python
@register("GelSight_mini")
@register("GelSight")
class GelSightAdapter:
    ...
```

### 5. Place the file

```
haptix/
  sensors/
    __init__.py   # existing registry code
    digit.py      # existing DIGIT adapter
    gelsight.py   # NEW: your adapter
    biotac.py     # NEW: future adapter
    ...
```

The `__init__.py` uses lazy imports — your module is automatically discovered
when `list_sensors()` or `get_sensor()` is called.

## Complete example

Here's a minimal GelSight adapter:

```python
"""GelSight sensor adapter."""

from pathlib import Path
import numpy as np
from PIL import Image

from haptix.core import HaptData, RawData, SensorMeta, InteractionMeta, Labels
from haptix.sensors import register


@register("GelSight")
class GelSightAdapter:
    """Adapter for GelSight tactile sensor data."""

    sensor_type = "GelSight"

    def can_load(self, path: Path) -> bool:
        if path.is_dir():
            images = sorted(path.glob("*.png"))
            return len(images) > 0
        return False

    def load(
        self,
        path: Path,
        interaction: InteractionMeta,
        labels: Labels,
        sensor_meta: SensorMeta | None = None,
    ) -> HaptData:
        images = sorted(path.glob("*.png"))
        if not images:
            raise FileNotFoundError(f"No PNG frames found in {path}")

        frames = []
        for img_path in images:
            img = np.array(Image.open(img_path))
            frames.append(img)

        array = np.stack(frames, axis=0)  # [T, H, W, C]

        if sensor_meta is None:
            sensor_meta = SensorMeta(type="GelSight")

        raw = RawData(
            array=array,
            checksum=RawData.compute_checksum(array),
            dtype=str(array.dtype),
            shape=array.shape,
        )

        return HaptData(
            raw=raw,
            sensor=sensor_meta,
            modality="imaging",
            sampling_rate_hz=60.0,
            interaction=interaction,
            labels=labels,
        )
```

## Testing your adapter

Write tests in `tests/` following the pattern in `test_roundtrip.py`:

```python
class TestGelSightAdapter:
    def test_load_image_directory(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            # Create synthetic frames
            for i in range(3):
                img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                Image.fromarray(img).save(tmp / f"frame_{i:04d}.png")

            adapter = GelSightAdapter()
            assert adapter.can_load(tmp)

            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="sliding", speed_mm_s=50),
                labels=Labels(material="test"),
            )
            assert data.raw.shape == (3, 240, 320, 3)
            assert data.sensor.type == "GelSight"
        finally:
            shutil.rmtree(tmp)
```

Your adapter is also tested implicitly by the round-trip guarantee:

```python
data = adapter.load(...)
saved = haptix.save(data, "out.hapt")
reloaded = haptix.load(saved)
assert data.raw.checksum == reloaded.raw.checksum
```

## Best practices

1. **Always compute checksums** — `RawData.compute_checksum()` is a one-liner.
2. **Frame order** — sort image filenames lexicographically so frame order
   is deterministic.
3. **Provide good defaults** — sensor type, framerate, and calibration info
   should have sensible defaults but allow caller override.
4. **Handle missing dependencies gracefully** — if your adapter needs optional
   packages (e.g. `cv2` for video), raise `ImportError` with an install hint.
5. **Test with real data** — if possible, include a test fixture or download
   script for canonical native files.
