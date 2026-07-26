# haptix API Reference

> Complete reference for all public APIs. Private modules and underscore-prefixed
> items are internal — not part of the public contract.

---

## Top-level Module (`haptix`)

```python
import haptix
```

### `haptix.load(path: str | Path) -> HaptData`

Load a `.hapt` file from disk. Automatically verifies SHA-256 checksum on read.

**Args:**
| Parameter | Type | Description |
|---|---|---|
| `path` | `str` or `Path` | Path to a `.hapt` directory (or future: `.hapt` archive) |

**Raises:**
| Exception | Condition |
|---|---|
| `FileNotFoundError` | Path does not exist |
| `HaptFormatError` | Missing `manifest.json`, `raw/`, or `labels.json` |
| `ChecksumError` | Raw data does not match stored checksum |

**Example:**
```python
data = haptix.load("experiments/sandpaper_80.hapt")
```

### `haptix.save(data: HaptData, path: str | Path) -> Path`

Save a `HaptData` object to disk as a `.hapt` directory. Creates the directory
if it does not exist. Always writes checksum alongside raw data.

**Args:**
| Parameter | Type | Description |
|---|---|---|
| `data` | `HaptData` | In-memory data to serialize |
| `path` | `str` or `Path` | Output directory (will be created) |

**Returns:** The `Path` to the created directory.

**Example:**
```python
path = haptix.save(data, "outputs/experiment.hapt")
assert path.exists()
```

### `haptix.get_sensor(sensor_type: str) -> SensorAdapter`

Retrieve a registered sensor adapter by type name.

**Args:**
| Parameter | Type | Description |
|---|---|---|
| `sensor_type` | `str` | Registered sensor type (e.g., `"DIGIT"`, `"DIGIT_v2"`) |

**Raises:**
| Exception | Condition |
|---|---|
| `ValueError` | Unknown sensor type |

**Example:**
```python
digit = haptix.get_sensor("DIGIT")
data = digit.load("frames/", interaction=..., labels=...)
```

### `haptix.list_sensors() -> list[str]`

List all registered sensor type names. Triggers lazy import of all adapter
modules to populate the registry.

**Returns:** List of sensor type strings.

**Example:**
```python
haptix.list_sensors()
# ['DIGIT', 'DIGIT_v2']
```

---

## Core Data Classes (`haptix.core`)

### `class HaptData`

Immutable in-memory representation of `.hapt` file contents. Constructed once,
read-only afterward. Use `haptix.load()` or sensor adapters to create instances.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `raw` | `RawData` | Raw sensor array with checksum |
| `sensor` | `SensorMeta` | Sensor type and calibration info |
| `modality` | `Modality` | Signal modality (`"imaging"`, `"dynamic"`, `"force"`, `"multimodal"`) |
| `sampling_rate_hz` | `float` | Sampling rate in Hertz |
| `interaction` | `InteractionMeta` | Touch interaction parameters |
| `labels` | `Labels` | Annotation labels |
| `unified` | `Optional[UnifiedData]` | Cross-sensor unified representation |
| `version` | `str` | Format version (default: `"0.1.0"`) |

**Properties:**

| Property | Type | Description |
|---|---|---|
| `.raw` | `RawData` | Raw sensor data (see below) |
| `.sensor` | `SensorMeta` | Sensor metadata |
| `.modality` | `Modality` | Signal modality type |
| `.sampling_rate_hz` | `float` | Sampling rate |
| `.interaction` | `InteractionMeta` | Interaction parameters |
| `.labels` | `Labels` | Annotation labels |
| `.unified` | `Optional[UnifiedData]` | Unified rep, if present |
| `.version` | `str` | Format version |

**String representation:**
```python
HaptData(sensor=DIGIT_v2, modality=imaging, shape=(300, 480, 640, 3), labels=sandpaper_grit_80)
```

---

### `class RawData`

Immutable wrapper for raw sensor data with integrity verification.

**Fields:**
| Field | Type | Description |
|---|---|---|
| `.array` | `np.ndarray` | The raw sensor data |
| `.checksum` | `str` | SHA-256 hex digest of raw data |
| `.dtype` | `str` | NumPy dtype name |
| `.shape` | `tuple` | Array shape |

**Methods:**

#### `RawData.numpy() -> np.ndarray`

Return a read-only view of the raw data. Call `.copy()` if you need a mutable copy.

```python
frames = data.raw.numpy()   # read-only view
mutable = data.raw.numpy().copy()  # writable copy
```

#### `RawData.verify() -> bool`

Verifies that the current array matches the stored SHA-256 checksum.

```python
assert data.raw.verify()
```

#### `RawData.compute_checksum(arr: np.ndarray) -> str`

Static method: compute SHA-256 hex digest of any NumPy array.

```python
checksum = RawData.compute_checksum(my_array)
```

---

### `class SensorMeta`

Immutable sensor metadata.

**Fields:**
| Field | Type | Description |
|---|---|---|
| `.type` | `str` | Sensor type name (e.g., `"DIGIT_v2"`) |
| `.serial` | `Optional[str]` | Sensor serial number |
| `.calibration_date` | `Optional[str]` | ISO date of last calibration |
| `.calibration_params` | `dict` | Sensor-specific calibration parameters |

**Methods:**

#### `SensorMeta.to_dict() -> dict`

Serialize to JSON-compatible dict.

#### `SensorMeta.from_dict(d: dict) -> SensorMeta`

Deserialize from dict.

---

### `class InteractionMeta`

Immutable interaction parameters. **This is the metadata that differentiates
`.hapt` from a generic container format.** Interaction type is required;
all other fields are optional but recommended.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `.type` | `str` | Interaction type: `"sliding"`, `"pressing"`, `"grasping"`, or `"static"` |
| `.speed_mm_s` | `Optional[float]` | Tangential speed in mm/s |
| `.normal_force_N` | `Optional[float]` | Normal force in Newtons |
| `.approach_angle_deg` | `Optional[float]` | Approach angle in degrees |
| `.temperature_C` | `Optional[float]` | Ambient temperature in Celsius |
| `.humidity_pct` | `Optional[float]` | Relative humidity in percent |

**Methods:**

#### `InteractionMeta.to_dict() -> dict`

Serialize to dict (omits `None` fields).

#### `InteractionMeta.from_dict(d: dict) -> InteractionMeta`

Deserialize from dict.

---

### `class Labels`

Immutable annotation labels.

**Fields:**
| Field | Type | Description |
|---|---|---|
| `.material` | `Optional[str]` | Surface material name |
| `.material_category` | `Optional[str]` | Material category (e.g., `"abrasive"`, `"fabric"`) |
| `.object_name` | `Optional[str]` | Object name |
| `.object_category` | `Optional[str]` | Object category |
| `.task` | `Optional[str]` | Task description |
| `.custom_tags` | `list[str]` | Free-form tag list |

**Methods:**

#### `Labels.to_dict() -> dict`

Serialize to dict (omits `None` fields).

#### `Labels.from_dict(d: dict) -> Labels`

Deserialize from dict.

---

### `class UnifiedData`

Optional cross-sensor unified representation. Not always present — check
`data.unified is not None` before accessing.

**Fields:**
| Field | Type | Description |
|---|---|---|
| `.array` | `np.ndarray` | Unified data array |
| `.method` | `str` | Transform method name |
| `.source_modality` | `str` | Source modality |
| `.target_modality` | `str` | Target modality |
| `.is_lossy` | `bool` | Whether transform loses information |
| `.checksum` | `str` | SHA-256 of unified array |

**Methods:**

#### `UnifiedData.numpy() -> np.ndarray`

Return a read-only view.

---

### Type Aliases

```python
Modality = Literal["imaging", "dynamic", "force", "multimodal"]
```

---

## I/O Module (`haptix.io`)

### `class ChecksumError(ValueError)`

Raised when raw data checksum does not match stored checksum.

### `class HaptFormatError(ValueError)`

Raised when `.hapt` file structure is invalid (missing required files).

---

## Sensors Module (`haptix.sensors`)

### `class SensorAdapter` (Protocol)

Protocol for sensor format adapters. Any object implementing these methods
is a valid adapter.

```python
@runtime_checkable
class SensorAdapter(Protocol):
    sensor_type: str

    def can_load(self, path: Path) -> bool: ...
    def load(self, path: Path, interaction: InteractionMeta, labels: Labels) -> HaptData: ...
```

#### `can_load(path: Path) -> bool`

Check if this adapter can handle the given file or directory.

#### `load(path, interaction, labels) -> HaptData`

Load native sensor data and return a fully populated `HaptData`.

### `register(sensor_type: str)`

Decorator to register a sensor adapter class.

```python
from haptix.sensors import register

@register("MySensor")
class MyAdapter:
    ...
```

### `get_sensor(sensor_type: str) -> SensorAdapter`

Get an instance of a registered sensor adapter.

### `list_sensors() -> list[str]`

List all registered sensor type names.

---

## DIGIT Adapter (`haptix.sensors.digit`)

### `class DigitAdapter`

Adapter for DIGIT / DIGIT v2 tactile sensor data.

**Sensor types registered:** `"DIGIT"`, `"DIGIT_v2"`

**Supported formats:**
- Directory of `.png` or `.jpg` image frames
- `.mp4` / `.avi` video (requires `opencv-python`, coming in v0.2.0)

**Loading:**
```python
adapter = DigitAdapter()
adapter.can_load("path/to/frames/")  # True if has PNG/JPG files

data = adapter.load(
    path="path/to/frames/",
    interaction=InteractionMeta(type="sliding", speed_mm_s=50),
    labels=Labels(material="test"),
    sensor_meta=SensorMeta(type="DIGIT_v2", serial="SN-001"),  # optional
)
```
