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

---

## Datasets Module (`haptix.datasets`)

### `list_datasets() -> list[str]`

List all dataset names in the catalog.

### `get_dataset_info(name: str) -> dict`

Return catalog metadata for a dataset: `name`, `url`, `size_bytes`,
`sensor_type`, `modality`, `license`, `citation`, and optionally `sha256`
+ `provenance` for checksum-verified datasets.

### `download_dataset(name, cache_dir=None, force=False, extract=True) -> Path`

Download a dataset from the catalog into the local cache
(`~/.haptix/cache/datasets/` by default) and return the dataset directory.

- Automatically verifies the pinned SHA-256 checksum when the catalog entry
  has one; raises `ChecksumError` on mismatch.
- Archives (`.tar.gz`, `.zip`) are extracted into the dataset directory.
- `force=True` re-downloads even if cached; `extract=False` keeps the
  archive file as-is.

```python
path = haptix.download_dataset("haptix_demo_sample")
```

### `cached_datasets(cache_dir=None) -> list[str]`

List datasets already present in the local cache.

### `cache_info(cache_dir=None) -> dict`

Return `cache_path`, `total_datasets`, and `total_bytes` for the cache.

### `clear_cache(cache_dir=None) -> None`

Delete the dataset cache directory.

### `verify_checksum(path, expected_sha256) -> bool`

Streaming SHA-256 verification. Raises `ChecksumError` on mismatch, returns
`True` on match. Used internally by `download_dataset` and available to
contributors for weight-file verification.

---

## Streaming & Windowing Module (`haptix.streaming`)

Long recordings (hours of 30 Hz tactile data → 100K+ frames) cannot be
loaded eagerly. `open_archive()` is the lazy counterpart to `load()`: it
reads metadata only and materializes raw frames on demand.

### `open_archive(path) -> HaptArchive`

Opens a `.hapt` directory, `.hapt.zarr`, or `.hapt.zip` (auto-detected)
without loading the raw array. Use as a context manager:

```python
import haptix

with haptix.open_archive("long.hapt") as arc:
    print(arc.n_frames, arc.shape)              # metadata only
    for win in arc.iter_windows(window_size=256, stride=128):
        loader = DataLoader(win.to_torch(batch_size=32, label="material"))
        ...
```

### `class HaptArchive`

Lazy handle exposing recording metadata (`sensor`, `modality`,
`sampling_rate_hz`, `interaction`, `labels`, `provenance`,
`coordinate_frame`, `version`, `shape`, `dtype`, `n_frames`,
`timestamps_s`) and windowing:

- **`window(start, stop) -> HaptData`** — materialize frames `[start, stop)`
  as a standalone `HaptData` (own checksum, sliced timestamps, unified
  slice when present). Saveable and independently verifiable.
- **`iter_windows(window_size, stride=None, *, start=0, stop=None,
  drop_last=False) -> Iterator[HaptData]`** — temporal windows along the
  time axis; `stride` defaults to `window_size` (non-overlapping);
  `stride < window_size` yields overlapping windows.
- **`window_count(...)`** — how many windows iteration yields, without I/O.
- **`frame_index_at(time_s) -> int`** — frame nearest a recording time
  (uses `timestamps_s` when present, else `rate_hz` spacing).
- **`verify() -> bool`** — streaming SHA-256 verification (memory-bounded;
  raises `ChecksumError` on mismatch, matching `load()` semantics).

Memory behavior by format: directory → memory-mapped (`O(window)` per
window); zarr → chunked reads (`O(window)` + decompression); zip → raw
member decompressed once at open (`O(full array)` — prefer directory/zarr
for very long recordings).

---

## Unified Encoders Module (`haptix.unified`)

### `class UnifiedEncoder` (Protocol)

Protocol for cross-sensor encoders. Implementations must be deterministic
and expose a `version` string plus `encode(data: HaptData) -> UnifiedData`.

### `class SharedForceEncoder`

Untrained surrogate encoder (zero dependencies). Maps any sensor's data into
a fixed-dimension embedding via spatial resize (imaging) or pad/truncate
(dynamic). Useful for shape checks, pipeline tests, and as a fallback before
trained encoders exist.

```python
enc = haptix.SharedForceEncoder(embedding_dim=128)
embedding = enc.encode(haptix.load("sample.hapt"))  # UnifiedData, .array [T, 128]
```

### `class CrossModalEncoder`

Trained cross-sensor encoder: learns per-modality linear projections into a
shared latent space via CCA + Procrustes alignment over class centroids.
Weights are serializable to a single `.npz` file (`save`/`load`), so
encoders can be pre-trained, versioned, and shipped.

```python
enc = haptix.CrossModalEncoder(embedding_dim=64)
enc.fit(records, label_key="material")     # list[HaptData] with labels
shared = enc.encode(haptix.load("sample.hapt"))  # UnifiedData in shared space
enc.save("weights/cca_v0.2.npz")
```

**Introspection:** `enc.fitted`, `enc.classes`, `enc.n_records`.

---

## Encoder Registry Module (`haptix.encoders`)

Per-sensor encoders — the v0.3 "pre-trained encoders for common sensor types"
roadmap item (design: [`docs/encoder-registry.md`](encoder-registry.md)).
A per-sensor encoder is the front-end of the encoding stack: raw sensor data
→ fixed-dimensional embedding `[T, D]`. The alignment layer
(`CrossModalEncoder`) composes these embeddings into a shared space, so a
per-sensor encoder never needs to know about other sensors.

**Dimension convention:** 256 for imaging sensors (GelSight, DIGIT), 128 for
dynamic sensors (CoroCapacitive, BioTac_SP, TacTip). Once published, an
encoder's dim is stable for that sensor type — it never changes without a
version bump.

### `class SensorEncoder` (Protocol)

```python
class SensorEncoder(Protocol):
    sensor_type: str      # "GelSight", "DIGIT", "CoroCapacitive", ...
    modality: str         # "imaging" | "dynamic" | "force" | "multimodal"
    embedding_dim: int    # fixed output dim; 256 imaging / 128 dynamic
    version: str          # "encoders/gelsight/v0.1" (untrained) / ".../v1.0" (trained)
    trained: bool         # False until fit()/load() supplies learned weights

    def encode(self, data: HaptData) -> np.ndarray: ...   # [T, ...] -> [T, D]
    def fit(self, records: list[HaptData], label_key: str = "material") -> ...: ...
    def save(self, path: Path) -> None: ...               # single .npz
    @classmethod
    def load(cls, path: Path) -> "SensorEncoder": ...     # ready-to-encode instance
    def benchmark(self, dataset: str = "unavailable") -> dict: ...  # structured report
```

`encode()` is deterministic: same input → same embedding. `isinstance(enc,
SensorEncoder)` works at runtime (`@runtime_checkable`).

### `register_encoder(sensor_type, modality="imaging")`

Decorator to register a per-sensor encoder class. Sets `sensor_type` and
`modality` on the class and adds it to the registry. Contribution = drop a
module under `haptix/encoders/` (or register programmatically) — no core
edits. The registry is **decoupled** from the adapter registry: an encoder may
exist without a matching adapter; a contributed encoder's `benchmark()` is the
evidence a data path works.

### `get_encoder(sensor_type) -> SensorEncoder`

Returns the best available encoder for a sensor type:

- the registered encoder, if any;
- otherwise a deterministic **surrogate fallback** (version tag
  `unified/shared-force/v0.1/surrogate` — placeholder embeddings are never
  mistaken for learned ones).

Never raises for unknown sensor types (falls back to the dynamic dim, 128).

```python
import haptix

haptix.list_encoders()          # ["BioTac_SP", "CoroCapacitive", "DIGIT", "GelSight", "TacTip"]
enc = haptix.get_encoder("GelSight")
emb = enc.encode(haptix.load("sample.hapt"))   # np.ndarray [T, 256]
assert enc.version == "encoders/gelsight/v0.1"
```

### `list_encoders() -> list[str]`

Lists sensor types with a registered encoder. Triggers lazy import of every
module under `haptix/encoders/`.

### `load_trained(sensor_type, weights_dir=None) -> SensorEncoder`

Loads a **trained** encoder (learned weights) from a local `.npz` file.
Looks for `<weights_dir>/<sensor_type>_v1.0.npz`; the default weights dir is
`haptix/encoders/weights/` (gitignored — the local weights home).

```python
enc = haptix.load_trained("GelSight")       # trained=True, encoders/gelsight/v1.0
emb = enc.encode(haptix.load("sample.hapt"))  # learned projection applied
```

Raises `FileNotFoundError` if no trained weights exist for the sensor — the
registry entry is still served untrained via `get_encoder()`. Train weights
with `encoder.fit(records)` + `encoder.save(path)` (see
[`examples/train_encoders.py`](../examples/train_encoders.py)).

### Training encoders (`fit()`)

Every registered encoder implements `fit(records, label_key="material")`:

1. Extracts per-frame features (same path `encode()` uses).
2. Learns `mean` + `W`: **PCA whitening** followed by an **LDA-style
   class-aligned rotation** (pure numpy, deterministic).
3. Sets `trained=True`, bumps `version` to `.../v1.0`, and computes an honest
   **leave-one-record-out nearest-centroid benchmark** (per-fold refit — the
   eval record never leaks into the projection). Unsupervised fits (single
   class) report a `whitening_decorrelation` score instead.

```python
records = [haptix.load(p) for p in hapt_paths]     # labeled HaptData
enc = haptix.get_encoder("GelSight").fit(records, label_key="material")
enc.save("weights/GelSight_v1.0.npz")              # single .npz (mean, W, report)
loaded = haptix.load_trained("GelSight")           # reload later
```

### Registered encoders (v0.1 untrained / v1.0 trained)

All five concrete encoders honor the fixed-dim contract. Untrained
(`trained is False`): imaging sensors are grayscale-Lanczos-resized to a
`sqrt(D)` grid and flattened to exactly `[T, D]`; dynamic sensors pad or crop
features to exactly `[T, D]`. Trained (`fit()`/`load_trained()`): the learned
projection replaces the raw projection; the output dim never changes.

| Encoder | sensor_type | Modality | dim | version |
|---|---|---|---|---|
| `GelSightEncoder` | GelSight | imaging | 256 | `encoders/gelsight/v0.1` → `v1.0` |
| `DIGITEncoder` | DIGIT | imaging | 256 | `encoders/digit/v0.1` → `v1.0` |
| `CoroCapacitiveEncoder` | CoroCapacitive | dynamic | 128 | `encoders/coro/v0.1` → `v1.0` |
| `BioTacSPEncoder` | BioTac_SP | dynamic | 128 | `encoders/biotac/v0.1` → `v1.0` |
| `TacTipEncoder` | TacTip | dynamic | 128 | `encoders/tactip/v0.1` → `v1.0` |

Weights trained on real data (2026-08-10, via
[`examples/train_encoders.py`](../examples/train_encoders.py)):

- **GelSight v1.0** — YCB-Sight real frames (6 objects × 80 frames, 24
  temporal segments): 79.8% leave-one-record-out nearest-centroid accuracy.
- **CoroCapacitive v1.0** — real Lab-CORO CSV (786 frames): whitening
  decorrelates features (mean |off-diagonal corr| 0.180 → 0.000).

Weights live in the gitignored `haptix/encoders/weights/` dir; publication to
the Hugging Face Hub is a pending release decision (design doc §7).

### `benchmark()` (contributor contract)

Every encoder implements `benchmark() -> dict` returning a structured report
(`dataset`, `metric`, `score`, `split`). Untrained encoders report
`score=None` with a note; trained encoders return the honest held-out report
computed at `fit()` time (see the design doc §4).

---

## Deprecations

- **Flat `.hapt` files (legacy):** deprecated and rejected with
  `HaptFormatError`. Use directory format, `.hapt.zarr`, or `.hapt.zip`
  (see `haptix.load`).
- **`RuntimeError`-based `haptix.datasets.ChecksumError`:** removed as a
  separate class — `ChecksumError` is now defined once in `haptix.io`
  (subclass of `ValueError`) and re-exported from `haptix.datasets`.
  Existing `from haptix.datasets import ChecksumError` imports keep working.
