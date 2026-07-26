# Quickstart

> Get from zero to tactile data in 5 minutes.

## Installation

```bash
pip install haptix            # core (numpy, Pillow)
pip install haptix[torch]     # with PyTorch integration
pip install haptix[all]       # everything (torch, jax, video tools)
pip install haptix[dev]       # with dev dependencies (pytest, black, ruff)
```

## Load a .hapt file

The `load()` function reads any `.hapt` directory and returns a `HaptData` object
with automatic checksum verification.

```python
import haptix

data = haptix.load("experiments/sandpaper_80.hapt")
print(data)
# HaptData(sensor=DIGIT_v2, modality=imaging, shape=(300, 480, 640, 3), labels=sandpaper_grit_80)
```

### Access metadata

Tactile-specific metadata is mandatory in the `.hapt` format — every file
records the *interaction parameters* that make tactile data meaningful.

```python
# Interaction parameters (how the touch happened)
print(data.interaction.type)           # "sliding"
print(data.interaction.speed_mm_s)     # 50.0
print(data.interaction.normal_force_N) # 2.0

# Sensor metadata
print(data.sensor.type)                # "DIGIT_v2"
print(data.sensor.serial)              # "SN-2026-001" (optional)

# Annotations
print(data.labels.material)            # "sandpaper_grit_80"
print(data.labels.material_category)   # "abrasive"
print(data.labels.task)                # "sliding"
```

### Access raw data

Raw sensor data is stored as a NumPy array. The `RawData` wrapper provides
a read-only view so you can't accidentally mutate the canonical data.

```python
# Get the NumPy array (read-only view)
frames = data.raw.numpy()
print(frames.shape)  # (300, 480, 640, 3) — [T, H, W, C]

# Individual frame
frame_0 = frames[0]
```

### Verify integrity

Every `.hapt` file carries a SHA-256 checksum of its raw data.

```python
# Explicit verification
assert data.raw.verify()

# load() already verified on read — checksum mismatch raises ChecksumError
```

## Save a .hapt file

Round-trip is lossless and guaranteed.

```python
data.save("copy.hapt")
reloaded = haptix.load("copy.hapt")
assert data.raw.checksum == reloaded.raw.checksum
```

## Convert native sensor data

Use a sensor adapter to convert native formats into `.hapt`.

```python
from haptix import get_sensor
from haptix.core import InteractionMeta, Labels

adapter = get_sensor("DIGIT")
data = adapter.load(
    path="recordings/digit_session/",
    interaction=InteractionMeta(
        type="sliding",
        speed_mm_s=50.0,
        normal_force_N=2.0,
    ),
    labels=Labels(
        material="sandpaper_grit_80",
        task="sliding",
    ),
)
haptix.save(data, "outputs/sandpaper_80.hapt")
```

## Use with PyTorch

Convert `HaptData` to a PyTorch dataset for training.

```python
from torch.utils.data import DataLoader

dataset = data.to_torch(batch_size=32)
loader = DataLoader(dataset, shuffle=True)

for batch in loader:
    # batch is a dict with keys: raw, sensor, labels, ...
    frames = batch["raw"]  # torch.Tensor shape [32, 3, 480, 640]
    pass
```

## List available sensors

```python
haptix.list_sensors()
# ['DIGIT', 'DIGIT_v2']
```
