# Hands-On Walkthrough — haptix

> Written 2026-08-12 for Ronald's first hands-on evaluation (updated 2026-09-17
> so every snippet is copy-paste runnable). Goal: understand what haptix IS, how
> it works, and what it can do — in ~20 minutes, one command at a time. After
> this, we define v1.0.0 and un-gate X marketing + TouchNet outreach.

## Environment (verified working)

```bash
PY=/opt/anaconda3/bin/python3     # Python 3.13.5, haptix 0.2.0 installed (editable)
# Run from the haptix repository root.
```

One-command tour of sections 3–5 + 7 (no interactive browser):

```bash
$PY examples/walkthrough.py
```

Expect: ends with `WALKTHROUGH OK (<seconds>)` — about 40 s on this laptop
(the three storage backends each copy the ~74 MB sample; first run may
download ~2.7 MB to `~/.haptix/cache/datasets/`).

## 1. What is haptix? (30 seconds)

haptix = "the JPEG + ImageNet for touch". A file format (`.hapt`) + Python
library for tactile sensor data — the storage/interchange layer between data
producers (sensors, robots) and ML consumers (PyTorch, JAX).

Three pillars:
- **Format**: `.hapt` is a directory with raw data + mandatory metadata
  (sensor type, interaction parameters, provenance, checksums). Immutable,
  SHA-256 verified on every load.
- **Sensors**: adapters for GelSight, DIGIT, Lab-CORO, BioTac SP, TacTip
  (registry: `haptix.get_sensor` / `haptix.list_sensors`).
- **ML bridge**: `.to_torch()` / `.to_jax()` / `WindowedDataset` + trained
  per-sensor encoders + cross-sensor alignment (`CrossModalEncoder`).

## 2. Run the end-to-end demo (real data → .hapt → PyTorch training)

```bash
$PY examples/end_to_end_demo.py
```

Downloads real GelSight + Coro data when needed (SHA-256 verified), packages
into `.hapt`, trains a tiny CNN, and fits a `CrossModalEncoder` — about
45 s on this laptop, most of it CNN training (first run downloads ~2.7 MB to
`~/.haptix/cache/datasets/`). Prints train/test accuracy and
`Demo complete — pipeline is working end-to-end.`

## 3. Explore a .hapt file by hand

```python
$PY - <<'EOF'
from pathlib import Path

import haptix
from haptix.core import InteractionMeta, Labels

demo = haptix.download_dataset("haptix_demo_sample")
print("cache:", haptix.cache_info())

frames = demo / "gelsight" / "002_master_chef_can"
data = haptix.get_sensor("GelSight").load(
    frames,
    interaction=InteractionMeta(type="pressing"),
    labels=Labels(material="can", object_name="002_master_chef_can"),
)
haptix.save(data, Path("/tmp/my_first.hapt"))

loaded = haptix.load("/tmp/my_first.hapt")
print(loaded)
print(loaded.sensor)
print(loaded.interaction)
print(loaded.labels)
print(loaded.provenance)

ds = loaded.to_torch()
batch = next(iter(ds))
print("torch batch:", batch[0].shape)
EOF
```

Expect: `cache: {'cache_path': '.../.haptix/cache/datasets', 'total_datasets': 1, ...}`
Expect: `HaptData(sensor=GelSight, modality=imaging, shape=(80, 480, 640, 3), labels=can)`
Expect: `Provenance(file_hash='387bfbd7...', ..., created_by='haptix/0.2.0')`
Expect: `torch batch: torch.Size([480, 640, 3])`

## 4. Encoders — the "trained" part (v0.3 headline)

```python
$PY - <<'EOF'
import numpy as np

import haptix
from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta

print(haptix.list_encoders())

demo = haptix.download_dataset("haptix_demo_sample")
data = haptix.get_sensor("GelSight").load(
    demo / "gelsight" / "002_master_chef_can",
    interaction=InteractionMeta(type="pressing"),
    labels=Labels(material="can", object_name="002_master_chef_can"),
)

print(haptix.get_encoder("GelSight").version)
trained = haptix.load_trained("GelSight")  # YCB-Sight v1.0 weights (79.8% LOO)
emb = trained.encode(data)
print("embedding:", emb.shape)

# CrossModalEncoder must be .fit()-ted on paired imaging+dynamic records first
records = []
for mi, material in enumerate(["metal", "rubber"]):
    for trial in range(2):
        rng = np.random.RandomState(mi * 10 + trial)
        img = (rng.randint(40, 200, (4, 32, 40, 3)) + mi * 20).astype(np.uint8)
        records.append(
            HaptData(
                raw=RawData(
                    array=img,
                    checksum=RawData.compute_checksum(img),
                    dtype="uint8",
                    shape=img.shape,
                ),
                sensor=SensorMeta(type="GelSight"),
                modality="imaging",
                sampling_rate_hz=30.0,
                interaction=InteractionMeta(type="pressing"),
                labels=Labels(material=material),
            )
        )
        dyn = (rng.randn(4, 29) + mi * 5.0).astype(np.float32)
        records.append(
            HaptData(
                raw=RawData(
                    array=dyn,
                    checksum=RawData.compute_checksum(dyn),
                    dtype="float32",
                    shape=dyn.shape,
                ),
                sensor=SensorMeta(type="CoroCapacitive"),
                modality="dynamic",
                sampling_rate_hz=30.0,
                interaction=InteractionMeta(type="pressing"),
                labels=Labels(material=material),
            )
        )

align = haptix.CrossModalEncoder(embedding_dim=64).fit(records)
print(f"fitted: {align.n_records} records, classes={align.classes}")
shared = align.encode(data)
print("shared space:", shared.array.shape, shared.method)
EOF
```

Expect: `['BioTac_SP', 'CoroCapacitive', 'DIGIT', 'GelSight', 'TacTip']`
Expect: `encoders/gelsight/v0.1`
Expect: `embedding: (80, 256)`
Expect: `fitted: 8 records, classes=['metal', 'rubber']`
Expect: `shared space: (80, 64) unified/cross-modal/v0.2/GelSight`

## 5. Windows / episodes (robot-learning integration)

```python
$PY - <<'EOF'
import haptix
from haptix import WindowedDataset
from haptix.core import InteractionMeta, Labels

demo = haptix.download_dataset("haptix_demo_sample")
data = haptix.get_sensor("GelSight").load(
    demo / "gelsight" / "002_master_chef_can",
    interaction=InteractionMeta(type="pressing"),
    labels=Labels(material="can", object_name="002_master_chef_can"),
)
haptix.save(data, "/tmp/my_first.hapt")

ds = WindowedDataset("/tmp/my_first.hapt", window_size=32, stride=8)
print("windows:", len(ds))
batch = next(iter(ds))
print("window shape:", batch.shape)
EOF
```

Expect: `windows: 10`
Expect: `window shape: torch.Size([32, 480, 640, 3])`

## 6. Live capture (optional — requires hardware)

```bash
$PY examples/live_capture.py --sensor coro --out /tmp/live.hapt
```

Requires a connected Lab-CORO (or other supported) sensor. `HaptRecorder`
increments frames and produces a valid `.hapt` on close. Skip if you have no
hardware on this machine.

## 7. Compression modes

```python
$PY - <<'EOF'
import haptix
from haptix.core import InteractionMeta, Labels

demo = haptix.download_dataset("haptix_demo_sample")
data = haptix.get_sensor("GelSight").load(
    demo / "gelsight" / "002_master_chef_can",
    interaction=InteractionMeta(type="pressing"),
    labels=Labels(material="can"),
)

haptix.save(data, "/tmp/x.hapt")          # plain directory
haptix.save(data, "/tmp/x.hapt.zarr")     # Zarr+Zstd (compact, lazy)
haptix.save(data, "/tmp/x.hapt.zip")      # single-file stdlib archive

for path in ("/tmp/x.hapt", "/tmp/x.hapt.zarr", "/tmp/x.hapt.zip"):
    loaded = haptix.load(path)
    print(path, "checksum=", loaded.raw.checksum[:16], "ok=", loaded.raw.verify())
EOF
```

Expect: three lines, same `checksum=` prefix, each with `ok=True`
(directory ~74 MB; `.hapt.zarr` ~22 MB; `.hapt.zip` ~20 MB for this sample).

## 8. Browser (shipped)

Interactive Streamlit gallery for `.hapt` / `.hapt.zip` / `.hapt.zarr`
episodes — see `docs/browser.md`.

```bash
pip install -e ".[browser]"       # streamlit + plotly extra
haptix-browser /tmp               # gallery over local .hapt files
# or: haptix-browser ~/.haptix/cache
```

Opens http://localhost:8501. Library helpers (`haptix.scan_directory`,
`haptix.episode_summary`, …) work without Streamlit.

## After your hands-on

Tell the CLI agent:
1. What made sense / what didn't
2. Whether the demo felt like a real product
3. Your instinct on v1.0.0 scope

Then we record the v1.0.0 definition, and X marketing + TouchNet outreach
un-gate.
