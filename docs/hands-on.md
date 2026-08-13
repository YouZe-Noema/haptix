# Hands-On Walkthrough — haptix

> Written 2026-08-12 for Ronald's first hands-on evaluation. Goal: understand
> what haptix IS, how it works, and what it can do — in ~20 minutes, one
> command at a time. After this, we define v1.0.0 and un-gate X marketing +
> TouchNet outreach.

## Environment (verified working)

```bash
PY=/opt/anaconda3/bin/python3     # Python 3.13.5, haptix 0.2.0 installed (editable)
cd /Users/ronaldxia/Documents/incubator/project_2
```

## 1. What is haptix? (30 seconds)

haptix = "the JPEG + ImageNet for touch". A file format (`.hapt`) + Python
library for tactile sensor data — the storage/interchange layer between data
producers (sensors, robots) and ML consumers (PyTorch, JAX).

Three pillars:
- **Format**: `.hapt` is a directory with raw data + mandatory metadata
  (sensor type, interaction parameters, provenance, checksums). Immutable,
  SHA-256 verified on every load.
- **Sensors**: adapters for GelSight, DIGIT, Lab-CORO, BioTac SP, TacTip.
- **ML bridge**: `.to_torch()` / `.to_jax()` / `WindowedDataset` + trained
  per-sensor encoders + cross-sensor alignment (`CrossModalEncoder`).

## 2. Run the end-to-end demo (real data → .hapt → PyTorch training)

```bash
$PY examples/end_to_end_demo.py
```

This downloads real GelSight + Coro data (SHA-256 verified), packages into
`.hapt`, and trains a classifier — ~12s on CPU. This is the flagship "show,
don't tell" proof. (First run downloads ~2.7MB to ~/.cache/haptix.)

## 3. Explore a .hapt file by hand

```python
$PY - <<'EOF'
import haptix

# 1) Get real data (checksum-verified download)
haptix.datasets.download_dataset("haptix_demo_sample")
print("cache:", haptix.cache_info())

# 2) Convert a real recording into .hapt
from haptix import GelSightAdapter   # check actual adapter name below
data = GelSightAdapter().load("<path to extracted gelsight frames>")
data.save("/tmp/my_first.hapt")

# 3) Load it back — checksum verified
loaded = haptix.load("/tmp/my_first.hapt")
print(loaded)                       # HaptData(sensor=..., shape=..., labels=...)
print(loaded.metadata)              # mandatory interaction metadata
print(loaded.provenance)            # file hashes, derivation chain

# 4) Straight into ML
ds = loaded.to_torch()              # torch.utils.data.Dataset
batch = next(iter(ds))
print("torch batch:", batch[0].shape)
EOF
```

(Adapter class names: check `haptix.adapters` — e.g. `GelSightAdapter`,
`CoroCapacitiveAdapter`, `BioTacAdapter`. The demo script shows exact usage.)

## 4. Encoders — the "trained" part (v0.3 headline)

```python
$PY - <<'EOF'
import haptix

# Registered encoders per sensor type
print(haptix.list_encoders())

# Trained v1.0 weights (GelSight: 79.8% LOO nearest-centroid on real YCB-Sight)
enc = haptix.get_encoder("GelSight")
trained = haptix.load_trained("GelSight")      # loads real-data weights
emb = trained.encode(data)                     # [T, 256] embedding
print("embedding:", emb.shape)

# Cross-sensor alignment into a shared space
align = haptix.CrossModalEncoder(embedding_dim=64)
shared = align.encode_from_embedding(data, emb)
print("shared space:", shared.shape)           # [T, 64]
EOF
```

## 5. Windows / episodes (robot-learning integration, just shipped)

```python
$PY - <<'EOF'
import haptix
from haptix import WindowedDataset

# Stream long recordings as training windows (memory-mapped, no full load)
ds = WindowedDataset("/tmp/my_first.hapt", window=32, stride=8)
print("windows:", len(ds))
batch = next(iter(ds))
print("window shape:", batch[0].shape)         # [32, ...]
EOF
```

## 6. Live capture (optional — needs a real sensor)

```python
$PY examples/live_capture.py --sensor coro --out /tmp/live.hapt
```

`HaptRecorder` increments frames and produces a valid `.hapt` on close.

## 7. Compression modes

```python
data.save("/tmp/x.hapt")          # default: plain directory
data.save("/tmp/x.hapt.zarr")     # Zarr+Zstd (compact, lazy)
data.save("/tmp/x.hapt.zip")      # single-file stdlib archive
```

## 8. Browser (arriving 2026-08-12 evening run)

```bash
pip install -e ".[browser]"       # streamlit + plotly extra
haptix-browser ~/.cache/haptix    # interactive episode gallery
```

## After your hands-on

Tell the CLI agent:
1. What made sense / what didn't
2. Whether the demo felt like a real product
3. Your instinct on v1.0.0 scope

Then we record the v1.0.0 definition, and X marketing + TouchNet outreach
un-gate.
