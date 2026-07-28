# haptix

> Tactile data infrastructure for the ML era — the JPEG+ImageNet for touch.

[![CI](https://github.com/YouZe-Noema/haptix/actions/workflows/ci.yml/badge.svg)](https://github.com/YouZe-Noema/haptix/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/haptix/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`haptix` provides a **unified container format** (`.hapt`), a **Python SDK**, and direct **ML framework integration** (PyTorch / JAX) for tactile sensor data. It does for touch what `.jpg` + ImageNet did for vision: a standard way to store, share, and feed tactile data into learning pipelines.

---

## Why

Tactile sensing is fragmented. Every lab uses different sensors (DIGIT, GelSight, BioTac, custom capacitive arrays), different file formats, different metadata conventions. There's no `torchvision` for touch. `haptix` fixes that:

- **One format** — any tactile sensor → `.hapt`. Raw data preserved verbatim, never transcoded.
- **One API** — `haptix.load()`, `data.to_torch(batch_size=32)`. Done.
- **ML-first** — time-first arrays `[T, ...]`, immutable raw data, labels and interaction metadata required by design.
- **Lossless by construction** — SHA-256 checksums on every load. Round-trip guarantee: save + reload produces identical data.

---

## Quick Start

```python
import haptix

# Load any supported tactile sensor file
data = haptix.load("experiment/sandpaper_80.hapt")

# Access rich metadata
print(data.interaction.speed_mm_s)   # 50.0 (contact speed)
print(data.labels.material)          # "sandpaper_grit_80"

# Get data as numpy
frames = data.raw.numpy()            # ndarray [T, H, W, C]

# Feed directly to PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(data.to_torch(batch_size=32, label="material"))

# Lossless round-trip
haptix.save(data, "copy.hapt")
reloaded = haptix.load("copy.hapt")
assert data.raw.checksum == reloaded.raw.checksum  # always true
```

Convert from native sensor formats:

```python
from haptix.sensors import get_sensor
from haptix.core import InteractionMeta, Labels

adapter = get_sensor("DIGIT_v2")
data = adapter.load(
    "path/to/frames/",
    interaction=InteractionMeta(type="sliding", speed_mm_s=50, normal_force_N=2.0),
    labels=Labels(material="rubber", task="grasp_stability"),
)
haptix.save(data, "experiment.hapt")
```

---

## Architecture

```
Native Sensor File          .hapt Container              ML Framework
(DIGIT dir, CSV, etc.)      (manifest + raw + labels)    (PyTorch / JAX)

  ┌──────────┐              ┌──────────────────┐         ┌─────────────┐
  │  DIGIT   │──┐           │  manifest.json   │         │ DataLoader  │
  └──────────┘  │           │  raw/data.npy    │         │ TensorDataset│
                ├─ adapter ─│  raw/checksum    │── .to_torch() ─│             │
  ┌──────────┐  │           │  labels.json     │         │ .to_jax()   │
  │ GelSight │──┘           └──────────────────┘         └─────────────┘
  └──────────┘
  ┌──────────┐
  │ Coro CSV │── adapter ──► same .hapt container
  └──────────┘
```

---

## Supported Sensors

| Sensor | Modality | Native Format | Status |
|--------|----------|---------------|--------|
| DIGIT / DIGIT v2 | imaging | PNG/JPEG dir, .mp4 | ✅ Supported (video stubbed) |
| GelSight / GelSight Mini | imaging | PNG/JPEG dir | ✅ Supported |
| Lab-CORO Capacitive | dynamic | CSV (57-taxel) | ✅ Supported (real + simulated) |
| BioTac (SynTouch) | dynamic | Electrode impedances | 🔜 Planned |
| TacTip (Bristol) | imaging | Optical markers | 🔜 Planned |
| NeuTouch | dynamic | Event-driven spikes | 🔜 Planned |
| ATI Nano17 / load cells | force | 6-DOF force/torque | 🔜 Planned |

**Want to add a sensor?** See the [Adapter Authoring Guide](docs/adapters.md) — it's a two-method protocol with auto-discovery.

---

## Installation

```bash
pip install haptix           # core (numpy, pillow, pandas, pyyaml)
pip install haptix[torch]    # + PyTorch integration
pip install haptix[jax]      # + JAX integration
pip install haptix[all]      # everything + h5py, zarr
pip install haptix[dev]      # development (pytest, black, ruff, mypy)
```

Python 3.10+ required.

---

## Roadmap

### v0.1 — Foundation ✅ (current)
- [x] `.hapt` container format spec (v0.1)
- [x] Core data model: `HaptData`, `RawData`, `SensorMeta`, `InteractionMeta`, `Labels`
- [x] IO: `save()` / `load()` with checksum verification and round-trip guarantee
- [x] Sensor adapter registry with auto-discovery (`@register` decorator)
- [x] DIGIT, GelSight, Lab-CORO adapters
- [x] PyTorch integration: `.to_torch()` → Dataset / DataLoader
- [x] JAX integration: `.to_jax()` → JAX arrays
- [x] Dataset catalog infrastructure (download, cache, info)
- [x] CI/CD: lint + test on Python 3.10–3.12

### v0.2 — Real Data & Coverage (in progress)
- [ ] Real sensor validation (Lab-CORO, DIGIT, GelSight capture data)
- [ ] PyPI publication (`pip install haptix`)
- [ ] BioTac, TacTip adapters
- [ ] Video loading for DIGIT (.mp4)
- [ ] Hosted dataset catalog with download
- [ ] Compression mode (`.hapt.zip`)

### v0.3 — Unified Representations
- [ ] Cross-sensor latent space (shared embedding across modalities)
- [ ] `unified/` directory in `.hapt` container (transform metadata, model versioning)
- [ ] Pre-trained encoders for common sensor types
- [ ] Foundation model for tactile data

### Beyond
- [ ] Streaming / temporal windowing for long recordings
- [ ] Real-time data collection toolkit
- [ ] Integration with robot learning frameworks (Diffusion Policy, ACT, etc.)
- [ ] Tactile data browser / visualization tool
- [ ] Community sensor adapter contributions

---

## Format

The `.hapt` specification is in [`spec/hapt-spec-v0.1.md`](spec/hapt-spec-v0.1.md). Key properties:

- **Immutable raw data** — `RawData` is a frozen dataclass. Once written, sensor data cannot be modified.
- **Checksum-verified** — SHA-256 on every load. Corrupted files raise `ChecksumError`.
- **Interaction metadata required** — no tactile data is meaningful without contact parameters (speed, force, angle, temperature).
- **Modality-aware** — `imaging`, `dynamic`, `force`, `multimodal`. Each has a canonical shape convention.

---

## Development

```bash
git clone https://github.com/YouZe-Noema/haptix.git
cd haptix
pip install -e ".[dev]"
pytest -v                        # 130+ tests
ruff check haptix/ tests/        # lint
black haptix/ tests/             # format
```

This project is under active autonomous development via [Hermes Agent](https://hermes-agent.nousresearch.com).

---

## License

MIT © Ronald Xia
