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

haptix operates at two layers:

```
                          L0: Sensor Abstraction
  ┌──────────┐     ┌─────────────────────────────────┐
  │  DIGIT   │────▶│  DigitAdapter.can_load()        │
  └──────────┘     │  DigitAdapter.load() → np.array │
  ┌──────────┐     │                                 │
  │ GelSight │────▶│  GelSightAdapter                │
  └──────────┘     │                                 │
  ┌──────────┐     │  CoroCapacitiveAdapter           │
  │ Coro CSV │────▶│  (auto-discovered via @register)│
  └──────────┘     └───────────────┬─────────────────┘
                                   │
                    ┌──────────────▼─────────────────┐
                    │      L1: Container Format       │
                    │  ┌───────────────────────────┐  │
                    │  │ experiment.hapt/          │  │
                    │  │  ├── manifest.json        │  │
                    │  │  ├── provenance.json      │  │
                    │  │  ├── raw/data.npy         │  │
                    │  │  ├── raw/checksum.sha256  │  │
                    │  │  ├── labels.json          │  │
                    │  │  └── unified/  (optional) │  │
                    │  └───────────────────────────┘  │
                    └───────────────┬─────────────────┘
                                    │
                    ┌──────────────▼─────────────────┐
                    │       ML Framework             │
                    │  .to_torch() → DataLoader      │
                    │  .to_jax()   → JAX arrays      │
                    └────────────────────────────────┘
```

- **L0** (`haptix/sensors/`): Adapters that know how to parse each sensor's native format into a numpy array. Two methods: `can_load()` + `load()`. Auto-discovered via `@register`.
- **L1** (`haptix/core.py`, `io.py`): The `.hapt` format — immutable raw data, checksums, provenance tracking, content-addressable file identity.

L2 (sim-to-real, data augmentation) and L3 (downstream benchmarks) live above `.hapt` — the format stays out of the way.

---

## Supported Sensors

| Sensor | Modality | Native Format | Status |
|--------|----------|---------------|--------|
| DIGIT / DIGIT v2 | imaging | PNG/JPEG dir, .mp4 | ✅ Supported (video stubbed) |
| GelSight / GelSight Mini | imaging | PNG/JPEG dir | ✅ Supported |
| Lab-CORO Capacitive | dynamic | CSV (57-taxel) | ✅ Supported (real + simulated) |
| BioTac SP (SynTouch) | dynamic | CSV (19 electrodes + PDC/PAC/TDC/TAC) | ✅ Supported |
| TacTip (Bristol) | imaging / dynamic | PNG/JPEG dir or CSV pin positions | ✅ Supported |
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

### v0.1 — Foundation ✅
- [x] `.hapt` container format spec (v0.1)
- [x] Core data model: `HaptData`, `RawData`, `SensorMeta`, `InteractionMeta`, `Labels`
- [x] IO: `save()` / `load()` with checksum verification and round-trip guarantee
- [x] Sensor adapter registry with auto-discovery (`@register` decorator)
- [x] DIGIT, GelSight, Lab-CORO adapters
- [x] PyTorch integration: `.to_torch()` → Dataset / DataLoader
- [x] JAX integration: `.to_jax()` → JAX arrays
- [x] Dataset catalog infrastructure (download, cache, info)
- [x] CI/CD: lint + test on Python 3.10–3.12

### v0.2 — Real Data & Coverage ✅
- [x] Provenance tracking: `provenance.json` with file hashes, derivation chain, processing history
- [x] Content-addressable file identity (`file_hash` = SHA-256 of directory)
- [x] `coordinate_frame` field in manifest (world / sensor_local / robot_base / object)
- [x] `timestamps_s` per-frame timestamps in manifest (always present, null for equal spacing)
- [x] Spec v0.2: [`spec/hapt-spec-v0.2.md`](spec/hapt-spec-v0.2.md)
- [x] Real sensor validation — Coro + GelSight validated with real data. DIGIT structurally identical.
- [x] PyPI publication — haptix 0.2.0 live on PyPI (2026-08-03), verified fresh-venv install + round-trip
- [x] BioTac SP, TacTip adapters
- [x] End-to-end demo: sensor data → .hapt → PyTorch training loop (~12s on CPU)
- [x] Zarr+Zstd compression mode (`.hapt.zarr`)
- [x] ZIP archive mode (`.hapt.zip`) — single-file stdlib archive, no extra deps
- [x] Hosted dataset catalog with provenance + checksum linking

### v0.3 — Unified Representations
- [x] Cross-sensor latent space (SharedForceEncoder prototype — surrogate projections)
- [x] CrossModalEncoder — trained cross-sensor alignment via CCA + Procrustes (weights serializable to `.npz`)
- [x] `unified/` directory in `.hapt` container (transform metadata, model versioning, checksum)
- [ ] Pre-trained encoders for common sensor types (design: [`docs/encoder-registry.md`](docs/encoder-registry.md))
- [ ] Foundation model for tactile data

### Beyond
- [ ] Streaming / temporal windowing for long recordings
- [ ] Real-time data collection toolkit
- [ ] Integration with robot learning frameworks (Diffusion Policy, ACT, etc.)
- [ ] Tactile data browser / visualization tool
- [ ] Community sensor adapter contributions

---

## Related Projects & Differentiation

haptix doesn't compete with these — it complements them. Our role is the storage and interchange layer that connects data producers to ML consumers.

| Project | Relationship | Notes |
|---------|-------------|-------|
| **TouchNet** (Eric / 一木科技) | Upstream / complementary | TouchNet solves data collection + annotation + model training. If TouchNet outputs `.hapt`, any `.hapt` reader can directly consume TouchNet data. We plan to actively pursue interoperability. |
| **TactiDex** (Ni et al., 2026) | Reference benchmark | Real-world tactile-guided dexterous manipulation benchmark. Shows what downstream tasks need — haptix provides the data format they'd consume. |
| **ViTacWorld** (Huang et al., 2026) | Reference method | Scaling visuo-tactile world models. Their world model could be stored as `.hapt/unified/` representations. |
| **OPENTOUCH** (Song, Li, Fu et al., MIT/CMU) | Reference dataset | First in-the-wild egocentric full-hand tactile dataset. Natural candidate for haptix catalog hosting. |
| **Touch and Go** (Yang et al., 2022) | Reference dataset | Paired egocentric video + tactile. Shows the need for cross-modal alignment — exactly what `unified/` targets. |
| **LeRobot** (Hugging Face) | Complementary | Robot learning datasets. haptix could provide tactile format support for LeRobot's dataset ecosystem. |
| **Open X-Embodiment** (Google DeepMind) | Complementary | Large-scale robot manipulation datasets. Currently vision+proprioception dominant — haptix could add standardized tactile. |
| **SSVTP** / **TouchNet-Bench** | Reference benchmark | Tactile perception benchmarks. Their evaluation protocol would benefit from a format that guarantees checksummed reproducibility. |

---

## Format

The `.hapt` specification is in [`spec/hapt-spec-v0.2.md`](spec/hapt-spec-v0.2.md). Key properties:

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
pytest -v                        # 235 tests
ruff check haptix/ tests/        # lint
black haptix/ tests/             # format
```

This project is under active autonomous development by 幽赜 (Noema), a Hermes-based autonomous agent by Ronald Xia.

---

## License

- **Code**: MIT © Ronald Xia
- **Format specification**: MIT (same as code)
- **Datasets and pre-trained models** (future): Individual licenses specified in each dataset's `provenance.json`. Models may use OpenRAIL-M or similar layered licenses.
- **Contributions**: By submitting a PR, you agree to license your contribution under MIT. See [CONTRIBUTING.md](CONTRIBUTING.md).
