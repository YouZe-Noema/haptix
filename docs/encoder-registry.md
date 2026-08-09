# Encoder Registry — Design Note (v0.1, implemented 2026-08-09)

> **Status:** Design implemented. Registry + untrained per-sensor encoders
> landed in `haptix/encoders/` (commit df82b8b); trained weights pending.
> **Strategy:** per-sensor encoders first, community-contributed; a foundation
> model only when sensor coverage + alignment data justify it.
> **Companion docs:** `docs/adapters.md` (sensor adapters), `docs/api.md`.

---

## 1. Why a registry, why now

haptix is a growing open-source project. The v0.3 roadmap item "pre-trained
encoders for common sensor types" is the natural next layer on top of the
sensor adapter system. Two design goals drive this proposal:

1. **Accessibility / programmability.** Anyone with data from *their* sensor
   should be able to add a usable encoder without touching core code — the
   same story `@register` already tells for adapters.
2. **A credible path to a foundation model later.** We deliberately do NOT
   train a FM now (data is 99% one sensor family; a "FM" trained on that is
   just a single-sensor encoder with extra steps). Instead we build the *seam*
   a future FM can slot into: a stable encoder interface + shared embedding
   dimension + alignment layer. When coverage reaches the mainstream sensor
   set (GelSight, DIGIT, Coro, BioTac, TacTip, FreeTacMan-sensor, ...) with
   real training data, a FM becomes a drop-in `CompositeEncoder` — non-breaking.

The registry is the contract contributors build against, and the thing that
keeps the future FM swap-in non-breaking.

---

## 2. Guiding principles

- **One encoder per sensor family, one interface for all.**
  Imaging (GelSight, DIGIT) and dynamic (Coro, BioTac, TacTip) sensors look
  different natively but must expose the *same* `encode()` contract so the
  alignment layer can compose them.
- **Fixed output dimension is the contract.** Every encoder emits `[T, D]`
  with a declared, registry-validated `embedding_dim`. Defaults: 256 for
  imaging sensors, 128 for dynamic sensors (see §3.1). The invariant that
  matters: once an encoder's dim is published it is **stable for that sensor
  type** — never changes without a version bump. The alignment layer handles
  heterogeneous dims natively (CCA supports D_a ≠ D_b), so mixed 256/128
  encoders compose cleanly.
- **Versioned and reproducible.** Encoder weights are serialized to `.npz`,
  version strings travel with every embedding (`UnifiedData.method`), and a
  checksum pins each weight file — reuse `verify_checksum` from
  `haptix.datasets`.
- **Deterministic.** Same input → same embedding. Required for provenance and
  for regression tests on contributed encoders.
- **Community-first.** Third-party encoders live in the same registry, gated
  by a lightweight quality contract (docs + checksum + eval hook), not by
  maintainer review of the training run.

---

## 3. API sketch

### 3.1 Protocol (`haptix/encoders/base.py`)

```python
@runtime_checkable
class SensorEncoder(Protocol):
    """Protocol for per-sensor encoders.

    Mirrors SensorAdapter: one class per sensor family, registered
    by sensor_type. The registry contract is the fixed embedding_dim.
    """

    sensor_type: str          # "GelSight", "DIGIT", "CoroCapacitive", ...
    modality: str             # "imaging" | "dynamic" | "force" | "multimodal"
    embedding_dim: int        # fixed output dim; 256 imaging / 128 dynamic
    version: str              # "encoders/gelsight/v1.0"

    def encode(self, data: HaptData) -> np.ndarray:
        """[T, ...] -> [T, embedding_dim]. Deterministic."""
        ...

    def save(self, path: Path) -> None:
        """Serialize weights (+ config) to a single .npz file."""
        ...

    @classmethod
    def load(cls, path: Path) -> "SensorEncoder":
        """Load weights from .npz, returning a ready-to-encode instance."""
        ...
```

Rationale for `encode(HaptData) -> np.ndarray` (vs the existing
`UnifiedEncoder.encode -> UnifiedData`): the per-sensor encoder is the
*front-end* — raw sensor → embedding. The alignment layer (see §5) owns the
`UnifiedData` envelope and shared-space semantics. Keeping them separate means
a per-sensor encoder never needs to know about other sensors.

### 3.2 Registry (`haptix/encoders/__init__.py`)

Mirror the proven `haptix/sensors` pattern exactly:

```python
_registry: dict[str, type[SensorEncoder]] = {}

def register_encoder(sensor_type: str, modality: str = "imaging"):
    """Decorator to register a per-sensor encoder."""
    def decorator(cls):
        cls.sensor_type = sensor_type
        cls.modality = modality
        _registry[sensor_type] = cls
        return cls
    return decorator

def get_encoder(sensor_type: str) -> SensorEncoder:
    """Get (and lazily import) the encoder for a sensor type.

    Returns the best available encoder:
      - the registered trained encoder for *sensor_type*, if any;
      - otherwise :class:`SharedForceEncoder` as a deterministic
        surrogate fallback (version tag carries "/surrogate" so callers
        can distinguish placeholder embeddings from learned ones).

    Encoders may exist without a matching :class:`SensorAdapter` — the two
    registries are intentionally decoupled. A contributed encoder's
    ``benchmark()`` is the evidence that a working data path exists.
    """
    _lazy_import_encoders()
    if sensor_type in _registry:
        return _registry[sensor_type]()
    # Surrogate fallback: deterministic, zero-deps, shape-correct.
    return SharedForceEncoder(embedding_dim=_default_dim_for(sensor_type))

def list_encoders() -> list[str]:
    """List sensor types with a registered encoder."""
    _lazy_import_encoders()
    return list(_registry.keys())
```

`_lazy_import_encoders()` = `pkgutil.iter_modules` over `haptix/encoders/`,
identical to `haptix.sensors._lazy_import_adapters`. Contribution = drop a
module in the package (or register programmatically) — no core edits.

Dimension helper (used by the surrogate fallback):

```python
_IMAGING_DIM, _DYNAMIC_DIM = 256, 128

def _default_dim_for(sensor_type: str) -> int:
    """Default embedding dim for a sensor type: 256 imaging / 128 dynamic."""
    from haptix.sensors import get_sensor  # lazy, decoupled
    try:
        modality = get_sensor(sensor_type).modality
    except ValueError:
        modality = "dynamic"  # unknown sensor → conservative default
    return _IMAGING_DIM if modality == "imaging" else _DYNAMIC_DIM
```

(If a modality is unavailable without an adapter, `get_encoder` falls back
to the dynamic dim — shape-correct, never raises.)

### 3.3 Layout

```
haptix/encoders/
├── __init__.py      # registry: register_encoder / get_encoder / list_encoders
├── base.py          # SensorEncoder protocol, _DEFAULT_EMBEDDING_DIM, helpers
├── gelsight.py      # @register_encoder("GelSight")    — imaging
├── digit.py         # @register_encoder("DIGIT")       — imaging
├── coro.py          # @register_encoder("CoroCapacitive") — dynamic
├── biotac.py        # @register_encoder("BioTac_SP")   — dynamic
├── tactip.py        # @register_encoder("TacTip")      — dynamic
└── weights/         # (gitignored) trained .npz; or fetched via catalog
```

### 3.4 Public exports (`haptix/__init__.py`)

```python
from haptix.encoders import get_encoder, list_encoders, register_encoder
```

Usage:

```python
import haptix

haptix.list_encoders()          # ["BioTac_SP", "CoroCapacitive", "DIGIT", ...]
enc = haptix.get_encoder("GelSight")
emb = enc.encode(load("sample.hapt"))   # np.ndarray [T, 256]
```

---

## 4. What a contributor ships

The community contribution path mirrors `docs/adapters.md`:

1. Implement `SensorEncoder` for their sensor (or fine-tune the shared
   backbone, see §6).
2. `@register_encoder("TheirSensor", modality="imaging")` in a module under
   `haptix/encoders/`.
3. Ship trained weights as `.npz` (single file, versioned name
   `their_sensor_v1.0.npz`) + a published SHA-256 (reuse the datasets
   checksum tooling; weights hosted on the Hugging Face Hub — see §7).
4. **Mandatory `benchmark()`** — a method (or script in the encoder module)
   that evaluates the encoder on a public dataset (e.g. YCB-Sight for
   GelSight) and returns a small structured report: dataset, metric, score,
   split. An encoder is not "done" without it.
5. **Maintainer verification gate** — before a contributed encoder is
   accepted into the registry, maintainers verify: (a) the `benchmark()`
   result is reproducible, (b) the weights checksum matches, (c) the encoder
   passes the deterministic-input regression test. Contributions that fail
   verification can still live as community forks, but are not listed in
   `haptix.list_encoders()`.

Registry entries that ship without weights are still valid — they document
the architecture and let users train the encoder on their own data
(`encoder.fit(records)` reusing the `CrossModalEncoder` training loop).

---

## 5. Alignment layer — the composer

Per-sensor encoders produce *per-sensor* embeddings. The existing
`CrossModalEncoder` (CCA + Procrustes, `haptix/unified/encoder.py`) becomes
the alignment layer that maps any per-sensor embedding into the *shared*
space, exactly as it does today for raw features:

```python
align = CrossModalEncoder(embedding_dim=64).fit(labeled_records)

# Pipeline for any sensor:
emb = get_encoder(data.sensor.type).encode(data)   # [T, 256] per-sensor
shared = align.encode_from_embedding(data, emb)    # [T, 64] shared space
```

This preserves the existing public API (`CrossModalEncoder.encode(HaptData)`
keeps working for raw data; a new `encode_from_embedding` accepts pre-encoded
per-sensor embeddings). The alignment layer is where the *cross-sensor*
science lives, and it is the piece that will eventually be replaced/absorbed
by a FM — while per-sensor encoders stay stable.

---

## 6. Shared backbone (recommended recipe, not required)

To get transfer benefits without multi-sensor data (the Sparsh finding:
multi-sensor pre-training + per-sensor fine-tune beats from-scratch even with
a frozen backbone), the default recipe is:

- Base: a pretrained vision backbone (DINOv2 / MAE / ResNet) frozen.
- Per-sensor head: a small trainable projector → `embedding_dim` (256).
- Dynamic sensors: a small temporal MLP/1D-CNN over `[T, F]` → 256.

The protocol does not mandate this — contributors may bring any backbone as
long as `encode()` is deterministic and emits `[T, 256]`. The recipe is a
convention documented in `docs/encoders.md` (this file, §7 extension), not an
enforcement.

---

## 7. Weights hosting: Hugging Face Hub

**Decision (2026-08-09):** encoder weights are hosted on the Hugging Face Hub,
starting with the first stable release. Rationale:

- HF Hub is the de-facto community distribution channel for model weights;
  contributors already have accounts, and `huggingface_hub` gives versioned
  downloads + a built-in audit trail.
- It matches the catalog pattern: a catalog entry per encoder pins
  `weights_url` (HF Hub resolve URL) + `weights_sha256`, and the existing
  `verify_checksum` flow validates integrity on download — identical to how
  `haptix_demo_sample` works today, just pointed at HF instead of a GitHub
  release.
- GitHub releases stay for dataset archives (as today); HF Hub is the
  weights home. No overlap, no ambiguity.

Catalog schema addition (per sensor, optional until weights exist):

```python
"encoder": {
    "weights_url": "https://huggingface.co/YouZe-Noema/haptix-encoders/resolve/main/gelsight_v1.0.npz",
    "weights_sha256": "64-char-hex",
    "embedding_dim": 256,
}
```

## 8. Foundation-model evolution path (non-breaking)

A future FM is just another registered encoder — a `CompositeEncoder`:

```python
@register_encoder("foundation", modality="multimodal")
class FoundationEncoder:  # handles any sensor, dispatches internally
    ...
```

Because the public contract is `get_encoder(sensor_type).encode(data) ->
[T, D]` with fixed `D`, swapping per-sensor encoders for a FM is invisible to
downstream users. Trigger conditions (from design discussion, 2026-08-09):

1. ≥3 sensors with 100K+ real frames each in the catalog, AND
2. the alignment layer shows capacity limits (CCA/Procrustes error stops
   improving; nonlinear cross-modal mapping needed), AND
3. a benchmark proves the FM beats the per-sensor ensemble on a held-out
   sensor.

Until then: per-sensor encoders + alignment is the product.

---

## 9. Open questions (decide before implementation)

**Resolved (2026-08-09):**

- **Dim convention:** 256 for imaging sensors, 128 for dynamic sensors
  (rationale in §3.1). Not a universal 256 — per-sensor defaults, stable
  once published.
- **Weights hosting:** Hugging Face Hub, from the first stable release
  (see §7). Catalog schema gains `weights_url` + `weights_sha256`.
- **Contribution gate:** contributed encoders require maintainer
  verification (reproducible `benchmark()`, checksum match, determinism
  test) before appearing in `list_encoders()` (see §4).

**Status: all resolved (2026-08-09).** Final decisions on the remaining
items:

- **Encoder ↔ adapter coupling:** decoupled by design. Encoders may precede
  adapters; `get_encoder()` never hard-validates against the adapter
  registry (see §3.2). `benchmark()` is the proof a data path works.
- **`SharedForceEncoder`:** stays as the automatic surrogate fallback for
  sensors without a registered trained encoder, version-tagged
  `.../surrogate` so placeholder embeddings are never mistaken for learned
  ones (see §3.2). Retired only if/when trained encoders cover the full
  catalog.

---

## 10. Relationship to existing code (no changes to current API)

- `haptix/sensors/*` — untouched. Adapters remain the input layer.
- `haptix/unified/encoder.py` — `SharedForceEncoder` and `CrossModalEncoder`
  remain; `SharedForceEncoder` becomes the automatic surrogate fallback
  served by `get_encoder()` for sensors without a registered trained encoder
  (see §3.2); `CrossModalEncoder` gains `encode_from_embedding` (+ alignment
  semantics).
- `haptix/datasets/*` — `verify_checksum` reused for encoder weight files;
  catalog schema gains optional `weights_url`/`weights_sha256` per sensor.
- `.hapt` spec v0.2 `unified/` directory — `method` tags already versioned
  (`unified/cross-modal/v0.2`); per-sensor encoders add
  `unified/encoders/<sensor>/vX.Y` tags. No spec change needed.
