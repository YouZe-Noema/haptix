# Encoder Registry — Design Note (v0.1, not implemented)

> **Status:** Design proposal. No code landed yet.
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
  with the same `D` (default 256). The alignment layer and downstream
  consumers depend on this, not on any sensor's internals.
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
    embedding_dim: int        # fixed output dim; 256 by convention
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
    """Get (and lazily import) the encoder for a sensor type."""
    _lazy_import_encoders()
    if sensor_type not in _registry:
        raise ValueError(
            f"Unknown encoder for sensor type: {sensor_type}. "
            f"Available: {list(_registry.keys())}"
        )
    return _registry[sensor_type]()

def list_encoders() -> list[str]:
    """List sensor types with a registered encoder."""
    _lazy_import_encoders()
    return list(_registry.keys())
```

`_lazy_import_encoders()` = `pkgutil.iter_modules` over `haptix/encoders/`,
identical to `haptix.sensors._lazy_import_adapters`. Contribution = drop a
module in the package (or register programmatically) — no core edits.

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
   checksum tooling; optionally host weights on the haptix GitHub release,
   same as `haptix_demo_sample`).
4. Add a `benchmark()` method or doc section reporting eval on a public
   dataset (e.g. YCB-Sight for GelSight) so quality is checkable.

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

## 7. Foundation-model evolution path (non-breaking)

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

## 8. Open questions (decide before implementation)

- **Dim convention:** 256 as `_DEFAULT_EMBEDDING_DIM`? (existing
  `SharedForceEncoder` uses 128, `CrossModalEncoder` 64 — registry should
  standardize on one for the per-sensor layer.)
- **Weights hosting:** GitHub releases (like `haptix_demo_sample`) vs HF Hub
  vs both. Affects the catalog schema (`weights_url` + `sha256` fields).
- **Encoder ↔ adapter naming:** should `get_encoder` validate that a matching
  `SensorAdapter` exists, or are encoders allowed to precede adapters?
- **Trained-from-scratch vs backbone-finetune gating:** is a contributed
  encoder "done" with weights, or is a `benchmark()` hook mandatory?
- **Where does `SharedForceEncoder` go?** Keep as fallback surrogate for
  sensors without a registered encoder (deterministic, zero deps), or retire
  once real encoders cover all catalog sensors?

---

## 9. Relationship to existing code (no changes to current API)

- `haptix/sensors/*` — untouched. Adapters remain the input layer.
- `haptix/unified/encoder.py` — `SharedForceEncoder` and `CrossModalEncoder`
  remain; `CrossModalEncoder` gains `encode_from_embedding` (+ alignment
  semantics), `SharedForceEncoder` stays as the no-weights fallback.
- `haptix/datasets/*` — `verify_checksum` reused for encoder weight files;
  catalog schema gains optional `weights_url`/`weights_sha256` per sensor.
- `.hapt` spec v0.2 `unified/` directory — `method` tags already versioned
  (`unified/cross-modal/v0.2`); per-sensor encoders add
  `unified/encoders/<sensor>/vX.Y` tags. No spec change needed.
