# Changelog

All notable changes to `haptix` are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). This is the human-readable
release history; see the README roadmap for the broader plan. Per-version git tags are
not yet created.

## [Unreleased]

Work on `main` after the 0.2.0 PyPI release (2026-08-03) that is not yet published.
(`SharedForceEncoder` / `unified/` landed just before that release; the items below
complete the v0.3 stack from the README.)

### Added

- Cross-sensor latent space: `unified/` representations, `SharedForceEncoder`
  prototype (surrogate projections), and `CrossModalEncoder` trained alignment via
  CCA + Procrustes with weights serializable to `.npz`.
- Encoder registry with untrained per-sensor encoders (`haptix/encoders/`,
  `@register_encoder` / `get_encoder`).
- Trained v1.0 encoder weights for GelSight and CoroCapacitive, with catalog-pinned
  SHA-256 digests and checksum-verified auto-download wiring for the Hugging Face
  Hub repo `YouZe-Noema/haptix-encoders` via `haptix.load_trained("GelSight")`
  (and `"CoroCapacitive"`).
- Streaming and temporal windowing for long recordings: `haptix.open_archive()`,
  `HaptArchive.iter_windows()`, `window()`, and `verify()`.
- Real-time capture toolkit: `haptix.HaptRecorder` incremental recording to a valid
  `.hapt` on close.
- PyTorch-native `WindowedDataset` for windowed episode training (Diffusion Policy /
  ACT-style loops); `TemporalDataset` is an alias of `WindowedDataset`.
- Streamlit tactile data browser: `haptix-browser` entry point and `haptix[browser]`
  extra (episode gallery, frame scrubbing, metadata / labels / provenance / unified).
- Hands-on walkthrough docs (`docs/hands-on.md`) for first-user evaluation.

### Changed

- Packaging metadata completed for PyPI; `haptix[all]` now includes the browser extra.
- Lint tooling pinned (`ruff==0.16.0`, `black==24.10.0`) to match CI.
- CI hardened: the test matrix now includes Python 3.13, the lint job also covers
  `scripts/`, and the `package` job now smoke-tests the built wheel
  (`scripts/wheel_smoke.py`) by installing it into a clean venv with no extras and
  verifying the version, the PEP 561 `py.typed` marker, every `haptix.__all__` export,
  the `haptix-browser` console-script registration, and a save/load round-trip —
  `twine check` alone never installed the wheel, so a broken artifact could ship.

### Fixed

- CI `types` gate: pin the type-check environment to `numpy<2.5` (`numpy` ≥ 2.5
  ships PEP 695 `type` aliases in its stubs, unparseable under the
  `python_version = "3.10"` target) and pin `mypy==2.3.1`.
- Streamlit browser tests: `st.image` lookup now handles both AppTest API
  generations (`get("image")` on 1.6x, `get("imgs")` on 1.45), so the suite no
  longer depends on which Streamlit the `browser` extra resolves to.
- Content-addressable `file_hash` computation on directory save; documented behaviour
  across storage backends.
- Manifest `created` / `created_by` fields write real timestamps and derive
  `created_by` from the package version.
- CI restored on Python 3.10–3.12 (ruff 0.16 import sorting, py3.10 `tomllib` /
  `tomli` handling).
- Zarr 2.x / 3.x API compatibility in the `.hapt.zarr` save path.
- Streamlit browser: session_state key conflict on Streamlit 1.6x; tolerate a missing
  browse root.
- Demo GelSight loader scans per-object YCB-Sight subfolders; demo classifier uses
  textured synthetic data and deterministic training for stability.
- Packaging metadata rejected by modern setuptools (PEP 639): the legacy
  `License :: OSI Approved :: MIT License` classifier conflicted with
  `license = "MIT"`, so `python -m build` failed and CI `pip install -e ".[dev]"`
  broke on Python 3.10–3.12. Raised the build floor to `setuptools>=77` and added
  a CI `package` job (`build` + `twine check`).

## [0.2.0] - 2026-08-03

### Added

- Provenance tracking via `provenance.json` (file hashes, derivation chain, processing
  history).
- Content-addressable file identity: `file_hash` as SHA-256 of the container directory.
- `coordinate_frame` in the manifest (`world` / `sensor_local` / `robot_base` /
  `object`).
- Per-frame `timestamps_s` in the manifest (always present; null when equally spaced).
- Spec v0.2 ([`spec/hapt-spec-v0.2.md`](spec/hapt-spec-v0.2.md)).
- Real-sensor validation for Coro and GelSight; DIGIT structurally identical.
- PyPI publication of haptix 0.2.0.
- BioTac SP and TacTip sensor adapters.
- End-to-end demo: sensor data → `.hapt` → PyTorch training loop.
- Zarr + Zstd compression mode (`.hapt.zarr`).
- ZIP archive mode (`.hapt.zip`) — single-file stdlib archive, no extra deps.
- Hosted dataset catalog with provenance and checksum linking.

## [0.1.0] - 2026-07-31

### Added

- `.hapt` container format specification (v0.1).
- Core data model: `HaptData`, `RawData`, `SensorMeta`, `InteractionMeta`, `Labels`.
- IO `save()` / `load()` with checksum verification and round-trip guarantee.
- Sensor adapter registry with auto-discovery (`@register`).
- DIGIT, GelSight, and Lab-CORO adapters.
- PyTorch integration via `.to_torch()` and JAX integration via `.to_jax()`.
- Dataset catalog infrastructure (download, cache, info).
- CI on Python 3.10–3.12 (lint + tests).
