# Release process

Checklist for cutting a haptix release to PyPI. Run from a clean clone of `main`.
Do not bump the version or publish unless an owner has explicitly approved the
release.

## 1. Pre-flight

```bash
git checkout main
git status                  # working tree clean
git pull
python -m pytest -v --tb=short
pytest tests/test_roundtrip.py -v -k roundtrip
ruff check haptix/ tests/ scripts/
black --check haptix/ tests/ scripts/
```

Confirm CI is green on Python 3.10 / 3.11 / 3.12 / 3.13
([`.github/workflows/ci.yml`](../.github/workflows/ci.yml) is the source of truth:
`pytest -v --tb=short`, then the round-trip job; lint pins `ruff==0.16.0` and
`black==24.10.0`).

## 2. Version bump

Update **both** version locations to the same string:

- `haptix/_version.py` — `__version__`
- `pyproject.toml` — `[project] version`

**Pitfall:** these two must stay in sync. Bumping only one leaves installs and
`haptix.__version__` disagreeing. Do not bump without an explicit human gate.

## 3. Changelog

In [`CHANGELOG.md`](../CHANGELOG.md), move everything under `## [Unreleased]` into a
new dated section:

```markdown
## [X.Y.Z] - YYYY-MM-DD
```

Leave an empty `## [Unreleased]` heading (with no empty `###` stubs) for the next
cycle.

## 4. Build and verify locally

The package CI job already builds the wheel and runs
[`scripts/wheel_smoke.py`](../scripts/wheel_smoke.py) against a core-only
install (no extras). The venv snippet below is an optional extra local check.

```bash
rm -rf dist/ build/ *.egg-info
python -m build
twine check dist/*
python -m venv /tmp/haptix-verify && source /tmp/haptix-verify/bin/activate
pip install dist/haptix-*.whl
python -c "
import haptix
from haptix.core import HaptData, RawData, SensorMeta, InteractionMeta, Labels
import numpy as np
# minimal round-trip smoke — adapt to a small fixture if preferred
print('haptix', haptix.__version__)
"
deactivate
```

## 5. Publish

Requires PyPI credentials (API token or trusted publisher).

```bash
twine upload dist/*
python -m venv /tmp/haptix-pypi && source /tmp/haptix-pypi/bin/activate
pip install haptix==X.Y.Z
python -c "import haptix; print(haptix.__version__)"
deactivate
```

## 6. Tag and GitHub release

```bash
git tag -a vX.Y.Z -m "haptix X.Y.Z"
git push origin vX.Y.Z
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file - <<'EOF'
Paste the CHANGELOG section for X.Y.Z here.
EOF
```

## 7. Post-release (encoder weights)

If encoder architecture or trained weights changed:

1. Re-upload the `.npz` files to the Hugging Face Hub repo
   [`YouZe-Noema/haptix-encoders`](https://huggingface.co/YouZe-Noema/haptix-encoders).
2. Update the pinned `weights_sha256` (and URL if needed) in
   `haptix/datasets/catalog.py`.
3. Follow the publishing notes in [`docs/encoder-registry.md`](encoder-registry.md).

## 8. Not yet done / gated — 0.3.0

The **0.3.0 release is pending a human decision**. The version bump, PyPI publish,
and git tags are owner-gated and have **not** been performed.

Already staged for 0.3.0 (see `## [Unreleased]` in [`CHANGELOG.md`](../CHANGELOG.md)):

- Cross-sensor latent space (`unified/`, `SharedForceEncoder`, `CrossModalEncoder`)
- Encoder registry + trained GelSight / CoroCapacitive v1.0 weights + HF auto-download
- Streaming / temporal windowing, `HaptRecorder`, `WindowedDataset` / `TemporalDataset`
- Streamlit browser (`haptix-browser`), hands-on docs, packaging / CI fixes

Do not run the steps above for 0.3.0 until that gate is cleared.
