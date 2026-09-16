# Contributing to haptix

haptix is autonomously developed by 幽赜 (Noema), a Hermes-based agent, with human review by Ronald Xia. This document explains how to contribute.

## How Development Works

- **Autonomous sessions**: Noema runs daily development sessions at 19:00 Beijing time. Each session pulls latest, works on the next roadmap task, runs tests + lint, commits, and pushes.
- **Human review**: Ronald reviews direction, strategy, and architecture decisions. He drives the roadmap and answers questions the agent can't resolve alone.
- **CI enforcement**: Every push to `main` triggers lint (ruff + black), type-checking (`mypy haptix/`), and tests (Python 3.10/3.11/3.12/3.13). The package job builds the wheel and smoke-tests a core-only install of it (`scripts/wheel_smoke.py`). Failures block further autonomous work until fixed. The package ships `py.typed` (PEP 561) so downstream users are type-checked against haptix.

## How to Contribute

### Reporting Bugs

Open an issue on GitHub. Include:
- haptix version (`python -c "import haptix; print(haptix.__version__)"`)
- Python version
- Minimal reproduction code
- Error traceback

The GitHub issue templates will prompt for exactly this information.

### Submitting PRs

1. Fork the repo
2. Create a branch (`feat/my-sensor` or `fix/describe-bug`)
3. Write code following the [Adapter Authoring Guide](docs/adapters.md) if adding a sensor
4. Run tests: `pytest -v`
5. Run lint: `ruff check haptix/ tests/ && black --check haptix/ tests/`
6. Run types: `mypy haptix/` (core install + mypy is enough; optional extras are skipped)
7. Open a PR against `main`

By submitting a PR, you agree to license your contribution under the MIT license.

### Changelog

User-visible changes should add an entry under `## [Unreleased]` in
[`CHANGELOG.md`](CHANGELOG.md), following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The release process is documented in [`docs/release.md`](docs/release.md).

### Adding a New Sensor Adapter

See [docs/adapters.md](docs/adapters.md) for the full guide. The TL;DR:

1. Create `haptix/sensors/yoursensor.py`
2. Implement `can_load(path)` and `load(path, interaction, labels)`
3. Add `@register("YourSensorName")` decorator
4. Write tests in `tests/test_yoursensor.py`
5. PR it

Your adapter will be auto-discovered — no need to touch the registry.

## Code Style

- Follow existing patterns in `haptix/sensors/` for adapters
- Use frozen dataclasses for data containers
- All public APIs must have docstrings
- Line length: 100 characters (configured in `pyproject.toml`)

## License

All contributions are MIT-licensed. See [LICENSE](LICENSE) for the full text.
