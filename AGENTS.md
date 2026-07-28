# haptix Project Notes

## Autonomous Development

This project uses Hermes Agent for autonomous development.

### Active Profiles
- **main**: DeepSeek V4 Pro — planning, architecture, QC
- **haptix-dev**: implementation — coding, tests, CI

### Kanban Board
- Backlog: features, sensors, datasets
- In Progress: current sprint
- Review: QC gate before merge

### Cron Jobs
- `haptix-evening`: Daily autonomous development at 19:00 BJT. Works through phases in order:
  - Phase 1: Real sensor data validation
  - Phase 2: PyPI publication
  - Phase 3: Sensor coverage (BioTac, TacTip)
  - Phase 4: Unified representations
  - Demo prerequisite: end-to-end pipeline (real data → .hapt → PyTorch training)

### Deferred
- **Contact Eric / TouchNet** — waiting for working demo (real data → .hapt → PyTorch, end-to-end). Tracked in [docs/TODO.md](docs/TODO.md).
