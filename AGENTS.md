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
- `haptix-evening`: Daily autonomous development at 19:00 BJT. **Chained-session model**: works through roadmap items in order, no task cap, no turn cap (max_turns=1000). After each completed task it posts a short checkpoint to Discord (#general). **Decision gate**: if a task needs a decision only Ronald can make, the session stops, writes the question to `~/.hermes/cron/open-questions-haptix.md` (context + options + recommendation), and reports "DECISION REQUIRED". Ronald records the answer in that file (or via the CLI agent) and the next session applies it and continues. Sessions must read that file first at start.
