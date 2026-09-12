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

  Note (2026-09-12): the gate rule was since relaxed — an unanswered
  question blocks **only its own item**; the session skips it and continues
  with independent work (docs/tests/refactors). Only end a session early if
  *every* remaining task depends on an unanswered question.

## Implementation Lane (Cursor CLI)

Implementation work is delegated to the Cursor CLI in an isolated git
worktree; the orchestrator only reviews/applies/commits.

```bash
cd /Users/ronaldxia/Documents/incubator/project_2
git worktree add -b cursor/<task> /tmp/haptix-lane main
cat > /tmp/lane_prompt_<task>.md   # self-contained prompt
cd /tmp/haptix-lane && AGENT_CLI_CREDENTIAL_STORE=file CURSOR_API_KEY=$CURSOR_API_KEY \
  <CURSOR_AGENT_BIN> -p --trust --force --model auto "$(cat /tmp/lane_prompt_<task>.md)"
```

**Pitfall — host is macOS 12.7.6 (Monterey) and the newest cursor-agent
does not run there.** Versions `>= 2026.09.02` ship a bundled node built for
macOS 13+; launching them dies with:

```
dyld: Symbol not found: (__ZNSt3__122__libcpp_verbose_abortEPKcz)
  Referenced from: .../cursor-agent/versions/<ver>/node
```

The `agent` / `cursor-agent` symlinks in `~/.local/bin` point at the newest
(broken) version, so the old instruction "just run `agent`" silently fails.
Use a known-good version binary directly instead:

```bash
CURSOR_AGENT_BIN=/Users/ronaldxia/.local/share/cursor-agent/versions/2026.08.31-4057e58/cursor-agent
```

Verify with `<bin> --version` before launching a lane. Also use a
**unique lane-prompt filename** (`/tmp/lane_prompt_<task>.md`) — concurrent
Hermes sessions share `/tmp/lane_prompt.md` and will clobber each other.
If no version runs, fall back to implementing directly and say so in the
session report.
