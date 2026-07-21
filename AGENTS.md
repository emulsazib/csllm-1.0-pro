# AGENTS.md — Governance Contract (managed by knbase)

This project is governed by **knbase**. Any AI agent working here MUST follow
this workflow. Do not skip steps. This exists so any agent can safely extend the
project from existing code and design without re-deriving context.

## Governance documents (memory-bank/)

- `memory-bank/prd.md` — What we are building and why. The single source of product truth.
- `memory-bank/architecture.md` — High-level system structure, components, and data flow.
- `memory-bank/design.md` — Detailed design decisions, interfaces, and conventions.
- `memory-bank/phase.md` — Where the project is now and what comes next.
- `memory-bank/rules.md` — Hard rules every agent must obey when working in this project.
- `memory-bank/memory.md` — Running knowledge base updated after every task so future agents can extend the project.

Plus system artifacts in `.knbase/` (index, mind map, activity log) — do not edit by hand.

## Required workflow

1. **Read before acting.** Call the `start_session` MCP tool first. It returns a
   compact mind map, per-file summaries, and the current phase. Fetch a full doc
   only when needed via `get_context(files=[...], full=true)` to conserve tokens.
2. **Bootstrap if missing.** If `start_session` reports `NEEDS_BOOTSTRAP`, author
   every missing file with `write_governance_file` (all required sections) based on
   your understanding of the user's request, BEFORE doing any other work.
3. **Gate every task.** Call `begin_task` before making changes and `complete_task`
   after. `begin_task` refuses until context is loaded; `complete_task` refuses
   until `memory.md` was updated for that task.
4. **Update knowledge after each task.** Append what changed to `memory.md`
   ("Recent Changes", "Learnings & Gotchas") and advance `phase.md` when relevant.

## Without MCP (any agent / shell)

- Initialize: `npx knbase init`
- Check status / gate: `knbase status` or `knbase check`
- Run gated commands: `knbase guard -- <your command>` (refuses until context is loaded)
- A git `pre-commit` hook (via `knbase install-hooks`) blocks commits unless a task
  was completed with a memory update.

## Token discipline

Prefer summaries and the mind map over full files. Only load full document content
when a task truly requires it.
