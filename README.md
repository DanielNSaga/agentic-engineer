# Agentic Engineer (WIP)

> **Heads up:** Agentic Engineer is an experimental playground and very much a work in progress. It is not very effective right now, has some rough edges, and while it can occasionally spin up tiny coding experiments, it is just as likely to create small problems along the way. Do not point it at any repository you care about; use a throwaway workspace instead.

## What This Project Tries To Do
- Drive an autonomous-ish coding loop that plans, analyzes, designs, implements, and verifies changes inside a repository.
- Provide a Typer-based CLI (`ae`) that can initialise project scaffolding, bootstrap plans, and run an iteration loop against a stored plan.
- Persist agent memory (plans, tasks, checkpoints, incidents) in SQLite via `MemoryStore` so future runs can resume where they left off.
- Build lightweight context packages (via `ContextBuilder`) that gather repo snippets, symbol data, policy text, and guidance for the LLM.
- Offer deterministic local fallbacks for every phase when a real GPT-5 endpoint is unavailable.

The orchestrator wiring lives under `src/ae/`:
- `cli.py` — entry point exposing `init`, `index`, `iterate`, and `status`.
- `orchestrator.py` — coordinates phase execution, patch application, policy/static checks, and pytest runs.
- `planning/` — bootstrap logic that asks the planner LLM (or local heuristics) to seed plans, tasks, and decisions.
- `phases/` — structured request/response models plus local heuristics for analyze, design, implement, diagnose, fix-violations, plan, and plan-adjust.
- `memory/` — SQLite-backed persistence and a simple code-symbol index.
- `tools/` — wrappers for git hygiene, patch application, pytest execution, cove rage snapshots, and static gate parsing.

## Quick Start
- **Prerequisites:** Python 3.11+, a virtual environment, and a GPT-5 API key exported as `OPENAI_API_KEY` or `GPT5_API_KEY` (the CLI falls back to an offline stub)
- **Install dependencies:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .
  ```
- **Workspace caution:** start from an empty directory and run the CLI there; it writes scaffolding, config, and state files automatically. Avoid running it inside existing projects with code you want to keep.
- **Seed project state:** run `ae init --goal "<goal>"`. The command generates `config.yaml`, scaffolding, the policy capsule, and initial planning artifacts for you.

Key files written by the CLI live under `data/` (ignored by git) for caches, the sqlite database, and log output.

## CLI Commands
- `ae init --goal "<goal>"` — write/update the config file, ensure repo scaffolding exists. Run this in a clean directory so the agent can take over the workspace and plan how to achieve the goal given. 
- `ae iterate` — run tasks from the stored plan.
- `ae status` — report current plan/task counts pulled from the sqlite memory store.

All commands expect to run at the repo root so paths like `data/` and `policy/` are resolved correctly.

## Project Layout
- `src/ae/context_builder.py` — assembles prompt sections, snippets, and instructions under a token budget.
- `src/ae/models/` — LLM client abstraction plus a thin GPT-5 JSON responses adapter.
- `src/ae/tools/` — implementation helpers (patch handling, pytest runner, static gates, git hygiene, scaffold creation).
- `tests/` — pytest suite covering CLI flows, storage, scaffolding, orchestration smoke tests, and tool utilities.
- `data/` — runtime database, log, and cache directories created on demand (not committed).

## Development Workflow
- Run the test suite with `pytest -q`.
- Use `rg` (ripgrep) for searching; many helpers expect it to be available.
- Policy guidance and static parser configuration are controlled via `config.yaml` and `policy/capsule.txt` (created by the scaffold).

## Current Limitations
- Planning and iteration loops assume narrow Python projects and a cooperative git workspace; coverage outside that scope is minimal.
- Patch application and static gate parsing are brittle and may fail on complex diffs or non-Python outputs.
- Expect the agent to need manual supervision—the system is currently better at generating small experiments than shipping reliable fixes.

