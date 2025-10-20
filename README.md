# Agentic Engineer

Agentic Engineer is a Typer-based command line toolkit for running autonomous coding iterations. It wraps an orchestration loop, planning primitives, memory store, and developer tooling so you can experiment with agentic workflows locally or connect them to hosted GPT-5 endpoints.

## Features
- Plan-driven iterations that coordinate analyze → design → implement cycles through the `Orchestrator`.
- SQLite-backed `MemoryStore` that persists plans, tasks, checkpoints, test runs, and decisions between invocations.
- Rich `ContextBuilder` that stitches together policy guidance, repository summaries, static findings, and embeddings when preparing prompts.
- Flexible GPT-5 client abstraction with an offline stub for local development and scripted demos.
- Command suite for bootstrapping configs, building code indexes, replaying captured phases, and inspecting runtime status.

## Project Layout
- `src/ae/cli.py` – Typer entrypoint exposing the `ae` CLI.
- `src/ae/orchestrator.py` – Core iteration loop, patch application, test execution, and policy gating.
- `src/ae/context_builder.py` – Bounded context assembly for LLM calls (symbols, embeddings, snippets).
- `src/ae/memory/` – Data access layer (SQLite schema, reflection logs, code indexes).
- `src/ae/phases/` – Typed request/response definitions for each agent phase.
- `src/ae/tools/` – Utilities for scaffolding repos, running pytest, collecting snippets, and interacting with git.
- `tests/` – Pytest coverage for CLI flows, stores, policy gates, and helper utilities.

## Installation
1. Work from a fresh, empty directory that does not contain a `.git` folder. The CLI handles repository scaffolding itself—do **not** run it inside the `agentic-engineer` source tree.
2. Ensure Python 3.11+ is available.
3. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

The `ae` console script is registered via `pyproject.toml`, so it will be available on your `$PATH` once installation succeeds.

## Quick Start
Set either `OPENAI_API_KEY` or `GPT5_API_KEY` in the environment before you begin—`ae init` and later commands call GPT-5 unless you explicitly opt into the offline stub (tests/demos only).

1. From the empty, non-git directory created above, initialise a configuration and optional iteration-zero plan:
   ```bash
   ae init --goal "Ship iteration zero" --plan
   ```
   Add `--no-plan` when you only need the scaffold and config file.
2. Rebuild the code symbol/embedding index at any time:
   ```bash
   ae index
   ```
3. Run a demo iteration loop against the recorded plan (uses GPT-5 by default; pass `--no-use-remote` for the offline stub that exists solely for tests and local demos):
   ```bash
   ae iterate
   ```
4. Inspect stored plans and READY tasks:
   ```bash
   ae status
   ```
5. Reconstruct a previous phase in an isolated workspace using a saved log:
   ```bash
   ae replay-phase logs/phase-log.json --label diagnose-demo
   ```

All commands accept `--config` to point at a non-default configuration file.

## Configuration
`ae init` creates a YAML configuration (defaults to `config.yaml`). Key sections include:
- `project`: Name, description, and repository root.
- `iteration`: Current iteration counters plus default goal and plan identifier.
- `policy`: Capsule file path and guardrail toggles.
- `sandbox`: Defaults for filesystem/network/approval behaviour.
- `models`: Model aliases, timeouts, and connection settings.
- `git`: Auto-clean/push policies for managed worktrees.
- `context`: Guidance lines injected into the system prompt.
- `paths`: Locations for data, database, logs, and cached indexes.

Example:
```yaml
project:
  name: Example Repo
  repo_root: .
iteration:
  goal: bootstrap
  plan_id: demo-plan
models:
  default: gpt-5-offline
paths:
  data: data
  db_path: data/ae.sqlite
```

## Data & Indexes
- Runtime state is stored in `data/ae.sqlite` by default (configurable via `paths.db_path`).
- Symbol, embedding, and graph indexes live under `data/index/` and are refreshed with `ae index` or automatically as part of `ae init` and `ae iterate`.
- Replay workspaces and logs are created beneath the configured `data` and `logs` directories.

## Development
- Run the automated test suite with:
  ```bash
  pytest
  ```
- Formatting defaults to Black (`line-length = 100`) and import ordering follows `isort`'s Black profile.
- The repository uses `typer`, `pydantic`, `libcst`, and `PyYAML` as core runtime dependencies (see `pyproject.toml`).

## Current Limitations
- Work-in-progress tooling that can be unreliable, especially on larger or more complex repositories.
- Runtime has not been performance-optimized yet—use caution when pointing it at production codebases.
- Codebase is currently bloated and due for refactors; expect tighter modules and leaner dependencies in upcoming iterations.
- Designed for greenfield sandboxes; running it inside existing repositories (including this `agentic-engineer` repo) can lead to unexpected behaviour.

