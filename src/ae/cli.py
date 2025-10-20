"""CLI commands for managing Agentic Engineer projects and iterations."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml

from .context_builder import ContextBuilder
from .memory.code_index.indexer import CodeIndexer
from .memory.schema import Plan, PlanStatus, TaskStatus
from .memory.store import MemoryStore
from .models import GPT5Client, LLMClient, LLMClientError
from .models.llm_client import LLMRequest
from .orchestrator import CodingIterationPlan, CodingIterationResult, Orchestrator
from .phases import PhaseName
from .phases.analyze import AnalyzeRequest
from .phases.design import DesignRequest
from .phases.implement import ImplementRequest
from .phases.plan_adjust import PlanAdjustmentItem
from .planning import bootstrap_initial_plan
from .planning.executor import PlanExecutionSummary, PlanExecutor
from .tools.phase_logs import load_phase_log
from .tools.phase_replay import PhaseReplayConfig, prepare_replay_workspace
from .tools.scaffold import ensure_project_scaffold
from .tools.vcs import GitError, GitRepository
from .utils.slug import slugify

APP_HELP = "Agentic Engineer CLI entry point."
DEFAULT_CONFIG_NAME = "config.yaml"

DEFAULT_CONFIG_TEMPLATE: Dict[str, Any] = {
    "project": {
        "name": "",
        "description": "",
        "repo_root": ".",
    },
    "iteration": {
        "current": 0,
        "max_calls": 300,
        "max_cycles": 20,
        "backoff_ms": 500,
        "enable_autofix": True,
        "goal": "",
        "plan_id": "",
    },
    "policy": {
        "capsule_path": "policy/capsule.txt",
        "enable_checks": True,
        "fail_fast": False,
        "no_implicit_optional": False,
    },
    "sandbox": {
        "mode": "workspace-write",
        "network": "enabled",
        "approvals": "on-request",
    },
    "models": {
        "default": "gpt-5-mini",
        "planner": "gpt-5-mini",
        "embedding": "text-embedding-3-large",
        "timeout": 250,
    },
    "git": {
        "auto_clean": {
            "enabled": True,
            "include_untracked": True,
            "max_workspaces": 5,
            "retention_hours": 24,
            "message_template": "ae:auto-clean:{task_slug}-{timestamp}",
        },
        "auto_push": {
            "enabled": False,
            "remote": "origin",
            "set_upstream": True,
            "branch_template": "{plan_slug}-{task_slug}",
        },
    },
    "context": {
        "guidance": [
            "Act as the sole engineer, make decisions independently, avoid questions, and record assumptions in your final summary.",
            "Treat config.yaml and pyproject.toml as read-only.",
            "Do not create separate planning docs; use concise inline comments only where needed and summarize key decisions at the end of each run.",
            "Rebuild state from repo files each iteration and restate critical decisions so future cycles inherit context.",
            "Place code in src/ and tests in tests/; reorganize modules only when it improves maintainability.",
            "Always add or update automated tests when changing code; create new test files as needed.",
            "Assume commands run in the workspace clone; use repo-root-relative paths for files and tests.",
            "Git shows committed state only inspect files directly for uncommitted edits.",
            "Review files before editing; prefer focused diffs unless a broader rewrite is more robust.",
            "Add dependencies to requirements.txt for automatic installation.",
            "The /data folder is ignored by git and reserved for workspace use, never store files there, use a new folder instead.",
            "Do NOT run pytest; tests run automatically.",
            "When using argparse for CLIs, avoid add_subparsers; implement manual routing to handle unknown commands gracefully.",
            "You have created all the code in the repository so don't be scared to delete files and folders if they are redundant, or not needed."
        ]

    },
    "paths": {
        "data": "data",
        "db_path": "data/ae.sqlite",
        "logs": "data/logs",
        "cache": "data/cache",
        "config": DEFAULT_CONFIG_NAME,
    },
}


@dataclass(slots=True)
class ProjectMetadata:
    """Human-friendly project metadata derived from high-level goals."""

    name: str
    description: str


def _copy_config_template() -> Dict[str, Any]:
    """Return a deep copy of the default configuration template."""
    return copy.deepcopy(DEFAULT_CONFIG_TEMPLATE)


def _write_config(config_path: Path, config_data: Dict[str, Any]) -> None:
    """Persist configuration data to disk with stable formatting."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False)


def _fallback_metadata(goal: str) -> ProjectMetadata:
    """Generate project metadata heuristically when the LLM is unavailable."""
    tokens = re.findall(r"[A-Za-z0-9]+", goal)
    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "to",
        "of",
        "with",
        "on",
        "in",
        "by",
        "from",
        "into",
        "initial",
        "version",
        "build",
        "create",
        "make",
        "design",
        "implement",
    }
    filtered = [token for token in tokens if token.lower() not in stopwords]
    if not filtered:
        filtered = tokens
    if not filtered:
        filtered = ["Agentic", "Engineer"]

    words = [token.capitalize() for token in filtered[:4]]
    if len(words) < 2:
        words.append("Initiative")

    suffix_map = [
        ("manager", "Vault"),
        ("assistant", "Assistant"),
        ("tracker", "Tracker"),
        ("planner", "Planner"),
        ("analysis", "Insights"),
        ("agent", "Agent"),
        ("api", "Gateway"),
        ("monitor", "Monitor"),
        ("automation", "Automation"),
    ]
    lower_goal = goal.lower()
    suffix = None
    for needle, candidate in suffix_map:
        if needle in lower_goal:
            suffix = candidate
            break
    if suffix is None:
        suffix = "Program"
    if suffix not in words:
        words.append(suffix)

    name = " ".join(words[:5])
    trimmed_goal = goal.strip().rstrip(".")
    if trimmed_goal:
        remainder = trimmed_goal[0].lower() + trimmed_goal[1:] if len(trimmed_goal) > 1 else trimmed_goal.lower()
        description = f"This project will {remainder}."
    else:
        description = "Agentic Engineer project."
    return ProjectMetadata(name=name, description=description)


def _generate_project_metadata(client: LLMClient, goal: str) -> ProjectMetadata:
    """Request project naming metadata from the LLM with sane fallbacks."""
    cleaned_goal = goal.strip()
    fallback = _fallback_metadata(cleaned_goal or goal)
    if not cleaned_goal:
        return fallback

    request = LLMRequest(
        prompt=(
            "You are helping set up an autonomous coding agent.\n"
            f"Goal: {cleaned_goal}\n\n"
            "Respond with JSON containing two fields:\n"
            "- name: A distinctive Title Case project name (3-6 words) that captures the goal without simply repeating it.\n"
            "- description: One sentence describing the outcome the project delivers to achieve the goal.\n"
        ),
        response_model=ProjectMetadata,
        metadata={
            "phase": "project-metadata",
            "request": {"goal": cleaned_goal},
        },
        temperature=0.2,
    )
    try:
        metadata = client.invoke(request)
    except LLMClientError:
        metadata = fallback

    metadata.name = metadata.name.strip() or fallback.name
    metadata.description = metadata.description.strip() or fallback.description
    if not metadata.description.endswith("."):
        metadata.description = f"{metadata.description}."
    return metadata


def _relative_to_repo(path: Path, repo_root: Path) -> Optional[Path]:
    """Return ``path`` relative to ``repo_root`` or ``None`` if not inside."""
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None


def _commit_bootstrap_config(repo: GitRepository, config_path: Path) -> None:
    """Attempt to commit a freshly created configuration file."""
    relative = _relative_to_repo(config_path, repo.root)
    if relative is None:
        typer.echo("Warning: configuration path is outside the repository; skipping auto-commit.")
        return
    try:
        repo.git("add", relative.as_posix())
        repo.git("commit", "-m", "ae: bootstrap config")
        typer.echo("Committed configuration to git.")
    except GitError as error:
        message = str(error).lower()
        if "nothing to commit" in message:
            return
        typer.echo(f"Warning: failed to commit configuration automatically: {error}")


app = typer.Typer(help=APP_HELP)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration from disk and return it as a dictionary."""
    if not config_path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as error:
        typer.echo(f"Failed to parse config: {error}")
        raise typer.Exit(code=1) from error

    if not isinstance(data, dict):
        typer.echo("Configuration must be a mapping at the top level.")
        raise typer.Exit(code=1)

    return data


def _resolve_repo_root(config: Dict[str, Any], config_path: Path) -> Path:
    """Resolve the repository root from configuration."""

    project_cfg = config.get("project") or {}
    repo_root_value = project_cfg.get("repo_root", ".")
    repo_root_path = Path(repo_root_value)
    if not repo_root_path.is_absolute():
        repo_root_path = (config_path.parent / repo_root_path).resolve()
    return repo_root_path


def _resolve_data_root(config: Dict[str, Any], config_path: Path, repo_root: Path) -> Path:
    """Resolve the data root directory for the current configuration."""

    paths_cfg = config.get("paths") or {}
    data_value = paths_cfg.get("data")
    if isinstance(data_value, str) and data_value.strip():
        candidate = Path(data_value.strip())
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        return candidate

    # Fallback to ContextBuilder defaults to handle legacy configs.
    builder = ContextBuilder.from_config(config, repo_root=repo_root)
    return Path(builder.data_root).resolve()


def _default_replay_identifier(phase: str, plan_id: Optional[str], task_id: Optional[str]) -> str:
    """Construct a stable identifier for replay workspaces."""

    parts: list[str] = []
    if phase:
        parts.append(phase)
    if plan_id:
        parts.append(plan_id)
    if task_id:
        parts.append(task_id)
    label = "-".join(parts) if parts else "phase-replay"
    return slugify(label, fallback="phase-replay", lowercase=True)


def _ensure_initial_commit(config: Dict[str, Any], config_path: Path) -> None:
    """Create an empty initial commit when the repository has no HEAD."""

    repo_root_path = _resolve_repo_root(config, config_path)

    try:
        repo = GitRepository(repo_root_path)
    except GitError:
        return

    head_probe = repo.git("rev-parse", "--verify", "HEAD", check=False)
    if head_probe.returncode == 0 and head_probe.stdout.strip():
        return

    try:
        repo.git("commit", "--allow-empty", "-m", "ae: initial commit")
        typer.echo(f"Initialized empty git history at {repo_root_path}.")
    except GitError as error:
        typer.echo(
            "Failed to create initial git commit automatically. "
            "Configure git user.name/user.email and run "
            "'git commit --allow-empty -m \"Initial commit\"'."
        )
        typer.echo(f"Details: {error}")


def _refresh_code_index(config: Dict[str, Any], config_path: Path) -> List[Path]:
    """Update the code index using the configured repository/data roots."""

    indexer = CodeIndexer.from_config(config, config_path.resolve())
    return indexer.reindex()


def _build_client(config: Dict[str, Any], *, use_remote: bool) -> LLMClient:
    """Select either the real GPT-5 client or the offline stub."""
    models_cfg = config.get("models") or {}
    model_name = str(models_cfg.get("default", "gpt-5-nano"))
    model_name_key = model_name.lower()
    offline_model = model_name_key in {"offline", "gpt-5-offline"} or model_name_key.endswith(
        "-offline"
    )

    if use_remote and not offline_model:
        typer.echo(f"Using GPT-5 client ({model_name}).")
        client_kwargs: Dict[str, Any] = {}
        timeout_value = models_cfg.get("timeout")
        if isinstance(timeout_value, (int, float)) and timeout_value > 0:
            client_kwargs["timeout"] = float(timeout_value)
        max_attempts_value = models_cfg.get("max_attempts")
        if isinstance(max_attempts_value, int) and max_attempts_value > 0:
            client_kwargs["max_attempts"] = max_attempts_value
        retry_delay_value = models_cfg.get("retry_delay")
        if isinstance(retry_delay_value, (int, float)) and retry_delay_value >= 0:
            client_kwargs["retry_delay"] = float(retry_delay_value)
        base_url_value = models_cfg.get("base_url")
        if isinstance(base_url_value, str) and base_url_value.strip():
            client_kwargs["base_url"] = base_url_value.strip()
        api_key_value = models_cfg.get("api_key")
        if isinstance(api_key_value, str) and api_key_value.strip():
            client_kwargs["api_key"] = api_key_value.strip()
        try:
            return GPT5Client(model=model_name, **client_kwargs)
        except ValueError as error:
            message = str(error)
            if "api key" in message.lower():
                typer.echo(
                    "No API key given. Set OPENAI_API_KEY or GPT5_API_KEY, "
                    "or re-run with --no-use-remote to use the offline stub."
                )
            else:
                typer.echo(f"Failed to initialise GPT-5 client: {error}")
            raise typer.Exit(code=1)
        except LLMClientError as error:
            typer.echo(f"Failed to initialise GPT-5 client: {error}")
            raise typer.Exit(code=1)

    if use_remote and offline_model:
        typer.echo(f"Model '{model_name}' is offline-only; using offline stub client.")
    else:
        typer.echo("Using offline stub client.")
    return _OfflineLLMClient()


def _sample_iteration_plan(config: Dict[str, Any]) -> CodingIterationPlan:
    """Produce a representative plan for the demo iterate command."""
    iteration = config.get("iteration") or {}
    goal = str(iteration.get("goal", "bootstrap"))
    plan_id = str(iteration.get("plan_id", "demo-plan"))
    task_id = "TASK-0001"
    product_spec = "### Product Vision\n- Goal: bootstrap\n- Current task: TASK-0001 demo"
    analyze = AnalyzeRequest(
        task_id=task_id,
        goal=goal,
        context="Sample repository state summary.",
        constraints=["Stay within coding guidelines."],
        product_spec=product_spec,
    )
    design = DesignRequest(
        task_id=task_id,
        goal=goal,
        proposed_interfaces=["ae.router.PhaseRouter"],
        constraints=["Remain backwards compatible with the CLI."],
        product_spec=product_spec,
    )
    implement = ImplementRequest(
        task_id=task_id,
        diff_goal="Demonstrate structured phase invocation.",
        touched_files=["src/ae/cli.py"],
        test_plan=[],
        notes=["Focus on non-destructive changes."],
        product_spec=product_spec,
    )
    return CodingIterationPlan(analyze=analyze, design=design, implement=implement, plan_id=plan_id)


def _render_iteration_result(result: CodingIterationResult) -> None:
    """Display a concise summary of the coding iteration run."""
    typer.echo("Iteration summary:")
    typer.echo(
        f"- Checkpoint: {result.checkpoint_label or '(unknown)'} "
        f"(rolled back: {'yes' if result.rolled_back else 'no'})"
    )
    if result.analyze:
        typer.echo(f"- Analyze: {result.analyze.summary}")
    if result.design:
        typer.echo(f"- Design: {result.design.design_summary}")
    if result.patch.attempted:
        typer.echo(f"- Patch: {'applied' if result.patch.applied else 'skipped'}")
        if result.patch.error:
            typer.echo(f"  Reason: {result.patch.error}")
    else:
        typer.echo("- Patch: not attempted")
    if result.gates.report:
        gate_status = "passed" if result.gates.ok else "failed"
        typer.echo(f"- Gates: {gate_status}")
        typer.echo(result.gates.report.format_summary())
    elif result.gates.error:
        typer.echo(f"- Gates: error :: {result.gates.error}")
    if result.tests:
        typer.echo(f"- Pytest: {result.tests.status} (exit {result.tests.exit_code})")
    if result.fix_violations:
        reason = result.fix_violations.reason or "Automatic remediation prepared."
        primary = result.fix_violations.touched_files[0] if result.fix_violations.touched_files else ""
        if primary:
            typer.echo(f"- Fix Violations suggestion: {reason} ({primary})")
        else:
            typer.echo(f"- Fix Violations suggestion: {reason}")
    if result.diagnose:
        causes = ", ".join(result.diagnose.suspected_causes)
        typer.echo(f"- Diagnose: {causes or 'no suspected causes'}")
    if result.commit_sha:
        label = (result.commit_message or "").strip()
        label = label[:80] if label else ""
        typer.echo(f"- Commit: {result.commit_sha[:7]}{(' ' + label) if label else ''}")
    if result.pushed_remote or result.pushed_branch:
        remote = result.pushed_remote or "(unknown remote)"
        branch = result.pushed_branch or "(unknown branch)"
        typer.echo(f"- Push: {remote} {branch}")
    if result.artifact_path:
        typer.echo(f"- Artifact: {result.artifact_path.as_posix()}")
    if result.errors:
        typer.echo("Issues detected:")
        for entry in result.errors:
            typer.echo(f"  - {entry}")
    outcome = "success" if result.ok else "incomplete"
    typer.echo(f"Outcome: {outcome}")


def _render_plan_execution(summary: PlanExecutionSummary) -> None:
    """Render a concise summary for a plan execution run."""
    typer.echo(f"Plan {summary.plan.id} [{summary.plan.status}]")
    if not summary.iterations:
        typer.echo("No READY tasks to execute.")
        return

    for outcome in summary.iterations:
        status_label = "DONE" if outcome.result.ok else "BLOCKED"
        typer.echo(f"- {outcome.task.id}: {outcome.task.title} -> {status_label}")

        if outcome.result.errors:
            for entry in outcome.result.errors:
                typer.echo(f"    ! {entry}")

        if outcome.result.gates.violations:
            typer.echo("    gate violations:")
            for violation in outcome.result.gates.violations:
                typer.echo(f"      - {violation}")

        tests = outcome.result.tests
        if tests is not None and not tests.ok:
            typer.echo(f"    tests: {tests.status} (exit {tests.exit_code})")

        if outcome.result.commit_sha:
            label = (outcome.result.commit_message or "").strip()
            label = label[:80] if label else ""
            typer.echo(f"    commit: {outcome.result.commit_sha[:7]}{(' ' + label) if label else ''}")

        if outcome.plan_adjustment and outcome.plan_adjustment.adjustments:
            typer.echo("    adjustments:")
            for adjustment in outcome.plan_adjustment.adjustments:
                if isinstance(adjustment, PlanAdjustmentItem):
                    typer.echo(f"      - {adjustment.render()}")
                else:
                    typer.echo(f"      - {adjustment}")

        if outcome.follow_up_tasks:
            typer.echo("    new tasks:")
            for follow_up in outcome.follow_up_tasks:
                status = follow_up.status.value
                depends = f" (waits on {', '.join(follow_up.depends_on)})" if follow_up.depends_on else ""
                typer.echo(f"      - [{status}] {follow_up.id}: {follow_up.title}{depends}")

        if outcome.result.artifact_path:
            typer.echo(f"    artifact: {outcome.result.artifact_path.as_posix()}")

    typer.echo(f"Plan status: {summary.plan.status}")
    if summary.completed:
        typer.echo("Goal achieved: plan marked complete.")
    if summary.tasks:
        typer.echo("Task backlog:")
        for task in summary.tasks:
            depends = f" (waits on {', '.join(task.depends_on)})" if task.depends_on else ""
            typer.echo(f"    - [{task.status.value}] {task.id}: {task.title}{depends}")


def _select_plan(store: MemoryStore, plan_id_hint: Optional[str]) -> Optional[Plan]:
    """Select an appropriate plan from the memory store."""
    if plan_id_hint:
        plan = store.get_plan(plan_id_hint)
        if plan is not None:
            return plan

    plans = store.list_plans()
    if not plans:
        return None

    active = [item for item in plans if item.status in {PlanStatus.ACTIVE, PlanStatus.DRAFT}]
    if active:
        return active[-1]
    return plans[-1]


class _OfflineLLMClient(LLMClient):
    """Local stub that synthesizes deterministic JSON responses for demos/tests."""

    def __init__(self) -> None:
        super().__init__("offline", max_attempts=1)
        self.supports_local_logic = True

    def _raw_invoke(self, payload: Dict[str, Any]) -> str:
        metadata = payload.get("metadata") or {}
        phase = metadata.get("phase", "unknown")
        request_data = metadata.get("request") or {}
        if isinstance(request_data, str):
            try:
                request_data = json.loads(request_data)
            except json.JSONDecodeError:
                request_data = {}
        response = self._build_response(str(phase), request_data)
        return json.dumps(response)

    def _build_response(self, phase: str, request: Dict[str, Any]) -> Dict[str, Any]:
        if phase == "project-metadata":
            goal = str((request or {}).get("goal") or "").strip()
            metadata = _fallback_metadata(goal or "Agentic Engineer goal")
            return {
                "name": metadata.name,
                "description": metadata.description,
            }
        if phase == PhaseName.PLAN.value:
            goal = request.get("goal") or "goal"
            constraints = request.get("constraints") or []
            deliverables = request.get("deliverables") or []
            notes = request.get("notes") or []
            tasks = [
                {
                    "id": "plan::understand-requirements",
                    "title": "Understand requirements",
                    "summary": f"Review repository state and clarify acceptance for goal '{goal}'.",
                    "depends_on": [],
                    "constraints": constraints,
                    "deliverables": ["Requirements summary"],
                    "acceptance_criteria": [
                        "Document current behaviour related to the goal.",
                        "Surface unknowns or missing context.",
                    ],
                    "notes": notes,
                },
                {
                    "id": "plan::implement-solution",
                    "title": "Design and implement solution",
                    "summary": f"Propose and implement the changes necessary to achieve '{goal}'.",
                    "depends_on": ["plan::understand-requirements"],
                    "constraints": constraints,
                    "deliverables": deliverables or ["Code updates"],
                    "acceptance_criteria": [
                        "Implementation satisfies all deliverables.",
                        "Relevant tests are updated or added.",
                    ],
                },
                {
                    "id": "plan::validate-outcome",
                    "title": "Validate outcome",
                    "summary": "Run automated checks and document risks before completion.",
                    "depends_on": ["plan::implement-solution"],
                    "deliverables": ["Validation notes"],
                    "acceptance_criteria": [
                        "All automated tests pass.",
                        "Known risks are recorded with mitigations.",
                    ],
                },
            ]
            return {
                "plan_name": f"Plan for {goal}",
                "plan_summary": f"Three-step model-generated plan to accomplish '{goal}'.",
                "tasks": tasks,
                "decisions": [
                    {
                        "id": "plan::decision-scope",
                        "title": "Initial scope assumptions",
                        "content": f"Focus on the primary repository components impacted by '{goal}'.",
                        "kind": "assumption",
                    }
                ],
                "risks": [
                    {
                        "id": "plan::risk-regression",
                        "description": "Changes may introduce regressions in untouched modules.",
                        "mitigation": "Add targeted tests and rely on automated gates.",
                        "impact": "medium",
                        "likelihood": "medium",
                    }
                ],
            }

        if phase == PhaseName.ANALYZE.value:
            constraints = request.get("constraints") or []
            return {
                "summary": f"Analyze task {request.get('task_id', 'TASK')} targeting {request.get('goal', 'goal')}.",
                "plan_steps": [
                    "Review relevant files and recent index snapshots.",
                    "Outline implementation approach tied to the goal.",
                ],
                "risks": [f"Constraint: {item}" for item in constraints],
            }

        if phase == PhaseName.DESIGN.value:
            interfaces = request.get("proposed_interfaces") or []
            return {
                "design_summary": f"Design proposal for task {request.get('task_id', 'TASK')}.",
                "interface_changes": interfaces,
                "rationale": [
                    f"Supports goal: {request.get('goal', 'goal')}.",
                    "Keeps compatibility with existing modules.",
                ],
                "validation_plan": [
                    "Document new interfaces.",
                    "Review with maintainers.",
                ],
            }

        if phase == PhaseName.IMPLEMENT.value:
            task_slug = str(request.get("task_id", "task")).strip().lower() or "task"
            task_slug = task_slug.replace(" ", "-")
            new_file = f"notes/{task_slug}-iteration.md"
            content_lines = [
                f"# Offline iteration for {request.get('task_id', 'TASK')}",
                "This file is created by the CLI stub to demonstrate a valid structured implement response.",
                f"Goal: {request.get('diff_goal', 'N/A')}.",
                "",
                "The production agent would edit real files listed in `touched_files`.",
                "For now we emit a safe placeholder payload that host tooling can convert into a patch.",
            ]
            structured_content = "\n".join(content_lines) + "\n"
            test_commands: list[str] = []
            common_payload = {
                "summary": f"Implements goal: {request.get('diff_goal', 'N/A')}.",
                "test_commands": test_commands,
                "follow_up": (request.get("notes") or []) + [f"Inspect generated file {new_file}."],
            }
            return {
                **common_payload,
                "files": [
                    {
                        "path": new_file,
                        "content": structured_content,
                        "encoding": "utf-8",
                        "executable": False,
                    }
                ],
                "edits": [],
            }

        if phase == PhaseName.FIX_VIOLATIONS.value:
            violations = request.get("violations") or []
            report_path = "notes/fix-violations.md"
            lines = ["# Fix Violations Summary", ""]
            if violations:
                lines.append("## Reported violations")
                lines.extend(f"- {item}" for item in violations)
            else:
                lines.append("No violations supplied; this is a placeholder update.")
            lines.append("")
            lines.append("Autofix: review and re-run static gates.")
            content = "\n".join(lines) + "\n"
            return {
                "rationale": [f"Resolve {item}" for item in violations] or ["Nothing to resolve."],
                "follow_up": ["Re-run static gates."],
                "touched_files": (report_path,),
                "files": [
                    {
                        "path": report_path,
                        "content": content,
                        "encoding": "utf-8",
                        "executable": False,
                    }
                ],
                "edits": [],
            }

        if phase == PhaseName.DIAGNOSE.value:
            failing = request.get("failing_tests") or []
            suspected = [f"Investigate failure in {test}" for test in failing] or [
                "No failing tests provided."
            ]
            return {
                "suspected_causes": suspected,
                "recommended_fixes": ["Inspect recent diffs.", "Add targeted logging."],
                "additional_tests": list(failing),
                "confidence": 0.4,
            }

        if phase == PhaseName.PLAN_ADJUST.value:
            blockers = request.get("blockers") or []
            adjustments = request.get("suggested_changes") or [
                f"Revisit blockers: {', '.join(blockers)}" if blockers else "Maintain current plan."
            ]
            return {
                "adjustments": adjustments,
                "new_tasks": [f"Follow up on {item}" for item in blockers],
                "drop_tasks": [],
                "risks": ["Schedule slip"] if blockers else [],
                "notes": [request.get("reason", "No stated reason.")],
            }

        return {"summary": "Unsupported phase.", "plan_steps": [], "risks": []}

@app.command()
def init(
    config: str = typer.Option(
        DEFAULT_CONFIG_NAME,
        "--config",
        "-c",
        help="Path to the agent configuration file.",
    ),
    plan: bool = typer.Option(
        True,
        "--plan/--no-plan",
        help="Generate iteration-zero planning artifacts for the planning workflow.",
    ),
    goal: Optional[str] = typer.Option(
        None,
        "--goal",
        "-g",
        help="Top-level goal for the planning phase.",
    ),
    constraint: List[str] = typer.Option(
        None,
        "--constraint",
        "-C",
        help="Constraint to enforce (repeatable).",
    ),
    deliverable: List[str] = typer.Option(
        None,
        "--deliverable",
        "-D",
        help="Deliverable expected from the plan (repeatable).",
    ),
    deadline: Optional[str] = typer.Option(
        None,
        "--deadline",
        help="Optional deadline for completing the goal.",
    ),
    note: List[str] = typer.Option(
        None,
        "--note",
        "-N",
        help="Additional contextual note for planning (repeatable).",
    ),
    use_remote: bool = typer.Option(
        True,
        "--use-remote/--no-use-remote",
        help="Call the GPT-5 API instead of the offline stub (requires API key).",
    ),
) -> None:
    """Initialize project state (placeholder)."""
    config_path = Path(config)
    config_exists = config_path.exists()
    config_data = load_config(config_path) if config_exists else _copy_config_template()

    project_cfg = config_data.setdefault("project", {})
    iteration_cfg = config_data.setdefault("iteration", {})
    paths_cfg = config_data.setdefault("paths", {})

    config_dirty = not config_exists
    config_created = not config_exists
    config_written = False

    if paths_cfg.get("config") != config_path.name:
        paths_cfg["config"] = config_path.name
        config_dirty = True

    if not str(project_cfg.get("repo_root") or "").strip():
        project_cfg["repo_root"] = "."
        config_dirty = True

    goal_candidate = goal or iteration_cfg.get("goal")
    goal_text = str(goal_candidate).strip() if goal_candidate is not None else ""
    if not goal_text:
        raise typer.BadParameter(
            "A goal is required. Provide --goal or set iteration.goal in the config.",
            param_hint="--goal",
        )

    if goal_text != str(iteration_cfg.get("goal") or "").strip():
        iteration_cfg["goal"] = goal_text
        config_dirty = True

    planner_client: Optional[LLMClient] = None

    def ensure_client() -> LLMClient:
        nonlocal planner_client
        if planner_client is None:
            planner_client = _build_client(config_data, use_remote=use_remote)
        return planner_client

    metadata_needed = (
        not str(project_cfg.get("name") or "").strip()
        or not str(project_cfg.get("description") or "").strip()
        or config_created
    )
    if metadata_needed:
        metadata = _generate_project_metadata(ensure_client(), goal_text)
        if str(project_cfg.get("name") or "").strip() != metadata.name:
            project_cfg["name"] = metadata.name
            config_dirty = True
        if str(project_cfg.get("description") or "").strip() != metadata.description:
            project_cfg["description"] = metadata.description
            config_dirty = True

    repo_root_path = _resolve_repo_root(config_data, config_path)

    def flush_config() -> None:
        nonlocal config_dirty, config_created, config_exists, config_written
        if not config_dirty:
            return
        _write_config(config_path, config_data)
        action = "Created" if config_created else "Updated"
        typer.echo(f"{action} configuration at {config_path}.")
        config_dirty = False
        config_created = False
        config_exists = True
        config_written = True

    flush_config()

    repo: GitRepository | None = None
    try:
        repo = GitRepository(repo_root_path)
    except GitError:
        try:
            repo = GitRepository.initialise(repo_root_path)
            typer.echo(f"Initialized git repository at {repo_root_path}.")
        except GitError as error:
            typer.echo(
                f"Failed to initialize git repository at {repo_root_path}: {error}"
            )
            raise typer.Exit(code=1) from error

    assert repo is not None  # Narrow for type checkers.

    def commit_config_if_needed() -> None:
        nonlocal config_written
        if not config_written:
            return
        _commit_bootstrap_config(repo, config_path)
        config_written = False

    commit_config_if_needed()

    scaffold_created = ensure_project_scaffold(
        repo.root,
        project_name=project_cfg.get("name"),
    )
    if scaffold_created:
        typer.echo("Seeded project scaffold:")
        for path in scaffold_created:
            typer.echo(f"- {path.as_posix()}")
        created_files = [
            path
            for path in scaffold_created
            if (repo.root / path).is_file() or (repo.root / path).is_symlink()
        ]
        if created_files:
            try:
                repo.git("add", *[path.as_posix() for path in created_files])
                repo.git("commit", "-m", "ae: scaffold project structure")
                typer.echo("Committed scaffold files to git.")
            except GitError as error:
                typer.echo(f"Warning: failed to commit scaffold files automatically: {error}")

    _ensure_initial_commit(config_data, config_path)

    index_updates = _refresh_code_index(config_data, config_path)
    if index_updates:
        typer.echo(f"Refreshed code index ({len(index_updates)} file(s) processed).")
    else:
        typer.echo("Code index already up to date.")

    if not plan:
        typer.echo("Repository ready. Re-run with --plan to generate planning artifacts.")
        return

    constraints = list(constraint or [])
    deliverables = list(deliverable or [])
    notes = list(note or [])

    planner_client = ensure_client()

    with MemoryStore.from_config(config_data) as store:
        artifacts = bootstrap_initial_plan(
            store,
            config_data,
            planner_client,
            goal=goal_text,
            constraints=constraints,
            deliverables=deliverables,
            deadline=deadline,
            notes=notes,
            config_path=config_path,
            repo_root=repo.root,
        )

    ready_tasks = sum(1 for task in artifacts.tasks if task.status == TaskStatus.READY)
    typer.echo(f"Created plan {artifacts.plan.id} targeting '{artifacts.plan.goal}'.")
    typer.echo(f"Plan summary: {artifacts.plan.summary}")
    typer.echo(f"Recorded {len(artifacts.tasks)} task(s) ({ready_tasks} READY).")
    typer.echo(
        f"Captured {len(artifacts.decisions)} planning artifact(s) and "
        f"checkpoint {artifacts.checkpoint.label}."
    )
    plan_identifier = artifacts.plan.id
    if str(iteration_cfg.get("plan_id") or "").strip() != plan_identifier:
        iteration_cfg["plan_id"] = plan_identifier
        config_dirty = True
        flush_config()
        commit_config_if_needed()


@app.command()
def index(
    config: str = typer.Option(
        DEFAULT_CONFIG_NAME,
        "--config",
        "-c",
        help="Path to the agent configuration file.",
    )
) -> None:
    """Build or refresh the code index."""
    config_path = Path(config)
    config_data = load_config(config_path)

    processed = _refresh_code_index(config_data, config_path)
    if processed:
        typer.echo(f"Indexed {len(processed)} file(s):")
        for path in processed:
            typer.echo(f"- {path.as_posix()}")
    else:
        typer.echo("Code index already up to date.")


@app.command()
def iterate(
    config: str = typer.Option(
        DEFAULT_CONFIG_NAME,
        "--config",
        "-c",
        help="Path to the agent configuration file.",
    ),
    use_remote: bool = typer.Option(
        True,
        "--use-remote/--no-use-remote",
        help="Call the GPT-5 API instead of the offline stub (requires API key).",
    ),
) -> None:
    """Run a sample coding iteration using the core orchestrator wiring."""
    config_path = Path(config)
    config_data = load_config(config_path)
    iteration_cfg = config_data.setdefault("iteration", {})

    index_updates = _refresh_code_index(config_data, config_path)
    if index_updates:
        typer.echo(f"Refreshed code index ({len(index_updates)} file(s) processed).")

    client = _build_client(config_data, use_remote=use_remote)
    orchestrator = Orchestrator.from_client(
        client,
        config=config_data,
        config_path=config_path,
    )

    typer.echo("Running plan-driven iteration loop...")

    plan_hint = iteration_cfg.get("plan_id")
    plan_id = str(plan_hint).strip() if isinstance(plan_hint, str) and plan_hint.strip() else None

    with MemoryStore.from_config(config_data) as store:
        executor = PlanExecutor(
            store,
            orchestrator,
            config_path=config_path,
            config=config_data,
            revert_on_exit=False,
        )
        summary = executor.execute(plan_id=plan_id)

        if summary is None:
            goal_text = str(iteration_cfg.get("goal") or "").strip()
            if not goal_text:
                typer.echo(
                    "No plan found and no iteration goal configured. "
                    "Run `ae init --plan --goal \"...\"` or set iteration.goal in the config."
                )
                raise typer.Exit(code=1)

            repo_root = _resolve_repo_root(config_data, config_path)
            typer.echo("No plan found in memory. Bootstrapping a plan automatically...")
            artifacts = bootstrap_initial_plan(
                store,
                config_data,
                client,
                goal=goal_text,
                repo_root=repo_root,
                config_path=config_path,
            )
            new_plan_id = artifacts.plan.id
            if str(iteration_cfg.get("plan_id") or "").strip() != new_plan_id:
                iteration_cfg["plan_id"] = new_plan_id
                _write_config(config_path, config_data)
                typer.echo(f"Recorded plan ID {new_plan_id} in {config_path}.")
            store.reopen()
            summary = executor.execute(plan_id=new_plan_id)

        if summary is None:
            typer.echo(
                "Plan bootstrap failed to produce an executable plan. "
                "Inspect the planning store and configuration."
            )
            raise typer.Exit(code=1)

    _render_plan_execution(summary)


@app.command()
def replay_phase(
    log_path: Path = typer.Argument(..., help="Path to a stored phase log JSON file."),
    config: str = typer.Option(
        DEFAULT_CONFIG_NAME,
        "--config",
        "-c",
        help="Path to the agent configuration file.",
    ),
    restore_workspace: bool = typer.Option(
        True,
        "--restore-workspace/--no-restore-workspace",
        help="Apply the recorded workspace snapshot when available.",
    ),
    keep_existing: bool = typer.Option(
        False,
        "--keep-existing",
        help="Fail if the replay workspace already exists instead of replacing it.",
    ),
    label: Optional[str] = typer.Option(
        None,
        "--label",
        help="Optional identifier used when naming the replay workspace directory.",
    ),
) -> None:
    """Materialise a replay workspace for a stored phase log."""

    config_path = Path(config)
    config_data = load_config(config_path)
    repo_root = _resolve_repo_root(config_data, config_path)
    data_root = _resolve_data_root(config_data, config_path, repo_root)

    log_entry = load_phase_log(log_path)
    identifier = label.strip() if isinstance(label, str) and label else None
    if not identifier:
        identifier = _default_replay_identifier(log_entry.phase, log_entry.plan_id, log_entry.task_id)

    try:
        repo = GitRepository(repo_root)
    except GitError as error:
        typer.echo(f"Failed to open repository at {repo_root}: {error}")
        raise typer.Exit(code=1)

    replay_config = PhaseReplayConfig(
        repo=repo,
        data_root=data_root,
        keep_existing=keep_existing,
        identifier=identifier,
    )

    try:
        workspace = prepare_replay_workspace(
            replay_config,
            log_entry,
            apply_snapshot=restore_workspace,
            allow_partial=True,
        )
    except (GitError, FileExistsError) as error:
        typer.echo(f"Unable to create replay workspace: {error}")
        raise typer.Exit(code=1)

    typer.echo(f"Replay workspace available at {workspace.root}")
    if restore_workspace:
        if workspace.snapshot_applied:
            typer.echo("Applied workspace snapshot from log metadata.")
        else:
            typer.echo("Log does not include a workspace snapshot; using a clean clone.")
    else:
        typer.echo("Workspace snapshot application disabled by flag.")


@app.command()
def status(
    config: str = typer.Option(
        DEFAULT_CONFIG_NAME,
        "--config",
        "-c",
        help="Path to the agent configuration file.",
    )
) -> None:
    """Validate configuration and report basic status information."""
    config_path = Path(config)
    config_data = load_config(config_path)

    project = config_data.get("project") or {}
    iteration = config_data.get("iteration") or {}

    typer.echo(f"Loaded configuration from {config_path}")
    typer.echo(f"Project: {project.get('name', 'unnamed')}")
    typer.echo(f"Iteration: {iteration.get('current', 0)}")

    plan_hint_value = iteration.get("plan_id")
    plan_id_hint = None
    if isinstance(plan_hint_value, str) and plan_hint_value.strip():
        plan_id_hint = plan_hint_value.strip()

    with MemoryStore.from_config(config_data) as store:
        plan = _select_plan(store, plan_id_hint)
        if plan:
            tasks = store.list_tasks(plan_id=plan.id)
            ready_count = sum(1 for task in tasks if task.status == TaskStatus.READY)
            blocked_count = sum(1 for task in tasks if task.status == TaskStatus.BLOCKED)
            done_count = sum(1 for task in tasks if task.status == TaskStatus.DONE)
            typer.echo(f"Plan: {plan.id} [{plan.status}] goal='{plan.goal}'")
            typer.echo(
                f"Tasks: total {len(tasks)} | READY {ready_count} | BLOCKED {blocked_count} | DONE {done_count}"
            )
        ready_tasks = store.list_ready_tasks(plan_id=plan.id if plan else None)

    if ready_tasks:
        typer.echo("READY tasks:")
        for task in ready_tasks:
            typer.echo(f"- ({task.plan_id}) {task.title} [{task.id}]")
    else:
        typer.echo("No READY tasks in memory.")


if __name__ == "__main__":
    app()
