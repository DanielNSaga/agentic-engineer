"""High-level orchestration loop driving iterative coding cycles."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import tempfile
import time
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

from .context_builder import ContextBuilder
from .memory.schema import Checkpoint, TestRun, TestStatus
from .memory.store import MemoryStore
from .models.llm_client import LLMClient, LLMClientError
from .phases import PhaseName
from .phases.analyze import AnalyzeRequest, AnalyzeResponse
from .phases.design import DesignRequest, DesignResponse
from .phases.diagnose import DiagnoseRequest, DiagnoseResponse
from .phases.fix_violations import FixViolationsRequest, FixViolationsResponse
from .phases.implement import ImplementRequest, ImplementResponse
from .phases.plan_adjust import PlanAdjustResponse
from .router import PhaseRouter
from .tools.gates import GateReport, run_policy_and_static
from .tools.hygiene import ensure_workspace_hygiene
from .tools.patch import (
    PatchError,
    PatchTelemetry,
    apply_patch,
    build_structured_patch,
    canonicalise_unified_diff,
)
from .tools.pytest_runner import PytestResult, ensure_requirements_installed, run_pytest
from .tools.scaffold import ensure_project_scaffold
from .tools.snippets import (
    Snippet,
    SnippetRequest,
    StaticFinding,
    build_requests_from_findings,
    collect_snippets,
    normalize_static_findings,
)
from .tools.static_output import resolve_static_parser
from .tools.vcs import GitCheckpoint, GitError, GitRepository

if TYPE_CHECKING:
    from .planning.executor import AutoAppliedAdjustment


_FAILED_LINE_RE = re.compile(r"(FAILED|ERROR)\s+([^\s]+)")

_EXPECTED_FILE_EXTENSIONS: set[str] = {
    ".py",
    ".pyi",
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".ts",
    ".js",
    ".tsx",
    ".jsx",
    ".css",
    ".scss",
    ".html",
}

_PATH_TOKEN_SPLIT_RE = re.compile(r"[^\w./\\-]+")

_PYTEST_OPTIONS_EXPECT_VALUE: set[str] = {
    "-c",
    "-k",
    "-m",
    "-n",
    "-o",
    "-p",
    "--basetemp",
    "--confcutdir",
    "--cov",
    "--cov-config",
    "--cov-report",
    "--durations",
    "--html",
    "--junitxml",
    "--log-cli-level",
    "--log-level",
    "--looponfail-command",
    "--maxfail",
    "--result-log",
    "--rootdir",
    "--self-contained-html",
}

_HIGH_RISK_EDIT_BASENAMES: set[str] = {
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "requirements.in",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "npm-shrinkwrap.json",
}

_DEPENDENCY_FILENAMES: set[str] = {
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-dev.in",
    "requirements-test.txt",
    "constraints.txt",
    "constraints.in",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
}

_DEPENDENCY_SUFFIXES: set[str] = {".in", ".txt"}

_PLACEHOLDER_TEST_LINES: set[str] = {
    "pass",
    "...",
    "return True",
    "return False",
    "return",
    "raise NotImplementedError",
}

_PLACEHOLDER_ASSERTIONS: set[str] = {
    "True",
    "False",
    "0",
    "1",
    "1 == 1",
    "0 == 0",
}


@dataclass(slots=True)
class PatchApplicationResult:
    """Outcome of attempting to apply a patch."""

    attempted: bool = False
    applied: bool = False
    touched_paths: tuple[Path, ...] = ()
    error: str | None = None
    telemetry: PatchTelemetry | dict[str, Any] | None = None
    no_op_reason: str | None = None


@dataclass(slots=True)
class GateRunSummary:
    """Summary of running static gates for an iteration."""

    ok: bool = False
    report: GateReport | None = None
    error: str | None = None
    violations: list[str] = field(default_factory=list)
    suspect_files: list[str] = field(default_factory=list)
    static_findings: list[StaticFinding] = field(default_factory=list)


@dataclass(slots=True)
class CycleAdjustment:
    """Adjustment applied automatically during an iteration cycle."""

    source: str
    description: str
    priority: str | None = None
    touched_paths: tuple[Path, ...] = ()


@dataclass(slots=True)
class IterationCycle:
    """Telemetry for a single implement → gates/tests attempt."""

    cycle_index: int
    reason: str
    gates_ok: bool
    tests_ok: bool | None
    adjustments: list[CycleAdjustment] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IterationSettings:
    """Runtime configuration for the auto-loop."""

    max_cycles: int = 5
    backoff_seconds: float = 0.0
    enable_autofix: bool = True
    structured_edits_only: bool = False


@dataclass(slots=True)
class GitAutomationState:
    """Runtime git automation configuration and bookkeeping."""

    auto_clean_enabled: bool = False
    include_untracked: bool = True
    workspace_path: Path | None = None
    workspace_method: str | None = None
    original_repo_root: Path | None = None
    original_config_path: Path | None = None
    remote_urls: dict[str, str] = field(default_factory=dict)
    preflight_error: str | None = None
    push_enabled: bool = False
    push_remote: str = "origin"
    push_branch_template: str | None = None
    push_branch: str | None = None
    push_force: bool = False
    push_set_upstream: bool = False


@dataclass(slots=True)
class CodingIterationResult:
    """Aggregated results for a single coding iteration attempt."""

    run_id: str = field(default_factory=lambda: uuid4().hex)
    analyze: AnalyzeResponse | None = None
    design: DesignResponse | None = None
    implement: ImplementResponse | None = None
    patch: PatchApplicationResult = field(default_factory=PatchApplicationResult)
    gates: GateRunSummary = field(default_factory=GateRunSummary)
    fix_violations: FixViolationsResponse | None = None
    tests: PytestResult | None = None
    diagnose: DiagnoseResponse | None = None
    plan_adjustments: list[PlanAdjustResponse] = field(default_factory=list)
    diagnose_attempts: list[str] = field(default_factory=list)
    cycles: list[IterationCycle] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    checkpoint_label: str | None = None
    rolled_back: bool = False
    artifact_path: Path | None = None
    commit_sha: str | None = None
    commit_message: str | None = None
    pushed_remote: str | None = None
    pushed_branch: str | None = None

    @property
    def ok(self) -> bool:
        patch_ok = True
        if self.patch.attempted and not self.patch.applied:
            patch_ok = False
        tests_ok = True
        if self.tests is not None:
            tests_ok = self.tests.ok
        return (
            patch_ok
            and self.gates.ok
            and tests_ok
            and not self.errors
        )

    @property
    def diff(self) -> PatchApplicationResult:
        """Backward compatible alias for the applied patch."""
        return self.patch


@dataclass(slots=True)
class CodingIterationPlan:
    """Phase inputs required to run a coding iteration."""

    analyze: AnalyzeRequest
    design: DesignRequest
    implement: ImplementRequest
    plan_id: str | None = None

    @property
    def task_id(self) -> str:
        return self.analyze.task_id

    @property
    def goal(self) -> str:
        return self.analyze.goal


class Orchestrator:
    """Coordinator that can execute individual phases or a full iteration loop."""

    _MAX_GATE_FIX_ATTEMPTS = 3
    _MAX_DIAGNOSE_ATTEMPTS = 2

    def __init__(
        self,
        *,
        client: LLMClient,
        config: Mapping[str, Any] | None = None,
        repo_root: Path | str | None = None,
        context_builder: ContextBuilder | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        self._client = client
        self._config = dict(config or {})
        config_hint = self._resolve_config_path_hint(config_path)
        self._repo_root = self._normalise_repo_root(repo_root, config_hint)
        self._context_builder = context_builder or ContextBuilder.from_config(
            self._config,
            repo_root=self._repo_root,
        )
        project_section = self._config.get("project")
        if isinstance(project_section, Mapping):
            self._project_name = project_section.get("name")
        else:
            self._project_name = None
        self._config_path = self._determine_config_path(config_hint)
        self._router = PhaseRouter(client=client, context_builder=self._context_builder)

    @classmethod
    def from_client(
        cls,
        client: LLMClient,
        config: Mapping[str, Any] | None = None,
        *,
        repo_root: Path | str | None = None,
        config_path: Path | str | None = None,
    ) -> "Orchestrator":
        """Convenience constructor used by the CLI."""
        return cls(
            client=client,
            config=config,
            repo_root=repo_root,
            config_path=config_path,
        )

    def run_phase(self, phase: PhaseName | str, payload: Any) -> Any:
        """Execute a single phase and return its structured response."""
        return self._router.dispatch(phase, payload)

    def available_phases(self) -> list[PhaseName]:
        """Return the registered phases in deterministic order."""
        return list(self._router.available_phases())

    def _rebind_runtime_environment(
        self,
        *,
        repo_root: Path,
        config_path: Path | None = None,
    ) -> None:
        """Update cached roots and rebuild helpers when the target repo changes."""

        repo_root_resolved = Path(repo_root).resolve()
        config_path_resolved: Path | None = None
        if config_path is not None:
            candidate = Path(config_path)
            if not candidate.is_absolute():
                candidate = (repo_root_resolved / candidate).resolve()
            else:
                candidate = candidate.resolve()
            config_path_resolved = candidate

        if repo_root_resolved == self._repo_root:
            if config_path_resolved is not None and config_path_resolved != self._config_path:
                self._config_path = config_path_resolved
            return

        self._repo_root = repo_root_resolved
        if config_path_resolved is not None:
            self._config_path = config_path_resolved
        else:
            self._config_path = self._determine_config_path(None)

        self._context_builder = ContextBuilder.from_config(
            self._config,
            repo_root=self._repo_root,
        )
        self._router = PhaseRouter(client=self._client, context_builder=self._context_builder)

    def run_coding_iteration(
        self,
        plan: CodingIterationPlan,
        *,
        config_path: Path | str | None = None,
        repo_root: Path | str | None = None,
        revert_on_exit: bool = True,
    ) -> CodingIterationResult:
        """Execute the coding loop using the provided plan."""
        result = CodingIterationResult()
        config_hint = self._resolve_config_path_hint(config_path) if config_path is not None else None
        repo_override: Path | None = None
        if repo_root is not None or config_hint is not None:
            repo_override = self._normalise_repo_root(repo_root, config_hint)
            self._rebind_runtime_environment(
                repo_root=repo_override,
                config_path=config_hint,
            )
        repo = self._resolve_repo(repo_override)
        plan_id = self._resolve_plan_id(plan)
        task_id = plan.task_id
        store: MemoryStore | None = None
        db_path: Path | None = None
        store_pre_existing = False
        persisted_checkpoint = False
        persisted_tests = False
        settings = self._resolve_iteration_settings(plan)
        repo, git_state = self._prepare_git_session(repo, plan_id, task_id)
        if git_state.preflight_error:
            result.errors.append(git_state.preflight_error)
            result.patch.no_op_reason = git_state.preflight_error
            self._finalize_git_session(repo, git_state, result, revert_on_exit=revert_on_exit)
            return result
        if git_state.workspace_method == "clone" and git_state.workspace_path:
            workspace_config = self._resolve_workspace_config_path(
                workspace_root=git_state.workspace_path,
                original_root=git_state.original_repo_root,
                original_config=git_state.original_config_path,
            )
            self._rebind_runtime_environment(
                repo_root=git_state.workspace_path,
                config_path=workspace_config,
            )
            repo = self._resolve_repo(git_state.workspace_path)

        while True:
            try:
                repo.ensure_clean(include_untracked=False)
                break
            except GitError as error:
                status_entries = repo.status_entries()
                if self._should_reinitialise_repository(status_entries):
                    repo = self._reinitialise_repository(repo)
                    continue
                dirty_paths = [path for status, path in status_entries if status != "??"]
                message = self._format_dirty_tree_error(dirty_paths, str(error))
                result.errors.append(message)
                result.patch.no_op_reason = message
                self._finalize_git_session(repo, git_state, result, revert_on_exit=revert_on_exit)
                return result

        ensure_project_scaffold(repo.root, project_name=self._project_name)
        self._ensure_scaffold_baseline(repo)

        try:
            db_path = self._resolve_db_path()
            store_pre_existing = db_path.exists()
            store = MemoryStore(db_path)
        except Exception as error:  # pragma: no cover - defensive
            result.errors.append(f"Memory store unavailable: {error}")
            store = None

        checkpoint = repo.create_checkpoint(label=f"ae:{plan.task_id}")
        result.checkpoint_label = checkpoint.label
        if store is not None:
            try:
                persisted_checkpoint = self._persist_checkpoint(store, checkpoint, plan_id, task_id)
            except Exception as error:  # pragma: no cover - defensive
                result.errors.append(f"Failed to record checkpoint: {error}")

        resolved_config_path = self._prepare_config_path(config_path, repo)

        applied_adjustment_hashes: set[str] = set()
        snippets: list[Snippet] = []
        static_findings: list[StaticFinding] = []
        changed_paths: set[Path] = set()
        needs_implement = True
        iteration_success = False

        try:
            result.analyze = self.run_phase(PhaseName.ANALYZE, plan.analyze)
            result.design = self.run_phase(PhaseName.DESIGN, plan.design)
        except (LLMClientError, PatchError, GitError) as error:
            result.errors.append(str(error))
        except Exception as error:  # pragma: no cover - defensive
            result.errors.append(str(error))
        else:
            cycle_index = 0
            while cycle_index < settings.max_cycles:
                if settings.backoff_seconds > 0 and cycle_index > 0:
                    time.sleep(settings.backoff_seconds)
                cycle_index += 1

                cycle_adjustments: list[CycleAdjustment] = []
                cycle_errors: list[str] = []

                if needs_implement:
                    execute_method = self._execute_implement_phase
                    func_obj = getattr(execute_method, "__func__", execute_method)
                    if getattr(func_obj, "_ae_native", False):
                        implement_response, patch_result, snippets_result = execute_method(
                            plan=plan,
                            repo=repo,
                            snippets=snippets,
                            static_findings=static_findings,
                            structured_only=settings.structured_edits_only,
                        )
                    else:
                        implement_response, patch_result = execute_method(plan.implement, repo)
                        snippets_result = list(snippets)
                    snippets = snippets_result
                    result.implement = implement_response
                    result.patch = patch_result

                    if not patch_result.applied:
                        message = patch_result.error or "Implement response did not include any updates."
                        self._record_error(result, message)
                        cycle_errors.append(message)
                        result.cycles.append(
                            self._make_cycle(
                                cycle_index,
                                message,
                                result.gates,
                                result.tests,
                                cycle_adjustments,
                                cycle_errors,
                            )
                        )
                        break

                    install_ok, install_error = self._ensure_requirements_for_changes(
                        repo=repo,
                        touched_paths=patch_result.touched_paths,
                    )
                    if not install_ok:
                        message = install_error or "Failed to install repository requirements."
                        patch_result.applied = False
                        patch_result.error = message
                        patch_result.touched_paths = ()
                        self._record_error(result, message)
                        cycle_errors.append(message)
                        result.cycles.append(
                            self._make_cycle(
                                cycle_index,
                                message,
                                result.gates,
                                result.tests,
                                cycle_adjustments,
                                cycle_errors,
                            )
                        )
                        break
                    changed_paths.update(patch_result.touched_paths)

                gate_method = self._ensure_gates_clean
                gate_func = getattr(gate_method, "__func__", gate_method)
                if getattr(gate_func, "_ae_native", False):
                    gate_summary, gate_adjustments, needs_implement, static_findings = gate_method(
                        repo=repo,
                        config_path=resolved_config_path,
                        plan=plan,
                        plan_id=plan_id,
                        task_id=task_id,
                        result=result,
                        changed_paths=changed_paths,
                        enable_autofix=settings.enable_autofix,
                        applied_adjustment_hashes=applied_adjustment_hashes,
                    )
                else:
                    gate_summary, gate_adjustments = gate_method(
                        repo=repo,
                        config_path=resolved_config_path,
                        plan=plan,
                        result=result,
                        changed_paths=changed_paths,
                    )
                    # Legacy stubs do not signal re-entry or updated findings.
                    needs_implement = True
                    try:
                        from .planning.executor import apply_adjustments_and_reenter as _compat_apply
                    except ImportError:  # pragma: no cover - defensive
                        pass
                    else:
                        _compat_apply(
                            adjustment=None,
                            apply_updates=lambda files, edits: PatchApplicationResult(),
                            applied_hashes=applied_adjustment_hashes,
                        )
                result.gates = gate_summary
                cycle_adjustments.extend(gate_adjustments)

                if not gate_summary.ok:
                    reason = self._infer_gate_reason(gate_summary)
                    error_message = "Static gates failed."
                    self._record_error(result, error_message)
                    cycle_errors.append(error_message)
                    if gate_summary.error:
                        self._record_error(result, gate_summary.error)
                        cycle_errors.append(gate_summary.error)
                    result.cycles.append(
                        self._make_cycle(
                            cycle_index,
                            reason,
                            gate_summary,
                            result.tests,
                            cycle_adjustments,
                            cycle_errors,
                        )
                    )
                    if needs_implement:
                        continue
                    break

                (
                    tests_result,
                    diagnose_response,
                    test_adjustments,
                    tests_ok,
                    static_findings,
                    restart_requested,
                ) = self._run_tests_with_recovery(
                    plan=plan,
                    plan_id=plan_id,
                    task_id=task_id,
                    repo=repo,
                    config_path=resolved_config_path,
                    checkpoint=checkpoint,
                    changed_paths=changed_paths,
                    result=result,
                    implement=result.implement,
                    enable_autofix=settings.enable_autofix,
                    applied_adjustment_hashes=applied_adjustment_hashes,
                    static_findings=static_findings,
                )

                if tests_result is not None:
                    result.tests = tests_result
                    if store is not None and tests_ok:
                        try:
                            persisted_tests = self._persist_test_run(store, tests_result, plan_id, task_id)
                        except Exception as error:  # pragma: no cover - defensive
                            self._record_error(result, f"Failed to record test run: {error}")
                    if not tests_ok:
                        error_message = "Pytest reported failures."
                        self._record_error(result, error_message)
                        cycle_errors.append(error_message)

                if diagnose_response is not None:
                    result.diagnose = diagnose_response

                cycle_adjustments.extend(test_adjustments)

                if restart_requested:
                    restart_summary = ""
                    if diagnose_response is not None:
                        restart_summary = (diagnose_response.restart_summary or "").strip()
                        if not restart_summary:
                            lessons = getattr(diagnose_response, "iteration_lessons", []) or []
                            if lessons:
                                restart_summary = "; ".join(lesson.strip() for lesson in lessons if lesson and str(lesson).strip())
                    if not restart_summary:
                        restart_summary = "Diagnose requested workspace reset; restarting from clean clone."
                    snippets = []
                    static_findings = []
                    changed_paths.clear()
                    applied_adjustment_hashes.clear()
                    needs_implement = True
                    cycle_errors.append(restart_summary)
                    result.cycles.append(
                        self._make_cycle(
                            cycle_index,
                            "diagnose restart",
                            gate_summary,
                            tests_result or result.tests,
                            cycle_adjustments,
                            cycle_errors,
                        )
                    )
                    continue

                if tests_result is not None and not tests_ok:
                    reason = self._infer_test_reason(tests_result)
                    result.cycles.append(
                        self._make_cycle(
                            cycle_index,
                            reason,
                            gate_summary,
                            tests_result,
                            cycle_adjustments,
                            cycle_errors,
                        )
                    )
                    continue

                iteration_success = True
                result.cycles.append(
                    self._make_cycle(
                        cycle_index,
                        "success",
                        gate_summary,
                        result.tests,
                        cycle_adjustments,
                        cycle_errors,
                    )
                )
                break

            if not iteration_success and cycle_index >= settings.max_cycles:
                final_message = f"Auto-loop exceeded {settings.max_cycles} cycles without success."
                filtered_errors = [error for error in result.errors if error != "Static gates failed."]
                if final_message not in filtered_errors:
                    filtered_errors.append(final_message)
                result.errors = filtered_errors or [final_message]
            elif iteration_success and not result.errors and not revert_on_exit:
                try:
                    sha, message = self._auto_commit(
                        repo=repo,
                        plan_id=plan_id,
                        task_id=task_id,
                        implement=result.implement,
                    )
                    result.commit_sha = sha
                    result.commit_message = message
                    if sha and git_state.push_enabled:
                        try:
                            remote, branch = self._maybe_auto_push(
                                repo=repo,
                                git_state=git_state,
                                plan_id=plan_id,
                                task_id=task_id,
                            )
                        except GitError as push_error:
                            self._record_error(result, f"Push failed: {push_error}")
                        else:
                            result.pushed_remote = remote
                            result.pushed_branch = branch
                except GitError as commit_error:
                    self._record_error(result, f"Commit failed: {commit_error}")
        finally:
            if store is not None:
                store.close()
            if revert_on_exit:
                try:
                    checkpoint.rollback()
                    result.rolled_back = True
                except GitError as rollback_error:
                    result.errors.append(f"Rollback failed: {rollback_error}")
                    result.rolled_back = False
            else:
                result.rolled_back = False
            self._finalize_git_session(repo, git_state, result, revert_on_exit=revert_on_exit)
            if (
                revert_on_exit
                and not store_pre_existing
                and not (persisted_checkpoint or persisted_tests)
                and db_path is not None
            ):
                try:
                    db_path.unlink(missing_ok=True)
                except OSError:
                    pass
                parent = db_path.parent
                if parent.exists():
                    try:
                        next(parent.iterdir())
                    except StopIteration:
                        try:
                            parent.rmdir()
                        except OSError:
                            pass
                    except OSError:
                        pass
            try:
                result.artifact_path = self._write_iteration_artifact(
                    plan_id=plan_id,
                    task_id=task_id,
                    goal=plan.goal,
                    iteration_result=result,
                    revert_on_exit=revert_on_exit,
                )
            except Exception as artifact_error:  # pragma: no cover - defensive
                result.errors.append(f"Failed to write iteration artifact: {artifact_error}")

        return result

    # ------------------------------------------------------------------ helpers
    def _execute_implement_phase(
        self,
        *,
        plan: CodingIterationPlan,
        repo: GitRepository,
        snippets: Sequence[Snippet],
        static_findings: Sequence[StaticFinding],
        structured_only: bool,
    ) -> tuple[ImplementResponse, PatchApplicationResult, list[Snippet]]:
        """Backward compatible entry point for implement execution."""

        return self._execute_implement_cycle(
            plan=plan,
            repo=repo,
            snippets=snippets,
            static_findings=static_findings,
            structured_only=structured_only,
        )

    _execute_implement_phase._ae_native = True  # type: ignore[attr-defined]

    def _execute_implement_cycle(
        self,
        *,
        plan: CodingIterationPlan,
        repo: GitRepository,
        snippets: Sequence[Snippet],
        static_findings: Sequence[StaticFinding],
        structured_only: bool,
    ) -> tuple[ImplementResponse, PatchApplicationResult, list[Snippet]]:
        """Run the implement phase, serving snippet requests as needed."""

        active_snippets = self._refresh_snippet_cache(repo, snippets)
        findings = normalize_static_findings(static_findings)
        seen_requests: set[tuple[str, int | None, int | None]] = set()

        while True:
            request = replace(
                plan.implement,
                snippets=list(active_snippets),
                static_findings=list(findings),
                structured_edits_only=structured_only or plan.implement.structured_edits_only,
            )
            if findings:
                self._seed_static_code_requests(request, findings)
            response = self.run_phase(PhaseName.IMPLEMENT, request)

            code_requests = [entry for entry in (response.code_requests or []) if isinstance(entry, SnippetRequest)]
            if code_requests:
                new_requests: list[SnippetRequest] = []
                for entry in code_requests:
                    key = (entry.path.strip(), entry.start_line, entry.end_line)
                    if key in seen_requests:
                        continue
                    seen_requests.add(key)
                    new_requests.append(entry)
                if new_requests:
                    collected = collect_snippets(
                        repo.root,
                        new_requests,
                        static_findings=findings,
                    )
                    added = 0
                    existing = {(snippet.path, snippet.start_line, snippet.end_line) for snippet in active_snippets}
                    for snippet in collected:
                        key = (snippet.path, snippet.start_line, snippet.end_line)
                        if key in existing:
                            continue
                        active_snippets.append(snippet)
                        existing.add(key)
                        added += 1
                    if added > 0:
                        # Serve the snippet-enhanced request to the phase.
                        continue

            missing_high_risk = self._detect_high_risk_blind_replacements(
                repo=repo,
                files=list(getattr(response, "files", []) or []),
                served_snippets=active_snippets,
            )
            if missing_high_risk is not None:
                _, blocked_paths = missing_high_risk
                high_risk_requests = [
                    SnippetRequest(
                        path=path,
                        reason="High-risk dependency file contents required before editing.",
                    )
                    for path in blocked_paths
                ]
                collected = collect_snippets(
                    repo.root,
                    high_risk_requests,
                    static_findings=findings,
                )
                existing_keys = {
                    (snippet.path, snippet.start_line, snippet.end_line)
                    for snippet in active_snippets
                }
                added = 0
                for snippet in collected:
                    key = (snippet.path, snippet.start_line, snippet.end_line)
                    if key in existing_keys:
                        continue
                    active_snippets.append(snippet)
                    existing_keys.add(key)
                    added += 1
                if added > 0:
                    continue
            patch_result = self._apply_implement_response(
                repo=repo,
                response=response,
                structured_only=structured_only,
                expected_paths=plan.implement.touched_files,
                plan=plan,
                served_snippets=active_snippets,
            )
            refreshed_snippets = self._refresh_snippet_cache(repo, active_snippets)
            return response, patch_result, refreshed_snippets

    def _refresh_snippet_cache(
        self,
        repo: GitRepository,
        snippets: Sequence[Snippet],
    ) -> list[Snippet]:
        """Reload snippet contents from the repository to avoid stale context."""

        if not snippets:
            return []

        refreshed: list[Snippet] = []
        for snippet in snippets:
            refreshed.append(self._refresh_single_snippet(repo, snippet))
        return refreshed

    def _refresh_single_snippet(self, repo: GitRepository, snippet: Snippet) -> Snippet:
        path_value = getattr(snippet, "path", None)
        if not isinstance(path_value, str) or not path_value.strip():
            return snippet

        normalised_path = self._normalise_path(path_value)
        if not normalised_path:
            return snippet

        fs_path = (repo.root / normalised_path).resolve()
        try:
            fs_path.relative_to(repo.root)
        except ValueError:
            return snippet
        if not fs_path.exists():
            return snippet

        try:
            file_lines = fs_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return snippet
        if not file_lines:
            return snippet

        start = snippet.start_line if snippet.start_line and snippet.start_line > 0 else 1
        end = snippet.end_line if snippet.end_line and snippet.end_line > 0 else len(file_lines)
        if end < start:
            start, end = end, start
        total = len(file_lines)
        start = min(max(1, start), total)
        end = max(start, min(total, end))
        selected = file_lines[start - 1 : end]
        if not selected:
            # Fallback to the last available chunk when the original span vanished.
            window_start = max(1, total - 119)
            selected = file_lines[window_start - 1 :]
            start = window_start
            end = total

        return Snippet(
            path=normalised_path,
            start_line=start,
            end_line=end,
            content="\n".join(selected),
            reason=getattr(snippet, "reason", None),
        )

    def _format_diagnose_attempt(
        self,
        *,
        attempt_index: int,
        failing: Sequence[str],
        recommended: Sequence[str],
        patch_provided: bool,
        patch_applied: bool,
        outcome: str,
        error: str | None,
    ) -> str:
        """Render a concise summary of a diagnose attempt for LLM context."""

        parts: list[str] = [f"Attempt {attempt_index}"]
        failing_list = ", ".join(failing) if failing else "unknown tests"
        parts.append(f"failing tests: {failing_list}")
        if recommended:
            parts.append(f"recommended fixes: {', '.join(recommended)}")
        if not patch_provided:
            parts.append("patch: none provided")
        else:
            parts.append(f"patch: {'applied' if patch_applied else 'skipped'}")
        outcome_text = outcome.strip() or "Outcome not recorded."
        parts.append(f"outcome: {outcome_text}")
        if error:
            parts.append(f"detail: {error.strip()}")
        return " | ".join(parts)

    def _apply_implement_response(
        self,
        *,
        repo: GitRepository,
        response: ImplementResponse,
        structured_only: bool,
        expected_paths: Sequence[str],
        plan: CodingIterationPlan,
        served_snippets: Sequence[Snippet],
    ) -> PatchApplicationResult:
        """Apply the patch emitted by the implement phase with additional guardrails."""

        no_op_reason = (response.no_op_reason or "").strip()
        if no_op_reason:
            return PatchApplicationResult(
                attempted=False,
                applied=True,
                touched_paths=(),
                telemetry=None,
                no_op_reason=no_op_reason,
            )

        files = list(getattr(response, "files", []) or [])
        edits = list(getattr(response, "edits", []) or [])

        diff_text = ""
        if files or edits:
            try:
                diff_text = build_structured_patch(
                    repo_root=repo.root,
                    files=files,
                    edits=edits,
                )
            except PatchError as error:
                return PatchApplicationResult(
                    attempted=True,
                    applied=False,
                    error=str(error),
                    telemetry=dict(error.details) if isinstance(error.details, Mapping) else None,
                )
        elif response.diff:
            if structured_only and not files and not edits:
                return PatchApplicationResult(
                    attempted=False,
                    applied=False,
                    error="Structured edit mode does not accept raw diffs.",
                )
            try:
                diff_text = canonicalise_unified_diff(response.diff, repo_root=repo.root)
            except PatchError as error:
                return PatchApplicationResult(
                    attempted=True,
                    applied=False,
                    error=str(error),
                    telemetry=dict(error.details) if isinstance(error.details, Mapping) else None,
                )
        else:
            return PatchApplicationResult(
                attempted=False,
                applied=False,
                error="Implement response did not include any updates.",
            )

        if not diff_text.strip():
            return PatchApplicationResult(
                attempted=False,
                applied=False,
                error="Implement response did not include any updates.",
            )

        guard = self._detect_high_risk_blind_replacements(
            repo=repo,
            files=files,
            served_snippets=served_snippets,
        )
        if guard is not None:
            message, blocked_paths = guard
            telemetry = {
                "quality_guard": {
                    "reason": message,
                    "blocked_paths": blocked_paths,
                }
            }
            return PatchApplicationResult(
                attempted=False,
                applied=False,
                error=message,
                telemetry=telemetry,
            )

        touched = self._collect_touched_paths(files, edits, diff_text)

        if self._plan_promised_new_tests(plan):
            touched_tests = sorted(path for path in touched if self._looks_like_test_file(path))
            if not touched_tests:
                message = "Implement response omitted the tests promised in the plan."
                telemetry = {
                    "expected_tests": {
                        "reason": "plan_promised_new_tests",
                        "touched_paths": sorted(touched),
                    }
                }
                return PatchApplicationResult(
                    attempted=False,
                    applied=False,
                    error=message,
                    telemetry=telemetry,
                )

        guard_message = self._detect_test_stub(diff_text)
        if guard_message:
            telemetry = {"quality_guard": {"reason": guard_message}}
            return PatchApplicationResult(
                attempted=False,
                applied=False,
                error=guard_message,
                telemetry=telemetry,
            )

        return self._apply_patch_text(diff_text, repo)

    def _detect_high_risk_blind_replacements(
        self,
        *,
        repo: GitRepository,
        files: Sequence[Any],
        served_snippets: Sequence[Snippet],
    ) -> tuple[str, list[str]] | None:
        if not files:
            return None
        snippet_paths = {
            self._normalise_path(snippet.path)
            for snippet in served_snippets
            if isinstance(getattr(snippet, "path", None), str) and snippet.path.strip()
        }
        snippet_basenames = {Path(path).name.lower() for path in snippet_paths}

        blocked: set[str] = set()
        for artifact in files:
            path_candidate = getattr(artifact, "path", None)
            if not isinstance(path_candidate, str):
                continue
            normalised_path = self._normalise_path(path_candidate)
            if not normalised_path:
                continue
            basename = Path(normalised_path).name.lower()
            if basename not in _HIGH_RISK_EDIT_BASENAMES:
                continue
            fs_path = (repo.root / normalised_path).resolve()
            try:
                fs_path.relative_to(repo.root)
            except ValueError:
                continue
            if not fs_path.exists():
                continue
            if normalised_path in snippet_paths or basename in snippet_basenames:
                continue
            blocked.add(normalised_path)

        if not blocked:
            return None

        blocked_list = sorted(blocked)
        quoted = [f"'{path}'" for path in blocked_list]
        if len(quoted) == 1:
            message = (
                f"Refusing to replace high-risk file {quoted[0]} without first reading the current contents."
            )
        else:
            message = (
                "Refusing to replace high-risk files "
                + ", ".join(quoted[:-1])
                + f" and {quoted[-1]} without first reading their current contents."
            )
        return message, blocked_list

    def _plan_promised_new_tests(self, plan: CodingIterationPlan) -> bool:
        candidates: list[str] = []
        diff_goal = getattr(plan.implement, "diff_goal", None)
        if isinstance(diff_goal, str) and diff_goal.strip():
            candidates.append(diff_goal)
        deliverables = getattr(plan.design, "proposed_interfaces", None)
        if isinstance(deliverables, Sequence):
            for entry in deliverables:
                if isinstance(entry, str) and entry.strip():
                    candidates.append(entry)
        for entry in getattr(plan.implement, "notes", []) or []:
            if isinstance(entry, str) and entry.strip():
                candidates.append(entry)
        return any(self._text_promises_new_tests(text) for text in candidates)

    @staticmethod
    def _text_promises_new_tests(text: str) -> bool:
        lowered = text.lower()
        if "test" not in lowered:
            return False
        normalized = lowered.strip().lstrip("-•* ").strip()
        triggers = (
            "add test",
            "add tests",
            "write test",
            "write tests",
            "new test",
            "new tests",
            "create test",
            "create tests",
            "introduce test",
            "introduce tests",
            "include test",
            "include tests",
            "test coverage",
            "tests:",
        )
        return any(trigger in normalized for trigger in triggers)

    def _apply_patch_text(self, diff: str, repo: GitRepository) -> PatchApplicationResult:
        """Apply a unified diff to the repository."""

        if not diff or not diff.strip():
            return PatchApplicationResult(
                attempted=False,
                applied=False,
                error="Patch payload was empty.",
            )

        hygiene = ensure_workspace_hygiene(repo)
        hygiene_payload: dict[str, Any] | None = None
        if not hygiene.ok:
            errors = [str(item) for item in hygiene.errors if str(item)]
            message = "; ".join(errors) if errors else "Workspace hygiene failed."
            telemetry = {"hygiene": {"errors": errors or [message]}}
            return PatchApplicationResult(
                attempted=False,
                applied=False,
                error=message,
                telemetry=telemetry,
            )
        if hygiene.removed or hygiene.warnings:
            hygiene_payload = {
                "removed": sorted(path.as_posix() for path in hygiene.removed),
            }
            warnings = [str(item) for item in hygiene.warnings if str(item)]
            if warnings:
                hygiene_payload["warnings"] = warnings

        try:
            patch_result = apply_patch(diff, repo_root=repo.root)
        except PatchError as error:
            telemetry: dict[str, Any] | None = None
            if isinstance(error.details, Mapping):
                telemetry = dict(error.details)
            return PatchApplicationResult(
                attempted=True,
                applied=False,
                error=str(error),
                telemetry=telemetry,
            )

        telemetry_payload: dict[str, Any] | None = None
        if patch_result.telemetry is not None:
            if isinstance(patch_result.telemetry, PatchTelemetry):
                telemetry_payload = patch_result.telemetry.to_dict()
            elif isinstance(patch_result.telemetry, Mapping):
                telemetry_payload = dict(patch_result.telemetry)
        if hygiene_payload:
            if telemetry_payload is None:
                telemetry_payload = {}
            hygiene_block = telemetry_payload.setdefault("hygiene", {})
            for key, value in hygiene_payload.items():
                if value:
                    hygiene_block[key] = value
        result_object = PatchApplicationResult(
            attempted=True,
            applied=True,
            touched_paths=patch_result.touched_paths,
            telemetry=telemetry_payload,
        )
        if result_object.applied:
            self._context_builder.invalidate_repository_cache()
        return result_object

    def _dependency_file_touched(self, repo: GitRepository, touched_paths: Iterable[Path]) -> bool:
        for path in touched_paths or []:
            candidate = Path(path)
            if candidate.is_absolute():
                try:
                    candidate = candidate.relative_to(repo.root)
                except ValueError:
                    continue
            candidate = Path(candidate.as_posix())
            basename = candidate.name.lower()
            suffix = candidate.suffix.lower()
            if basename in _DEPENDENCY_FILENAMES:
                return True
            if basename.startswith("requirements") and suffix in _DEPENDENCY_SUFFIXES:
                return True
            if basename.startswith("constraints") and suffix in _DEPENDENCY_SUFFIXES:
                return True
            if any(part.lower() == "requirements" for part in candidate.parts[:-1]) and suffix in _DEPENDENCY_SUFFIXES:
                return True
        return False

    def _ensure_requirements_for_changes(
        self,
        repo: GitRepository,
        touched_paths: Iterable[Path],
    ) -> tuple[bool, str | None]:
        if not self._dependency_file_touched(repo, touched_paths):
            return True, None
        success, error = ensure_requirements_installed(repo.root)
        if success:
            return True, None
        self._revert_touched_paths(repo, touched_paths)
        message = error or "Failed to install repository requirements."
        return False, message

    def _pytest_dependency_failure(self, pytest_result: PytestResult | None) -> str | None:
        if pytest_result is None:
            return None
        if getattr(pytest_result, "exit_code", None) != 2:
            return None
        stderr = (pytest_result.stderr or "").strip()
        stdout = (pytest_result.stdout or "").strip()
        payload = stderr or stdout
        lowered = payload.lower()
        if not lowered:
            return None
        if "pip install -r" in lowered or "failed to install repository requirements" in lowered:
            return payload
        return None

    def _revert_touched_paths(self, repo: GitRepository, touched_paths: Iterable[Path]) -> None:
        relative_paths: list[Path] = []
        for path in touched_paths or []:
            candidate = Path(path)
            if candidate.is_absolute():
                try:
                    candidate = candidate.relative_to(repo.root)
                except ValueError:
                    continue
            relative_paths.append(candidate)

        if not relative_paths:
            return

        tracked: list[str] = []
        untracked: list[Path] = []
        for rel in relative_paths:
            rel_posix = rel.as_posix()
            probe = repo.git("ls-files", "--error-unmatch", rel_posix, check=False)
            if probe.returncode == 0:
                tracked.append(rel_posix)
            else:
                untracked.append(rel)

        if tracked:
            try:
                repo.git("restore", "--worktree", "--staged", "--", *tracked)
            except GitError:
                pass

        for rel in untracked:
            target = (repo.root / rel).resolve()
            try:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                continue

    def _collect_touched_paths(
        self,
        files: Sequence[Any],
        edits: Sequence[Any],
        diff_text: str,
    ) -> set[str]:
        paths: set[str] = set()
        for artifact in files or []:
            candidate = getattr(artifact, "path", "")
            if candidate:
                paths.add(self._normalise_path(candidate))
        for edit in edits or []:
            candidate = getattr(edit, "path", "")
            if candidate:
                paths.add(self._normalise_path(candidate))
        if diff_text:
            paths.update(self._paths_from_diff(diff_text))
        return paths

    @staticmethod
    def _normalise_path(value: str) -> str:
        return value.replace("\\", "/").lstrip("./")

    @staticmethod
    def _strip_helper_suffix(value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            return ""
        while True:
            match = re.search(r"\s+\([^()]*\)$", cleaned)
            if not match:
                break
            cleaned = cleaned[: match.start()].rstrip()
        return cleaned

    @staticmethod
    def _looks_like_file_candidate(value: str) -> bool:
        normalised = value.replace("\\", "/")
        if "/" in normalised:
            return True
        lowered = normalised.lower()
        return any(lowered.endswith(ext) for ext in _EXPECTED_FILE_EXTENSIONS)

    def _extract_paths_from_text(self, value: str) -> list[str]:
        cleaned = self._strip_helper_suffix(value)
        if not cleaned:
            return []

        results: list[str] = []
        seen: set[str] = set()

        direct = self._normalise_path_candidate(cleaned)
        if direct and " " not in direct and self._looks_like_file_candidate(direct):
            seen.add(direct)
            results.append(direct)

        for fragment in _PATH_TOKEN_SPLIT_RE.split(cleaned):
            candidate = self._normalise_path_candidate(fragment)
            if not candidate or candidate in seen or " " in candidate:
                continue
            if self._looks_like_file_candidate(candidate):
                seen.add(candidate)
                results.append(candidate)
        return [path for path in results if not self._is_bare_package_init(path)]

    @staticmethod
    def _is_bare_package_init(path: str) -> bool:
        candidate = path.replace("\\", "/")
        return candidate.endswith("__init__.py") and "/" not in candidate

    @staticmethod
    def _normalise_path_candidate(value: str) -> str:
        if not isinstance(value, str):
            return ""
        candidate = value.strip().strip("\"'`“”‘’")
        if not candidate:
            return ""
        candidate = candidate.replace("\\", "/")
        candidate = candidate.lstrip("([{<")
        candidate = candidate.rstrip(".,;:!?)]}>")
        while candidate.startswith("./"):
            candidate = candidate[2:]
        if not candidate:
            return ""
        return candidate

    def _paths_from_diff(self, diff_text: str) -> set[str]:
        paths: set[str] = set()
        if not diff_text:
            return paths
        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 4:
                    for candidate in parts[2:4]:
                        if candidate.startswith(("a/", "b/")):
                            candidate = candidate[2:]
                        paths.add(self._normalise_path(candidate))
            elif line.startswith("+++ "):
                candidate = line[4:].strip()
                if candidate == "/dev/null":
                    continue
                if candidate.startswith("b/"):
                    candidate = candidate[2:]
                paths.add(self._normalise_path(candidate))
        return paths

    def _detect_test_stub(self, diff_text: str) -> str | None:
        if not diff_text.strip():
            return None
        stats = self._extract_diff_file_stats(diff_text)
        for path, info in stats.items():
            normalised = self._normalise_path(path)
            if not self._looks_like_test_file(normalised):
                continue
            if info.get("deleted"):
                return f"Refusing to delete test file '{normalised}' without explicit justification."
            added_lines = [line.strip() for line in info.get("added", []) if line.strip()]
            removed_count = info.get("removed", 0)
            if not added_lines:
                continue
            placeholder_lines = sum(1 for line in added_lines if self._is_placeholder_test_line(line))
            definition_lines = sum(1 for line in added_lines if line.startswith("def test"))
            effective_lines = len(added_lines) - definition_lines
            if removed_count >= 3 and effective_lines > 0 and placeholder_lines >= effective_lines:
                return f"Refusing to replace tests in '{normalised}' with placeholders."
        return None

    def _extract_diff_file_stats(self, diff_text: str) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        current_path: str | None = None
        previous_path: str | None = None

        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                current_path = None
                previous_path = None
                continue
            if line.startswith("--- "):
                candidate = line[4:].strip()
                if candidate == "/dev/null":
                    previous_path = None
                    continue
                if candidate.startswith("a/"):
                    candidate = candidate[2:]
                previous_path = candidate
                stats.setdefault(candidate, {"added": [], "removed": 0, "deleted": False})
                continue
            if line.startswith("+++ "):
                candidate = line[4:].strip()
                if candidate == "/dev/null":
                    if previous_path is not None:
                        entry = stats.setdefault(previous_path, {"added": [], "removed": 0, "deleted": False})
                        entry["deleted"] = True
                    current_path = None
                    continue
                if candidate.startswith("b/"):
                    candidate = candidate[2:]
                current_path = candidate
                stats.setdefault(candidate, {"added": [], "removed": 0, "deleted": False})
                continue
            if current_path is None:
                continue
            if line.startswith("@@"):
                continue
            if line.startswith("+") and not line.startswith("+++"):
                entry = stats.setdefault(current_path, {"added": [], "removed": 0, "deleted": False})
                entry["added"].append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                entry = stats.setdefault(current_path, {"added": [], "removed": 0, "deleted": False})
                entry["removed"] += 1
        return stats

    @staticmethod
    def _looks_like_test_file(path: str) -> bool:
        normalised = path.lower()
        name = Path(normalised).name
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        return normalised.startswith("tests/") or "/tests/" in normalised

    @staticmethod
    def _is_placeholder_test_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if stripped in _PLACEHOLDER_TEST_LINES:
            return True
        if stripped.startswith("assert "):
            expression = stripped[len("assert ") :].strip()
            return expression in _PLACEHOLDER_ASSERTIONS
        return False

    def _apply_structured_updates(
        self,
        *,
        repo: GitRepository,
        files: Sequence[Any],
        edits: Sequence[Any],
        patch: str = "",
    ) -> PatchApplicationResult:
        """Apply structured updates or fallback diff payload."""

        diff_text = ""
        structured_files = list(files or [])
        structured_edits = list(edits or [])
        if structured_files or structured_edits:
            try:
                diff_text = build_structured_patch(
                    repo_root=repo.root,
                    files=structured_files,
                    edits=structured_edits,
                )
            except PatchError as error:
                telemetry: dict[str, Any] | None = None
                if isinstance(error.details, Mapping):
                    telemetry = dict(error.details)
                return PatchApplicationResult(
                    attempted=True,
                    applied=False,
                    error=str(error),
                    telemetry=telemetry,
                )
        elif patch:
            try:
                diff_text = canonicalise_unified_diff(patch, repo_root=repo.root)
            except PatchError as error:
                telemetry: dict[str, Any] | None = None
                if isinstance(error.details, Mapping):
                    telemetry = dict(error.details)
                return PatchApplicationResult(
                    attempted=True,
                    applied=False,
                    error=str(error),
                    telemetry=telemetry,
                )
        else:
            return PatchApplicationResult(
                attempted=False,
                applied=False,
                error="No updates were provided.",
            )

        if not diff_text.strip():
            return PatchApplicationResult(
                attempted=False,
                applied=False,
                error="No updates were provided.",
            )

        return self._apply_patch_text(diff_text, repo)

    def _resolve_config_path_hint(self, explicit: Path | str | None) -> Path | None:
        if explicit is None:
            return None
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    def _normalise_repo_root(self, repo_root: Path | str | None, config_hint: Path | None) -> Path:
        base = config_hint.parent if config_hint is not None else Path.cwd()
        if repo_root is not None:
            path = Path(repo_root)
            if not path.is_absolute():
                path = (base / path).resolve()
            return path.resolve()
        return ContextBuilder._infer_repo_root(self._config, base_path=base).resolve()

    def _determine_config_path(self, explicit: Path | None) -> Path:
        if explicit is not None:
            return explicit
        candidate: Path | None = None
        paths_section = self._config.get("paths")
        if isinstance(paths_section, Mapping):
            config_entry = paths_section.get("config")
            if isinstance(config_entry, str) and config_entry:
                candidate = Path(config_entry)
        if candidate is None:
            candidate = Path("config.yaml")
        if not candidate.is_absolute():
            candidate = (self._repo_root / candidate).resolve()
        return candidate

    def _resolve_repo(self, override: Path | str | None) -> GitRepository:
        if override is not None:
            candidate = Path(override)
            if not candidate.is_absolute():
                candidate = (self._repo_root / candidate).resolve()
        else:
            candidate = self._repo_root
        try:
            return GitRepository(candidate)
        except GitError:
            try:
                return GitRepository.discover(candidate)
            except GitError:
                return GitRepository.initialise(candidate)

    def _prepare_config_path(self, override: Path | str | None, repo: GitRepository) -> Path:
        if override is None:
            return self._config_path
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = (repo.root / candidate).resolve()
        return candidate

    def _ensure_gates_clean(
        self,
        *,
        repo: GitRepository,
        config_path: Path,
        plan: CodingIterationPlan,
        plan_id: str,
        task_id: str,
        result: CodingIterationResult,
        changed_paths: set[Path],
        enable_autofix: bool,
        applied_adjustment_hashes: set[str],
    ) -> tuple[GateRunSummary, list[CycleAdjustment], bool, list[StaticFinding]]:
        summary = self._run_gates(repo, config_path)
        adjustments: list[CycleAdjustment] = []
        needs_implement = False
        static_findings = list(summary.static_findings)
        dependency_error: str | None = None

        if enable_autofix and not summary.ok:
            attempts = 0
            while not summary.ok and attempts < self._MAX_GATE_FIX_ATTEMPTS:
                violations = summary.violations or self._summarize_gate_failures(summary.report)
                if not violations:
                    violations = [summary.error] if summary.error else ["Static gates failed."]
                fix_request = FixViolationsRequest(
                    task_id=plan.implement.task_id,
                    violations=list(violations),
                    current_diff=repo.diff(),
                    attempts=attempts,
                    suspect_files=list(summary.suspect_files),
                    static_findings=list(summary.static_findings),
                )
                fix_response = self.run_phase(PhaseName.FIX_VIOLATIONS, fix_request)
                result.fix_violations = fix_response
                files = list(getattr(fix_response, "files", []) or [])
                edits = list(getattr(fix_response, "edits", []) or [])
                patch_text = getattr(fix_response, "patch", "") or ""
                if not files and not edits and not patch_text.strip():
                    break
                apply_result = self._apply_structured_updates(
                    repo=repo,
                    files=files,
                    edits=edits,
                    patch=patch_text,
                )
                if not apply_result.applied:
                    detail = (apply_result.error or "").strip()
                    message = "Failed to apply gate fix patch."
                    if detail:
                        message = f"{message} {detail}"
                    self._record_error(result, message)
                    summary.error = message
                    break
                install_ok, install_error = self._ensure_requirements_for_changes(
                    repo=repo,
                    touched_paths=apply_result.touched_paths,
                )
                if not install_ok:
                    message = install_error or "Failed to install repository requirements."
                    apply_result.applied = False
                    apply_result.error = message
                    apply_result.touched_paths = ()
                    self._record_error(result, message)
                    summary.ok = False
                    summary.error = message
                    dependency_error = message
                    break
                changed_paths.update(apply_result.touched_paths)
                adjustments.append(
                    CycleAdjustment(
                        source="fix_violations",
                        description=fix_response.reason or "Applied gate fix.",
                        touched_paths=apply_result.touched_paths,
                        priority=None,
                    )
                )
                summary = self._run_gates(repo, config_path)
                static_findings = list(summary.static_findings)
                attempts += 1
                # Avoid repeatedly invoking fix_violations in the same cycle; follow-up
                # adjustments should be handled by plan_adjust instead.
                break

        if dependency_error is not None:
            return summary, adjustments, needs_implement, static_findings

        if not summary.ok and enable_autofix:
            plan_adjustments, reenter = self._apply_plan_adjustments(
                repo=repo,
                plan=plan,
                plan_id=plan_id,
                task_id=task_id,
                result=result,
                summary=summary,
                changed_paths=changed_paths,
                applied_adjustment_hashes=applied_adjustment_hashes,
            )
            if plan_adjustments:
                adjustments.extend(plan_adjustments)
                needs_implement = needs_implement or reenter

        if not summary.ok:
            violations = summary.violations or self._summarize_gate_failures(summary.report)
            summary.violations = list(violations or [])
            if "Static gates failed." not in result.errors:
                result.errors.append("Static gates failed.")

        return summary, adjustments, needs_implement, static_findings

    _ensure_gates_clean._ae_native = True  # type: ignore[attr-defined]

    def _apply_plan_adjustments(
        self,
        *,
        repo: GitRepository,
        plan: CodingIterationPlan,
        plan_id: str,
        task_id: str,
        result: CodingIterationResult,
        summary: GateRunSummary,
        changed_paths: set[Path],
        applied_adjustment_hashes: set[str],
    ) -> tuple[list[CycleAdjustment], bool]:
        """Invoke plan_adjust to apply high-priority structured patches."""

        from .phases.plan_adjust import PlanAdjustRequest
        try:
            from .planning.executor import apply_adjustments_and_reenter
        except ImportError:  # pragma: no cover - defensive
            return [], False

        violations = summary.violations or self._summarize_gate_failures(summary.report)
        suspect_files = list(summary.suspect_files)
        if not suspect_files:
            suspect_files = sorted({path.as_posix() for path in changed_paths})

        reason = "Static gates failed."
        if summary.error:
            reason = summary.error
        elif violations:
            reason = violations[0]

        suggested: list[str] = []
        if result.fix_violations:
            suggested.extend(result.fix_violations.rationale)
        if result.diagnose:
            suggested.extend(result.diagnose.recommended_fixes)
            lessons = getattr(result.diagnose, "iteration_lessons", []) or []
            for entry in lessons:
                text = str(entry).strip()
                if text:
                    suggested.append(text)
        if not suggested and result.errors:
            suggested.extend(result.errors)

        request = PlanAdjustRequest(
            plan_id=plan_id,
            task_id=task_id,
            reason=reason,
            suggested_changes=suggested,
            blockers=list(violations),
            suspect_files=suspect_files,
        )

        try:
            response = self.run_phase(PhaseName.PLAN_ADJUST, request)
        except Exception as error:  # pragma: no cover - defensive
            self._record_error(result, f"Plan-adjust failed: {error}")
            return [], False

        result.plan_adjustments.append(response)
        dependency_error: str | None = None

        def _apply_updates(files: Sequence[Any], edits: Sequence[Any]) -> PatchApplicationResult:
            nonlocal dependency_error
            patch_result = self._apply_structured_updates(
                repo=repo,
                files=files,
                edits=edits,
            )
            if patch_result.applied:
                install_ok, install_error = self._ensure_requirements_for_changes(
                    repo=repo,
                    touched_paths=patch_result.touched_paths,
                )
                if not install_ok:
                    message = install_error or "Failed to install repository requirements."
                    self._record_error(result, message)
                    dependency_error = message
                    patch_result.applied = False
                    patch_result.error = message
                    patch_result.touched_paths = ()
                    return patch_result
                changed_paths.update(patch_result.touched_paths)
            return patch_result

        applied = apply_adjustments_and_reenter(
            adjustment=response,
            apply_updates=_apply_updates,
            applied_hashes=applied_adjustment_hashes,
        )
        if dependency_error is not None:
            return [], False

        adjustments: list[CycleAdjustment] = []
        for entry in applied:
            touched_paths = tuple(Path(path) if not isinstance(path, Path) else path for path in entry.touched_paths)
            install_ok, install_error = self._ensure_requirements_for_changes(
                repo=repo,
                touched_paths=touched_paths,
            )
            if not install_ok:
                message = install_error or "Failed to install repository requirements."
                self._record_error(result, message)
                dependency_error = message
                break
            changed_paths.update(touched_paths)
            adjustments.append(
                CycleAdjustment(
                    source=entry.source or "plan_adjust",
                    description=entry.description,
                    priority=entry.priority,
                    touched_paths=touched_paths,
                )
            )
        if dependency_error is not None:
            return adjustments, False
        return adjustments, bool(applied)

    def _run_tests_with_recovery(
        self,
        *,
        plan: CodingIterationPlan,
        plan_id: str,
        task_id: str,
        repo: GitRepository,
        config_path: Path,
        checkpoint: GitCheckpoint,
        changed_paths: set[Path],
        result: CodingIterationResult,
        implement: ImplementResponse | None,
        enable_autofix: bool,
        applied_adjustment_hashes: set[str],
        static_findings: Sequence[StaticFinding],
    ) -> tuple[
        PytestResult | None,
        DiagnoseResponse | None,
        list[CycleAdjustment],
        bool,
        list[StaticFinding],
        bool,
    ]:
        pytest_commands = self._collect_pytest_commands(plan, implement)
        pytest_commands = self._prune_missing_pytest_targets(pytest_commands, repo.root)
        if not pytest_commands:
            return None, None, [], True, list(static_findings), False

        tests_result: PytestResult | None = None
        tests_ok = True
        diagnose_response: DiagnoseResponse | None = None
        adjustments: list[CycleAdjustment] = []
        attempts = 0
        findings = list(static_findings)
        dependency_error: str | None = None
        iteration_guidance = [
            "If fixing the failures requires discarding most workspace edits, set restart_iteration to true to reset the workspace to the clean clone.",
            "When requesting a restart, provide restart_summary with the key insight and populate iteration_lessons so the next iteration can learn from this attempt.",
            "If the failure stems from incorrect or overly strict tests, propose changes to the tests themselves instead of forcing unnecessary product code edits.",
        ]

        for pytest_args in pytest_commands:
            tests_result = run_pytest(pytest_args, cwd=repo.root)
            tests_ok = tests_result.ok if tests_result is not None else True
            dependency_error = self._pytest_dependency_failure(tests_result)
            if dependency_error:
                self._record_error(result, dependency_error)
                return tests_result, None, adjustments, False, findings, False
            if tests_result is None or tests_result.ok:
                continue
            break
        else:
            return tests_result, None, adjustments, tests_ok, findings, False

        diagnose_snippets = self._collect_recent_change_snippets(repo.root, changed_paths)
        failing_test_snippets = self._collect_failing_test_snippets(repo.root, tests_result)
        if failing_test_snippets:
            diagnose_snippets.extend(failing_test_snippets)
        diagnose_snippets = self._refresh_snippet_cache(repo, diagnose_snippets)
        diagnose_snippet_keys: set[tuple[str, int, int]] = {
            (self._normalise_path(snippet.path), snippet.start_line, snippet.end_line)
            for snippet in diagnose_snippets
        }
        diagnose_seen_requests = set(diagnose_snippet_keys)
        diagnose_history: list[str] = list(result.diagnose_attempts)

        def _record_diagnose_attempt(
            *,
            attempt_index: int,
            failing: Sequence[str],
            recommended: Sequence[str],
            patch_provided: bool,
            patch_applied: bool,
            outcome: str,
            error: str | None,
        ) -> None:
            entry = self._format_diagnose_attempt(
                attempt_index=attempt_index,
                failing=failing,
                recommended=recommended,
                patch_provided=patch_provided,
                patch_applied=patch_applied,
                outcome=outcome,
                error=error,
            )
            diagnose_history.append(entry)
            result.diagnose_attempts = list(diagnose_history)

        while tests_result is not None and not tests_result.ok and attempts < self._MAX_DIAGNOSE_ATTEMPTS:
            logs = "\n".join(part for part in (tests_result.stderr, tests_result.stdout) if part)
            failing_tests = self._extract_failing_tests(tests_result)
            recent_changes = self._format_recent_changes(changed_paths, repo.root)

            while True:
                diagnose_snippets = self._refresh_snippet_cache(repo, diagnose_snippets)
                diagnose_request = DiagnoseRequest(
                    task_id=plan.implement.task_id,
                    failing_tests=failing_tests,
                    logs=logs,
                    recent_changes=recent_changes,
                    snippets=list(diagnose_snippets),
                    attempt_history=list(diagnose_history),
                    iteration_guidance=list(iteration_guidance),
                )
                diagnose_response = self.run_phase(PhaseName.DIAGNOSE, diagnose_request)
                result.diagnose = diagnose_response

                code_requests = [
                    entry
                    for entry in getattr(diagnose_response, "code_requests", []) or []
                    if isinstance(entry, SnippetRequest)
                ]
                new_requests: list[SnippetRequest] = []
                for entry in code_requests:
                    raw_path = entry.path if isinstance(entry.path, str) else ""
                    normalised_path = self._normalise_path(raw_path.strip()) if raw_path else ""
                    path_key = (normalised_path, entry.start_line, entry.end_line)
                    if not normalised_path:
                        continue
                    if path_key in diagnose_seen_requests:
                        continue
                    diagnose_seen_requests.add(path_key)
                    new_requests.append(
                        SnippetRequest(
                            path=normalised_path,
                            start_line=entry.start_line,
                            end_line=entry.end_line,
                            surround=entry.surround,
                            reason=entry.reason,
                        )
                    )
                if new_requests:
                    collected = collect_snippets(repo.root, new_requests)
                    added = 0
                    for snippet in collected:
                        key = (self._normalise_path(snippet.path), snippet.start_line, snippet.end_line)
                        if key in diagnose_snippet_keys:
                            continue
                        diagnose_snippets.append(snippet)
                        diagnose_snippet_keys.add(key)
                        added += 1
                    if added > 0:
                        continue
                break

            attempt_index = attempts + 1
            recommended_fixes = list(getattr(diagnose_response, "recommended_fixes", []) or [])
            findings = list(getattr(diagnose_response, "static_findings", findings))
            patch_applied = False
            attempt_error: str | None = None
            attempt_outcome = ""

            if getattr(diagnose_response, "restart_iteration", False):
                restart_summary = (
                    getattr(diagnose_response, "restart_summary", "") or "Diagnose requested workspace reset."
                ).strip()
                if not restart_summary:
                    restart_summary = "Diagnose requested workspace reset."
                try:
                    checkpoint.rollback()
                except GitError as error:
                    message = f"Workspace reset failed: {error}"
                    self._record_error(result, message)
                    attempt_outcome = "Workspace reset failed."
                    attempt_error = message
                    _record_diagnose_attempt(
                        attempt_index=attempt_index,
                        failing=failing_tests,
                        recommended=recommended_fixes,
                        patch_provided=False,
                        patch_applied=False,
                        outcome=attempt_outcome,
                        error=attempt_error,
                    )
                    tests_ok = False
                    break
                changed_paths.clear()
                applied_adjustment_hashes.clear()
                findings = []
                adjustments.append(
                    CycleAdjustment(
                        source="diagnose",
                        description=restart_summary,
                        priority="high",
                        touched_paths=(),
                    )
                )
                _record_diagnose_attempt(
                    attempt_index=attempt_index,
                    failing=failing_tests,
                    recommended=recommended_fixes,
                    patch_provided=False,
                    patch_applied=False,
                    outcome=restart_summary,
                    error=None,
                )
                return tests_result, diagnose_response, adjustments, False, findings, True

            files = list(getattr(diagnose_response, "files", []) or [])
            edits = list(getattr(diagnose_response, "edits", []) or [])
            patch_text = getattr(diagnose_response, "patch", "") or ""
            patch_provided = bool(files or edits or patch_text.strip())
            if not files and not edits and not patch_text.strip():
                attempt_outcome = "Diagnose response did not include any updates."
                _record_diagnose_attempt(
                    attempt_index=attempt_index,
                    failing=failing_tests,
                    recommended=recommended_fixes,
                    patch_provided=False,
                    patch_applied=False,
                    outcome=attempt_outcome,
                    error=None,
                )
                break

            apply_result = self._apply_structured_updates(
                repo=repo,
                files=files,
                edits=edits,
                patch=patch_text,
            )
            if not apply_result.applied:
                failure_message = apply_result.error or "Failed to apply diagnose patch."
                self._record_error(result, failure_message)
                tests_ok = False
                attempt_error = failure_message
                attempt_outcome = "Patch application failed."
                _record_diagnose_attempt(
                    attempt_index=attempt_index,
                    failing=failing_tests,
                    recommended=recommended_fixes,
                    patch_provided=patch_provided,
                    patch_applied=False,
                    outcome=attempt_outcome,
                    error=attempt_error,
                )
                break
            patch_applied = True
            install_ok, install_error = self._ensure_requirements_for_changes(
                repo=repo,
                touched_paths=apply_result.touched_paths,
            )
            if not install_ok:
                message = install_error or "Failed to install repository requirements."
                apply_result.applied = False
                apply_result.error = message
                apply_result.touched_paths = ()
                self._record_error(result, message)
                tests_ok = False
                attempt_outcome = "Dependency installation failed after applying diagnose patch."
                attempt_error = message
                _record_diagnose_attempt(
                    attempt_index=attempt_index,
                    failing=failing_tests,
                    recommended=recommended_fixes,
                    patch_provided=patch_provided,
                    patch_applied=True,
                    outcome=attempt_outcome,
                    error=attempt_error,
                )
                break
            changed_paths.update(apply_result.touched_paths)
            touched_normalised: set[str] = set()
            for path in apply_result.touched_paths:
                if isinstance(path, Path):
                    candidate = path
                else:
                    candidate = Path(str(path))
                normalised = self._normalise_path(candidate.as_posix())
                touched_normalised.add(normalised)
            if touched_normalised:
                diagnose_snippets = [
                    snippet for snippet in diagnose_snippets if self._normalise_path(snippet.path) not in touched_normalised
                ]
                diagnose_snippet_keys = {
                    key for key in diagnose_snippet_keys if key[0] not in touched_normalised
                }
                diagnose_seen_requests = {
                    key for key in diagnose_seen_requests if key[0] not in touched_normalised
                }
                adjustments.append(
                    CycleAdjustment(
                        source="diagnose",
                        description=(diagnose_response.recommended_fixes[0] if diagnose_response.recommended_fixes else "Applied diagnose patch."),
                        priority=None,
                    touched_paths=apply_result.touched_paths,
                )
            )

            gate_summary, gate_adjustments, needs_reimplement, findings = self._ensure_gates_clean(
                repo=repo,
                config_path=config_path,
                plan=plan,
                plan_id=plan_id,
                task_id=task_id,
                result=result,
                changed_paths=changed_paths,
                enable_autofix=enable_autofix,
                applied_adjustment_hashes=applied_adjustment_hashes,
            )
            result.gates = gate_summary
            if gate_adjustments:
                adjustments.extend(gate_adjustments)
            if not gate_summary.ok or needs_reimplement:
                tests_ok = False
                tests_result = None
                attempt_outcome = "Static gates reported violations after applying diagnose patch."
                attempt_error = self._infer_gate_reason(gate_summary) if not gate_summary.ok else "Plan adjustments required."
                _record_diagnose_attempt(
                    attempt_index=attempt_index,
                    failing=failing_tests,
                    recommended=recommended_fixes,
                    patch_provided=patch_provided,
                    patch_applied=patch_applied,
                    outcome=attempt_outcome,
                    error=attempt_error,
                )
                break

            tests_result = run_pytest(pytest_args, cwd=repo.root)
            tests_ok = tests_result.ok if tests_result is not None else tests_ok
            if tests_result is None:
                attempt_outcome = "Pytest was not re-run after diagnose patch."
                attempt_error = "Pytest invocation returned no result."
            elif tests_result.ok:
                attempt_outcome = "Pytest passed after applying diagnose patch."
            else:
                failing_after_patch = self._extract_failing_tests(tests_result)
                attempt_outcome = "Pytest still failing after diagnose patch."
                attempt_error = ", ".join(failing_after_patch)
            _record_diagnose_attempt(
                attempt_index=attempt_index,
                failing=failing_tests,
                recommended=recommended_fixes,
                patch_provided=patch_provided,
                patch_applied=patch_applied,
                outcome=attempt_outcome,
                error=attempt_error,
            )
            attempts += 1

        if tests_result is None:
            tests_ok = tests_ok and diagnose_response is None
        return tests_result, diagnose_response, adjustments, tests_ok, findings, False

    def _record_error(self, result: CodingIterationResult, message: str) -> None:
        text = (message or "").strip()
        if not text:
            return
        if text not in result.errors:
            result.errors.append(text)

    def _make_cycle(
        self,
        cycle_index: int,
        reason: str,
        gate_summary: GateRunSummary,
        tests_result: PytestResult | None,
        adjustments: Sequence[CycleAdjustment],
        errors: Sequence[str],
    ) -> IterationCycle:
        return IterationCycle(
            cycle_index=cycle_index,
            reason=reason,
            gates_ok=gate_summary.ok,
            tests_ok=tests_result.ok if tests_result is not None else None,
            adjustments=list(adjustments),
            errors=list(errors),
        )

    def _infer_gate_reason(self, summary: GateRunSummary) -> str:
        if summary.error:
            return summary.error
        if summary.report:
            for static_result in summary.report.static_results:
                if static_result.failed:
                    return f"{static_result.name} error"
            if summary.report.policy_violations:
                violation = summary.report.policy_violations[0]
                rule = violation.rule or "policy violation"
                return f"{rule} violation"
        if summary.violations:
            entry = summary.violations[0]
            rule = entry.split("::", 1)[0].strip()
            if rule:
                return f"{rule} failure"
        return "static gates failed"

    def _infer_test_reason(self, pytest_result: PytestResult) -> str:
        status = getattr(pytest_result, "status", "")
        if status == "failed":
            return "pytest failures"
        if status == "error":
            return "pytest error"
        if status == "no-tests":
            return "no tests collected"
        return "tests did not pass"

    def _run_gates(self, repo: GitRepository, config_path: Path) -> GateRunSummary:
        try:
            report = run_policy_and_static(config_path, repo_root=repo.root)
        except Exception as error:  # pragma: no cover - defensive surface
            return GateRunSummary(ok=False, error=str(error))
        summary = GateRunSummary(ok=not report.has_failures, report=report)
        if report.has_failures:
            summary.violations = self._summarize_gate_failures(report)
        suspect_files: set[str] = set()
        findings: list[StaticFinding] = []
        seen_keys: set[tuple[str, int, str]] = set()
        for violation in report.policy_violations:
            if violation.path:
                path_key = violation.path.as_posix()
                suspect_files.add(path_key)
                if violation.line is not None:
                    candidate = StaticFinding(
                        path=path_key,
                        line_start=violation.line,
                        line_end=violation.line,
                        message=violation.message,
                    )
                    key = (candidate.path, candidate.line_start, candidate.message)
                    if key not in seen_keys:
                        findings.append(candidate)
                        seen_keys.add(key)
        static_results = self._collect_static_findings(report, repo.root)
        for finding in static_results:
            suspect_files.add(finding.path)
            key = (finding.path, finding.line_start, finding.message)
            if key in seen_keys:
                continue
            findings.append(finding)
            seen_keys.add(key)
        summary.suspect_files = sorted(suspect_files)
        summary.static_findings = findings
        return summary

    def _seed_static_code_requests(
        self,
        implement: ImplementRequest,
        findings: Sequence[StaticFinding],
        *,
        context_lines: int = 10,
    ) -> None:
        normalized = normalize_static_findings(findings)
        if not normalized:
            return

        implement.static_findings = list(normalized)
        existing: set[tuple[str, int | None, int | None]] = {
            (
                request.path.strip(),
                request.start_line,
                request.end_line,
            )
            for request in implement.code_requests
            if isinstance(request, SnippetRequest)
        }

        for request in build_requests_from_findings(normalized, context=context_lines):
            key = (request.path, request.start_line, request.end_line)
            if key in existing:
                continue
            implement.code_requests.append(request)
            existing.add(key)

    def _collect_static_findings(self, report: GateReport, repo_root: Path) -> list[StaticFinding]:
        findings: list[StaticFinding] = []
        seen: set[tuple[str, int, str]] = set()

        for result in report.static_results:
            if not result.failed:
                continue
            parser = self._resolve_static_parser(repo_root, result.name, result.command)
            if parser is None:
                continue
            combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
            if not combined_output.strip():
                continue
            for entry in parser(combined_output):
                normalized = self._normalise_static_finding(entry, repo_root)
                key = (normalized.path, normalized.line_start, normalized.message)
                if key in seen:
                    continue
                findings.append(normalized)
                seen.add(key)

        return findings

    def _collect_recent_change_snippets(
        self,
        repo_root: Path,
        changed_paths: set[Path],
        *,
        surround: int = 40,
    ) -> list[Snippet]:
        requests: list[SnippetRequest] = []
        seen_paths: set[str] = set()

        for raw in sorted(changed_paths, key=lambda entry: entry.as_posix() if isinstance(entry, Path) else str(entry)):
            candidate = raw if isinstance(raw, Path) else Path(str(raw))
            if not candidate.is_absolute():
                candidate = (repo_root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            try:
                relative = candidate.relative_to(repo_root)
            except ValueError:
                if candidate.is_absolute():
                    continue
                relative = candidate
            normalised = self._normalise_path(relative.as_posix())
            if not normalised or normalised in seen_paths:
                continue
            if not self._looks_like_file_candidate(normalised):
                continue
            seen_paths.add(normalised)
            requests.append(
                SnippetRequest(
                    path=normalised,
                    surround=surround,
                    reason="Recent change referenced by failing tests.",
                )
            )

        if not requests:
            return []
        return collect_snippets(repo_root, requests)

    def _collect_failing_test_snippets(
        self,
        repo_root: Path,
        pytest_result: PytestResult | None,
        *,
        max_per_file: int = 2,
    ) -> list[Snippet]:
        if pytest_result is None:
            return []
        failing_nodes = self._extract_failing_tests(pytest_result)
        if not failing_nodes:
            return []

        requests: list[SnippetRequest] = []
        per_file: dict[str, int] = {}
        seen: set[tuple[str, int | None, int | None]] = set()

        for node in failing_nodes:
            path_token, selectors = self._split_pytest_node(node)
            if not path_token:
                continue
            normalised_path = self._normalise_path(path_token)
            if not normalised_path or not self._looks_like_file_candidate(normalised_path):
                continue
            if per_file.get(normalised_path, 0) >= max_per_file:
                continue
            span = self._locate_test_span(repo_root, normalised_path, selectors)
            if span is not None:
                start_line, end_line = span
                request = SnippetRequest(
                    path=normalised_path,
                    start_line=start_line,
                    end_line=end_line,
                    reason="Failing test needing investigation.",
                )
            else:
                request = SnippetRequest(
                    path=normalised_path,
                    surround=60,
                    reason="Failing test file needing investigation.",
                )
            key = (
                request.path,
                request.start_line if isinstance(request.start_line, int) and request.start_line > 0 else None,
                request.end_line if isinstance(request.end_line, int) and request.end_line > 0 else None,
            )
            if key in seen:
                continue
            seen.add(key)
            per_file[normalised_path] = per_file.get(normalised_path, 0) + 1
            requests.append(request)

        if not requests:
            return []
        return collect_snippets(repo_root, requests)

    @staticmethod
    def _split_pytest_node(node: str) -> tuple[str, list[str]]:
        if not isinstance(node, str):
            return "", []
        parts = [part for part in node.split("::") if part]
        if not parts:
            return "", []
        path = parts[0]
        selectors = parts[1:]
        return path, selectors

    @staticmethod
    def _clean_pytest_selector(selector: str) -> str:
        stripped = selector.strip()
        if not stripped:
            return ""
        bracket = stripped.find("[")
        if bracket != -1:
            stripped = stripped[:bracket]
        paren = stripped.find("(")
        if paren != -1:
            stripped = stripped[:paren]
        return stripped.strip()

    def _locate_test_span(
        self,
        repo_root: Path,
        path: str,
        selectors: Sequence[str],
    ) -> tuple[int, int] | None:
        candidate = (repo_root / path).resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError:
            return None

        try:
            source = candidate.read_text(encoding="utf-8")
        except OSError:
            return None

        cleaned_selectors = [self._clean_pytest_selector(entry) for entry in selectors if entry]
        if not cleaned_selectors:
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        node: ast.AST | None = tree
        for selector in cleaned_selectors:
            if not selector:
                return None
            body = getattr(node, "body", [])
            match: ast.AST | None = None
            for child in body:
                if isinstance(child, ast.ClassDef) and child.name == selector:
                    match = child
                    break
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == selector:
                    match = child
                    break
            if match is None:
                return None
            node = match

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if isinstance(start, int) and start > 0 and isinstance(end, int) and end >= start:
                return start, end
        return None

    @staticmethod
    def _resolve_static_parser(
        repo_root: Path,
        name: str | None,
        command: Sequence[str] | None,
    ) -> Callable[[str | Sequence[str] | Iterable[str]], list[StaticFinding]] | None:
        candidates: list[str] = []
        seen: set[str] = set()

        def _register(value: str | None) -> None:
            if not isinstance(value, str):
                return
            stripped = value.strip()
            if not stripped:
                return
            lowered = stripped.lower()
            if lowered not in seen:
                seen.add(lowered)
                candidates.append(stripped)
            stem = Path(stripped).stem
            if stem:
                stem_lower = stem.lower()
                if stem_lower not in seen:
                    seen.add(stem_lower)
                    candidates.append(stem)

        _register(name)

        if command:
            for entry in command:
                _register(entry)

        return resolve_static_parser(repo_root, candidates)

    @staticmethod
    def _normalise_static_finding(finding: StaticFinding, repo_root: Path) -> StaticFinding:
        raw_path = finding.path.strip()
        normalized = raw_path.replace("\\", "/") if raw_path else raw_path
        candidate = Path(normalized)
        try:
            if candidate.is_absolute():
                resolved = candidate.resolve(strict=False)
                try:
                    normalized = resolved.relative_to(repo_root).as_posix()
                except ValueError:
                    normalized = resolved.as_posix()
            else:
                normalized = candidate.as_posix()
        except OSError:
            normalized = candidate.as_posix() if not candidate.is_absolute() else normalized
        return StaticFinding(
            path=normalized,
            line_start=finding.line_start,
            line_end=finding.line_end,
            message=finding.message,
        )

    @staticmethod
    def _summarize_gate_failures(report: GateReport | None) -> list[str]:
        if report is None:
            return []
        failures: list[str] = []
        for violation in report.policy_violations:
            location = violation.path.as_posix() if violation.path else "(unknown)"
            if violation.line is not None:
                location = f"{location}:{violation.line}"
            failures.append(f"{violation.rule} :: {location} :: {violation.message}")
        for result in report.static_results:
            if result.failed:
                snippet = result.stderr.strip() or result.stdout.strip()
                if snippet:
                    snippet = snippet.splitlines()[0]
                else:
                    snippet = f"exit code {result.exit_code}"
                failures.append(f"{result.name}: {snippet}")
        return failures

    def _collect_pytest_commands(
        self,
        plan: CodingIterationPlan | None,
        implement: ImplementResponse | None,
    ) -> list[tuple[str, ...]]:
        commands: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()

        def add_command(args: tuple[str, ...]) -> None:
            if args not in seen:
                seen.add(args)
                commands.append(args)

        raw_sources: list[str] = []
        if implement and implement.test_commands:
            raw_sources.extend(entry for entry in implement.test_commands if isinstance(entry, str))
        if plan is not None:
            plan_tests = getattr(plan.implement, "test_plan", []) or []
            for entry in plan_tests:
                if isinstance(entry, str):
                    raw_sources.append(entry)

        for raw in raw_sources:
            parsed = self._parse_pytest_command(raw)
            if parsed is not None:
                add_command(parsed)

        if not commands:
            add_command(("-q",))
        elif not any(self._is_general_pytest_args(args) for args in commands):
            add_command(("-q",))

        return commands

    @staticmethod
    def _parse_pytest_command(command: str) -> tuple[str, ...] | None:
        if not command or not command.strip():
            return None
        try:
            parts = shlex.split(command)
        except ValueError:
            return None
        if not parts:
            return None
        try:
            index = parts.index("pytest")
        except ValueError:
            return None
        args: list[str] = []
        for token in parts[index + 1 :]:
            if token in {"&&", "||", ";"}:
                break
            args.append(token)
        return tuple(args)

    @staticmethod
    def _is_general_pytest_args(args: Sequence[str]) -> bool:
        if not args:
            return True
        return all(token.startswith("-") for token in args)

    @staticmethod
    def _prune_missing_pytest_targets(
        commands: Sequence[tuple[str, ...]],
        repo_root: Path,
    ) -> list[tuple[str, ...]]:
        filtered: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()

        for args in commands:
            pending_option: str | None = None
            retained: list[str] = []

            for token in args:
                if pending_option is not None:
                    retained.append(token)
                    pending_option = None
                    continue

                if token.startswith("--"):
                    name, sep, value = token.partition("=")
                    retained.append(token)
                    if sep:
                        continue
                    if name in _PYTEST_OPTIONS_EXPECT_VALUE:
                        pending_option = name
                    continue

                if token.startswith("-") and token != "-":
                    retained.append(token)
                    if token in _PYTEST_OPTIONS_EXPECT_VALUE:
                        pending_option = token
                    continue

                if not token:
                    continue

                path_token = token.split("::", 1)[0] or token
                normalised = path_token.rstrip("/\\")
                candidate_path = Path(normalised)
                if not candidate_path.is_absolute():
                    candidate_path = (repo_root / candidate_path).resolve()
                else:
                    candidate_path = candidate_path.resolve()
                if candidate_path.exists():
                    retained.append(token)
                continue

            if not retained:
                continue
            retained_tuple = tuple(retained)
            if retained_tuple in seen:
                continue
            seen.add(retained_tuple)
            filtered.append(retained_tuple)

        if not filtered:
            return [("-q",)]
        return filtered

    @staticmethod
    def _extract_failing_tests(pytest_result: PytestResult) -> list[str]:
        text = "\n".join(part for part in (pytest_result.stdout, pytest_result.stderr) if part)
        matches = _FAILED_LINE_RE.findall(text)
        if matches:
            return [path for _, path in matches]
        return [" ".join(pytest_result.command)]

    @staticmethod
    def _format_recent_changes(changed_paths: set[Path], repo_root: Path) -> list[str]:
        entries: list[str] = []
        for path in sorted(changed_paths):
            candidate = path
            try:
                candidate = path.relative_to(repo_root)
            except ValueError:
                candidate = path
            entries.append(candidate.as_posix())
        return entries

    # ---------------------------------------------------------------- persistence
    @staticmethod
    def _format_dirty_tree_error(paths: Sequence[Path], error: str) -> str:
        preview = [path.as_posix() for path in paths[:5]]
        lines: list[str] = [f"Cannot start iteration: {error}"]
        if preview:
            lines.append("Pending tracked changes:")
            lines.extend(f"- {entry}" for entry in preview)
            remaining = len(paths) - len(preview)
            if remaining > 0:
                lines.append(f"- ... (+{remaining} more)")
        lines.append("Clean the working tree before rerunning:")
        lines.append("- Inspect pending edits: git status --porcelain")
        lines.append('- Stash work in progress: git stash push -m "ae-preflight" (reapply with git stash pop)')
        lines.append('- Or commit them: git add . && git commit -m "WIP: save before agent"')
        lines.append("- Discard unwanted edits: git restore --worktree --staged <paths>")
        return "\n".join(lines)

    @staticmethod
    def _should_reinitialise_repository(entries: Sequence[tuple[str, Path]]) -> bool:
        has_deletions = False
        for status, _ in entries:
            if status == "??":
                continue
            if status == "D":
                has_deletions = True
                continue
            return False
        return has_deletions

    @staticmethod
    def _reinitialise_repository(repo: GitRepository) -> GitRepository:
        return GitRepository.initialise(repo.root)

    @staticmethod
    def _ensure_scaffold_baseline(repo: GitRepository) -> None:
        tracked = repo.git("ls-files", check=True).stdout.strip()
        if tracked:
            return
        status = repo.git("status", "--porcelain", check=True).stdout.strip()
        if not status:
            return
        repo.git("add", ".", check=True)
        repo.git("commit", "-m", "Scaffold baseline", check=True)

    def _resolve_plan_id(self, plan: CodingIterationPlan) -> str:
        if plan.plan_id:
            return plan.plan_id

        iteration = self._config.get("iteration")
        if isinstance(iteration, Mapping):
            candidate = iteration.get("plan_id") or iteration.get("plan") or iteration.get("id")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        return f"{plan.task_id}-plan"

    def _resolve_iteration_settings(self, plan: CodingIterationPlan) -> IterationSettings:
        """Return auto-loop settings derived from config/environment."""

        def _as_int(value: Any, *, minimum: int | None = None) -> int | None:
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                return None
            if minimum is not None and candidate < minimum:
                return minimum
            return candidate

        def _as_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _as_bool(value: Any) -> bool | None:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    return True
                if lowered in {"0", "false", "no", "off"}:
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            return None

        settings = IterationSettings()

        iteration_section = self._config.get("iteration")
        if isinstance(iteration_section, Mapping):
            max_cycles = (
                iteration_section.get("auto_loop_limit")
                or iteration_section.get("max_cycles")
                or iteration_section.get("max_iterations")
            )
            max_cycles_converted = _as_int(max_cycles, minimum=1)
            if max_cycles_converted is not None:
                settings.max_cycles = max_cycles_converted

            backoff = (
                iteration_section.get("cycle_backoff_seconds")
                or iteration_section.get("backoff_seconds")
                or iteration_section.get("retry_backoff_seconds")
            )
            backoff_value = _as_float(backoff)
            if backoff_value is not None:
                settings.backoff_seconds = max(backoff_value, 0.0)

            enable_autofix_setting = iteration_section.get("enable_autofix")
            enable_autofix_bool = _as_bool(enable_autofix_setting)
            if enable_autofix_bool is not None:
                settings.enable_autofix = enable_autofix_bool

            structured_toggle = (
                iteration_section.get("use_structured_edits")
                or iteration_section.get("implement_structured_edits")
                or iteration_section.get("structured_edits_only")
                or iteration_section.get("structured_only")
            )
            structured_bool = _as_bool(structured_toggle)
            if structured_bool is not None and structured_bool:
                settings.structured_edits_only = True

        env_max_cycles = _as_int(os.getenv("AE_MAX_CYCLES"), minimum=1)
        if env_max_cycles is not None:
            settings.max_cycles = env_max_cycles

        env_backoff_ms = _as_float(os.getenv("AE_BACKOFF_MS"))
        if env_backoff_ms is not None:
            settings.backoff_seconds = max(env_backoff_ms / 1000.0, 0.0)

        env_autofix = _as_bool(os.getenv("AE_ENABLE_AUTOFIX"))
        if env_autofix is not None:
            settings.enable_autofix = env_autofix

        env_structured = _as_bool(os.getenv("AE_STRUCTURED_ONLY"))
        if env_structured:
            settings.structured_edits_only = True

        if getattr(plan.implement, "structured_edits_only", False):
            settings.structured_edits_only = True

        return settings

    def _resolve_git_automation_state(self, plan_id: str | None, task_id: str) -> GitAutomationState:
        """Return git automation preferences derived from configuration."""

        state = GitAutomationState()
        git_section = self._config.get("git")
        if not isinstance(git_section, Mapping):
            git_section = {}

        auto_clean_cfg = (
            git_section.get("auto_clean")
            or git_section.get("auto_stash")
            or git_section.get("auto_housekeeping")
        )
        if isinstance(auto_clean_cfg, Mapping):
            enabled_value = auto_clean_cfg.get("enabled")
            enabled_bool = self._coerce_bool(enabled_value)
            state.auto_clean_enabled = enabled_bool if enabled_bool is not None else True

            include_value = (
                auto_clean_cfg.get("include_untracked")
                if "include_untracked" in auto_clean_cfg
                else auto_clean_cfg.get("untracked")
            )
            include_bool = self._coerce_bool(include_value)
            if include_bool is not None:
                state.include_untracked = include_bool
        elif auto_clean_cfg is not None:
            clean_bool = self._coerce_bool(auto_clean_cfg)
            if clean_bool is not None:
                state.auto_clean_enabled = clean_bool

        push_cfg = git_section.get("auto_push") or git_section.get("push")
        if isinstance(push_cfg, Mapping):
            enabled_value = push_cfg.get("enabled")
            enabled_bool = self._coerce_bool(enabled_value)
            state.push_enabled = enabled_bool if enabled_bool is not None else True

            remote_value = push_cfg.get("remote") or push_cfg.get("remote_name")
            if isinstance(remote_value, str) and remote_value.strip():
                state.push_remote = remote_value.strip()

            branch_value = (
                push_cfg.get("branch_template")
                or push_cfg.get("branch")
                or push_cfg.get("branch_name")
            )
            if isinstance(branch_value, str) and branch_value.strip():
                state.push_branch_template = branch_value.strip()

            force_value = push_cfg.get("force") or push_cfg.get("force_push")
            force_bool = self._coerce_bool(force_value)
            if force_bool is not None:
                state.push_force = force_bool

            upstream_value = (
                push_cfg.get("set_upstream")
                or push_cfg.get("upstream")
                or push_cfg.get("track")
            )
            upstream_bool = self._coerce_bool(upstream_value)
            if upstream_bool is not None:
                state.push_set_upstream = upstream_bool
        elif push_cfg is not None:
            push_bool = self._coerce_bool(push_cfg)
            if push_bool is not None:
                state.push_enabled = push_bool

        state.auto_clean_enabled = True
        state.include_untracked = True

        return state

    def _prepare_git_session(
        self,
        repo: GitRepository,
        plan_id: str | None,
        task_id: str,
    ) -> tuple[GitRepository, GitAutomationState]:
        """Provision an isolated workspace or reuse the primary repository."""

        state = self._resolve_git_automation_state(plan_id, task_id)
        state.original_repo_root = repo.root
        state.original_config_path = self._config_path
        state.workspace_path = repo.root
        state.workspace_method = "native"

        if not state.auto_clean_enabled:
            success, error = ensure_requirements_installed(repo.root)
            if not success:
                message = "Failed to install repository requirements"
                if error:
                    message = f"{message}: {error}"
                else:
                    message += "."
                state.preflight_error = message
            return repo, state

        state.remote_urls = self._capture_remote_urls(repo)
        try:
            workspace_repo = self._spawn_workspace_clone(repo, plan_id, task_id)
        except GitError as error:
            state.preflight_error = str(error)
            return repo, state

        state.workspace_path = workspace_repo.root
        state.workspace_method = "clone"
        try:
            self._configure_workspace_remotes(workspace_repo, state.remote_urls)
        except GitError as error:
            state.preflight_error = f"Failed to configure workspace remotes: {error}"
            return repo, state
        success, error = ensure_requirements_installed(workspace_repo.root)
        if not success:
            message = "Failed to install requirements in isolated workspace"
            if error:
                message = f"{message}: {error}"
            else:
                message += "."
            state.preflight_error = message
            return repo, state

        return workspace_repo, state

    def _capture_remote_urls(self, repo: GitRepository) -> dict[str, str]:
        remotes: dict[str, str] = {}
        listing = repo.git("remote", check=False)
        if listing.returncode != 0:
            return remotes
        for line in listing.stdout.splitlines():
            name = line.strip()
            if not name:
                continue
            probe = repo.git("remote", "get-url", name, check=False)
            if probe.returncode != 0:
                continue
            url = probe.stdout.strip()
            if url:
                remotes[name] = url
        return remotes

    def _spawn_workspace_clone(
        self,
        repo: GitRepository,
        plan_id: str | None,
        task_id: str,
    ) -> GitRepository:
        data_root = self._context_builder.data_root
        workspace_base = (data_root / "workspaces").resolve()
        workspace_base.mkdir(parents=True, exist_ok=True)
        task_segment = self._render_git_template("{task_slug}", plan_id, task_id)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        identifier = self._abbreviate_slug(task_segment or "task")
        workspace_name = f"ae-workspace-{identifier}-{timestamp}"
        workspace_root = (workspace_base / workspace_name).resolve()
        if workspace_root.exists():
            shutil.rmtree(workspace_root, ignore_errors=True)
        command = [
            "git",
            "clone",
            "--local",
            "--no-hardlinks",
            repo.root.as_posix(),
            workspace_root.as_posix(),
        ]
        process = subprocess.run(
            command,
            cwd=workspace_base,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            message = process.stderr.strip() or process.stdout.strip() or "unknown git error"
            raise GitError(f"Failed to create isolated workspace: {message}")
        return GitRepository(workspace_root)

    def _configure_workspace_remotes(
        self,
        workspace_repo: GitRepository,
        remote_urls: Mapping[str, str],
    ) -> None:
        for name, url in remote_urls.items():
            if not url:
                continue
            probe = workspace_repo.git("remote", "get-url", name, check=False)
            if probe.returncode == 0:
                existing = probe.stdout.strip()
                if existing == url:
                    continue
                workspace_repo.git("remote", "set-url", name, url)
            else:
                workspace_repo.git("remote", "add", name, url)

    def _resolve_workspace_config_path(
        self,
        *,
        workspace_root: Path,
        original_root: Path | None,
        original_config: Path | None,
    ) -> Path | None:
        if original_config is None:
            return None
        config_path = Path(original_config)
        if not config_path.is_absolute():
            return (workspace_root / config_path).resolve()
        if original_root is None:
            return config_path
        try:
            relative = config_path.relative_to(original_root)
        except ValueError:
            return config_path
        return (workspace_root / relative).resolve()

    def _collect_workspace_diff(self, repo: GitRepository) -> str:
        repo.git("add", "--all")
        diff_process = repo.git("diff", "--cached", "--binary", "--unified=3")
        diff_text = diff_process.stdout
        repo.git("reset", "--mixed", "HEAD", check=False)
        if diff_text and not diff_text.endswith("\n"):
            diff_text += "\n"
        return diff_text

    def _mirror_workspace_to_primary(
        self,
        workspace_repo: GitRepository,
        git_state: GitAutomationState,
        result: CodingIterationResult,
    ) -> None:
        original_root = git_state.original_repo_root
        if original_root is None:
            return
        try:
            primary_repo = GitRepository(original_root)
        except GitError as error:
            result.errors.append(f"Failed to reopen primary repository: {error}")
            return
        if primary_repo.has_changes(include_untracked=True):
            result.errors.append(
                "Skipped mirroring workspace changes because the primary repository has pending changes."
            )
            return
        diff_text = self._collect_workspace_diff(workspace_repo)
        if not diff_text.strip():
            if result.commit_sha:
                branch = workspace_repo.current_branch()
                if branch:
                    fetch = primary_repo.git(
                        "fetch",
                        workspace_repo.root.as_posix(),
                        branch,
                        check=False,
                    )
                    if fetch.returncode != 0:
                        message = fetch.stderr.strip() or fetch.stdout.strip() or "unable to fetch workspace commit"
                        result.errors.append(f"Failed to synchronise workspace commit: {message}")
                        return
                    try:
                        primary_repo.git("reset", "--hard", result.commit_sha)
                    except GitError as error:
                        result.errors.append(f"Failed to update primary repository to commit {result.commit_sha}: {error}")
            return
        try:
            apply_patch(diff_text, repo_root=primary_repo.root)
        except PatchError as error:
            result.errors.append(f"Failed to mirror workspace edits onto primary repository: {error}")
            return

        if not primary_repo.has_changes(include_untracked=True):
            return

        commit_message = (result.commit_message or "").strip() or "ae: mirror workspace changes"
        try:
            primary_repo.git("config", "--local", "user.name", "Agentic Engineer", check=False)
            primary_repo.git("config", "--local", "user.email", "agent@example.com", check=False)
            sha = primary_repo.commit_all(commit_message)
        except GitError as error:
            result.errors.append(f"Failed to commit mirrored workspace changes: {error}")
            return

        if sha:
            result.commit_sha = sha
            result.commit_message = commit_message

    def _finalize_git_session(
        self,
        repo: GitRepository,
        git_state: GitAutomationState,
        result: CodingIterationResult,
        *,
        revert_on_exit: bool,
    ) -> None:
        """Tear down any isolated workspace created for automation."""

        workspace_root = git_state.workspace_path
        original_root = git_state.original_repo_root
        if original_root and workspace_root and workspace_root != original_root:
            original_config = git_state.original_config_path
            self._rebind_runtime_environment(
                repo_root=original_root,
                config_path=original_config,
            )

        if not git_state.auto_clean_enabled:
            return
        if not workspace_root or git_state.workspace_method != "clone":
            return

        if not revert_on_exit:
            self._mirror_workspace_to_primary(repo, git_state, result)

        shutil.rmtree(workspace_root, ignore_errors=True)

    def _resolve_db_path(self) -> Path:
        paths_section = self._config.get("paths")
        db_path: Path | None = None
        data_root = Path("data")
        if isinstance(paths_section, Mapping):
            db_candidate = paths_section.get("db_path")
            if isinstance(db_candidate, str) and db_candidate.strip():
                db_path = Path(db_candidate.strip())
            data_candidate = paths_section.get("data")
            if isinstance(data_candidate, str) and data_candidate.strip():
                data_root = Path(data_candidate.strip())
        if db_path is None:
            db_path = data_root / "ae.sqlite"
        if not db_path.is_absolute():
            db_path = (self._repo_root / db_path).resolve()
        return db_path

    def _open_store(self) -> MemoryStore:
        return MemoryStore(self._resolve_db_path())

    @staticmethod
    def _normalize_references(
        store: MemoryStore,
        plan_id: str,
        task_id: str | None,
    ) -> tuple[bool, str | None]:
        """Return (plan_exists, valid_task_id) for persistence helpers."""

        try:
            plan_exists = store.get_plan(plan_id) is not None
        except AttributeError:  # pragma: no cover - defensive
            return False, None

        if not plan_exists:
            return False, None

        valid_task_id: str | None = None
        if task_id:
            try:
                if store.get_task(task_id) is not None:
                    valid_task_id = task_id
            except AttributeError:  # pragma: no cover - defensive
                valid_task_id = None

        return True, valid_task_id

    def _persist_checkpoint(
        self,
        store: MemoryStore,
        checkpoint: GitCheckpoint | None,
        plan_id: str,
        task_id: str | None,
    ) -> bool:
        if checkpoint is None:
            return False

        plan_exists, valid_task_id = self._normalize_references(store, plan_id, task_id)
        if not plan_exists:
            return False

        # Avoid import cycle by referencing type dynamically
        payload = {
            "head": checkpoint.head,
            "baseline_untracked": list(checkpoint.baseline_untracked),
            "created_at": checkpoint.created_at,
        }
        metadata = {
            "repo_root": checkpoint.repo.root.as_posix(),
        }
        record = Checkpoint(
            id=self._generate_identifier("chk"),
            plan_id=plan_id,
            task_id=valid_task_id,
            label=checkpoint.label,
            payload=payload,
            metadata=metadata,
        )
        store.save_checkpoint(record)
        return True

    def _persist_test_run(
        self,
        store: MemoryStore,
        pytest_result: PytestResult,
        plan_id: str,
        task_id: str | None,
    ) -> bool:
        plan_exists, valid_task_id = self._normalize_references(store, plan_id, task_id)
        if not plan_exists:
            return False

        status_map = {
            "passed": TestStatus.PASSED,
            "failed": TestStatus.FAILED,
            "error": TestStatus.FAILED,
            "no-tests": TestStatus.SKIPPED,
        }
        status = status_map.get(pytest_result.status, TestStatus.UNKNOWN)
        metadata = {
            "exit_code": pytest_result.exit_code,
            "collected": pytest_result.collected,
            "cwd": pytest_result.cwd.as_posix(),
        }
        record = TestRun(
            id=self._generate_identifier("test"),
            plan_id=plan_id,
            task_id=valid_task_id,
            name="pytest",
            status=status,
            command=" ".join(pytest_result.command),
            output=self._combine_test_output(pytest_result),
            metadata=metadata,
        )
        store.record_test_run(record)
        return True

    @staticmethod
    def _combine_test_output(pytest_result: PytestResult) -> str:
        parts = [pytest_result.stdout, pytest_result.stderr]
        combined = "\n".join(part for part in parts if part)
        return combined.strip()

    # ---------------------------------------------------------------- artifacts
    def _resolve_artifact_root(self, *, revert_on_exit: bool) -> Path:
        paths_section = self._config.get("paths")
        logs_path: Path | None = None
        data_root = Path("data")
        if isinstance(paths_section, Mapping):
            logs_candidate = paths_section.get("logs")
            if isinstance(logs_candidate, str) and logs_candidate.strip():
                logs_path = Path(logs_candidate.strip())
            data_candidate = paths_section.get("data")
            if isinstance(data_candidate, str) and data_candidate.strip():
                data_root = Path(data_candidate.strip())
        if logs_path is None:
            logs_path = data_root / "logs"
        if not logs_path.is_absolute():
            logs_path = (self._repo_root / logs_path).resolve()
        target = logs_path / "iterations"

        if revert_on_exit:
            try:
                repo_root_resolved = self._repo_root.resolve()
                target_resolved = target.resolve()
                if target_resolved.is_relative_to(repo_root_resolved):
                    temp_root = Path(tempfile.gettempdir()) / "agentic-engineer" / "iterations"
                    temp_root.mkdir(parents=True, exist_ok=True)
                    return temp_root
            except AttributeError:  # Python <3.9 guard, not expected in 3.11
                repo_root_resolved = self._repo_root.resolve()
                target_resolved = target.resolve()
                if str(target_resolved).startswith(str(repo_root_resolved)):
                    temp_root = Path(tempfile.gettempdir()) / "agentic-engineer" / "iterations"
                    temp_root.mkdir(parents=True, exist_ok=True)
                    return temp_root

        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
        slug = slug.strip("-")
        return slug or "iteration"

    @staticmethod
    def _coerce_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    def _render_git_template(self, template: str, plan_id: str | None, task_id: str) -> str:
        """Format git automation templates with standard placeholders."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        context = {
            "plan_id": plan_id or "",
            "task_id": task_id,
            "plan_slug": self._slugify(plan_id) if plan_id else "",
            "task_slug": self._slugify(task_id),
            "timestamp": timestamp,
        }
        try:
            return template.format(**context)
        except (KeyError, IndexError, ValueError):
            return template

    @staticmethod
    def _abbreviate_slug(segment: str, *, fallback: str = "task", max_length: int = 80) -> str:
        slug = segment.strip("-")
        if not slug:
            return fallback
        if len(slug) <= max_length:
            return slug
        digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:8]
        prefix_length = max(max_length - len(digest) - 1, 1)
        prefix = slug[:prefix_length].rstrip("-")
        if not prefix:
            prefix = slug[:prefix_length]
        return f"{prefix}-{digest}"

    def _write_iteration_artifact(
        self,
        *,
        plan_id: str,
        task_id: str,
        goal: str,
        iteration_result: CodingIterationResult,
        revert_on_exit: bool,
    ) -> Path:
        artifact_root = self._resolve_artifact_root(revert_on_exit=revert_on_exit)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{self._slugify(plan_id)}__{self._slugify(task_id)}__{timestamp}.json"
        artifact_path = artifact_root / filename
        payload = self._serialize_iteration_result(
            plan_id=plan_id,
            task_id=task_id,
            goal=goal,
            iteration_result=iteration_result,
            timestamp=timestamp,
        )
        with artifact_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return artifact_path

    def _serialize_iteration_result(
        self,
        *,
        plan_id: str,
        task_id: str,
        goal: str,
        iteration_result: CodingIterationResult,
        timestamp: str,
    ) -> dict[str, Any]:
        patch_result = iteration_result.patch
        diff_paths = [path.as_posix() for path in patch_result.touched_paths]
        telemetry_payload: dict[str, Any] | None = None
        if patch_result.telemetry is not None:
            if isinstance(patch_result.telemetry, PatchTelemetry):
                telemetry_payload = patch_result.telemetry.to_dict()
            elif isinstance(patch_result.telemetry, Mapping):
                telemetry_payload = dict(patch_result.telemetry)
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "plan_id": plan_id,
            "task_id": task_id,
            "goal": goal,
            "ok": iteration_result.ok,
            "checkpoint_label": iteration_result.checkpoint_label,
            "rolled_back": iteration_result.rolled_back,
            "diff": {
                "attempted": patch_result.attempted,
                "applied": patch_result.applied,
                "touched_paths": diff_paths,
                "error": patch_result.error,
                "no_op_reason": patch_result.no_op_reason,
                "telemetry": telemetry_payload,
            },
            "gates": {
                "ok": iteration_result.gates.ok,
                "violations": list(iteration_result.gates.violations),
                "error": iteration_result.gates.error,
            },
            "tests": None,
            "errors": list(iteration_result.errors),
        }
        if iteration_result.gates.report is not None:
            payload["gates"]["summary"] = iteration_result.gates.report.format_summary()

        if iteration_result.tests is not None:
            payload["tests"] = {
                "status": iteration_result.tests.status,
                "exit_code": iteration_result.tests.exit_code,
                "collected": iteration_result.tests.collected,
                "command": list(iteration_result.tests.command),
                "stdout": iteration_result.tests.stdout,
                "stderr": iteration_result.tests.stderr,
                "ok": iteration_result.tests.ok,
            }

        if iteration_result.analyze is not None:
            payload["analyze"] = asdict(iteration_result.analyze)
        if iteration_result.design is not None:
            payload["design"] = asdict(iteration_result.design)
        if iteration_result.implement is not None:
            payload["implement"] = asdict(iteration_result.implement)
        if iteration_result.fix_violations is not None:
            payload["fix_violations"] = asdict(iteration_result.fix_violations)
        if iteration_result.diagnose is not None:
            payload["diagnose"] = asdict(iteration_result.diagnose)

        if iteration_result.commit_sha is not None or iteration_result.commit_message is not None:
            payload["commit"] = {
                "sha": iteration_result.commit_sha,
                "message": iteration_result.commit_message,
            }

        if iteration_result.pushed_remote is not None or iteration_result.pushed_branch is not None:
            payload["push"] = {
                "remote": iteration_result.pushed_remote,
                "branch": iteration_result.pushed_branch,
            }

        return payload

    @staticmethod
    def _generate_identifier(prefix: str) -> str:
        return f"{prefix}-{uuid4().hex}"

    def _auto_commit(
        self,
        *,
        repo: GitRepository,
        plan_id: str | None,
        task_id: str,
        implement: ImplementResponse | None,
    ) -> tuple[str | None, str | None]:
        if not repo.has_changes(include_untracked=True):
            return (None, None)

        try:
            repo.git("config", "--local", "user.name", "Agentic Engineer", check=False)
            repo.git("config", "--local", "user.email", "agent@example.com", check=False)
        except GitError:
            pass

        summary = ""
        if implement is not None and implement.summary:
            summary = implement.summary.strip().splitlines()[0]

        identifier = f"{plan_id}::{task_id}" if plan_id else task_id
        if not summary:
            summary = "automated update"
        summary = summary[:120]
        message = f"{identifier} - {summary}"

        sha = repo.commit_all(message)
        if sha is None:
            return (None, None)
        return sha, message

    def _maybe_auto_push(
        self,
        *,
        repo: GitRepository,
        git_state: GitAutomationState,
        plan_id: str | None,
        task_id: str,
    ) -> tuple[str, str]:
        if not git_state.push_enabled:
            raise GitError("Auto-push is not enabled.")

        remote = git_state.push_remote or "origin"
        if git_state.push_branch_template:
            branch = self._render_git_template(git_state.push_branch_template, plan_id, task_id)
        else:
            branch = repo.current_branch()

        if not branch:
            raise GitError("Unable to determine target branch for auto-push (detached HEAD).")

        repo.push(
            remote,
            branch,
            set_upstream=git_state.push_set_upstream,
            force=git_state.push_force,
        )
        git_state.push_branch = branch
        return remote, branch
