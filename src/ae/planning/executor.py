"""Plan executor wiring that drives orchestrator iterations and adjustments."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from pydantic import ValidationError

from ..memory.code_index.indexer import CodeIndexer
from ..memory.schema import (
    Decision,
    IncidentSeverity,
    Plan,
    PlanStatus,
    Task,
    TaskStatus,
    TestStatus,
    utc_now,
)
from ..memory.store import MemoryStore
from ..orchestrator import (
    CodingIterationPlan,
    CodingIterationResult,
    FailureCategory,
    Orchestrator,
    PatchApplicationResult,
)
from ..utils import slugify
from ..phases import PhaseName
from ..phases.analyze import AnalyzeRequest
from ..phases.design import DesignRequest
from ..phases.implement import ImplementRequest
from ..phases.plan import PlanRequest
from ..phases.plan_adjust import (
    PlanAdjustRequest,
    PlanAdjustResponse,
    PlanAdjustment,
    PlanAdjustmentItem,
)
from ..tools.coverage_map import CoverageMap
from .schemas import PlannerResponse, PlannerTask
from .task_filters import looks_like_test_execution, task_is_test_execution

LOGGER = logging.getLogger(__name__)

_FILE_EXTENSIONS: set[str] = {
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

_PYTEST_OPTIONS_EXPECT_VALUE = {
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


@dataclass(slots=True)
class AutoAppliedAdjustment:
    """Metadata about an automatically applied plan adjustment patch."""

    digest: str
    description: str
    priority: str | None = None
    touched_paths: tuple[Path, ...] = ()
    source: str | None = None


def build_plan_adjust_request(
    plan_id: str,
    task_id: str,
    result: CodingIterationResult,
) -> PlanAdjustRequest:
    """Construct a Plan-Adjust request from the latest iteration result."""

    blockers = list(result.gates.violations)
    if result.tests is not None and not result.tests.ok:
        failing = _extract_failing_tests(result.tests)
        blockers.append(f"Tests failed: {', '.join(failing)}")

    suggested: list[str] = []
    if result.fix_violations:
        suggested.extend(result.fix_violations.rationale)
    if result.diagnose:
        suggested.extend(result.diagnose.recommended_fixes)

    reason = "; ".join(result.errors) if result.errors else "Iteration did not complete successfully."

    suspect_files: list[str] = []
    seen_suspects: set[str] = set()

    def _normalise_suspect_path(raw: object) -> str | None:
        if raw is None:
            return None
        if isinstance(raw, Path):
            candidate = raw.as_posix()
        else:
            candidate = str(raw).strip()
        if not candidate:
            return None
        candidate = candidate.strip("\"'`“”‘’")
        candidate = candidate.replace("\\", "/")
        candidate = candidate.lstrip("([{<").rstrip(".,;:!?)]}>")
        while candidate.startswith("./"):
            candidate = candidate[2:]
        return candidate or None

    def _record_suspect(entry: object) -> None:
        normalised = _normalise_suspect_path(entry)
        if not normalised or normalised in seen_suspects:
            return
        seen_suspects.add(normalised)
        suspect_files.append(normalised)

    if result.gates.suspect_files:
        for candidate in result.gates.suspect_files:
            _record_suspect(candidate)

    touched_paths = getattr(result.patch, "touched_paths", ()) or ()
    for path in touched_paths:
        _record_suspect(path)

    if result.fix_violations:
        for path in getattr(result.fix_violations, "touched_files", []) or []:
            _record_suspect(path)

    for cycle in getattr(result, "cycles", []) or []:
        for adjustment in getattr(cycle, "adjustments", []) or []:
            for path in getattr(adjustment, "touched_paths", []) or []:
                _record_suspect(path)

    return PlanAdjustRequest(
        plan_id=plan_id,
        task_id=task_id,
        reason=reason,
        suggested_changes=suggested,
        blockers=blockers,
        suspect_files=suspect_files,
    )


def apply_adjustments_and_reenter(
    *,
    adjustment: PlanAdjustResponse | None,
    apply_updates: Callable[[Sequence[Any], Sequence[Any]], PatchApplicationResult],
    applied_hashes: set[str],
) -> list[AutoAppliedAdjustment]:
    """Apply high-priority adjustments and signal whether re-entry is warranted.

    Returns the list of successfully applied adjustments. Each entry includes
    the touching paths so callers can update downstream gate/test selection.
    """

    if adjustment is None:
        return []

    applied: list[AutoAppliedAdjustment] = []
    for entry in adjustment.adjustments:
        description, priority, files, edits, source = _extract_adjustment(entry)
        if not files and not edits:
            continue
        priority_normalised = (priority or "").lower()
        if priority_normalised and priority_normalised not in {"high", "p0", "urgent"}:
            continue
        digest = _digest_structured_payload(files, edits)
        if digest in applied_hashes:
            continue
        patch_result = apply_updates(files, edits)
        if not patch_result.applied:
            continue
        applied_hashes.add(digest)
        applied.append(
            AutoAppliedAdjustment(
                digest=digest,
                description=description,
                priority=priority_normalised or None,
                touched_paths=tuple(patch_result.touched_paths),
                source=source,
            )
        )
    return applied


def _extract_adjustment(entry: PlanAdjustment) -> tuple[str, str | None, list[Any], list[Any], str | None]:
    """Normalise plan adjustments into display text and structured payloads."""
    if isinstance(entry, PlanAdjustmentItem):
        description = entry.render()
        priority = entry.priority.strip() if entry.priority else None
        files = list(entry.files or [])
        edits = list(entry.edits or [])
        return description, priority, files, edits, entry.id

    description = str(entry).strip()
    priority = _infer_priority_from_text(description)
    return description, priority, [], [], None


def _digest_structured_payload(files: Sequence[Any], edits: Sequence[Any]) -> str:
    """Generate a stable hash for structured patch payloads."""
    payload = {
        "files": [
            {
                "path": getattr(item, "path", None),
                "content": getattr(item, "content", None),
                "executable": bool(getattr(item, "executable", False)),
                "encoding": getattr(item, "encoding", None),
            }
            for item in files
        ],
        "edits": [
            {
                "path": getattr(item, "path", None),
                "action": getattr(item, "action", None),
                "start_line": getattr(item, "start_line", None),
                "end_line": getattr(item, "end_line", None),
                "content": getattr(item, "content", None),
                "note": getattr(item, "note", None),
            }
            for item in edits
        ],
    }
    encoded = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _infer_priority_from_text(text: str) -> str | None:
    """Infer an adjustment priority from free-form text cues."""
    lowered = text.lower()
    if "priority: high" in lowered or "priority high" in lowered:
        return "high"
    if "[high priority]" in lowered or "urgent" in lowered:
        return "high"
    if "p0" in lowered:
        return "p0"
    return None


def _extract_failing_tests(pytest_result) -> list[str]:
    """Derive failing test identifiers from a pytest result object."""
    command = getattr(pytest_result, "command", ())
    if command:
        return [" ".join(str(part) for part in command)]
    return ["pytest"]


@dataclass(slots=True)
class IterationOutcome:
    """Summary of a single task iteration."""

    task: Task
    result: CodingIterationResult
    plan_adjustment: PlanAdjustResponse | None = None
    follow_up_tasks: list[Task] = field(default_factory=list)


@dataclass(slots=True)
class PlanExecutionSummary:
    """Aggregated summary for executing a plan loop."""

    plan: Plan
    iterations: list[IterationOutcome] = field(default_factory=list)
    completed: bool = False
    tasks: list[Task] = field(default_factory=list)


class PlanExecutor:
    """Execute READY tasks for a plan using the orchestrator."""

    _AUTO_REVIEW_MARKER = "code_review"
    _AUTO_REVIEW_TASK_SEED = "code-review"
    _AUTO_README_MARKER = "readme_polish"
    _AUTO_METADATA_KEY = "auto_generated"
    _AUTO_TASK_SEED = "polish-readme"
    _MAX_PENDING_PLAN_ADJUST_FOLLOWUPS = 3
    _AUTO_REVIEW_PLAN_LIMIT = 1
    _FINAL_ITERATION_THRESHOLD = 50

    def __init__(
        self,
        store: MemoryStore,
        orchestrator: Orchestrator,
        *,
        config_path: Path | None = None,
        config: Mapping[str, Any] | None = None,
        code_indexer: CodeIndexer | None = None,
        revert_on_exit: bool = False,
    ) -> None:
        self._store = store
        self._orchestrator = orchestrator
        self._config: dict[str, Any] = dict(config or {})
        self._config_path = Path(config_path).resolve() if config_path else None
        self._repo_root = self._determine_repo_root(self._config, self._config_path)
        self._revert_on_exit = revert_on_exit

        self._persist_coverage_map_enabled = False
        iteration_section = self._config.get("iteration")
        if isinstance(iteration_section, Mapping):
            persist_flag = iteration_section.get("persist_coverage_map")
            if isinstance(persist_flag, bool):
                self._persist_coverage_map_enabled = persist_flag

        self._coverage_map_path = (
            self._resolve_coverage_map_path() if self._persist_coverage_map_enabled else None
        )
        self._coverage_map = self._load_coverage_map()
        self._code_indexer = code_indexer or self._create_code_indexer()
        self._index_repo_root: Path | None = (
            self._code_indexer.repo_root if isinstance(self._code_indexer, CodeIndexer) else None
        )
        self._index_data_root: Path | None = (
            self._code_indexer.data_root if isinstance(self._code_indexer, CodeIndexer) else None
        )

    # ------------------------------------------------------------------ public
    def execute(self, plan_id: str | None = None) -> PlanExecutionSummary | None:
        plan = self._select_plan(plan_id)
        if plan is None:
            return None

        summary = PlanExecutionSummary(plan=plan)

        while True:
            task = self._store.get_ready_task(plan_id=plan.id)
            if task is None:
                if self._maybe_schedule_code_review_iteration(plan):
                    plan = self._refresh_plan(plan.id) or plan
                    continue
                if self._maybe_schedule_readme_iteration(plan):
                    plan = self._refresh_plan(plan.id) or plan
                    continue
                break

            self._refresh_code_index()
            planner_task = self._load_planner_task(task)
            iteration_plan = self._build_iteration_plan(plan, task, planner_task)
            if iteration_plan is None:
                self._store.update_task_status(task.id, TaskStatus.BLOCKED)
                continue

            try:
                result = self._orchestrator.run_coding_iteration(
                    iteration_plan,
                    config_path=self._config_path,
                    revert_on_exit=self._revert_on_exit,
                )
            finally:
                # Ensure newly applied workspace edits are reflected in the indices, even if
                # the iteration fails part-way through orchestration.
                self._refresh_code_index()
            try:
                self._store.reopen()
            except Exception:
                # If reopening fails, propagate so callers can handle persistence errors.
                raise
            self._update_coverage_map(planner_task, result)
            plan = self._record_iteration_metrics(plan, task, result)
            adjustment, followups = self._handle_iteration_result(plan, task, result)
            summary.iterations.append(
                IterationOutcome(
                    task=task,
                    result=result,
                    plan_adjustment=adjustment,
                    follow_up_tasks=followups,
                )
            )
            plan = self._refresh_plan(plan.id) or plan

        summary.plan, summary.completed = self._mark_plan_if_completed(plan)
        summary.tasks = self._store.list_tasks(plan_id=plan.id)
        return summary

    # ----------------------------------------------------------------- helpers
    def _select_plan(self, plan_id: str | None) -> Plan | None:
        if plan_id:
            plan = self._store.get_plan(plan_id)
            if plan is not None:
                return plan
        plans = self._store.list_plans()
        if not plans:
            return None
        # Prefer active plans and fall back to the most recent.
        active = [item for item in plans if item.status in {PlanStatus.ACTIVE, PlanStatus.DRAFT}]
        if active:
            return active[-1]
        return plans[-1]

    def _refresh_plan(self, plan_id: str) -> Plan | None:
        return self._store.get_plan(plan_id)

    def _build_iteration_plan(
        self,
        plan: Plan,
        task: Task,
        planner_task: PlannerTask,
    ) -> CodingIterationPlan | None:
        proposed_interfaces = self._derive_proposed_interfaces(planner_task)
        if not proposed_interfaces:
            fallback_interface = planner_task.title or task.title
            if isinstance(fallback_interface, str) and fallback_interface:
                proposed_interfaces = [fallback_interface.strip()] if fallback_interface.strip() else [fallback_interface]
        if not proposed_interfaces and isinstance(task.title, str) and task.title:
            proposed_interfaces = [task.title.strip()] if task.title.strip() else [task.title]
        structured_only = self._requires_structured_edits(planner_task)
        extra_notes = self._coerce_string_collection(
            self._lookup_metadata_value(planner_task.metadata or {}, ("implement_notes", "execution_notes"))
        )
        product_spec = self._compose_product_spec(plan, planner_task)

        analyze = AnalyzeRequest(
            task_id=task.id,
            goal=plan.goal,
            context=planner_task.summary,
            constraints=list(planner_task.constraints),
            product_spec=product_spec,
        )
        design = DesignRequest(
            task_id=task.id,
            goal=plan.goal,
            proposed_interfaces=proposed_interfaces,
            constraints=list(planner_task.constraints),
            product_spec=product_spec,
        )
        implement = self._build_implement_request(
            plan=plan,
            task=task,
            planner_task=planner_task,
            structured_only=structured_only,
            extra_notes=extra_notes,
            product_spec=product_spec,
        )
        return CodingIterationPlan(analyze=analyze, design=design, implement=implement, plan_id=plan.id)

    def _build_implement_request(
        self,
        *,
        plan: Plan,
        task: Task,
        planner_task: PlannerTask,
        structured_only: bool,
        extra_notes: Sequence[str],
        product_spec: str,
    ) -> ImplementRequest:
        diff_goal = self._compose_diff_goal(plan, planner_task)
        touched_files, related_files = self._assemble_file_hints(plan, planner_task)
        acceptance = self._coerce_string_collection(planner_task.acceptance_criteria)
        constraints = self._coerce_string_collection(planner_task.constraints)
        deliverables = self._coerce_string_collection(planner_task.deliverables)
        dependencies = self._coerce_string_collection(planner_task.depends_on)

        if self._final_iteration_mode_active(plan):
            constraints.append(
                "Final delivery mode: prioritise shipping a working baseline over polish or optional refactors."
            )

        failure_notes = self._collect_failure_notes(plan)
        incident_notes = self._collect_incident_notes(plan)
        decision_notes = self._collect_decision_notes(plan, task)
        metadata_notes = self._extract_metadata_notes(planner_task.metadata or {})

        combined_notes: list[str] = []
        combined_notes.extend(failure_notes)
        combined_notes.extend(incident_notes)
        combined_notes.extend(planner_task.notes or [])
        combined_notes.extend(extra_notes or [])
        combined_notes.extend(self._final_iteration_notes(plan))

        context_sections = self._build_implement_context(
            plan=plan,
            planner_task=planner_task,
            decision_notes=decision_notes,
            related_files=related_files,
            metadata_notes=metadata_notes,
        )

        return ImplementRequest(
            task_id=task.id,
            diff_goal=diff_goal,
            touched_files=touched_files,
            related_files=related_files,
            test_plan=self._select_test_plan(planner_task, plan),
            constraints=constraints,
            acceptance_criteria=acceptance,
            deliverables=deliverables,
            dependencies=dependencies,
            notes=combined_notes,
            context=context_sections,
            product_spec=product_spec,
            structured_edits_only=structured_only,
        )

    def _compose_product_spec(self, plan: Plan, planner_task: PlannerTask) -> str:
        sections: list[str] = []

        overview_lines: list[str] = []
        goal = (plan.goal or "").strip()
        plan_summary = (plan.summary or "").strip()
        task_title = (planner_task.title or "").strip()
        task_summary = (planner_task.summary or "").strip()

        if goal:
            overview_lines.append(f"- Goal: {goal}")
        if plan_summary and plan_summary.lower() != goal.lower():
            overview_lines.append(f"- Plan summary: {self._summarize_text(plan_summary, width=220)}")
        if task_title:
            overview_lines.append(f"- Current task: {task_title}")
        if task_summary:
            overview_lines.append(f"- Task summary: {self._summarize_text(task_summary, width=220)}")
        constraints = self._coerce_string_collection(planner_task.constraints)
        if constraints:
            overview_lines.append(f"- Constraints: {', '.join(constraints[:5])}")
        if overview_lines:
            sections.append("### Product Vision\n" + "\n".join(overview_lines))

        components = self._plan_analysis_components(plan)
        if components:
            component_lines: list[str] = []
            for component in components[:10]:
                name = str(component.get("name") or component.get("title") or "Component").strip()
                summary = self._summarize_text(component.get("summary"), width=200)
                primary_paths = [str(path).strip() for path in component.get("primary_paths") or [] if str(path).strip()]
                related_tests = [
                    str(entry).strip() for entry in component.get("related_tests") or [] if str(entry).strip()
                ]
                line = f"- {name}"
                details: list[str] = []
                if summary:
                    details.append(summary)
                if primary_paths:
                    details.append(f"Paths: {', '.join(primary_paths[:4])}")
                if related_tests:
                    details.append(f"Tests: {', '.join(related_tests[:3])}")
                if details:
                    line = f"{line} — {'; '.join(details)}"
                component_lines.append(line)
            if component_lines:
                sections.append("### Architecture Components\n" + "\n".join(component_lines))

        try:
            tasks = self._store.list_tasks(plan_id=plan.id)
        except Exception:
            tasks = []
        completed_section = self._render_task_section(
            tasks,
            heading="### Completed Milestones",
            statuses={TaskStatus.DONE},
            limit=8,
        )
        if completed_section:
            sections.append(completed_section)

        active_section = self._render_task_section(
            tasks,
            heading="### Active Backlog",
            statuses={TaskStatus.READY, TaskStatus.RUNNING},
            limit=8,
        )
        if active_section:
            sections.append(active_section)

        blocked_section = self._render_task_section(
            tasks,
            heading="### Blocked Tasks",
            statuses={TaskStatus.BLOCKED, TaskStatus.FAILED},
            limit=8,
        )
        if blocked_section:
            sections.append(blocked_section)

        diagnostics_lines = self._collect_diagnose_followups(tasks)
        if diagnostics_lines:
            sections.append("### Outstanding Diagnostics\n" + "\n".join(diagnostics_lines))

        decision_lines = self._collect_decision_summaries(plan)
        if decision_lines:
            sections.append("### Key Decisions\n" + "\n".join(decision_lines))

        risk_lines = self._collect_known_risks(plan)
        if risk_lines:
            sections.append("### Known Risks\n" + "\n".join(risk_lines))

        return "\n\n".join(section for section in sections if section).strip()

    def _render_task_section(
        self,
        tasks: Sequence[Task],
        *,
        heading: str,
        statuses: set[TaskStatus],
        limit: int,
    ) -> str:
        entries: list[str] = []
        for task in tasks:
            if task.status not in statuses:
                continue
            entries.append(self._summarize_task(task))
            if len(entries) >= limit:
                break
        if not entries:
            return ""
        return f"{heading}\n" + "\n".join(entries)

    def _summarize_task(self, task: Task) -> str:
        planner_task = self._load_planner_task(task)
        status_label = task.status.value.lower()
        summary_bits: list[str] = []
        summary_text = self._summarize_text(planner_task.summary or task.summary, width=160)
        if summary_text:
            summary_bits.append(summary_text)
        deliverables = self._coerce_string_collection(planner_task.deliverables)
        if deliverables:
            summary_bits.append(f"Deliverables: {', '.join(deliverables[:3])}")
        acceptance = self._coerce_string_collection(planner_task.acceptance_criteria)
        if acceptance:
            summary_bits.append(f"Acceptance: {', '.join(acceptance[:3])}")
        depends = planner_task.depends_on or task.depends_on or []
        if depends:
            summary_bits.append(f"Depends on: {', '.join(depends[:3])}")
        details = f" — {'; '.join(summary_bits)}" if summary_bits else ""
        return f"- {task.title} [{status_label}]{details}"

    def _collect_diagnose_followups(self, tasks: Sequence[Task]) -> list[str]:
        lines: list[str] = []
        for task in tasks:
            metadata = task.metadata or {}
            if metadata.get("source") != "diagnose_followup":
                continue
            signature = metadata.get("diagnose_signature")
            summary = self._summarize_text(task.summary, width=160)
            status = task.status.value.lower()
            line = f"- {summary or task.title} [{status}]"
            if signature:
                line = f"{line} (signature: {signature})"
            lines.append(line)
        return lines[:10]

    def _collect_decision_summaries(self, plan: Plan, limit: int = 8) -> list[str]:
        try:
            decisions = self._store.list_decisions(plan_id=plan.id)
        except Exception:
            return []
        lines: list[str] = []
        for decision in decisions:
            if decision.task_id:
                continue
            title = (decision.title or "").strip()
            summary = self._summarize_text(decision.content, width=180)
            if summary:
                lines.append(f"- {title or 'Decision'} — {summary}")
            elif title:
                lines.append(f"- {title}")
            if len(lines) >= limit:
                break
        return lines

    def _collect_known_risks(self, plan: Plan, limit: int = 8) -> list[str]:
        planner_meta = plan.metadata.get("planner") if isinstance(plan.metadata, Mapping) else {}
        collected: list[Any] = []
        if isinstance(planner_meta, Mapping):
            analysis = planner_meta.get("analysis")
            if isinstance(analysis, Mapping):
                analysis_risks = analysis.get("risks")
                if isinstance(analysis_risks, Sequence):
                    collected.extend(analysis_risks)
                elif isinstance(analysis_risks, Mapping):
                    collected.append(analysis_risks)
            general_risks = planner_meta.get("risks")
            if isinstance(general_risks, Sequence):
                collected.extend(general_risks)
            elif isinstance(general_risks, Mapping):
                collected.append(general_risks)
        elif isinstance(plan.metadata, Mapping):
            risks = plan.metadata.get("risks")
            if isinstance(risks, Sequence):
                collected.extend(risks)
            elif isinstance(risks, Mapping):
                collected.append(risks)

        lines: list[str] = []
        for entry in collected:
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
                for sub_entry in entry:
                    line = self._format_risk_entry(sub_entry)
                    if line:
                        lines.append(line)
                    if len(lines) >= limit:
                        return lines
                continue
            line = self._format_risk_entry(entry)
            if line:
                lines.append(line)
            if len(lines) >= limit:
                break
        return lines[:limit]

    def _format_risk_entry(self, entry: Any) -> str:
        if isinstance(entry, Mapping):
            description = str(entry.get("description") or entry.get("summary") or "").strip()
            impact = str(entry.get("impact") or "").strip()
            likelihood = str(entry.get("likelihood") or "").strip()
            extras: list[str] = []
            if impact:
                extras.append(f"impact: {impact}")
            if likelihood:
                extras.append(f"likelihood: {likelihood}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            return f"- {description}{suffix}" if description else ""
        if isinstance(entry, str):
            cleaned = entry.strip()
            return f"- {cleaned}" if cleaned else ""
        return ""

    def _compose_diff_goal(self, plan: Plan, planner_task: PlannerTask) -> str:
        summary = (planner_task.summary or "").strip()
        lines: list[str] = []
        if summary:
            lines.append(summary)
        goal = (plan.goal or "").strip()
        if goal and (not summary or goal.lower() not in summary.lower()):
            lines.append(f"Plan goal: {goal}")
        plan_summary = (plan.summary or "").strip()
        if plan_summary:
            dedupe_targets = {line.lower() for line in lines}
            if plan_summary.lower() not in dedupe_targets:
                lines.append(f"Plan summary: {plan_summary}")
        return "\n\n".join(lines).strip()

    def _assemble_file_hints(
        self,
        plan: Plan,
        planner_task: PlannerTask,
    ) -> tuple[list[str], list[str]]:
        base = self._dedupe_paths(self._extract_touched_files(planner_task))
        supplemental_candidates: list[str] = []
        extra_sources = (
            planner_task.deliverables or [],
            planner_task.acceptance_criteria or [],
            planner_task.notes or [],
        )
        for source in extra_sources:
            for entry in source:
                supplemental_candidates.extend(self._extract_paths_from_text(entry))

        metadata = planner_task.metadata or {}
        for key in ("additional_paths", "candidate_paths", "context_paths", "related_files"):
            value = metadata.get(key)
            if isinstance(value, Sequence) and not isinstance(value, str):
                for entry in value:
                    if isinstance(entry, str):
                        supplemental_candidates.extend(self._extract_paths_from_text(entry))

        supplemental = self._dedupe_paths(supplemental_candidates)
        analysis_paths = self._collect_analysis_primary_paths(plan)
        combined = self._merge_path_groups(base, supplemental, analysis_paths)
        if not combined:
            combined = analysis_paths or supplemental or base
        if not combined:
            combined = self._collect_repository_directories()

        related_candidates: list[str] = []
        for path in combined:
            related_candidates.extend(self._enumerate_parent_paths(path))
        for path in analysis_paths:
            related_candidates.append(path)
        related_candidates.extend(self._collect_repository_directories())
        related = self._dedupe_paths(related_candidates)
        return combined, related

    def _collect_failure_notes(self, plan: Plan, limit: int = 5) -> list[str]:
        try:
            runs = self._store.list_test_runs(plan_id=plan.id)
        except Exception:
            return []
        notes: list[str] = []
        seen: set[str] = set()
        for run in runs:
            if run.status != TestStatus.FAILED:
                continue
            identifier = f"{run.name}|{run.command}"
            if identifier in seen:
                continue
            seen.add(identifier)
            header = (run.name or "").strip()
            command = (run.command or "").strip()
            label = header or command or "pytest"
            summary = self._summarize_text(run.output)
            note = f"[TEST FAILURE] {label}"
            if command and command != label:
                note += f" (command: {command})"
            if summary:
                note += f" — {summary}"
            notes.append(note)
            if len(notes) >= limit:
                break
        return notes

    def _collect_incident_notes(self, plan: Plan, limit: int = 5) -> list[str]:
        try:
            incidents = self._store.list_incidents(plan_id=plan.id)
        except Exception:
            return []
        notes: list[str] = []
        for incident in incidents:
            if incident.severity not in {IncidentSeverity.ERROR, IncidentSeverity.CRITICAL}:
                continue
            summary = (incident.summary or "").strip()
            detail = self._summarize_text(incident.details)
            note = f"[INCIDENT:{incident.severity.value}] {summary}"
            if detail:
                note += f" — {detail}"
            notes.append(note)
            if len(notes) >= limit:
                break
        return notes

    def _collect_decision_notes(self, plan: Plan, task: Task, limit: int = 5) -> list[str]:
        try:
            decisions = self._store.list_decisions(plan_id=plan.id)
        except Exception:
            return []
        direct: list[str] = []
        general: list[str] = []
        for decision in decisions:
            content = self._summarize_text(decision.content, width=180)
            title = (decision.title or "").strip()
            if title and content:
                note = f"{title}: {content}"
            else:
                note = title or content
            if not note:
                continue
            if decision.task_id and decision.task_id == task.id:
                direct.append(note)
            elif not decision.task_id:
                general.append(note)
        ordered = direct + [entry for entry in general if entry not in direct]
        return ordered[:limit]

    def _extract_metadata_notes(self, metadata: Mapping[str, Any]) -> list[str]:
        if not metadata:
            return []
        notes: list[str] = []
        candidate_keys = ("background", "context", "rationale", "risks")
        for key in candidate_keys:
            value = metadata.get(key)
            if isinstance(value, str):
                summary = self._summarize_text(value, width=180)
                if summary:
                    notes.append(f"{key.capitalize()}: {summary}")
            elif isinstance(value, Sequence) and not isinstance(value, str):
                fragments: list[str] = []
                for entry in value:
                    if isinstance(entry, str):
                        summary = self._summarize_text(entry, width=100)
                        if summary:
                            fragments.append(summary)
                    if len(fragments) >= 3:
                        break
                if fragments:
                    notes.append(f"{key.capitalize()}: {', '.join(fragments)}")
        return notes[:5]

    def _build_implement_context(
        self,
        *,
        plan: Plan,
        planner_task: PlannerTask,
        decision_notes: Sequence[str],
        related_files: Sequence[str],
        metadata_notes: Sequence[str],
    ) -> str:
        sections: list[str] = []
        goal = (plan.goal or "").strip()
        plan_summary = (plan.summary or "").strip()
        task_title = (planner_task.title or "").strip()

        overview_lines: list[str] = []
        if task_title:
            overview_lines.append(f"- Task title: {task_title}")
        if goal:
            overview_lines.append(f"- Plan goal: {goal}")
        if plan_summary:
            summary_line = self._summarize_text(plan_summary, width=200)
            if summary_line:
                overview_lines.append(f"- Plan summary: {summary_line}")
        if overview_lines:
            sections.append("### Plan Overview\n" + "\n".join(overview_lines))

        if metadata_notes:
            sections.append("### Planner Notes\n" + "\n".join(f"- {note}" for note in metadata_notes))

        if decision_notes:
            sections.append("### Key Decisions\n" + "\n".join(f"- {note}" for note in decision_notes))

        analysis_paths = self._collect_analysis_primary_paths(plan)
        if analysis_paths:
            sections.append(
                "### Analysis Primary Paths\n" + "\n".join(f"- {path}" for path in analysis_paths[:12])
            )

        if related_files:
            sections.append("### Additional Relevant Paths\n" + "\n".join(f"- {path}" for path in related_files[:15]))

        repo_outline = self._collect_repo_tree_lines(max_depth=3, max_lines=60)
        if repo_outline:
            sections.append("### Repository Snapshot\n" + "\n".join(repo_outline))

        return "\n\n".join(section for section in sections if section).strip()

    def _collect_analysis_primary_paths(self, plan: Plan, limit: int = 30) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        for component in self._plan_analysis_components(plan):
            primary_paths = component.get("primary_paths") or []
            for entry in primary_paths:
                candidate = self._normalise_candidate_path(entry)
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                paths.append(candidate)
                if len(paths) >= limit:
                    return paths
        return paths

    def _collect_repository_directories(self, limit: int = 12) -> list[str]:
        root = getattr(self, "_repo_root", None)
        if not isinstance(root, Path):
            return []
        try:
            entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return []
        excluded = {
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            ".idea",
            ".vscode",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            "dist",
            "build",
            "tmp",
            "logs",
        }
        directories: list[str] = []
        for entry in entries:
            if not entry.is_dir():
                continue
            name = entry.name
            if name in excluded:
                continue
            if name.startswith(".") and name not in {".github"}:
                continue
            directories.append(f"{name}/")
            if len(directories) >= limit:
                break
        return directories

    def _enumerate_parent_paths(self, path: str) -> list[str]:
        normalised = self._normalise_candidate_path(path)
        if not normalised:
            return []
        candidate = Path(normalised)
        parents: list[str] = []
        for parent in reversed(candidate.parents):
            if not parent.parts:
                continue
            parent_key = parent.as_posix()
            if parent_key in {".", ""}:
                continue
            if not parent_key.endswith("/"):
                parent_key = f"{parent_key}/"
            parents.append(parent_key)
        return parents

    def _merge_path_groups(self, *groups: Sequence[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for entry in group or []:
                candidate = self._normalise_candidate_path(entry)
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                merged.append(candidate)
        return merged

    def _dedupe_paths(self, candidates: Sequence[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for entry in candidates or []:
            candidate = self._normalise_candidate_path(entry)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _summarize_text(value: str | None, *, width: int = 160) -> str:
        if not value:
            return ""
        text = value.strip()
        if not text:
            return ""
        first_line = text.splitlines()[0].strip()
        if not first_line:
            return ""
        return textwrap.shorten(first_line, width=width, placeholder="…")

    def _seed_diagnose_followups(
        self,
        plan: Plan,
        task: Task,
        result: CodingIterationResult,
    ) -> list[Task]:
        if result.failure_category in {
            FailureCategory.NO_PATCH,
            FailureCategory.DIAGNOSE_NO_PATCH,
            FailureCategory.SAFETY_STOP,
        }:
            return []
        diagnose = getattr(result, "diagnose", None)
        recommendations = list(getattr(diagnose, "recommended_fixes", []) or [])
        recommendations.extend(result.errors or [])
        cleaned: list[str] = []
        for entry in recommendations:
            if not isinstance(entry, str):
                continue
            trimmed = entry.strip()
            if trimmed and trimmed not in cleaned:
                cleaned.append(trimmed)
        if not cleaned:
            return []

        try:
            existing_tasks = self._store.list_tasks(plan_id=plan.id)
        except Exception:
            return []

        existing_signatures: set[str] = set()
        for existing in existing_tasks:
            metadata = existing.metadata or {}
            if metadata.get("source") != "diagnose_followup":
                continue
            signature = str(metadata.get("diagnose_signature") or "").strip().lower()
            if signature:
                existing_signatures.add(signature)
                continue
            fallback = (existing.summary or existing.title or "").strip().lower()
            if fallback:
                existing_signatures.add(fallback)

        created: list[Task] = []
        for entry in cleaned:
            signature = self._normalise_followup_signature(entry)
            if signature in existing_signatures:
                continue
            existing_signatures.add(signature)
            task_id = self._generate_task_id(plan.id, entry)
            title = self._summarize_text(entry, width=96) or "Address diagnose finding"
            metadata = {
                "source": "diagnose_followup",
                "origin_task_id": task.id,
                "diagnose_signature": signature,
                "recommendation": entry,
            }
            followup = Task(
                id=task_id,
                plan_id=plan.id,
                title=title,
                summary=entry,
                status=TaskStatus.READY,
                metadata=metadata,
                depends_on=[],
                priority=max(task.priority - 1, 0),
            )
            self._store.save_task(followup)
            created.append(followup)

        if created:
            current_depends = list(task.depends_on or [])
            for followup in created:
                if followup.id not in current_depends:
                    current_depends.append(followup.id)
            updated_metadata = dict(task.metadata or {})
            updated_metadata["diagnose_followups"] = [
                {"task_id": followup.id, "summary": followup.summary} for followup in created
            ]
            updated_task = task.model_copy(update={"depends_on": current_depends, "metadata": updated_metadata})
            self._store.save_task(updated_task)
        return created

    @staticmethod
    def _normalise_followup_signature(entry: str) -> str:
        lowered = entry.strip().lower()
        return re.sub(r"\s+", " ", lowered)

    @staticmethod
    def _task_signature(value: str | None) -> str:
        if not isinstance(value, str):
            return ""
        cleaned = value.strip().lower()
        if not cleaned:
            return ""
        cleaned = re.sub(r"[^\w\s./:-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _handle_iteration_result(
        self,
        plan: Plan,
        task: Task,
        result: CodingIterationResult,
    ) -> tuple[PlanAdjustResponse | None, list[Task]]:
        auto_review = self._is_auto_review_task(task)
        iteration_success = result.ok
        ignored_errors: set[str] = set()
        if not iteration_success and auto_review:
            iteration_success, ignored_errors = self._auto_review_iteration_success(result)
            if iteration_success and ignored_errors:
                result.errors = [error for error in result.errors if error not in ignored_errors]

        if iteration_success:
            self._store.update_task_status(task.id, TaskStatus.DONE)
            followups: list[Task] = []
            if auto_review:
                followups = self._spawn_review_followups(plan, task, result)
                if not followups:
                    active_plan = self._refresh_plan(plan.id) or plan
                    self._maybe_schedule_readme_iteration(active_plan)
            self._unlock_blocked_tasks(plan.id)
            return None, followups

        failure_category = result.failure_category
        terminal_categories = {
            FailureCategory.NO_PATCH,
            FailureCategory.DIAGNOSE_NO_PATCH,
            FailureCategory.SAFETY_STOP,
        }
        terminal_failure = failure_category in terminal_categories
        next_status = TaskStatus.FAILED if terminal_failure else TaskStatus.BLOCKED

        self._store.update_task_status(task.id, next_status)
        if terminal_failure:
            self._unlock_blocked_tasks(plan.id)
            return None, []

        self._seed_diagnose_followups(plan, task, result)
        adjustment = self._request_plan_adjustment(plan, task, result)
        followups = self._materialize_followups(plan, task, adjustment)
        self._unlock_blocked_tasks(plan.id)
        return adjustment, followups

    def _auto_review_iteration_success(self, result: CodingIterationResult) -> tuple[bool, set[str]]:
        """Handle code-review iterations that intentionally produce no repository edits."""

        implement = result.implement
        if implement is None:
            return False, set()

        patch = result.patch
        tolerated_errors: set[str] = set()
        if patch.applied or patch.no_op_reason:
            result.failure_category = None
            return True, tolerated_errors

        error_message = (patch.error or "").strip()
        if error_message == "Implement response did not include any updates.":
            tolerated_errors.add(error_message)
            result.failure_category = None
            return True, tolerated_errors

        return False, set()

    def _request_plan_adjustment(
        self,
        plan: Plan,
        task: Task,
        result: CodingIterationResult,
    ) -> PlanAdjustResponse | None:
        request = build_plan_adjust_request(plan.id, task.id, result)
        return self._orchestrator.run_phase(PhaseName.PLAN_ADJUST, request)

    def _materialize_followups(
        self,
        plan: Plan,
        task: Task,
        adjustment: PlanAdjustResponse | None,
    ) -> list[Task]:
        if adjustment is None:
            return []
        removed, skipped, unresolved = self._apply_plan_adjust_task_drops(plan, adjustment)
        filtered_new_tasks: list[str] = []
        ignored_new_tasks: list[str] = []
        for entry in adjustment.new_tasks:
            if not isinstance(entry, str):
                continue
            cleaned = entry.strip()
            if not cleaned:
                continue
            if looks_like_test_execution(cleaned):
                ignored_new_tasks.append(cleaned)
                continue
            filtered_new_tasks.append(cleaned)
        existing_tasks = self._store.list_tasks(plan_id=plan.id)
        global_signatures: set[str] = set()
        for existing in existing_tasks:
            if existing.status == TaskStatus.DONE:
                continue
            for candidate in (existing.title, existing.summary):
                signature = self._task_signature(candidate)
                if signature:
                    global_signatures.add(signature)

        pending_chain: list[str] = []
        pending_signatures: set[str] = set()
        for existing in existing_tasks:
            metadata = existing.metadata or {}
            if metadata.get("origin_task_id") != task.id:
                continue
            if existing.status == TaskStatus.DONE:
                continue
            pending_chain.append(existing.id)
            for candidate in (existing.title, existing.summary):
                signature = self._task_signature(candidate)
                if signature:
                    pending_signatures.add(signature)

        duplicate_new_tasks: list[str] = []
        capped_new_tasks: list[str] = []
        planned_followups: list[tuple[str, str, list[str], dict[str, Any], str]] = []
        base_priority = max(task.priority - 1, 0)
        for entry in filtered_new_tasks:
            signature = self._task_signature(entry)
            if signature and (signature in pending_signatures or signature in global_signatures):
                duplicate_new_tasks.append(entry)
                continue
            if len(pending_chain) >= self._MAX_PENDING_PLAN_ADJUST_FOLLOWUPS:
                capped_new_tasks.append(entry)
                continue
            task_id = self._generate_task_id(plan.id, entry)
            depends_on = list(dict.fromkeys(pending_chain))
            metadata: dict[str, Any] = {
                "source": "plan_adjust",
                "origin_task_id": task.id,
            }
            planned_followups.append((task_id, entry, depends_on, metadata, signature))
            pending_chain.append(task_id)
            if signature:
                pending_signatures.add(signature)
                global_signatures.add(signature)

        decision = self._record_plan_adjust_decision(
            plan,
            task,
            adjustment,
            removed_tasks=removed,
            skipped_drop_requests=skipped,
            unresolved_drop_requests=unresolved,
            ignored_new_tasks=ignored_new_tasks,
            duplicate_new_tasks=duplicate_new_tasks,
            capped_new_tasks=capped_new_tasks,
        )

        followups: list[Task] = []
        for task_id, entry, depends_on, metadata, signature in planned_followups:
            enriched_metadata = dict(metadata)
            if decision:
                enriched_metadata["plan_adjust_decision_id"] = decision.id
            status = TaskStatus.BLOCKED if depends_on else TaskStatus.READY
            followup = Task(
                id=task_id,
                plan_id=plan.id,
                title=entry or f"Follow up on {task.id}",
                summary=entry,
                status=status,
                metadata=enriched_metadata,
                depends_on=depends_on,
                priority=base_priority,
            )
            self._store.save_task(followup)
            followups.append(followup)
        return followups

    def _record_plan_adjust_decision(
        self,
        plan: Plan,
        task: Task,
        adjustment: PlanAdjustResponse,
        *,
        removed_tasks: Sequence[str] | None = None,
        skipped_drop_requests: Sequence[str] | None = None,
        unresolved_drop_requests: Sequence[str] | None = None,
        ignored_new_tasks: Sequence[str] | None = None,
        duplicate_new_tasks: Sequence[str] | None = None,
        capped_new_tasks: Sequence[str] | None = None,
    ) -> Decision | None:
        content_parts: list[str] = []
        if adjustment.adjustments:
            content_parts.append("Adjustments:")
            content_parts.extend(f"- {self._render_adjustment(item)}" for item in adjustment.adjustments)
        if removed_tasks:
            content_parts.append("")
            content_parts.append("Removed tasks:")
            content_parts.extend(f"- {item}" for item in removed_tasks)
        if skipped_drop_requests:
            content_parts.append("")
            content_parts.append("Skipped removals (already DONE):")
            content_parts.extend(f"- {item}" for item in skipped_drop_requests)
        if unresolved_drop_requests:
            content_parts.append("")
            content_parts.append("Unresolved removal references:")
            content_parts.extend(f"- {item}" for item in unresolved_drop_requests)
        if ignored_new_tasks:
            content_parts.append("")
            content_parts.append("Ignored new task requests (test execution only):")
            content_parts.extend(f"- {item}" for item in ignored_new_tasks)
        if duplicate_new_tasks:
            content_parts.append("")
            content_parts.append("Skipped duplicate new tasks:")
            content_parts.extend(f"- {item}" for item in duplicate_new_tasks)
        if capped_new_tasks:
            content_parts.append("")
            content_parts.append("Skipped new tasks due to pending backlog limit:")
            content_parts.extend(f"- {item}" for item in capped_new_tasks)
        if adjustment.risks:
            content_parts.append("")
            content_parts.append("Risks:")
            content_parts.extend(f"- {item}" for item in adjustment.risks)
        if adjustment.notes:
            content_parts.append("")
            content_parts.append("Notes:")
            content_parts.extend(f"- {item}" for item in adjustment.notes)
        if not content_parts:
            return None

        decision = Decision(
            id=self._generate_decision_id(plan.id),
            plan_id=plan.id,
            task_id=task.id,
            title=f"Plan adjustment for {task.id}",
            content="\n".join(content_parts),
            kind="plan_adjust",
            metadata={
                "plan_adjust_response": asdict(adjustment),
                "task_drop_summary": {
                    "removed": list(removed_tasks or []),
                    "skipped": list(skipped_drop_requests or []),
                    "unresolved": list(unresolved_drop_requests or []),
                },
                "ignored_new_tasks": list(ignored_new_tasks or []),
                "duplicate_new_tasks": list(duplicate_new_tasks or []),
                "capped_new_tasks": list(capped_new_tasks or []),
            },
        )
        self._store.record_decision(decision)
        return decision

    @staticmethod
    def _render_adjustment(item: PlanAdjustment) -> str:
        if isinstance(item, PlanAdjustmentItem):
            return item.render()
        return str(item)

    def _apply_plan_adjust_task_drops(
        self,
        plan: Plan,
        adjustment: PlanAdjustResponse,
    ) -> tuple[list[str], list[str], list[str]]:
        drop_requests = getattr(adjustment, "drop_tasks", None)
        if not drop_requests:
            return [], [], []

        existing_tasks = self._store.list_tasks(plan_id=plan.id)
        if not existing_tasks:
            return [], [], []

        by_id: dict[str, Task] = {task.id: task for task in existing_tasks}
        alias_map: dict[str, Task] = {}
        for task in existing_tasks:
            alias_map.setdefault(task.id.lower(), task)
            title = (task.title or "").strip().lower()
            if title:
                alias_map.setdefault(title, task)

        removed: list[str] = []
        removed_seen: set[str] = set()
        skipped: list[str] = []
        skipped_seen: set[str] = set()
        unresolved: list[str] = []
        unresolved_seen: set[str] = set()

        for reference in drop_requests:
            if not isinstance(reference, str):
                continue
            cleaned = reference.strip()
            if not cleaned:
                continue
            task = by_id.get(cleaned)
            if task is None:
                task = alias_map.get(cleaned.lower())
            if task is None:
                lowered = cleaned.lower()
                if lowered not in unresolved_seen:
                    unresolved.append(cleaned)
                    unresolved_seen.add(lowered)
                continue
            if task.status == TaskStatus.DONE:
                if task.id not in skipped_seen:
                    skipped.append(task.id)
                    skipped_seen.add(task.id)
                continue
            if task.id in removed_seen:
                continue
            removed.append(task.id)
            removed_seen.add(task.id)

        if not removed:
            return [], skipped, unresolved

        self._store.delete_tasks(removed)
        self._remove_dependencies_for_deleted_tasks(plan.id, removed)
        return removed, skipped, unresolved

    def _remove_dependencies_for_deleted_tasks(self, plan_id: str, removed_ids: Sequence[str]) -> None:
        if not removed_ids:
            return
        remaining_tasks = self._store.list_tasks(plan_id=plan_id)
        removed_set = set(removed_ids)
        for task in remaining_tasks:
            depends_on = list(task.depends_on or [])
            if not depends_on:
                continue
            filtered = [dependency for dependency in depends_on if dependency not in removed_set]
            if len(filtered) == len(depends_on):
                continue
            updated = task.model_copy(update={"depends_on": filtered})
            self._store.save_task(updated)

    def _unlock_blocked_tasks(self, plan_id: str) -> None:
        tasks = self._store.list_tasks(plan_id=plan_id)
        done: set[str] = {task.id for task in tasks if task.status == TaskStatus.DONE}
        for task in tasks:
            if task.status != TaskStatus.BLOCKED:
                continue
            if all(dep in done for dep in task.depends_on):
                self._store.update_task_status(task.id, TaskStatus.READY)

    def _mark_plan_if_completed(self, plan: Plan) -> tuple[Plan, bool]:
        tasks = self._store.list_tasks(plan_id=plan.id)
        if tasks and all(task.status == TaskStatus.DONE for task in tasks):
            if plan.status != PlanStatus.COMPLETED:
                updated = plan.model_copy(update={"status": PlanStatus.COMPLETED})
                self._store.create_plan(updated)
                return updated, True
            return plan, True
        return plan, False

    # ------------------------------------------------------- plan state helpers
    def _get_plan_metadata(self, plan: Plan) -> dict[str, Any]:
        metadata = plan.metadata or {}
        if isinstance(metadata, Mapping):
            return dict(metadata)
        return dict(metadata)

    def _update_plan_metadata(self, plan: Plan, mutator: Callable[[dict[str, Any]], None]) -> Plan:
        metadata = self._get_plan_metadata(plan)
        mutator(metadata)
        updated = plan.model_copy(update={"metadata": metadata})
        self._store.create_plan(updated)
        return updated

    def _final_iteration_mode_active(self, plan: Plan) -> bool:
        metadata = plan.metadata or {}
        final_mode = metadata.get("final_iteration_mode")
        if isinstance(final_mode, Mapping):
            return bool(final_mode.get("active"))
        if isinstance(final_mode, bool):
            return bool(final_mode)
        return False

    def _final_iteration_notes(self, plan: Plan) -> list[str]:
        if not self._final_iteration_mode_active(plan):
            return []
        metadata = plan.metadata or {}
        final_mode = metadata.get("final_iteration_mode") or {}
        activated_iter = final_mode.get("activated_at_iteration") or self._FINAL_ITERATION_THRESHOLD
        return [
            f"Final push mode is active after {activated_iter} iterations; deliver a working baseline users can run today.",
            "Defer cosmetic clean-up, TODOs, and low-impact refactors unless they block core functionality.",
            "Focus on ensuring primary workflows operate end-to-end and communicate any remaining gaps clearly.",
        ]

    def _get_auto_review_state(self, plan: Plan) -> dict[str, Any]:
        metadata = plan.metadata or {}
        state = metadata.get("auto_review_state")
        if isinstance(state, Mapping):
            return dict(state)
        return {}

    def _auto_review_can_create_tasks(self, plan: Plan) -> bool:
        if self._final_iteration_mode_active(plan):
            return False
        state = self._get_auto_review_state(plan)
        rounds = int(state.get("followup_plan_rounds", 0))
        return rounds < self._AUTO_REVIEW_PLAN_LIMIT

    def _record_auto_review_plan_round(
        self,
        plan: Plan,
        *,
        followup_count: int,
        digests: Sequence[str],
    ) -> None:
        timestamp = utc_now().isoformat()

        def _mutator(metadata: dict[str, Any]) -> None:
            state = dict(metadata.get("auto_review_state") or {})
            state["followup_plan_rounds"] = int(state.get("followup_plan_rounds", 0)) + 1
            state["last_round_at"] = timestamp
            state["last_followup_count"] = followup_count
            state["last_round_digests"] = list(digests)
            metadata["auto_review_state"] = state

        self._update_plan_metadata(plan, _mutator)

    def _record_auto_review_suppressed(
        self,
        plan: Plan,
        *,
        reason: str,
        digests: Sequence[str],
    ) -> None:
        timestamp = utc_now().isoformat()

        def _mutator(metadata: dict[str, Any]) -> None:
            state = dict(metadata.get("auto_review_state") or {})
            state["suppressed_rounds"] = int(state.get("suppressed_rounds", 0)) + 1
            state["last_suppressed_reason"] = reason
            state["last_suppressed_at"] = timestamp
            state["last_suppressed_digests"] = list(digests)
            metadata["auto_review_state"] = state

        self._update_plan_metadata(plan, _mutator)

    def _record_suppressed_review_followups(
        self,
        plan: Plan,
        review_task: Task,
        entries: Sequence[tuple[str, str]],
        *,
        reason: str,
    ) -> None:
        if not entries:
            return
        lines = [
            f"Auto code review follow-ups were suppressed ({reason}).",
            "Outstanding findings:",
        ]
        for entry, digest in entries:
            lines.append(f"- [{digest}] {entry}")
        decision = Decision(
            id=self._generate_decision_id(plan.id),
            plan_id=plan.id,
            task_id=review_task.id,
            title=f"Suppressed auto review follow-ups ({review_task.id})",
            content="\n".join(lines),
            kind="auto_review_followup_suppressed",
            metadata={
                "reason": reason,
                "digests": [digest for _, digest in entries],
            },
        )
        self._store.record_decision(decision)

    def _record_iteration_metrics(self, plan: Plan, task: Task, result: CodingIterationResult) -> Plan:
        timestamp = utc_now().isoformat()
        errors = list(result.errors or [])
        iteration_success = bool(result.ok)

        def _mutator(metadata: dict[str, Any]) -> None:
            metrics = dict(metadata.get("iteration_metrics") or {})
            total_iterations = int(metrics.get("total_iterations", 0)) + 1
            metrics["total_iterations"] = total_iterations
            metrics["last_iteration_at"] = timestamp
            metrics["last_task_id"] = task.id
            metrics["last_task_title"] = task.title
            metrics["last_result_ok"] = iteration_success
            if iteration_success:
                metrics["successful_iterations"] = int(metrics.get("successful_iterations", 0)) + 1
                metrics["last_success_at"] = timestamp
            else:
                metrics["failed_iterations"] = int(metrics.get("failed_iterations", 0)) + 1
                metrics["last_failure_at"] = timestamp
                if errors:
                    metrics["last_failure_reason"] = errors[-1]
            metadata["iteration_metrics"] = metrics

            final_mode = dict(metadata.get("final_iteration_mode") or {})
            if total_iterations >= self._FINAL_ITERATION_THRESHOLD:
                if not final_mode.get("active"):
                    final_mode["active"] = True
                    final_mode["reason"] = "iteration_limit"
                    final_mode["activated_at_iteration"] = total_iterations
                    final_mode["activated_at"] = timestamp
                metadata["final_iteration_mode"] = final_mode
            elif "final_iteration_mode" in metadata:
                metadata["final_iteration_mode"] = final_mode

        return self._update_plan_metadata(plan, _mutator)

    # --------------------------------------------------------------- utilities
    def _load_planner_task(self, task: Task) -> PlannerTask:
        payload = task.metadata.get("planner_task")
        if payload:
            try:
                return PlannerTask.model_validate(payload)
            except ValidationError:
                pass
        return PlannerTask(
            id=task.id,
            title=task.title,
            summary=task.summary,
        )

    def _create_code_indexer(self) -> CodeIndexer | None:
        try:
            if self._config:
                return CodeIndexer.from_config(self._config, self._config_path)
            data_root = (self._repo_root / "data").resolve()
            return CodeIndexer(repo_root=self._repo_root, data_root=data_root)
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.warning("Failed to initialise code indexer: %s", error)
            return None

    def _refresh_code_index(self) -> None:
        if self._code_indexer is None:
            return
        self._ensure_code_indexer()
        try:
            updated = self._code_indexer.reindex()
            if updated:
                LOGGER.debug("Refreshed code index for %d file(s)", len(updated))
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.warning("Failed to refresh code index: %s", error)

    @staticmethod
    def _extract_touched_files(planner_task: PlannerTask) -> list[str]:
        metadata = planner_task.metadata or {}
        candidate = metadata.get("touched_files")
        paths: list[str] = []
        if isinstance(candidate, Sequence):
            for item in candidate:
                if not isinstance(item, str):
                    continue
                paths.extend(PlanExecutor._extract_paths_from_text(item))
        if paths:
            seen: set[str] = set()
            deduped: list[str] = []
            for path in paths:
                if path not in seen:
                    seen.add(path)
                    deduped.append(path)
            return deduped

        inferred: list[str] = []
        for entry in planner_task.deliverables or []:
            if not isinstance(entry, str):
                continue
            inferred.extend(PlanExecutor._extract_paths_from_text(entry))

        seen: set[str] = set()
        deduped: list[str] = []
        for path in inferred:
            if path not in seen:
                seen.add(path)
                deduped.append(path)
        return deduped

    @staticmethod
    def _derive_proposed_interfaces(planner_task: PlannerTask) -> list[str]:
        deliverables = PlanExecutor._coerce_string_collection(planner_task.deliverables)
        optional = PlanExecutor._coerce_string_collection(planner_task.optional_deliverables)
        metadata = planner_task.metadata or {}
        prioritised_value = PlanExecutor._lookup_metadata_value(
            metadata,
            ("prioritised_deliverables", "prioritized_deliverables", "deliverable_priorities", "ordered_deliverables"),
        )
        prioritised = PlanExecutor._coerce_string_collection(prioritised_value)

        ordered: list[str] = []
        seen: set[str] = set()

        def _add_entries(entries: Sequence[str]) -> None:
            for entry in entries:
                if not entry or entry in seen:
                    continue
                seen.add(entry)
                ordered.append(entry)

        if prioritised:
            _add_entries(prioritised)
        _add_entries(deliverables)
        _add_entries(optional)

        if not ordered and isinstance(planner_task.title, str) and planner_task.title.strip():
            ordered.append(planner_task.title.strip())

        return ordered

    @staticmethod
    def _coerce_string_collection(value: Any) -> list[str]:
        if not value:
            return []

        results: list[str] = []
        dedup: set[str] = set()
        visited: set[int] = set()

        def _append(text: str | None) -> None:
            if not text:
                return
            cleaned = text.strip()
            if not cleaned or cleaned in dedup:
                return
            dedup.add(cleaned)
            results.append(cleaned)

        def _traverse(candidate: Any) -> None:
            if candidate is None:
                return
            if isinstance(candidate, str):
                _append(candidate)
                return
            if isinstance(candidate, (bytes, bytearray)):
                try:
                    decoded = candidate.decode("utf-8", errors="ignore")
                except Exception:
                    return
                _append(decoded)
                return

            candidate_id = id(candidate)
            if candidate_id in visited:
                return
            visited.add(candidate_id)

            if isinstance(candidate, Mapping):
                preferred_keys = (
                    "ordered",
                    "prioritised",
                    "prioritized",
                    "deliverables",
                    "items",
                    "values",
                    "primary",
                    "secondary",
                    "required",
                    "optional",
                    "list",
                )
                for key in preferred_keys:
                    if key in candidate:
                        _traverse(candidate.get(key))
                for value in candidate.values():
                    _traverse(value)
                return

            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                for item in candidate:
                    if isinstance(item, Mapping):
                        name = item.get("name") or item.get("title") or item.get("deliverable") or item.get("value")
                        _traverse(name)
                        for key in ("items", "values"):
                            if key in item:
                                _traverse(item.get(key))
                        continue
                    _traverse(item)
                return

        _traverse(value)
        return results

    @staticmethod
    def _lookup_metadata_value(metadata: Mapping[str, Any] | None, keys: Sequence[str]) -> Any:
        if not metadata or not keys:
            return None

        stack: list[Any] = [metadata]
        visited: set[int] = set()

        while stack:
            current = stack.pop()
            if isinstance(current, Mapping):
                current_id = id(current)
                if current_id in visited:
                    continue
                visited.add(current_id)
                for key in keys:
                    if key in current:
                        return current[key]
                for value in current.values():
                    if isinstance(value, Mapping):
                        stack.append(value)
                    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                        stack.extend(item for item in value if isinstance(item, Mapping))
            elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
                for item in current:
                    if isinstance(item, Mapping):
                        stack.append(item)
        return None

    @staticmethod
    def _requires_structured_edits(planner_task: PlannerTask) -> bool:
        metadata = planner_task.metadata or {}
        value = PlanExecutor._lookup_metadata_value(
            metadata,
            ("structured_edits_only", "structured_only", "require_structured_edits"),
        )
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "required", "1", "y"}:
                return True
            if lowered in {"false", "no", "optional", "0", "n"}:
                return False
            return bool(lowered)
        return bool(value)

    @staticmethod
    def _looks_like_file_path(value: str) -> bool:
        normalised = value.replace("\\", "/")
        if "/" in normalised:
            return True
        lowered = normalised.lower()
        return any(lowered.endswith(ext) for ext in _FILE_EXTENSIONS)

    @staticmethod
    def _is_likely_test_path(path: str) -> bool:
        normalised = path.replace("\\", "/").lower()
        if normalised.startswith("tests/") or "/tests/" in normalised:
            return True
        filename = normalised.rsplit("/", 1)[-1]
        return filename.startswith("test_") or filename.endswith("_test.py") or filename.endswith("_tests.py")

    @staticmethod
    def _strip_helper_suffix(value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            return ""
        # Trim trailing helper annotations such as "(may be extended)".
        while True:
            match = re.search(r"\s+\([^()]*\)$", cleaned)
            if not match:
                break
            cleaned = cleaned[: match.start()].rstrip()
        return cleaned

    @staticmethod
    def _extract_paths_from_text(value: str) -> list[str]:
        cleaned = PlanExecutor._strip_helper_suffix(value)
        if not cleaned:
            return []

        results: list[str] = []
        seen: set[str] = set()

        direct = PlanExecutor._normalise_candidate_path(cleaned)
        if direct and " " not in direct and PlanExecutor._looks_like_file_path(direct):
            seen.add(direct)
            results.append(direct)

        for fragment in _PATH_TOKEN_SPLIT_RE.split(cleaned):
            normalised = PlanExecutor._normalise_candidate_path(fragment)
            if not normalised or normalised in seen or " " in normalised:
                continue
            if PlanExecutor._looks_like_file_path(normalised):
                seen.add(normalised)
                results.append(normalised)
        return [path for path in results if not PlanExecutor._is_bare_package_init(path)]

    @staticmethod
    def _is_bare_package_init(path: str) -> bool:
        candidate = path.replace("\\", "/")
        return candidate.endswith("__init__.py") and "/" not in candidate

    @staticmethod
    def _normalise_candidate_path(value: str) -> str:
        if not isinstance(value, str):
            return ""
        normalised = value.strip().strip("\"'`“”‘’")
        if not normalised:
            return ""
        normalised = normalised.replace("\\", "/")
        normalised = normalised.lstrip("([{<")
        normalised = normalised.rstrip(".,;:!?)]}>")
        while normalised.startswith("./"):
            normalised = normalised[2:]
        if not normalised:
            return ""
        return normalised

    def _select_test_plan(self, planner_task: PlannerTask, plan: Plan) -> list[str]:
        commands: list[str] = []
        seen: set[str] = set()

        def add_command(raw: str | None) -> None:
            command = self._normalise_test_command(raw)
            if not command or command in seen:
                return
            seen.add(command)
            commands.append(command)

        metadata = planner_task.metadata or {}
        validation_commands = metadata.get("validation_commands")
        if isinstance(validation_commands, Sequence) and not isinstance(validation_commands, str):
            for entry in validation_commands:
                if isinstance(entry, str):
                    add_command(entry)

        touched_files = self._extract_touched_files(planner_task)
        if touched_files and self._coverage_map is not None:
            coverage_tests = sorted(self._coverage_map.affected_tests(touched_files))
            for nodeid in coverage_tests:
                add_command(f"pytest {nodeid}")

        related_tests = self._collect_plan_related_tests(plan, touched_files)
        for entry in related_tests:
            add_command(entry)

        if not commands:
            add_command("pytest -q")

        return commands

    def _update_coverage_map(self, planner_task: PlannerTask, result: CodingIterationResult) -> None:
        pytest_result = result.tests
        if self._coverage_map is None or pytest_result is None:
            return

        selectors = self._extract_pytest_selectors(pytest_result.command)
        if not selectors:
            return

        changed_paths = self._collect_changed_paths(planner_task, result)
        if not changed_paths:
            return

        for selector in selectors:
            self._coverage_map.record(selector, changed_paths)
        self._persist_coverage_map()

    def _collect_changed_paths(
        self,
        planner_task: PlannerTask,
        result: CodingIterationResult,
    ) -> set[str]:
        paths: set[str] = set()

        for path in self._extract_touched_files(planner_task):
            normalised = self._normalise_repo_path(path)
            if normalised:
                paths.add(normalised)

        patch_paths = getattr(result.patch, "touched_paths", None)
        if patch_paths:
            for path in patch_paths:
                normalised = self._normalise_repo_path(path)
                if normalised:
                    paths.add(normalised)

        if result.fix_violations:
            for path in result.fix_violations.touched_files or ():
                normalised = self._normalise_repo_path(path)
                if normalised:
                    paths.add(normalised)

        gate_suspects = getattr(result.gates, "suspect_files", None)
        if gate_suspects:
            for path in gate_suspects:
                normalised = self._normalise_repo_path(path)
                if normalised:
                    paths.add(normalised)

        for cycle in result.cycles or []:
            for adjustment in cycle.adjustments or []:
                for touched in adjustment.touched_paths or ():
                    normalised = self._normalise_repo_path(touched)
                    if normalised:
                        paths.add(normalised)

        return {path for path in paths if path}

    def _collect_plan_related_tests(self, plan: Plan, touched_files: Sequence[str]) -> list[str]:
        if not isinstance(plan.metadata, Mapping):
            return []

        touched_normalised = {entry.strip().lower() for entry in touched_files if isinstance(entry, str)}
        candidates: list[str] = []

        for component in self._plan_analysis_components(plan):
            primary_paths = component.get("primary_paths") or []
            if touched_normalised and primary_paths:
                component_paths = {
                    str(path).strip().lower()
                    for path in primary_paths
                    if isinstance(path, str) and path.strip()
                }
                if component_paths and component_paths.isdisjoint(touched_normalised):
                    continue
            for test_ref in component.get("related_tests") or []:
                if isinstance(test_ref, str) and test_ref.strip():
                    candidates.append(test_ref.strip())
        return candidates

    def _plan_analysis_components(self, plan: Plan) -> Sequence[Mapping[str, Any]]:
        if not isinstance(plan.metadata, Mapping):
            return []
        planner_meta = plan.metadata.get("planner")
        if not isinstance(planner_meta, Mapping):
            return []
        analysis = planner_meta.get("analysis")
        if isinstance(analysis, Mapping):
            components = analysis.get("components")
        elif isinstance(analysis, Sequence):
            components = analysis
        else:
            components = None
        if isinstance(components, Sequence):
            return [component for component in components if isinstance(component, Mapping)]
        return []

    def _extract_pytest_selectors(self, command: Sequence[str]) -> list[str]:
        if not command:
            return []
        selectors: list[str] = []
        skip_next = False
        for token in command[1:]:
            if skip_next:
                skip_next = False
                continue
            if token.startswith("--"):
                name, sep, value = token.partition("=")
                if name in _PYTEST_OPTIONS_EXPECT_VALUE and not sep:
                    skip_next = True
                continue
            if token.startswith("-"):
                if token in _PYTEST_OPTIONS_EXPECT_VALUE:
                    skip_next = True
                continue
            if token:
                selectors.append(token)
        return selectors

    def _resolve_coverage_map_path(self) -> Path:
        paths_section = self._config.get("paths")
        if isinstance(paths_section, Mapping):
            candidate = paths_section.get("coverage_map")
            if isinstance(candidate, str) and candidate.strip():
                path = Path(candidate.strip())
                if not path.is_absolute():
                    path = (self._repo_root / path).resolve()
                else:
                    path = path.resolve()
                return path
        return (self._repo_root / ".agentic-engineer" / "coverage-map.json").resolve()

    def _load_coverage_map(self) -> CoverageMap:
        if self._coverage_map_path is None:
            return CoverageMap()

        path = self._coverage_map_path
        try:
            return CoverageMap.load(path)
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load coverage map from %s: %s", path, error)
            return CoverageMap()

    def _persist_coverage_map(self) -> None:
        if self._coverage_map_path is None:
            return
        try:
            self._coverage_map.dump(self._coverage_map_path)
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.warning("Failed to persist coverage map to %s: %s", self._coverage_map_path, error)

    def _normalise_repo_path(self, path: str | Path) -> str | None:
        if path is None:
            return None
        text = str(path).strip()
        if not text:
            return None
        candidate = Path(text.replace("\\", "/"))
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve(strict=False)
            except OSError:
                resolved = candidate
            try:
                return resolved.relative_to(self._repo_root).as_posix()
            except ValueError:
                return resolved.as_posix()
        normalised = candidate.as_posix()
        while normalised.startswith("./"):
            normalised = normalised[2:]
        return normalised or None

    @staticmethod
    def _normalise_test_command(command: str | None) -> str | None:
        if not isinstance(command, str):
            return None
        cleaned = command.strip()
        if not cleaned:
            return None
        lowered = cleaned.lower()
        index = lowered.find("pytest")
        if index != -1:
            remainder = cleaned[index + len("pytest") :].strip()
            if remainder:
                return f"pytest {remainder}".strip()
            return "pytest"
        return f"pytest {cleaned}"

    @staticmethod
    def _extract_failing_tests(pytest_result) -> list[str]:
        return _extract_failing_tests(pytest_result)

    @staticmethod
    def _determine_repo_root(config: Mapping[str, Any] | None, config_path: Path | None) -> Path:
        """Resolve the repository root using configuration metadata when available."""

        repo_root_value = "."
        if isinstance(config, Mapping):
            project_section = config.get("project")
            if isinstance(project_section, Mapping):
                candidate = project_section.get("repo_root")
                if isinstance(candidate, str) and candidate.strip():
                    repo_root_value = candidate.strip()

        base_dir = config_path.parent if config_path is not None else Path.cwd()
        candidate_path = Path(repo_root_value)
        if not candidate_path.is_absolute():
            candidate_path = base_dir / candidate_path

        try:
            return candidate_path.resolve()
        except OSError:
            return candidate_path

    def _ensure_code_indexer(self) -> None:
        """Align the code indexer with the orchestrator's active repository workspace."""

        if not isinstance(self._code_indexer, CodeIndexer):
            return

        repo_root, data_root = self._current_repo_and_data_roots()

        if (
            self._index_repo_root is not None
            and self._index_data_root is not None
            and self._paths_equal(self._index_repo_root, repo_root)
            and self._paths_equal(self._index_data_root, data_root)
        ):
            return

        self._code_indexer = CodeIndexer(repo_root=repo_root, data_root=data_root)
        self._index_repo_root = repo_root
        self._index_data_root = data_root

    def _current_repo_and_data_roots(self) -> tuple[Path, Path]:
        """Return the repository/data roots that should drive code indexing."""

        repo_root = self._repo_root
        data_root = self._resolve_data_root_hint(repo_root)

        builder = getattr(self._orchestrator, "_context_builder", None)
        if builder is not None:
            try:
                repo_root = Path(builder.repo_root).resolve()
            except Exception:
                repo_root = repo_root.resolve()
            try:
                data_root = Path(builder.data_root).resolve()
            except Exception:
                data_root = self._resolve_data_root_hint(repo_root)
        else:
            repo_root = repo_root.resolve()
            data_root = data_root.resolve()

        return repo_root, data_root

    def _resolve_data_root_hint(self, repo_root: Path) -> Path:
        """Resolve the configured data root relative to the provided repository root."""

        paths_section = self._config.get("paths")
        if isinstance(paths_section, Mapping):
            candidate = paths_section.get("data")
            if isinstance(candidate, str) and candidate.strip():
                path = Path(candidate.strip())
                if not path.is_absolute():
                    path = (repo_root / path).resolve()
                else:
                    path = path.resolve()
                return path
        return (repo_root / "data").resolve()

    @staticmethod
    def _paths_equal(first: Path, second: Path) -> bool:
        """Return True if two paths refer to the same resolved location."""

        try:
            return first.resolve() == second.resolve()
        except OSError:
            return first == second

    def _generate_task_id(self, plan_id: str, seed: str) -> str:
        base = seed.strip() or uuid4().hex
        fallback_slug = uuid4().hex[:8]
        slug = slugify(base, fallback=fallback_slug, lowercase=True)
        identifier = f"{plan_id}::{slug}"
        counter = 2
        existing = {task.id for task in self._store.list_tasks(plan_id=plan_id)}
        while identifier in existing:
            identifier = f"{plan_id}::{slug}-{counter}"
            counter += 1
        return identifier

    def _generate_decision_id(self, plan_id: str) -> str:
        return f"{plan_id}::adj-{uuid4().hex[:8]}"

    # -------------------------------------------------------------- auto review
    def _maybe_schedule_code_review_iteration(self, plan: Plan) -> bool:
        if self._final_iteration_mode_active(plan):
            return False
        tasks = self._store.list_tasks(plan_id=plan.id)
        if not tasks:
            return False

        core_tasks = [
            task
            for task in tasks
            if not self._is_auto_review_task(task) and not self._is_auto_readme_task(task)
        ]
        if not core_tasks:
            return False
        if any(task.status != TaskStatus.DONE for task in core_tasks):
            return False

        review_tasks = [task for task in tasks if self._is_auto_review_task(task)]
        for review_task in review_tasks:
            if review_task.status in {TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.BLOCKED}:
                return False

        dependency_signature = self._dependency_signature(core_tasks)
        for review_task in review_tasks:
            if review_task.status != TaskStatus.DONE:
                continue
            if self._dependency_signature_from_ids(review_task.depends_on) == dependency_signature:
                return False

        review_task = self._create_code_review_task(plan, core_tasks)
        self._store.save_task(review_task)
        return True

    def _create_code_review_task(self, plan: Plan, completed_tasks: Sequence[Task]) -> Task:
        task_id = self._generate_task_id(plan.id, self._AUTO_REVIEW_TASK_SEED)
        summary = self._render_code_review_summary(plan, completed_tasks)
        touched_files = self._collect_recent_touched_files(completed_tasks)
        planner_metadata = {
            self._AUTO_METADATA_KEY: self._AUTO_REVIEW_MARKER,
        }
        if touched_files:
            planner_metadata["touched_files"] = touched_files

        planner_task = PlannerTask(
            id=task_id,
            title="Perform targeted code review",
            summary=summary,
            constraints=[
                "Investigate only findings that would block shipping: crashes, data loss, security issues, or major user confusion.",
                "Ignore stylistic nits, formatting, or optional cleanup unless they hide a functional bug.",
                "Do not modify repository files; limit the response to analysis only.",
                "For each substantive blocker, add a follow_up entry using the format 'Task Title :: Required fix'.",
            ],
            deliverables=["Actionable code review findings"],
            notes=[
                "Focus on correctness gaps, missing validation, data handling issues, and test omissions that break core workflows.",
                "Skip exhaustive edge-case coverage or performance micro-optimisations unless they cause observable failures.",
                "If no blocking issues remain, state that the implementation looks shippable and leave follow_up empty.",
            ],
            metadata={
                **planner_metadata,
            },
        )
        dependency_signature = self._dependency_signature(completed_tasks)
        metadata = {
            "planner_task": planner_task.model_dump(mode="python"),
            self._AUTO_METADATA_KEY: self._AUTO_REVIEW_MARKER,
            "source": "auto_code_review",
            "auto_review_dependency_signature": list(dependency_signature),
        }
        depends_on = [task.id for task in completed_tasks]
        base_priority = min((task.priority for task in completed_tasks), default=0)
        return Task(
            id=task_id,
            plan_id=plan.id,
            title=planner_task.title,
            summary=planner_task.summary,
            status=TaskStatus.READY,
            metadata=metadata,
            depends_on=depends_on,
            priority=base_priority - 1,
        )

    def _render_code_review_summary(self, plan: Plan, completed_tasks: Sequence[Task]) -> str:
        lines: list[str] = [
            "Perform a pragmatic code review over the completed tasks before finalising documentation.",
            "Focus on correctness, resiliency, security, and only those gaps that would break the product or materially confuse users.",
            "Skip exhaustive edge-case requests; highlight missing coverage only when it hides a likely failure.",
            "Surface actionable blockers and leave out cosmetic or low-impact polish.",
            "For each meaningful issue, record a follow_up entry as 'Task Title :: What must be fixed'.",
        ]

        structure_lines = self._collect_repo_tree_lines()
        if structure_lines:
            lines.append("Repository snapshot:")
            lines.extend(structure_lines)

        module_lines = self._collect_python_module_summaries(completed_tasks)
        if module_lines:
            lines.append("Key modules to audit:")
            lines.extend(module_lines)

        artifact_lines = self._collect_task_artifacts(completed_tasks)
        if artifact_lines:
            lines.append("Recent deliverables to scrutinise:")
            lines.extend(artifact_lines)

        return "\n".join(lines)

    def _spawn_review_followups(
        self,
        plan: Plan,
        task: Task,
        result: CodingIterationResult,
        ) -> list[Task]:
        implement = result.implement
        if implement is None:
            return []

        findings = self._extract_review_followups(implement)
        if not findings:
            return []

        existing_tasks = self._store.list_tasks(plan_id=plan.id)
        active_review_digests: set[str] = set()
        for existing in existing_tasks:
            metadata = existing.metadata or {}
            source = metadata.get("source")
            if source not in {
                "auto_code_review_followup",
                "auto_code_review_followup_plan",
            }:
                continue
            digests: list[str] = []
            single_digest = metadata.get("auto_review_finding_digest")
            if isinstance(single_digest, str) and single_digest:
                digests.append(single_digest)
            multiple_digests = metadata.get("auto_review_finding_digests")
            if isinstance(multiple_digests, Sequence):
                for value in multiple_digests:
                    if isinstance(value, str) and value:
                        digests.append(value)
            if not digests:
                continue
            if existing.status != TaskStatus.DONE:
                active_review_digests.update(digests)

        pending_entries: list[tuple[str, str]] = []
        for entry in findings:
            digest = hashlib.sha256(entry.strip().lower().encode("utf-8")).hexdigest()[:16]
            if digest in active_review_digests:
                continue
            pending_entries.append((entry, digest))

        if not pending_entries:
            return []

        allow_followups = self._auto_review_can_create_tasks(plan)
        if not allow_followups:
            reason = "final_iteration_mode" if self._final_iteration_mode_active(plan) else "followup_limit_reached"
            digests = [digest for _, digest in pending_entries]
            self._record_auto_review_suppressed(plan, reason=reason, digests=digests)
            self._record_suppressed_review_followups(plan, task, pending_entries, reason=reason)
            return []

        planned_followups = self._plan_review_followups(
            plan,
            task,
            result,
            pending_entries,
            existing_tasks=existing_tasks,
        )
        if planned_followups:
            return planned_followups
        if not self._auto_review_can_create_tasks(plan):
            return []

        followups: list[Task] = []
        for entry, digest in pending_entries:
            title, summary = self._parse_review_followup_entry(entry)
            seed = f"{self._AUTO_REVIEW_TASK_SEED}-{digest}"
            new_task = Task(
                id=self._generate_task_id(plan.id, seed),
                plan_id=plan.id,
                title=title,
                summary=summary,
                status=TaskStatus.READY,
                metadata={
                    "source": "auto_code_review_followup",
                    "origin_review_task_id": task.id,
                    "auto_review_finding": entry,
                    "auto_review_finding_digest": digest,
                    "auto_review_finding_digests": [digest],
                },
                depends_on=[],
                priority=task.priority,
            )
            self._store.save_task(new_task)
            followups.append(new_task)
        if followups:
            digests = [digest for _, digest in pending_entries]
            self._record_auto_review_plan_round(plan, followup_count=len(followups), digests=digests)
        return followups

    def _plan_review_followups(
        self,
        plan: Plan,
        review_task: Task,
        result: CodingIterationResult,
        findings: Sequence[tuple[str, str]],
        *,
        existing_tasks: Sequence[Task],
    ) -> list[Task]:
        planner_response = self._invoke_review_followup_planner(
            plan,
            review_task,
            result,
            findings,
            existing_tasks=existing_tasks,
        )
        if planner_response is None or not planner_response.tasks:
            return []

        planner_tasks = list(planner_response.tasks)
        filtered_tasks: list[PlannerTask] = []
        ignored_tasks: list[PlannerTask] = []
        for planner_task in planner_tasks:
            if task_is_test_execution(planner_task):
                ignored_tasks.append(planner_task)
                continue
            filtered_tasks.append(planner_task)

        if not filtered_tasks:
            digests = [digest for _, digest in findings]
            self._record_review_followup_plan(
                plan=plan,
                review_task=review_task,
                response=planner_response,
                findings=[entry for entry, _ in findings],
                digests=digests,
                tasks_override=[],
                ignored_tasks=ignored_tasks,
            )
            return []

        digests = [digest for _, digest in findings]
        followups = self._materialise_review_planner_tasks(
            plan=plan,
            review_task=review_task,
            planner_tasks=filtered_tasks,
            digests=digests,
        )
        if not followups:
            self._record_review_followup_plan(
                plan=plan,
                review_task=review_task,
                response=planner_response,
                findings=[entry for entry, _ in findings],
                digests=digests,
                tasks_override=filtered_tasks,
                ignored_tasks=ignored_tasks,
            )
            return []

        self._record_review_followup_plan(
            plan=plan,
            review_task=review_task,
            response=planner_response,
            findings=[entry for entry, _ in findings],
            digests=digests,
            tasks_override=filtered_tasks,
            ignored_tasks=ignored_tasks,
        )
        if followups:
            self._record_auto_review_plan_round(
                plan,
                followup_count=len(followups),
                digests=digests,
            )
        return followups

    def _invoke_review_followup_planner(
        self,
        plan: Plan,
        review_task: Task,
        result: CodingIterationResult,
        findings: Sequence[tuple[str, str]],
        *,
        existing_tasks: Sequence[Task],
    ) -> PlannerResponse | None:
        context = self._build_review_followup_context(
            plan,
            review_task,
            result,
            findings,
            existing_tasks=existing_tasks,
        )

        guidance_notes: list[str] = [
            "Transform the code review findings into focused implementation tasks.",
            "Each task should target the minimal code/test changes required to resolve a finding.",
            "Reference concrete files, modules, or tests so executors know where to act.",
        ]
        for entry, digest in findings:
            guidance_notes.append(f"[{digest}] {entry}")

        request = PlanRequest(
            goal=f"Address code review findings for plan {plan.id}",
            constraints=[
                "Limit work to the validated repository changes necessary to resolve the code review findings.",
                "Prefer augmenting existing modules, tests, and documentation instead of broad refactors.",
                "Include validation or test expectations whenever they are required to verify the fix.",
            ],
            notes=guidance_notes,
            known_context=context,
        )

        try:
            response = self._orchestrator.run_phase(PhaseName.PLAN, request)
        except Exception:
            return None

        if isinstance(response, PlannerResponse):
            return response
        if isinstance(response, Mapping):
            try:
                return PlannerResponse.model_validate(response)
            except ValidationError:
                return None
        return None

    def _build_review_followup_context(
        self,
        plan: Plan,
        review_task: Task,
        result: CodingIterationResult,
        findings: Sequence[tuple[str, str]],
        *,
        existing_tasks: Sequence[Task],
    ) -> dict[str, Any]:
        tasks_list = list(existing_tasks)
        completed_tasks = [item for item in tasks_list if item.status == TaskStatus.DONE]
        ready_tasks = [item for item in tasks_list if item.status == TaskStatus.READY]
        blocked_tasks = [item for item in tasks_list if item.status == TaskStatus.BLOCKED]

        def _task_snapshot(tasks: Sequence[Task], limit: int) -> list[dict[str, Any]]:
            snapshot: list[dict[str, Any]] = []
            for task in list(tasks)[:limit]:
                status_value = task.status.value if isinstance(task.status, TaskStatus) else str(task.status)
                snapshot.append(
                    {
                        "id": task.id,
                        "title": task.title,
                        "summary": task.summary,
                        "status": status_value,
                    }
                )
            return snapshot

        repository_tree = self._collect_repo_tree_lines()
        module_summaries = self._collect_python_module_summaries(completed_tasks)
        artifact_summaries = self._collect_task_artifacts(completed_tasks)

        finding_payload = [
            {"digest": digest, "summary": entry}
            for entry, digest in findings
        ]

        code_review_payload: dict[str, Any] = {
            "task_id": review_task.id,
            "task_title": review_task.title,
            "task_summary": review_task.summary,
            "findings": finding_payload,
        }
        if result.analyze and getattr(result.analyze, "summary", None):
            code_review_payload["analysis_summary"] = result.analyze.summary
        if result.design and getattr(result.design, "design_summary", None):
            code_review_payload["design_summary"] = result.design.design_summary
        if result.implement and getattr(result.implement, "summary", None):
            code_review_payload["implement_summary"] = result.implement.summary
        if result.implement and getattr(result.implement, "follow_up", None):
            code_review_payload["raw_follow_up"] = list(result.implement.follow_up or [])

        context: dict[str, Any] = {
            "plan": {
                "id": plan.id,
                "goal": plan.goal,
                "summary": plan.summary,
            },
            "code_review": code_review_payload,
            "tasks": {
                "completed": _task_snapshot(list(reversed(completed_tasks)), 12),
                "ready": _task_snapshot(ready_tasks, 8),
                "blocked": _task_snapshot(blocked_tasks, 8),
            },
        }

        if repository_tree:
            context["repository_tree"] = repository_tree
        if module_summaries:
            context["important_modules"] = module_summaries
        if artifact_summaries:
            context["recent_artifacts"] = artifact_summaries

        return context

    def _materialise_review_planner_tasks(
        self,
        plan: Plan,
        review_task: Task,
        planner_tasks: Sequence[PlannerTask],
        *,
        digests: Sequence[str],
    ) -> list[Task]:
        if not planner_tasks:
            return []

        existing_tasks = self._store.list_tasks(plan_id=plan.id)
        alias_map: dict[str, str] = {}
        for existing in existing_tasks:
            alias_map.setdefault(existing.id, existing.id)
            alias_map.setdefault(existing.title, existing.id)

        assigned: list[tuple[str, PlannerTask, int]] = []
        for index, planner_task in enumerate(planner_tasks, start=1):
            base_seed = planner_task.id or planner_task.title or f"followup-{index}"
            seed = f"{self._AUTO_REVIEW_TASK_SEED}-plan-{base_seed}"
            task_identifier = self._generate_task_id(plan.id, seed)
            assigned.append((task_identifier, planner_task, index))
            if planner_task.id:
                alias_map.setdefault(planner_task.id, task_identifier)
            alias_map.setdefault(planner_task.title, task_identifier)
            alias_map.setdefault(task_identifier, task_identifier)

        done_ids: set[str] = {task.id for task in existing_tasks if task.status == TaskStatus.DONE}
        created_ids: set[str] = set()
        followups: list[Task] = []

        for task_identifier, planner_task, order in assigned:
            resolved_dependencies: list[str] = []
            unresolved_dependencies: list[str] = []
            for dependency in planner_task.depends_on or []:
                resolved = alias_map.get(dependency)
                if resolved is None:
                    unresolved_dependencies.append(str(dependency))
                    continue
                if resolved not in resolved_dependencies:
                    resolved_dependencies.append(resolved)

            ready = True
            for dependency in resolved_dependencies:
                if dependency not in done_ids and dependency not in created_ids:
                    ready = False
                    break

            status = TaskStatus.READY if ready else TaskStatus.BLOCKED
            priority = planner_task.priority if planner_task.priority is not None else max(review_task.priority - 1, 0)

            planner_snapshot = planner_task.model_dump(mode="python", exclude_none=True)
            metadata: dict[str, Any] = {
                "planner_task": planner_snapshot,
                "source": "auto_code_review_followup_plan",
                "origin_review_task_id": review_task.id,
                "auto_review_finding_digests": list(digests),
            }
            if unresolved_dependencies:
                metadata["unresolved_dependencies"] = unresolved_dependencies

            new_task = Task(
                id=task_identifier,
                plan_id=plan.id,
                title=planner_task.title.strip(),
                summary=planner_task.summary.strip(),
                status=status,
                depends_on=resolved_dependencies,
                metadata=metadata,
                priority=priority,
            )
            self._store.save_task(new_task)
            followups.append(new_task)
            created_ids.add(task_identifier)

        return followups

    def _record_review_followup_plan(
        self,
        plan: Plan,
        review_task: Task,
        response: PlannerResponse,
        *,
        findings: Sequence[str],
        digests: Sequence[str],
        tasks_override: Sequence[PlannerTask] | None = None,
        ignored_tasks: Sequence[PlannerTask] | None = None,
    ) -> None:
        lines: list[str] = []
        summary_text = (response.plan_summary or "").strip()
        if summary_text:
            lines.append(summary_text)

        if findings:
            lines.append("Findings:")
            for digest, entry in zip(digests, findings):
                lines.append(f"- [{digest}] {entry}")

        displayed_tasks = (
            list(tasks_override)
            if tasks_override is not None
            else list(response.tasks or [])
        )
        if displayed_tasks:
            lines.append("Proposed tasks:")
            for planner_task in displayed_tasks[:8]:
                lines.append(f"- {planner_task.title}: {planner_task.summary}")
            if len(displayed_tasks) > 8:
                lines.append(f"- ... (+{len(displayed_tasks) - 8} additional tasks)")

        if ignored_tasks:
            lines.append("Ignored test-only proposals:")
            for planner_task in ignored_tasks[:8]:
                lines.append(f"- {planner_task.title}: {planner_task.summary}")
            if len(ignored_tasks) > 8:
                lines.append(f"- ... (+{len(ignored_tasks) - 8} additional entries)")

        if response.decisions:
            lines.append("Planner decisions:")
            for decision in response.decisions[:5]:
                lines.append(f"- {decision.title}: {decision.content}")
            if len(response.decisions) > 5:
                lines.append(f"- ... (+{len(response.decisions) - 5} additional decisions)")

        if response.risks:
            lines.append("Risks:")
            for risk in response.risks[:5]:
                risk_line = risk.description
                if risk.impact or risk.likelihood:
                    parts: list[str] = []
                    if risk.impact:
                        parts.append(f"impact={risk.impact}")
                    if risk.likelihood:
                        parts.append(f"likelihood={risk.likelihood}")
                    if parts:
                        risk_line = f"{risk_line} ({', '.join(parts)})"
                lines.append(f"- {risk_line}")
            if len(response.risks) > 5:
                lines.append(f"- ... (+{len(response.risks) - 5} additional risks)")

        if not lines:
            return

        decision = Decision(
            id=self._generate_decision_id(plan.id),
            plan_id=plan.id,
            task_id=review_task.id,
            title=f"Auto plan for code review follow-up ({review_task.id})",
            content="\n".join(lines),
            kind="code_review_followup_plan",
            metadata={
                "review_findings": [
                    {"digest": digest, "summary": entry}
                    for digest, entry in zip(digests, findings)
                ],
                "planner_response": response.model_dump(mode="python"),
                "ignored_test_only_tasks": [
                    {
                        "title": task.title,
                        "summary": task.summary,
                        "metadata": task.metadata or {},
                    }
                    for task in (ignored_tasks or [])
                ],
            },
        )
        self._store.record_decision(decision)

    @staticmethod
    def _extract_review_followups(implement: ImplementResponse) -> list[str]:
        entries: list[str] = []
        raw_followups = implement.follow_up or []
        for entry in raw_followups:
            if not isinstance(entry, str):
                continue
            cleaned = entry.strip()
            if not cleaned:
                continue
            cleaned = cleaned.lstrip("-*•").strip()
            if cleaned:
                if looks_like_test_execution(cleaned):
                    continue
                entries.append(cleaned)

        deduped: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            lowered = entry.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(entry)
        return deduped

    @staticmethod
    def _parse_review_followup_entry(entry: str) -> tuple[str, str]:
        cleaned = entry.strip()
        cleaned = cleaned.lstrip("-*•").strip()
        if "::" in cleaned:
            title, _, remainder = cleaned.partition("::")
            title = title.strip()
            summary = remainder.strip()
        elif " - " in cleaned:
            title, _, remainder = cleaned.partition(" - ")
            title = title.strip()
            summary = remainder.strip()
        else:
            title = cleaned[:80].strip()
            summary = cleaned
        if not title:
            title = "Address code review finding"
        if not summary:
            summary = title
        return title, summary

    @staticmethod
    def _dependency_signature(tasks: Sequence[Task]) -> tuple[str, ...]:
        return tuple(sorted(task.id for task in tasks))

    @staticmethod
    def _dependency_signature_from_ids(depends_on: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted(depends_on))

    # ------------------------------------------------------------- auto README
    def _maybe_schedule_readme_iteration(self, plan: Plan) -> bool:
        if self._final_iteration_mode_active(plan):
            return False
        tasks = self._store.list_tasks(plan_id=plan.id)
        if not tasks:
            return False

        non_readme_tasks = [task for task in tasks if not self._is_auto_readme_task(task)]
        if not non_readme_tasks:
            return False

        if any(task.status != TaskStatus.DONE for task in non_readme_tasks):
            return False

        readme_tasks = [task for task in tasks if self._is_auto_readme_task(task)]
        for readme_task in readme_tasks:
            if readme_task.status in {
                TaskStatus.READY,
                TaskStatus.RUNNING,
                TaskStatus.BLOCKED,
            }:
                return False

        dependency_signature = self._dependency_signature(non_readme_tasks)
        for readme_task in readme_tasks:
            if readme_task.status != TaskStatus.DONE:
                continue
            if self._dependency_signature_from_ids(readme_task.depends_on) == dependency_signature:
                return False

        readme_task = self._create_readme_polish_task(plan, non_readme_tasks)
        self._store.save_task(readme_task)
        return True

    def _create_readme_polish_task(self, plan: Plan, completed_tasks: Sequence[Task]) -> Task:
        task_id = self._generate_task_id(plan.id, self._AUTO_TASK_SEED)
        summary = self._render_readme_summary(plan, completed_tasks)
        planner_task = PlannerTask(
            id=task_id,
            title="Polish README documentation",
            summary=summary,
            constraints=[
                "Ensure README.md accurately reflects the implemented functionality.",
                "Include setup, usage, and testing guidance for new contributors.",
                "Preserve licensing or attribution details unless they are incorrect.",
            ],
            deliverables=["README.md"],
            notes=[
                "What steps must a user take to install or configure the project?",
                "How does someone run the primary workflows provided by the solution?",
                "Which commands or checks verify the system works as intended?",
            ],
            metadata={
                "touched_files": ["README.md"],
                self._AUTO_METADATA_KEY: self._AUTO_README_MARKER,
            },
        )
        metadata = {
            "planner_task": planner_task.model_dump(mode="python"),
            self._AUTO_METADATA_KEY: self._AUTO_README_MARKER,
            "source": "auto_readme_polish",
        }
        depends_on = [task.id for task in completed_tasks]
        base_priority = min((task.priority for task in completed_tasks), default=0)
        return Task(
            id=task_id,
            plan_id=plan.id,
            title=planner_task.title,
            summary=planner_task.summary,
            status=TaskStatus.READY,
            metadata=metadata,
            depends_on=depends_on,
            priority=base_priority - 1,
        )

    def _render_readme_summary(self, plan: Plan, completed_tasks: Sequence[Task]) -> str:
        lines: list[str] = [
            "Create or update README.md so it communicates the finished implementation clearly.",
            "Base the documentation strictly on the current repository contents—ignore speculative plans or unimplemented ideas.",
            "Cover project purpose, setup, configuration, primary workflows, interfaces, and how to run tests or validation checks.",
        ]

        structure_lines = self._collect_repo_tree_lines()
        if structure_lines:
            lines.append("Repository structure (truncated view):")
            lines.extend(structure_lines)

        module_lines = self._collect_python_module_summaries(completed_tasks)
        if module_lines:
            lines.append("Key Python modules and top-level symbols:")
            lines.extend(module_lines)

        artifact_lines = self._collect_task_artifacts(completed_tasks)
        if artifact_lines:
            lines.append("Artifacts confirmed by completed tasks:")
            lines.extend(artifact_lines)

        return "\n".join(lines)

    def _collect_repo_tree_lines(self, *, max_depth: int = 3, max_lines: int = 40) -> list[str]:
        root = getattr(self, "_repo_root", None)
        if not isinstance(root, Path) or not root.exists() or not root.is_dir():
            return []

        excluded_dirs = {
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            ".idea",
            ".vscode",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            "dist",
            "build",
            "tmp",
            "logs",
            "data",
        }
        excluded_files = {".DS_Store"}

        lines: list[str] = []
        truncated = False

        def walk(directory: Path, depth: int) -> None:
            nonlocal truncated
            if truncated or depth > max_depth:
                return
            try:
                entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except OSError:
                return
            for entry in entries:
                name = entry.name
                if entry.is_dir():
                    if name in excluded_dirs:
                        continue
                    if name.startswith(".") and name not in {".github"}:
                        continue
                else:
                    if name in excluded_files:
                        continue
                display_name = name if depth == 0 else entry.relative_to(root).as_posix()
                indent = "  " * depth
                if entry.is_dir():
                    if len(lines) >= max_lines:
                        truncated = True
                        return
                    lines.append(f"{indent}- {display_name}/")
                    walk(entry, depth + 1)
                    if truncated:
                        return
                else:
                    if len(lines) >= max_lines:
                        truncated = True
                        return
                    lines.append(f"{indent}- {display_name}")

        walk(root, 0)
        if truncated:
            lines.append("- ... (truncated)")
        return lines

    def _collect_python_module_summaries(
        self,
        completed_tasks: Sequence[Task],
        *,
        max_entries: int = 12,
    ) -> list[str]:
        candidate_paths = self._gather_candidate_python_paths(completed_tasks, limit=max_entries * 2)
        if not candidate_paths:
            return []

        lines: list[str] = []
        seen: set[str] = set()
        for path in candidate_paths:
            normalised = self._normalise_candidate_path(path)
            if not normalised or normalised in seen or not normalised.lower().endswith(".py"):
                continue
            summary = self._summarize_python_module(normalised)
            if not summary:
                continue
            seen.add(normalised)
            lines.append(f"  - {normalised}: {summary}")
            if len(lines) >= max_entries:
                break
        return lines

    def _collect_task_artifacts(self, completed_tasks: Sequence[Task], *, limit: int = 20) -> list[str]:
        artifacts: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        overflow = 0
        for task in completed_tasks:
            planner_task = self._load_planner_task(task)
            for path in self._extract_touched_files(planner_task):
                normalised = self._normalise_candidate_path(path)
                if not normalised or normalised in seen_paths:
                    continue
                seen_paths.add(normalised)
                source = task.title.strip() or task.id
                if len(artifacts) < limit:
                    artifacts.append((normalised, source))
                else:
                    overflow += 1

        if not artifacts:
            return []

        lines = [f"  - {path} (from task '{source}')" for path, source in artifacts]
        if overflow > 0:
            lines.append(f"  - ... (+{overflow} additional artifacts)")
        return lines

    def _collect_recent_touched_files(
        self,
        completed_tasks: Sequence[Task],
        *,
        limit: int = 40,
    ) -> list[str]:
        primary: list[str] = []
        tests: list[str] = []
        seen: set[str] = set()

        for task in completed_tasks:
            planner_task = self._load_planner_task(task)
            for path in self._extract_touched_files(planner_task):
                normalised = self._normalise_candidate_path(path)
                if not normalised or normalised in seen:
                    continue
                seen.add(normalised)
                bucket = tests if self._is_likely_test_path(normalised) else primary
                bucket.append(normalised)
                if len(primary) + len(tests) >= limit:
                    return primary + tests

            metadata_paths = task.metadata.get("touched_files")
            if isinstance(metadata_paths, Sequence) and not isinstance(metadata_paths, (str, bytes, bytearray)):
                for entry in metadata_paths:
                    if not isinstance(entry, str):
                        continue
                    normalised = self._normalise_candidate_path(entry)
                    if not normalised or normalised in seen:
                        continue
                    seen.add(normalised)
                    bucket = tests if self._is_likely_test_path(normalised) else primary
                    bucket.append(normalised)
                    if len(primary) + len(tests) >= limit:
                        return primary + tests

        return primary + tests

    def _gather_candidate_python_paths(
        self,
        completed_tasks: Sequence[Task],
        *,
        limit: int = 30,
    ) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for task in completed_tasks:
            planner_task = self._load_planner_task(task)
            for path in self._extract_touched_files(planner_task):
                normalised = self._normalise_candidate_path(path)
                if (
                    not normalised
                    or not normalised.lower().endswith(".py")
                    or normalised in seen
                ):
                    continue
                seen.add(normalised)
                candidates.append(normalised)
                if len(candidates) >= limit:
                    return candidates

        if candidates:
            return candidates
        return self._discover_python_files(limit=limit)

    def _discover_python_files(self, *, limit: int = 30) -> list[str]:
        root = getattr(self, "_repo_root", None)
        if not isinstance(root, Path) or not root.exists() or not root.is_dir():
            return []

        preferred_roots = ["src", "app", "backend", "service", "lib"]
        search_roots: list[Path] = []
        for name in preferred_roots:
            candidate = root / name
            if candidate.exists() and candidate.is_dir():
                search_roots.append(candidate)
        if not search_roots:
            search_roots = [root]

        excluded_dirs = {
            "__pycache__",
            ".git",
            ".hg",
            ".svn",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            ".idea",
            ".vscode",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            "dist",
            "build",
            "tmp",
            "logs",
            "data",
        }

        discovered: list[str] = []
        root_resolved = root.resolve() if root.exists() else root

        for base in search_roots:
            try:
                base_resolved = base.resolve()
            except OSError:
                continue
            for dirpath, dirnames, filenames in os.walk(base_resolved):
                rel_dir = Path(dirpath).relative_to(root_resolved)
                dirnames[:] = [
                    name for name in dirnames if name not in excluded_dirs and not name.startswith(".")
                ]
                for filename in sorted(filenames):
                    if not filename.endswith(".py") or filename.startswith("."):
                        continue
                    rel_path = (Path(dirpath) / filename).relative_to(root_resolved).as_posix()
                    if rel_path not in discovered:
                        discovered.append(rel_path)
                        if len(discovered) >= limit:
                            return discovered
        return discovered

    def _summarize_python_module(self, relative_path: str) -> str | None:
        root = getattr(self, "_repo_root", None)
        if not isinstance(root, Path):
            return None
        candidate = root / relative_path
        try:
            resolved_candidate = candidate.resolve()
            resolved_root = root.resolve()
        except OSError:
            return None
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError:
            return None
        try:
            text = resolved_candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        if len(text) > 200_000:
            return "large module (skipped for brevity)"
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return "unable to parse module for symbol outline"

        class_names: list[str] = []
        function_names: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    class_names.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    function_names.append(node.name)

        descriptors: list[str] = []
        if class_names:
            display = ", ".join(class_names[:5])
            if len(class_names) > 5:
                display += ", ..."
            descriptors.append(f"classes: {display}")
        if function_names:
            display = ", ".join(function_names[:6])
            if len(function_names) > 6:
                display += ", ..."
            descriptors.append(f"functions: {display}")

        if not descriptors:
            return "module-level helpers (no public classes/functions detected)"
        return "; ".join(descriptors)

    def _is_auto_review_task(self, task: Task) -> bool:
        marker = task.metadata.get(self._AUTO_METADATA_KEY)
        if marker == self._AUTO_REVIEW_MARKER:
            return True
        planner_payload = task.metadata.get("planner_task")
        if isinstance(planner_payload, Mapping):
            planner_meta = planner_payload.get("metadata")
            if isinstance(planner_meta, Mapping):
                planner_marker = planner_meta.get(self._AUTO_METADATA_KEY)
                if planner_marker == self._AUTO_REVIEW_MARKER:
                    return True
        return task.id.endswith(f"::{self._AUTO_REVIEW_TASK_SEED}")

    def _is_auto_readme_task(self, task: Task) -> bool:
        marker = task.metadata.get(self._AUTO_METADATA_KEY)
        if marker == self._AUTO_README_MARKER:
            return True
        planner_payload = task.metadata.get("planner_task")
        if isinstance(planner_payload, Mapping):
            planner_meta = planner_payload.get("metadata")
            if isinstance(planner_meta, Mapping):
                planner_marker = planner_meta.get(self._AUTO_METADATA_KEY)
                if planner_marker == self._AUTO_README_MARKER:
                    return True
        return task.id.endswith(f"::{self._AUTO_TASK_SEED}")
