"""Plan executor wiring that drives orchestrator iterations and adjustments."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from pydantic import ValidationError

from ..memory.schema import Decision, Plan, PlanStatus, Task, TaskStatus
from ..memory.store import MemoryStore
from ..orchestrator import CodingIterationPlan, CodingIterationResult, Orchestrator, PatchApplicationResult
from ..phases import PhaseName
from ..phases.analyze import AnalyzeRequest
from ..phases.design import DesignRequest
from ..phases.implement import ImplementRequest
from ..phases.plan_adjust import (
    PlanAdjustRequest,
    PlanAdjustResponse,
    PlanAdjustment,
    PlanAdjustmentItem,
)
from .schemas import PlannerTask

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
    if result.gates.suspect_files:
        suspect_files = list(dict.fromkeys(result.gates.suspect_files))

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


class PlanExecutor:
    """Execute READY tasks for a plan using the orchestrator."""

    _AUTO_REVIEW_MARKER = "code_review"
    _AUTO_REVIEW_TASK_SEED = "code-review"
    _AUTO_README_MARKER = "readme_polish"
    _AUTO_METADATA_KEY = "auto_generated"
    _AUTO_TASK_SEED = "polish-readme"

    def __init__(
        self,
        store: MemoryStore,
        orchestrator: Orchestrator,
        *,
        config_path: Path | None = None,
        revert_on_exit: bool = False,
    ) -> None:
        self._store = store
        self._orchestrator = orchestrator
        self._config_path = Path(config_path).resolve() if config_path else None
        repo_root = self._config_path.parent if self._config_path else Path.cwd()
        try:
            self._repo_root = repo_root.resolve()
        except OSError:
            self._repo_root = repo_root
        self._revert_on_exit = revert_on_exit

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

            iteration_plan = self._build_iteration_plan(plan, task)
            if iteration_plan is None:
                self._store.update_task_status(task.id, TaskStatus.BLOCKED)
                continue

            result = self._orchestrator.run_coding_iteration(
                iteration_plan,
                config_path=self._config_path,
                revert_on_exit=self._revert_on_exit,
            )
            try:
                self._store.reopen()
            except Exception:
                # If reopening fails, propagate so callers can handle persistence errors.
                raise
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

    def _build_iteration_plan(self, plan: Plan, task: Task) -> CodingIterationPlan | None:
        planner_task = self._load_planner_task(task)

        analyze = AnalyzeRequest(
            task_id=task.id,
            goal=plan.goal,
            context=planner_task.summary,
            constraints=list(planner_task.constraints),
            questions=list(planner_task.notes),
        )
        design = DesignRequest(
            task_id=task.id,
            goal=plan.goal,
            proposed_interfaces=list(planner_task.deliverables) or [planner_task.title],
            open_questions=list(planner_task.notes),
            constraints=list(planner_task.constraints),
        )
        implement = ImplementRequest(
            task_id=task.id,
            diff_goal=planner_task.summary,
            touched_files=self._extract_touched_files(planner_task),
            test_plan=self._select_test_plan(planner_task, plan),
            notes=list(planner_task.notes),
        )
        return CodingIterationPlan(analyze=analyze, design=design, implement=implement, plan_id=plan.id)

    def _handle_iteration_result(
        self,
        plan: Plan,
        task: Task,
        result: CodingIterationResult,
    ) -> tuple[PlanAdjustResponse | None, list[Task]]:
        if result.ok:
            self._store.update_task_status(task.id, TaskStatus.DONE)
            followups: list[Task] = []
            if self._is_auto_review_task(task):
                followups = self._spawn_review_followups(plan, task, result)
            self._unlock_blocked_tasks(plan.id)
            return None, followups

        self._store.update_task_status(task.id, TaskStatus.BLOCKED)
        adjustment = self._request_plan_adjustment(plan, task, result)
        followups = self._materialize_followups(plan, task, adjustment)
        self._unlock_blocked_tasks(plan.id)
        return adjustment, followups

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
        decision = self._record_plan_adjust_decision(plan, task, adjustment)
        followups: list[Task] = []
        for entry in adjustment.new_tasks:
            task_id = self._generate_task_id(plan.id, entry)
            metadata = {
                "source": "plan_adjust",
                "origin_task_id": task.id,
                "plan_adjust_decision_id": decision.id if decision else None,
            }
            followup = Task(
                id=task_id,
                plan_id=plan.id,
                title=entry.strip() or f"Follow up on {task.id}",
                summary=entry.strip() or entry,
                status=TaskStatus.READY,
                metadata=metadata,
                priority=max(task.priority - 1, 0),
            )
            self._store.save_task(followup)
            followups.append(followup)
        return followups

    def _record_plan_adjust_decision(
        self,
        plan: Plan,
        task: Task,
        adjustment: PlanAdjustResponse,
    ) -> Decision | None:
        content_parts: list[str] = []
        if adjustment.adjustments:
            content_parts.append("Adjustments:")
            content_parts.extend(f"- {self._render_adjustment(item)}" for item in adjustment.adjustments)
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
            },
        )
        self._store.record_decision(decision)
        return decision
    
    @staticmethod
    def _render_adjustment(item: PlanAdjustment) -> str:
        if isinstance(item, PlanAdjustmentItem):
            return item.render()
        return str(item)

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
    def _looks_like_file_path(value: str) -> bool:
        normalised = value.replace("\\", "/")
        if "/" in normalised:
            return True
        lowered = normalised.lower()
        return any(lowered.endswith(ext) for ext in _FILE_EXTENSIONS)

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
        return []

    @staticmethod
    def _extract_failing_tests(pytest_result) -> list[str]:
        return _extract_failing_tests(pytest_result)

    def _generate_task_id(self, plan_id: str, seed: str) -> str:
        base = seed.strip() or uuid4().hex
        slug = self._slugify(base)
        identifier = f"{plan_id}::{slug}"
        counter = 2
        existing = {task.id for task in self._store.list_tasks(plan_id=plan_id)}
        while identifier in existing:
            identifier = f"{plan_id}::{slug}-{counter}"
            counter += 1
        return identifier

    def _generate_decision_id(self, plan_id: str) -> str:
        return f"{plan_id}::adj-{uuid4().hex[:8]}"

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.lower())
        while "--" in cleaned:
            cleaned = cleaned.replace("--", "-")
        cleaned = cleaned.strip("-")
        return cleaned or uuid4().hex[:8]


    # -------------------------------------------------------------- auto review
    def _maybe_schedule_code_review_iteration(self, plan: Plan) -> bool:
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
        planner_task = PlannerTask(
            id=task_id,
            title="Perform targeted code review",
            summary=summary,
            constraints=[
                "Investigate only issues that could cause failures, regressions, security incidents, or major user confusion.",
                "Skip purely stylistic nitpicks unless they introduce functional risk.",
                "Do not modify repository files; limit the response to analysis only.",
                "For each substantive finding, add a follow_up entry using the format 'Task Title :: Required fix'.",
            ],
            deliverables=["Actionable code review findings"],
            notes=[
                "Prioritise correctness gaps, missing validation, data handling problems, and test omissions that create high-severity risk.",
                "Do not require exhaustive edge-case tests; only highlight missing coverage when it could conceal a serious defect.",
                "If no major issues remain, explain why the implementation appears sound and leave follow_up empty.",
            ],
            metadata={
                self._AUTO_METADATA_KEY: self._AUTO_REVIEW_MARKER,
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
            "Perform a high-severity code review over the completed tasks before finalising documentation.",
            "Focus on correctness, resiliency, security, and only those test coverage gaps that could break the product or confuse users.",
            "Skip requests for exhaustive edge-case tests; call out missing coverage only when it hides likely high-severity bugs.",
            "Only surface actionable problems; ignore cosmetic issues unless they impact functionality.",
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
            if (
                metadata.get("source") == "auto_code_review_followup"
                and metadata.get("auto_review_finding_digest")
            ):
                digest = str(metadata.get("auto_review_finding_digest"))
                if existing.status != TaskStatus.DONE:
                    active_review_digests.add(digest)

        followups: list[Task] = []
        for entry in findings:
            digest = hashlib.sha256(entry.strip().lower().encode("utf-8")).hexdigest()[:16]
            if digest in active_review_digests:
                continue
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
                },
                depends_on=[],
                priority=task.priority,
            )
            self._store.save_task(new_task)
            followups.append(new_task)
            active_review_digests.add(digest)
        return followups

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
        tasks = self._store.list_tasks(plan_id=plan.id)
        if not tasks:
            return False

        auto_tasks = [task for task in tasks if self._is_auto_readme_task(task)]
        if auto_tasks:
            return False

        review_tasks = [task for task in tasks if self._is_auto_review_task(task)]
        if any(task.status != TaskStatus.DONE for task in review_tasks):
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

        dependency_signature = self._dependency_signature(core_tasks)
        review_coverage = any(
            review.status == TaskStatus.DONE
            and self._dependency_signature_from_ids(review.depends_on) == dependency_signature
            for review in review_tasks
        )
        if not review_coverage:
            return False

        readme_task = self._create_readme_polish_task(plan, core_tasks)
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
