"""Bootstrap routines for seeding planner artifacts from repository state."""

from __future__ import annotations

import ast
import importlib
import json
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel

from ..context_builder import ContextBuilder, ContextPackage
from ..memory.code_index.symbol_index import SymbolIndex
from ..memory.schema import Checkpoint, Decision, Plan, PlanStatus, Task, TaskStatus
from ..memory.store import MemoryStore
from ..models.llm_client import LLMClient, LLMClientError, LLMRequest
from ..phases import PhaseName
from ..phases.plan import PlanRequest as PhasePlanRequest, run as run_plan
from ..tools.vcs import GitError, GitRepository
from .schemas import (
    PlannerAnalysis,
    PlannerDecision,
    PlannerCritique,
    PlannerResponse,
    PlannerRisk,
    PlannerTask,
)

try:  # pragma: no cover - prefer stdlib tomllib when available
    import tomllib as _native_tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback to optional dependency
    _native_tomllib = None  # type: ignore[assignment]

tomllib: ModuleType | None
if _native_tomllib is not None:
    tomllib = _native_tomllib
else:
    tomllib = None
    tomli_spec = importlib.util.find_spec("tomli")
    if tomli_spec is not None:
        tomllib = importlib.import_module("tomli")

TOMLDecodeError = getattr(tomllib, "TOMLDecodeError", ValueError) if tomllib else ValueError


_TARGET_PATH_RE = re.compile(r"(?<![\w./-])(?:[\w.-]+/)*[\w.-]+\.py\b")
_DATA_PATH_RE = re.compile(r"\bdata/[A-Za-z0-9_.\-/]+\b")

@dataclass(slots=True)
class PlanningArtifacts:
    """Container for the planning bootstrap output."""

    plan: Plan
    tasks: list[Task]
    decisions: list[Decision]
    checkpoint: Checkpoint
    raw_response: dict[str, Any]


@dataclass(slots=True)
class _PlannerPassResult:
    """Intermediate planner pass outcome including context and diagnostics."""

    name: str
    request_payload: dict[str, Any]
    response: BaseModel | None = None
    raw_response: dict[str, Any] | None = None
    context: ContextPackage | None = None
    payload_preview: dict[str, Any] | None = None
    error: str | None = None


def bootstrap_initial_plan(
    store: MemoryStore,
    config: Mapping[str, Any],
    planner_client: LLMClient,
    *,
    goal: str | None = None,
    constraints: Sequence[str] | None = None,
    deliverables: Sequence[str] | None = None,
    deadline: str | None = None,
    notes: Sequence[str] | None = None,
    repo_root: Path | None = None,
    config_path: Path | None = None,
) -> PlanningArtifacts:
    """
    Synthesize an execution plan by invoking the planning phase LLM.

    The generated response is validated, persisted via the MemoryStore, and returned
    alongside the raw model payload for traceability.
    """

    project_cfg = config.get("project") or {}
    iteration_cfg = config.get("iteration") or {}

    project_name = str(project_cfg.get("name") or "project").strip() or "project"
    goal_input = goal if goal is not None else iteration_cfg.get("goal") or "bootstrap"
    goal_normalised = str(goal_input).strip() or "bootstrap"

    plan_identifier = str(iteration_cfg.get("plan_id") or _default_plan_id(project_name))

    builder = ContextBuilder.from_config(
        config,
        repo_root=repo_root,
    )
    repo_root_path = builder.repo_root
    repo_summary = _collect_repo_summary(repo_root_path, config)

    constraint_list = list(constraints or [])
    deliverable_list = list(deliverables or [])
    notes_list = list(notes or [])

    prune_result = _prune_completed_goal_inputs(
        repo_root=repo_root_path,
        repo_summary=repo_summary,
        constraints=constraint_list,
        deliverables=deliverable_list,
    )
    constraint_list = prune_result.constraints
    deliverable_list = prune_result.deliverables
    if prune_result.pruned_constraints or prune_result.pruned_deliverables:
        repo_summary["pruned_inputs"] = {
            "constraints": prune_result.pruned_constraints,
            "deliverables": prune_result.pruned_deliverables,
        }

    base_plan_request = PhasePlanRequest(
        goal=goal_normalised,
        constraints=constraint_list,
        deliverables=deliverable_list,
        deadline=deadline,
        notes=notes_list,
        known_context=repo_summary,
    )
    supports_local_logic = bool(getattr(planner_client, "supports_local_logic", False))

    pass_results: list[_PlannerPassResult] = []

    analysis_result: _PlannerPassResult | None = None
    analysis_model: PlannerAnalysis | None = None
    analysis_payload: dict[str, Any] | None = None
    if not supports_local_logic:
        analysis_notes = _compose_analysis_notes(goal_normalised, deliverable_list)
        analysis_request, analysis_payload = _plan_request_for_pass(
            base_plan_request,
            pass_name="analysis",
            notes_extra=analysis_notes,
        )
        try:
            analysis_result = _execute_planner_pass(
                pass_name="analysis",
                phase=PhaseName.PLAN.value,
                request_payload=analysis_payload,
                response_model=PlannerAnalysis,
                client=planner_client,
                context_builder=builder,
                supports_local_logic=supports_local_logic,
                plan_request=analysis_request,
                allow_local_logic=False,
                temperature=0.15,
            )
        except LLMClientError as error:
            pass_results.append(
                _PlannerPassResult(
                    name="analysis",
                    request_payload=dict(analysis_payload or {}),
                    error=str(error),
                )
            )
        else:
            pass_results.append(analysis_result)
            if isinstance(analysis_result.response, PlannerAnalysis):
                analysis_model = analysis_result.response

    analysis_dump = analysis_model.model_dump(mode="python") if analysis_model else None
    synthesis_known_context: dict[str, Any] = {}
    if analysis_dump:
        synthesis_known_context["analysis"] = analysis_dump

    draft_notes = _compose_synthesis_notes(goal_normalised, deliverable_list, analysis_model)
    draft_request, draft_payload = _plan_request_for_pass(
        base_plan_request,
        pass_name="draft",
        known_context_extra=synthesis_known_context,
        notes_extra=draft_notes,
    )
    draft_result = _execute_planner_pass(
        pass_name="draft",
        phase=PhaseName.PLAN.value,
        request_payload=draft_payload,
        response_model=PlannerResponse,
        client=planner_client,
        context_builder=builder,
        supports_local_logic=supports_local_logic,
        plan_request=draft_request,
        allow_local_logic=True,
    )
    pass_results.append(draft_result)
    final_pass = draft_result

    draft_plan = draft_result.response if isinstance(draft_result.response, PlannerResponse) else None

    critique_model: PlannerCritique | None = None
    if not supports_local_logic and draft_plan is not None:
        critic_notes = _compose_critic_notes(deliverable_list, draft_plan, stage="draft")
        critic_known_context: dict[str, Any] = {}
        if analysis_dump:
            critic_known_context["analysis"] = analysis_dump
        critic_known_context["candidate_plan"] = draft_plan.model_dump(mode="python")
        critic_request, critic_payload = _plan_request_for_pass(
            base_plan_request,
            pass_name="critic",
            known_context_extra=critic_known_context,
            notes_extra=critic_notes,
        )
        try:
            critic_result = _execute_planner_pass(
                pass_name="critic",
                phase=PhaseName.PLAN.value,
                request_payload=critic_payload,
                response_model=PlannerCritique,
                client=planner_client,
                context_builder=builder,
                supports_local_logic=supports_local_logic,
                plan_request=critic_request,
                allow_local_logic=False,
            )
        except LLMClientError as error:
            pass_results.append(
                _PlannerPassResult(
                    name="critic",
                    request_payload=dict(critic_payload or {}),
                    error=str(error),
                )
            )
        else:
            pass_results.append(critic_result)
            if isinstance(critic_result.response, PlannerCritique):
                critique_model = critic_result.response

    directives: list[str] = []
    feedback: dict[str, Any] = {}
    if draft_plan is not None:
        directives, feedback = _build_refinement_guidance(
            draft_plan,
            analysis=analysis_model,
            critique=critique_model,
            deliverables=deliverable_list,
        )

    refinement_required = (
        not supports_local_logic
        and draft_plan is not None
        and _should_trigger_refinement(critique_model, feedback)
    )

    if refinement_required:
        refinement_context = {
            "draft_plan": draft_plan.model_dump(mode="python"),
            "draft_plan_summary": draft_plan.plan_summary,
            "stage_feedback": feedback,
        }
        if analysis_dump:
            refinement_context["analysis"] = analysis_dump
        if critique_model is not None:
            refinement_context["critic_report"] = critique_model.model_dump(mode="python")

        refine_request, refine_payload = _plan_request_for_pass(
            base_plan_request,
            pass_name="refine",
            known_context_extra=refinement_context,
            notes_extra=_compose_refinement_notes(directives),
        )
        refine_payload["draft_plan"] = refinement_context["draft_plan"]
        refine_payload["stage_feedback"] = feedback
        refine_payload["refinement_directives"] = directives
        if critique_model is not None:
            refine_payload["critic_report"] = refinement_context["critic_report"]
        try:
            refine_result = _execute_planner_pass(
                pass_name="refine",
                phase=PhaseName.PLAN.value,
                request_payload=refine_payload,
                response_model=PlannerResponse,
                client=planner_client,
                context_builder=builder,
                supports_local_logic=supports_local_logic,
                plan_request=refine_request,
                allow_local_logic=False,
            )
        except LLMClientError as error:
            pass_results.append(
                _PlannerPassResult(
                    name="refine",
                    request_payload=dict(refine_payload),
                    error=str(error),
                )
            )
        else:
            pass_results.append(refine_result)
            if isinstance(refine_result.response, PlannerResponse):
                final_pass = refine_result
                draft_plan = refine_result.response
                if not supports_local_logic:
                    final_critic_notes = _compose_critic_notes(deliverable_list, draft_plan, stage="final")
                    final_critic_context: dict[str, Any] = {
                        "analysis": analysis_dump,
                        "candidate_plan": draft_plan.model_dump(mode="python"),
                        "stage_feedback": feedback,
                    }
                    final_critic_request, final_critic_payload = _plan_request_for_pass(
                        base_plan_request,
                        pass_name="critic-final",
                        known_context_extra=final_critic_context,
                        notes_extra=final_critic_notes,
                    )
                    try:
                        final_critic_result = _execute_planner_pass(
                            pass_name="critic-final",
                            phase=PhaseName.PLAN.value,
                            request_payload=final_critic_payload,
                            response_model=PlannerCritique,
                            client=planner_client,
                            context_builder=builder,
                            supports_local_logic=supports_local_logic,
                            plan_request=final_critic_request,
                            allow_local_logic=False,
                        )
                    except LLMClientError as error:
                        pass_results.append(
                            _PlannerPassResult(
                                name="critic-final",
                                request_payload=dict(final_critic_payload or {}),
                                error=str(error),
                            )
                        )
                    else:
                        pass_results.append(final_critic_result)
                        if isinstance(final_critic_result.response, PlannerCritique):
                            critique_model = final_critic_result.response or critique_model

    plan_response = final_pass.response if isinstance(final_pass.response, PlannerResponse) else None
    if plan_response is None:
        raise RuntimeError("Planner did not return a response.")

    _enrich_plan_with_analysis(plan_response, analysis_model)

    context = final_pass.context
    if context is None:
        context = builder.build(PhaseName.PLAN.value, final_pass.request_payload)
    payload_preview = final_pass.payload_preview or {}
    plan_request_payload = final_pass.request_payload
    raw_payload = final_pass.raw_response or plan_response.model_dump()
    planner_pass_history = _serialise_pass_history(pass_results)
    raw_payload = _attach_pass_history(raw_payload, pass_results)

    if plan_response.plan_id:
        plan_identifier = _normalise_plan_id(str(plan_response.plan_id), fallback=plan_identifier)

    validation_report = _validate_plan_against_contracts(plan_response, repo_summary)
    if validation_report.conflicts or validation_report.duplicate_tasks:
        _apply_plan_validation_annotations(plan_response, validation_report)

    plan_metadata: dict[str, Any] = {
        "source": "ae init --plan",
        "workflow_version": 10,
        "project_name": project_name,
        "goal": goal_normalised,
        "user_input": {
            "goal": goal_input,
            "constraints": constraint_list,
            "deliverables": deliverable_list,
            "deadline": deadline,
            "notes": notes_list,
        },
        "planner": {
            "model": planner_client.model,
            "phase": PhaseName.PLAN.value,
            "request": plan_request_payload,
            "payload_preview": payload_preview,
            "prompt": {
                "system": context.system_prompt,
                "user": context.user_prompt,
                "metadata": context.metadata,
            },
            "response": raw_payload,
            "passes": planner_pass_history,
        },
        "planner_metadata": plan_response.metadata,
    }
    planner_section = plan_metadata["planner"]
    if analysis_model is not None:
        planner_section["analysis"] = analysis_model.model_dump(mode="python")
    if critique_model is not None:
        planner_section["critique"] = critique_model.model_dump(mode="python")
    if feedback:
        planner_section["quality_feedback"] = feedback
    validation_metadata = validation_report.to_metadata()
    if validation_metadata:
        plan_metadata["planner_validation"] = validation_metadata
    if config_path is not None:
        plan_metadata["config_path"] = str(config_path)

    plan = Plan(
        id=plan_identifier,
        name=_derive_plan_name(plan_response.plan_name, project_name),
        goal=goal_normalised,
        status=PlanStatus.ACTIVE,
        summary=plan_response.plan_summary.strip(),
        metadata=plan_metadata,
    )
    store.create_plan(plan)

    task_results = _persist_tasks(store, plan_identifier, plan_response.tasks)
    decision_results = _persist_decisions(
        store,
        plan_identifier,
        task_results.alias_map,
        plan_response.decisions,
        plan_response.risks,
    )
    decision_results = _ensure_baseline_decisions(
        store=store,
        plan=plan,
        decisions=decision_results,
    )

    checkpoint = _persist_checkpoint(
        store=store,
        plan_id=plan_identifier,
        goal=goal_normalised,
        plan_summary=plan_response.plan_summary,
        tasks=task_results.tasks,
        decisions=decision_results,
    )

    _persist_llm_exchange(
        config=config,
        repo_root=repo_root_path,
        plan_id=plan_identifier,
        goal=goal_normalised,
        request_payload=payload_preview,
        context_package=context,
        raw_response=raw_payload,
        parsed_response=plan_response.model_dump(mode="json"),
        model_name=planner_client.model,
    )

    return PlanningArtifacts(
        plan=plan,
        tasks=task_results.tasks,
        decisions=decision_results,
        checkpoint=checkpoint,
        raw_response=raw_payload,
    )


# --------------------------------------------------------------------------- helpers
def _plan_request_for_pass(
    base_request: PhasePlanRequest,
    *,
    pass_name: str,
    known_context_extra: Mapping[str, Any] | None = None,
    notes_extra: Sequence[str] | None = None,
) -> tuple[PhasePlanRequest, dict[str, Any]]:
    """Clone the base planner request with pass-specific guidance."""
    known_context = dict(getattr(base_request, "known_context", {}) or {})
    known_context["_planner_pass"] = pass_name
    if known_context_extra:
        for key, value in known_context_extra.items():
            known_context[str(key)] = value

    base_notes = [note.strip() for note in getattr(base_request, "notes", []) if isinstance(note, str) and note.strip()]
    pass_notes: list[str] = []
    if notes_extra:
        for note in notes_extra:
            if not isinstance(note, str):
                continue
            cleaned = note.strip()
            if not cleaned:
                continue
            base_notes.append(cleaned)
            pass_notes.append(cleaned)

    request = PhasePlanRequest(
        goal=base_request.goal,
        constraints=list(getattr(base_request, "constraints", []) or []),
        deliverables=list(getattr(base_request, "deliverables", []) or []),
        deadline=base_request.deadline,
        notes=base_notes,
        known_context=known_context,
    )

    payload = _planning_request_payload(
        goal=request.goal,
        constraints=request.constraints,
        deliverables=request.deliverables,
        deadline=request.deadline,
        notes=request.notes,
        known_context=request.known_context,
    )
    payload["stage"] = pass_name
    if pass_notes:
        payload["stage_notes"] = pass_notes
    return request, payload


def _execute_planner_pass(
    *,
    pass_name: str,
    phase: str,
    request_payload: Mapping[str, Any],
    response_model: type[BaseModel],
    client: LLMClient,
    context_builder: ContextBuilder,
    supports_local_logic: bool,
    plan_request: PhasePlanRequest | None = None,
    allow_local_logic: bool = False,
    temperature: float | None = None,
) -> _PlannerPassResult:
    """Invoke the planner for a specific planner pass and capture context."""
    context = context_builder.build(phase, dict(request_payload))
    metadata = dict(context.metadata)
    metadata["planner_pass"] = pass_name
    llm_request = LLMRequest(
        prompt=context.user_prompt,
        system_prompt=context.system_prompt,
        response_model=response_model,
        metadata=metadata,
        temperature=temperature or 0.0,
    )
    payload_preview = llm_request.to_payload(client.model)

    response_obj: BaseModel | None = None
    raw_response: dict[str, Any] | None = None
    if allow_local_logic and supports_local_logic and plan_request is not None and response_model is PlannerResponse:
        response_obj = run_plan(plan_request, client=client, context_builder=context_builder)
        raw_response = response_obj.model_dump()
    else:
        response_obj, raw_payload = client.invoke_structured(llm_request)
        if isinstance(raw_payload, Mapping):
            raw_response = dict(raw_payload)
        else:
            raw_response = {"raw": raw_payload}
    return _PlannerPassResult(
        name=pass_name,
        request_payload=dict(request_payload),
        response=response_obj,
        raw_response=raw_response,
        context=context,
        payload_preview=payload_preview,
    )


def _compose_analysis_notes(goal: str, deliverables: Sequence[str]) -> list[str]:
    """Guidance supplied to the analysis planner pass."""
    notes = [
        "Return valid JSON matching the PlannerAnalysis schema.",
        "Summarise the repository into components with relevant paths, key symbols, and nearby tests.",
        "Populate `deliverable_map` so each requested deliverable maps to concrete components or files.",
        "List validation assets (tests, scripts, or commands) that help verify future work.",
        "Document knowledge gaps or assumptions that downstream passes must address.",
    ]
    trimmed_goal = goal.strip()
    if trimmed_goal:
        notes.append(f"Focus on information that helps deliver the goal '{trimmed_goal}'.")
    if deliverables:
        joined = ", ".join(deliverables)
        notes.append(f"Ensure `deliverable_map` keys cover: {joined}.")
    notes.append("Keep entries concise and immediately actionable for subsequent planner passes.")
    return _dedupe_preserve_order(notes)


def _compose_synthesis_notes(
    goal: str,
    deliverables: Sequence[str],
    analysis: PlannerAnalysis | None,
) -> list[str]:
    """Guidance supplied to the synthesis (draft) planner pass."""
    notes = [
        "Return valid JSON matching the PlannerResponse schema.",
        "Produce at least three focused tasks with explicit dependencies and acceptance criteria.",
        "Populate each task's `metadata.touched_files` with concrete paths to inspect or modify.",
        "Populate each task's `metadata.validation_commands` with commands or tests that confirm success.",
        "Avoid overlapping scopes; each task should deliver a narrowly defined outcome.",
    ]
    trimmed_goal = goal.strip()
    if trimmed_goal:
        notes.append(f"Align the task breakdown with the goal '{trimmed_goal}'.")
    if analysis and analysis.components:
        component_names = ", ".join(component.name for component in analysis.components[:5])
        notes.append(f"Reference relevant analysis components where helpful: {component_names}.")
    if deliverables:
        notes.append("Ensure every deliverable is covered by at least one task.")
    return _dedupe_preserve_order(notes)


def _compose_critic_notes(
    deliverables: Sequence[str],
    plan_response: PlannerResponse,
    *,
    stage: str,
) -> list[str]:
    """Guidance supplied to critic passes evaluating a plan response."""
    notes = [
        "Return valid JSON matching the PlannerCritique schema.",
        f"Evaluate the {stage} plan for completeness, risk coverage, and dependency soundness.",
        "List blocking issues under `blockers` or as `must_fix` items in `issues`.",
        "Highlight missing deliverables, dependency conflicts, or validation gaps.",
        "Suggest actionable improvements in `recommendations` for non-blocking polish.",
        "Reference tasks by `id` when possible so downstream tooling can attribute follow-ups.",
    ]
    if deliverables:
        joined = ", ".join(deliverables)
        notes.append(f"Confirm all deliverables are satisfied; flag any gaps from: {joined}.")
    notes.append("Keep criticism concise but precise enough for a refinement pass to act on.")
    return _dedupe_preserve_order(notes)


def _should_trigger_refinement(
    critique: PlannerCritique | None,
    feedback: Mapping[str, Any],
) -> bool:
    """Decide whether a refinement pass is required based on feedback and critiques."""
    blocking: set[str] = set()
    blocking_entries = feedback.get("blocking_issues") if isinstance(feedback, Mapping) else None
    if isinstance(blocking_entries, Sequence):
        for entry in blocking_entries:
            if isinstance(entry, str):
                cleaned = entry.strip()
                if cleaned:
                    blocking.add(cleaned)
    if critique is not None:
        if critique.blockers or critique.missing_deliverables or critique.dependency_conflicts:
            return True
        for issue in critique.issues:
            if issue.severity == "must_fix":
                return True
    return bool(blocking)


def _build_refinement_guidance(
    plan_response: PlannerResponse,
    *,
    analysis: PlannerAnalysis | None = None,
    critique: PlannerCritique | None = None,
    deliverables: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Compute planner refinement directives based on analysis, critique, and plan quality."""
    directives: list[str] = []
    feedback: dict[str, Any] = {}
    blocking_reasons: list[str] = []

    tasks = list(plan_response.tasks)
    task_count = len(tasks)
    feedback["task_count"] = task_count
    if task_count < 3:
        directives.append(
            "Expand the plan so it contains at least three concrete tasks covering discovery, implementation, and validation."
        )
        feedback["min_tasks"] = {"current": task_count, "required": 3}
        blocking_reasons.append("insufficient_task_count")

    missing_fields: list[dict[str, Any]] = []
    for task in tasks:
        missing: list[str] = []
        if not task.deliverables:
            missing.append("deliverables")
        if not task.acceptance_criteria:
            missing.append("acceptance_criteria")
        if missing:
            missing_fields.append(
                {
                    "task": task.id or task.title,
                    "title": task.title,
                    "missing": missing,
                }
            )
    if missing_fields:
        directives.append(
            "Fill in deliverables and acceptance criteria for the tasks listed in `stage_feedback.missing_fields`."
        )
        feedback["missing_fields"] = missing_fields
        blocking_reasons.append("missing_fields")

    duplicates = _detect_duplicate_tasks(plan_response)
    if duplicates:
        directives.append("Remove duplicate tasks so each title and summary is unique.")
        feedback["duplicate_tasks"] = duplicates

    if deliverables:
        uncovered_deliverables = _detect_uncovered_deliverables(plan_response, deliverables)
        if uncovered_deliverables:
            directives.append(
                "Ensure each requested deliverable is owned by at least one task; see `stage_feedback.missing_deliverables`."
            )
            feedback["missing_deliverables"] = uncovered_deliverables
            blocking_reasons.append("missing_deliverables")

    metadata_gaps = _detect_metadata_gaps(plan_response)
    if metadata_gaps:
        directives.append(
            "Populate `metadata.touched_files` and `metadata.validation_commands` for every task listed in `stage_feedback.metadata_gaps`."
        )
        feedback["metadata_gaps"] = metadata_gaps
        blocking_reasons.append("missing_metadata")

    dependency_issues, dependency_cycles = _detect_dependency_conflicts(plan_response)
    if dependency_issues:
        directives.append(
            "Resolve unresolved dependencies listed in `stage_feedback.unresolved_dependencies` by using task IDs or titles from this plan."
        )
        feedback["unresolved_dependencies"] = dependency_issues
        blocking_reasons.append("dependency_issues")
    if dependency_cycles:
        directives.append(
            "Break dependency cycles identified in `stage_feedback.dependency_cycles` so execution order is linear."
        )
        feedback["dependency_cycles"] = dependency_cycles
        blocking_reasons.append("dependency_cycles")

    component_gaps = _detect_component_gaps(plan_response, analysis)
    if component_gaps:
        directives.append(
            "Introduce explicit coverage for analysis components listed in `stage_feedback.uncovered_components`."
        )
        feedback["uncovered_components"] = component_gaps

    if critique is not None:
        feedback["critique"] = critique.model_dump(mode="python")
        if critique.blockers:
            directives.append("Address blockers enumerated in `stage_feedback.critique.blockers` before execution.")
            blocking_reasons.append("critic_blockers")
        must_fix_issues = [issue for issue in critique.issues if issue.severity == "must_fix"]
        if must_fix_issues:
            directives.append("Resolve all `must_fix` issues under `stage_feedback.critique.issues`.")
            blocking_reasons.append("critic_must_fix")
        elif critique.issues:
            directives.append("Incorporate improvements suggested in `stage_feedback.critique.issues` where appropriate.")
        if critique.missing_deliverables:
            existing_missing = set(feedback.get("missing_deliverables", []))
            combined = sorted(existing_missing.union(critique.missing_deliverables))
            feedback["missing_deliverables"] = combined
            blocking_reasons.append("missing_deliverables")
        if critique.dependency_conflicts:
            feedback["dependency_conflicts"] = critique.dependency_conflicts
            directives.append("Fix dependency conflicts highlighted in `stage_feedback.critique.dependency_conflicts`.")
            blocking_reasons.append("dependency_conflicts")
        if critique.recommendations:
            directives.append("Consider optional improvements in `stage_feedback.critique.recommendations` once blockers are resolved.")

    directives.append("Tighten the task breakdown so each step is narrowly scoped and directly tied to the goal.")
    directives.append("Ensure dependencies are explicit, linear, and free of cycles or missing references.")

    if blocking_reasons:
        feedback["blocking_issues"] = sorted({reason for reason in blocking_reasons if reason})

    return _dedupe_preserve_order(directives), feedback


def _compose_refinement_notes(directives: Sequence[str]) -> list[str]:
    """Translate refinement directives into planner notes."""
    notes = [
        "Refine the initial planner draft using the guidance in `stage_feedback`.",
        "Return valid JSON matching the PlannerResponse schema.",
    ]
    notes.extend(directives)
    notes.append("Do not repeat the draft verbatim; improve clarity, ordering, and validation coverage.")
    return _dedupe_preserve_order(notes)


def _detect_uncovered_deliverables(
    plan_response: PlannerResponse,
    deliverables: Sequence[str],
) -> list[str]:
    """Identify deliverables that are not referenced by any task."""
    uncovered: list[str] = []
    tasks = list(plan_response.tasks)
    if not tasks:
        return list(deliverables)

    task_bodies = []
    for task in tasks:
        metadata = task.metadata or {}
        fragments: list[str] = []
        fragments.extend(task.deliverables or [])
        fragments.extend(task.optional_deliverables or [])
        fragments.append(task.title or "")
        fragments.append(task.summary or "")
        touched = metadata.get("touched_files")
        if isinstance(touched, Sequence) and not isinstance(touched, str):
            fragments.extend(str(item) for item in touched if isinstance(item, str))
        task_bodies.append(" ".join(fragment.lower() for fragment in fragments if isinstance(fragment, str)))

    for deliverable in deliverables:
        if not isinstance(deliverable, str):
            continue
        cleaned = deliverable.strip()
        if not cleaned:
            continue
        needle = cleaned.lower()
        if any(needle in body for body in task_bodies):
            continue
        uncovered.append(cleaned)
    return uncovered


def _detect_metadata_gaps(plan_response: PlannerResponse) -> list[dict[str, Any]]:
    """Highlight tasks missing required metadata fields."""
    expected_fields = ("touched_files", "validation_commands")
    gaps: list[dict[str, Any]] = []
    for task in plan_response.tasks:
        metadata = task.metadata or {}
        missing: list[str] = []
        for field in expected_fields:
            value = metadata.get(field)
            if not _looks_like_non_empty_sequence(value):
                missing.append(field)
        if missing:
            gaps.append(
                {
                    "task": task.id or task.title,
                    "missing": missing,
                }
            )
    return gaps


def _detect_dependency_conflicts(
    plan_response: PlannerResponse,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    """Return unresolved dependency references and dependency cycles."""
    alias_map = _build_alias_map(plan_response)
    unresolved: list[dict[str, Any]] = []
    graph: dict[str, list[str]] = {}

    for task in plan_response.tasks:
        canonical = _canonical_task_label(task)
        graph.setdefault(canonical, [])

    for task in plan_response.tasks:
        canonical = _canonical_task_label(task)
        unresolved_refs: list[str] = []
        for dependency in task.depends_on:
            normalised = _normalise_dependency_reference(dependency)
            if not normalised:
                continue
            resolved = alias_map.get(normalised)
            if resolved is None:
                unresolved_refs.append(dependency)
                continue
            if resolved == canonical:
                continue
            graph.setdefault(canonical, []).append(resolved)
        if unresolved_refs:
            unresolved.append(
                {
                    "task": task.id or task.title,
                    "unresolved": [ref.strip() for ref in unresolved_refs if isinstance(ref, str)],
                }
            )

    cycles = _find_dependency_cycles(graph)
    return unresolved, cycles


def _detect_component_gaps(
    plan_response: PlannerResponse,
    analysis: PlannerAnalysis | None,
) -> list[dict[str, Any]]:
    """Surface analysis components that are not clearly covered by tasks."""
    if analysis is None or not analysis.components:
        return []

    task_texts: list[str] = []
    for task in plan_response.tasks:
        metadata = task.metadata or {}
        fragments: list[str] = []
        fragments.extend(task.deliverables or [])
        fragments.extend(task.optional_deliverables or [])
        fragments.append(task.title or "")
        fragments.append(task.summary or "")
        touched = metadata.get("touched_files")
        if isinstance(touched, Sequence) and not isinstance(touched, str):
            fragments.extend(str(item) for item in touched if isinstance(item, str))
        task_texts.append(" ".join(fragment.lower() for fragment in fragments if isinstance(fragment, str)))

    uncovered: list[dict[str, Any]] = []
    for component in analysis.components:
        keywords = [
            component.name,
            *(component.primary_paths or []),
            *(component.key_symbols or []),
        ]
        keywords = [keyword.strip() for keyword in keywords if isinstance(keyword, str) and keyword.strip()]
        if not keywords:
            continue
        matched = any(
            any(keyword.lower() in task_text for keyword in keywords)
            for task_text in task_texts
        )
        if not matched:
            uncovered.append(
                {
                    "component": component.name,
                    "keywords": keywords[:5],
                }
            )
    return uncovered


def _enrich_plan_with_analysis(
    plan_response: PlannerResponse,
    analysis: PlannerAnalysis | None,
) -> None:
    """Augment planner tasks with metadata inferred from analysis results."""
    if analysis is None:
        return

    deliverable_map: dict[str, list[str]] = {}
    for key, values in (analysis.deliverable_map or {}).items():
        if not isinstance(key, str):
            continue
        normalized_key = key.strip().lower()
        if not normalized_key:
            continue
        cleaned_values = [
            str(value).strip()
            for value in values or []
            if isinstance(value, str) and str(value).strip()
        ]
        if cleaned_values:
            deliverable_map[normalized_key] = cleaned_values

    global_validation_assets = [
        asset.strip()
        for asset in (analysis.validation_assets or [])
        if isinstance(asset, str) and asset.strip()
    ]

    enriched_tasks: list[PlannerTask] = []
    for task in plan_response.tasks:
        existing_metadata = task.metadata or {}
        metadata = dict(existing_metadata)

        touched_files = metadata.get("touched_files")
        if not _looks_like_non_empty_sequence(touched_files):
            candidate_paths: set[str] = set()
            for deliverable in task.deliverables or []:
                if not isinstance(deliverable, str):
                    continue
                lookup = deliverable.strip().lower()
                candidate_paths.update(deliverable_map.get(lookup, []))
            summary_lower = (task.summary or "").lower()
            if analysis.components:
                for component in analysis.components:
                    if not isinstance(component.name, str):
                        continue
                    component_match = component.name.lower() in summary_lower
                    if not component_match and component.primary_paths:
                        component_match = any(
                            isinstance(path, str) and path.lower() in summary_lower
                            for path in component.primary_paths
                        )
                    if component_match and component.primary_paths:
                        candidate_paths.update(
                            path.strip()
                            for path in component.primary_paths
                            if isinstance(path, str) and path.strip()
                        )
            if candidate_paths:
                metadata["touched_files"] = sorted(candidate_paths)[:8]

        validation_commands = metadata.get("validation_commands")
        if not _looks_like_non_empty_sequence(validation_commands):
            candidate_commands: set[str] = set(global_validation_assets[:5])
            if analysis.components:
                summary_lower = (task.summary or "").lower()
                for component in analysis.components:
                    if not isinstance(component.name, str):
                        continue
                    component_match = component.name.lower() in summary_lower
                    if not component_match and component.primary_paths:
                        component_match = any(
                            isinstance(path, str) and path.lower() in summary_lower
                            for path in component.primary_paths
                        )
                    if component_match and component.related_tests:
                        for test_ref in component.related_tests:
                            if not isinstance(test_ref, str):
                                continue
                            cleaned = test_ref.strip()
                            if not cleaned:
                                continue
                            if cleaned.startswith("pytest "):
                                candidate_commands.add(cleaned)
                            else:
                                candidate_commands.add(f"pytest {cleaned}")
            if candidate_commands:
                metadata["validation_commands"] = sorted(candidate_commands)[:5]

        if metadata != existing_metadata:
            enriched_tasks.append(task.model_copy(update={"metadata": metadata}))
        else:
            enriched_tasks.append(task)

    plan_response.tasks = enriched_tasks


def _looks_like_non_empty_sequence(value: Any) -> bool:
    if isinstance(value, str) or value is None:
        return False
    if isinstance(value, Sequence):
        return any(isinstance(item, str) and item.strip() for item in value)
    return False


def _build_alias_map(plan_response: PlannerResponse) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for task in plan_response.tasks:
        canonical = _canonical_task_label(task)
        for candidate in (task.id, task.title, canonical):
            if isinstance(candidate, str) and candidate.strip():
                alias_map[_normalise_dependency_reference(candidate)] = canonical
    return alias_map


def _canonical_task_label(task: PlannerTask) -> str:
    for candidate in (task.id, task.title, task.summary):
        if isinstance(candidate, str):
            cleaned = candidate.strip()
            if cleaned:
                return cleaned
    return "task"


def _normalise_dependency_reference(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    return cleaned.lower()


def _find_dependency_cycles(graph: Mapping[str, Sequence[str]]) -> list[list[str]]:
    """Detect cycles in the dependency graph."""
    cycles: list[list[str]] = []
    state: dict[str, str] = {}
    path: list[str] = []

    def visit(node: str) -> None:
        marker = state.get(node)
        if marker == "permanent":
            return
        if marker == "temporary":
            try:
                start_index = path.index(node)
            except ValueError:
                start_index = 0
            cycle = path[start_index:] + [node]
            if cycle not in cycles:
                cycles.append(cycle)
            return
        state[node] = "temporary"
        path.append(node)
        for neighbour in graph.get(node, []):
            visit(neighbour)
        path.pop()
        state[node] = "permanent"

    for node in graph:
        if state.get(node) != "permanent":
            visit(node)
    return cycles


def _serialise_pass_history(passes: Sequence[_PlannerPassResult]) -> list[dict[str, Any]]:
    """Produce a JSON-friendly view of each planning pass."""
    history: list[dict[str, Any]] = []
    for pass_result in passes:
        entry: dict[str, Any] = {
            "pass": pass_result.name,
            "request": deepcopy(pass_result.request_payload),
        }
        if pass_result.error:
            entry["error"] = pass_result.error
        else:
            if pass_result.raw_response is not None:
                entry["response"] = deepcopy(pass_result.raw_response)
            if pass_result.context is not None:
                entry["prompt"] = {
                    "system": pass_result.context.system_prompt,
                    "user": pass_result.context.user_prompt,
                    "metadata": pass_result.context.metadata,
                }
        history.append(entry)
    return history


def _attach_pass_history(
    raw_payload: Mapping[str, Any],
    passes: Sequence[_PlannerPassResult],
) -> dict[str, Any]:
    """Embed planner pass history alongside the final raw payload."""
    if not isinstance(raw_payload, Mapping):
        return dict(raw_payload)

    try:
        combined = deepcopy(raw_payload)
    except Exception:  # pragma: no cover - defensive copy guard
        combined = dict(raw_payload)

    final_pass = None
    for pass_result in reversed(passes):
        if pass_result.response is not None:
            final_pass = pass_result
            break

    history: list[dict[str, Any]] = []
    for pass_result in passes:
        entry: dict[str, Any] = {
            "pass": pass_result.name,
            "request": deepcopy(pass_result.request_payload),
        }
        if pass_result.error:
            entry["error"] = pass_result.error
        elif pass_result is not final_pass and pass_result.raw_response is not None:
            entry["response"] = deepcopy(pass_result.raw_response)
        history.append(entry)

    combined["_ae_planner_pass_history"] = history
    return combined


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    """Return items with duplicates removed while preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


@dataclass(slots=True)
class _TaskPersistenceResult:
    """Captured tasks persisted to memory along with alias mappings."""

    tasks: list[Task]
    alias_map: dict[str, str]


@dataclass(slots=True)
class _PlannerInputDedupeResult:
    """Outcome of normalising planner inputs before persistence."""

    constraints: list[str]
    deliverables: list[str]
    pruned_constraints: list[dict[str, Any]]
    pruned_deliverables: list[dict[str, Any]]


@dataclass(slots=True)
class _PlanValidationReport:
    """Aggregated validation findings when evaluating a planner response."""

    conflicts: list[dict[str, Any]]
    duplicate_tasks: list[dict[str, Any]]

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.conflicts:
            payload["conflicts"] = self.conflicts
        if self.duplicate_tasks:
            payload["duplicate_tasks"] = self.duplicate_tasks
        return payload


def _split_optional_deliverables(planner_task: PlannerTask) -> tuple[list[str], list[str]]:
    """Separate required and optional deliverables from a planner task."""
    optional: list[str] = []
    for entry in planner_task.optional_deliverables:
        if isinstance(entry, str):
            stripped = entry.strip()
            if stripped:
                optional.append(stripped)

    required: list[str] = []
    for entry in planner_task.deliverables:
        if not isinstance(entry, str):
            continue
        stripped = entry.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("optional:"):
            payload = stripped.split(":", 1)[1].strip()
            optional.append(payload or stripped)
        elif "(optional)" in lowered:
            cleaned = stripped.replace("(optional)", "").strip()
            optional.append(cleaned or stripped)
        else:
            required.append(stripped)

    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    return _dedupe(required), _dedupe(optional)


def _persist_tasks(
    store: MemoryStore,
    plan_id: str,
    planner_tasks: Sequence[PlannerTask],
) -> _TaskPersistenceResult:
    """Materialise planner tasks into storage while ensuring stable IDs."""
    aliases: dict[str, str] = {}
    persisted: list[Task] = []
    assigned_ids: set[str] = set()
    task_infos: list[tuple[str, PlannerTask, int]] = []

    for index, planner_task in enumerate(planner_tasks, start=1):
        base_candidate = planner_task.id or planner_task.title or f"task-{index}"
        task_id = _scoped_identifier(plan_id, base_candidate, assigned_ids)
        assigned_ids.add(task_id)
        task_infos.append((task_id, planner_task, index))
        _register_alias(aliases, planner_task.id, task_id)
        _register_alias(aliases, planner_task.title, task_id)
        _register_alias(aliases, task_id, task_id)

    for task_id, planner_task, index in task_infos:
        resolved_dependencies: list[str] = []
        unresolved_dependencies: list[str] = []
        for dependency in planner_task.depends_on:
            resolved = _resolve_dependency(dependency, aliases)
            if resolved is None or resolved == task_id:
                if resolved is None and dependency:
                    unresolved_dependencies.append(dependency)
                continue
            if resolved not in resolved_dependencies:
                resolved_dependencies.append(resolved)

        status = TaskStatus.READY if not resolved_dependencies else TaskStatus.BLOCKED
        priority = planner_task.priority if planner_task.priority is not None else max(100 - index, 0)

        required_deliverables, optional_deliverables = _split_optional_deliverables(planner_task)
        if planner_task.deliverables != required_deliverables:
            planner_task.deliverables = list(required_deliverables)
        if planner_task.optional_deliverables != optional_deliverables:
            planner_task.optional_deliverables = list(optional_deliverables)

        planner_snapshot = planner_task.model_dump(exclude_none=True)

        metadata: dict[str, Any] = {
            "planner_task": planner_snapshot,
            "deliverables": required_deliverables,
        }
        if optional_deliverables:
            metadata["optional_deliverables"] = optional_deliverables
        if unresolved_dependencies:
            metadata["unresolved_dependencies"] = unresolved_dependencies

        task = Task(
            id=task_id,
            plan_id=plan_id,
            title=planner_task.title.strip(),
            summary=planner_task.summary.strip(),
            status=status,
            depends_on=resolved_dependencies,
            metadata=metadata,
            priority=priority,
        )
        store.save_task(task)
        persisted.append(task)

    return _TaskPersistenceResult(tasks=persisted, alias_map=aliases)


def _persist_decisions(
    store: MemoryStore,
    plan_id: str,
    alias_map: Mapping[str, str],
    planner_decisions: Sequence[PlannerDecision],
    planner_risks: Sequence[PlannerRisk],
) -> list[Decision]:
    """Persist planner decisions and risks, resolving task aliases along the way."""
    decisions: list[Decision] = []
    assigned_ids: set[str] = set()

    for index, planner_decision in enumerate(planner_decisions, start=1):
        decision_id = _scoped_identifier(plan_id, planner_decision.id or f"decision-{index}", assigned_ids)
        assigned_ids.add(decision_id)
        task_ref = _resolve_dependency(planner_decision.task_id, alias_map)
        decision = Decision(
            id=decision_id,
            plan_id=plan_id,
            task_id=task_ref,
            title=planner_decision.title.strip(),
            content=planner_decision.content.strip(),
            kind=planner_decision.kind or "general",
            metadata={
                "planner_decision": planner_decision.model_dump(exclude_none=True),
            },
        )
        store.record_decision(decision)
        decisions.append(decision)

    for index, planner_risk in enumerate(planner_risks, start=1):
        risk_id = _scoped_identifier(plan_id, planner_risk.id or f"risk-{index}", assigned_ids)
        assigned_ids.add(risk_id)
        lines = [planner_risk.description.strip()]
        mitigation = (planner_risk.mitigation or "").strip()
        if mitigation:
            lines.append(f"Mitigation: {mitigation}")
        impact = (planner_risk.impact or "").strip()
        likelihood = (planner_risk.likelihood or "").strip()
        if impact or likelihood:
            segments = []
            if impact:
                segments.append(f"Impact={impact}")
            if likelihood:
                segments.append(f"Likelihood={likelihood}")
            lines.append(" / ".join(segments))
        decision = Decision(
            id=risk_id,
            plan_id=plan_id,
            title=f"Risk: {planner_risk.description.strip()}",
            content="\n".join(lines),
            kind="risk",
            metadata={
                "planner_risk": planner_risk.model_dump(exclude_none=True),
            },
        )
        store.record_decision(decision)
        decisions.append(decision)

    return decisions


def _ensure_baseline_decisions(
    *,
    store: MemoryStore,
    plan: Plan,
    decisions: Sequence[Decision],
) -> list[Decision]:
    """Guarantee core decision records exist even when the planner omits them."""
    existing = {decision.kind for decision in decisions}
    required: dict[str, tuple[str, str]] = {
        "architecture": (
            "Architecture sketch",
            f"Baseline architecture context for goal '{plan.goal}'.",
        ),
        "contract": (
            "Interface contracts",
            "Document implicit interface contracts discovered during planning.",
        ),
        "risk": (
            "Risk register",
            "Track key risks and mitigations for this plan.",
        ),
    }
    updated = list(decisions)
    for kind, (title, content) in required.items():
        if kind in existing:
            continue
        decision = Decision(
            id=f"{plan.id}::decision-{kind}",
            plan_id=plan.id,
            title=title,
            content=content,
            kind=kind,
            metadata={"source": "bootstrap_fallback"},
        )
        store.record_decision(decision)
        updated.append(decision)
    return updated


def _persist_checkpoint(
    *,
    store: MemoryStore,
    plan_id: str,
    goal: str,
    plan_summary: str,
    tasks: Sequence[Task],
    decisions: Sequence[Decision],
) -> Checkpoint:
    """Capture the bootstrap checkpoint summarising tasks, decisions, and tests."""
    ready = [task.id for task in tasks if task.status == TaskStatus.READY]
    blocked = [task.id for task in tasks if task.status == TaskStatus.BLOCKED]
    decision_groups: dict[str, list[str]] = defaultdict(list)
    for decision in decisions:
        decision_groups[decision.kind].append(decision.id)

    checkpoint = Checkpoint(
        id=f"{plan_id}::checkpoint-0",
        plan_id=plan_id,
        label="checkpoint-0",
        payload={
            "goal": goal,
            "plan_summary": plan_summary,
            "ready_tasks": ready,
            "blocked_tasks": blocked,
            "decisions": dict(decision_groups),
        },
        metadata={
            "source": "ae init --plan",
            "workflow_version": 10,
        },
    )
    store.save_checkpoint(checkpoint)

    return checkpoint


def _planning_request_payload(
    *,
    goal: str,
    constraints: Sequence[str],
    deliverables: Sequence[str],
    deadline: str | None,
    notes: Sequence[str],
    known_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the structured payload sent to the planning phase."""
    constraints_list = list(constraints)
    for extra in _infer_source_layout_constraints(known_context):
        if extra not in constraints_list:
            constraints_list.append(extra)

    payload: dict[str, Any] = {
        "goal": goal,
        "constraints": constraints_list,
        "deliverables": list(deliverables),
        "known_context": known_context,
    }
    if deadline:
        payload["deadline"] = deadline
    if notes:
        payload["notes"] = list(notes)
    return payload


def _collect_repo_summary(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise repository layout details that inform planning prompts."""
    top_level_dirs = [
        entry.name
        for entry in sorted(repo_root.iterdir(), key=lambda p: p.name)
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    src_root = repo_root / "src"
    src_packages = []
    if src_root.exists():
        for path in sorted(src_root.iterdir(), key=lambda p: p.name):
            if path.is_dir() and not path.name.startswith("."):
                src_packages.append(path.relative_to(repo_root).as_posix())

    policy_cfg = config.get("policy") or {}
    policy_path = None
    if isinstance(policy_cfg, Mapping):
        candidate = policy_cfg.get("capsule_path")
        if isinstance(candidate, str) and candidate.strip():
            path = Path(candidate.strip())
            if not path.is_absolute():
                path = (repo_root / path).resolve()
            policy_path = path.as_posix()

    package_layout = _extract_package_layout(repo_root)

    summary = {
        "repo_root": repo_root.as_posix(),
        "top_level_dirs": top_level_dirs[:20],
        "src_packages": src_packages[:20],
        "tests_present": (repo_root / "tests").exists(),
        "config": {key: value for key, value in config.items() if key in {"project", "iteration"}},
    }
    if policy_path:
        summary["policy_capsule_path"] = policy_path
    if package_layout:
        summary["package_layout"] = package_layout
    contracts = _build_workspace_contracts(repo_root)
    if contracts:
        api_hints = _collect_test_api_hints(repo_root, config, contracts)
        if api_hints:
            contracts["api_hints"] = api_hints
        summary["workspace_contracts"] = contracts
    return summary


def _prune_completed_goal_inputs(
    *,
    repo_root: Path,
    repo_summary: Mapping[str, Any],
    constraints: Sequence[str],
    deliverables: Sequence[str],
) -> _PlannerInputDedupeResult:
    """Drop constraints or deliverables already satisfied in the workspace."""
    workspace_contracts = repo_summary.get("workspace_contracts")
    module_index: dict[str, str] = {}
    if isinstance(workspace_contracts, Mapping):
        raw_index = workspace_contracts.get("module_index")
        if isinstance(raw_index, Mapping):
            module_index = {
                str(name): str(path)
                for name, path in raw_index.items()
                if isinstance(name, str) and isinstance(path, str)
            }

    module_paths = {Path(path).as_posix() for path in module_index.values()}
    clean_paths = _resolve_clean_workspace_paths(repo_root, module_index)

    def _process_entries(entries: Sequence[str], *, kind: str) -> tuple[list[str], list[dict[str, Any]]]:
        kept: list[str] = []
        removed: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, str):
                continue
            trimmed = entry.strip()
            if not trimmed:
                continue
            targets = _extract_targets_from_text(trimmed, module_index, module_paths)
            if not targets:
                kept.append(entry)
                continue
            remaining = [target for target in targets if target not in clean_paths]
            if remaining:
                kept.append(entry)
                continue
            removed.append(
                {
                    "kind": kind,
                    "value": trimmed,
                    "targets": sorted(targets),
                }
            )
        return kept, removed

    filtered_constraints, pruned_constraints = _process_entries(constraints, kind="constraint")
    filtered_deliverables, pruned_deliverables = _process_entries(deliverables, kind="deliverable")

    return _PlannerInputDedupeResult(
        constraints=filtered_constraints,
        deliverables=filtered_deliverables,
        pruned_constraints=pruned_constraints,
        pruned_deliverables=pruned_deliverables,
    )


def _validate_plan_against_contracts(
    plan_response: PlannerResponse,
    repo_summary: Mapping[str, Any],
) -> _PlanValidationReport:
    """Compare planner output against repository contracts and detect issues."""
    conflicts = _detect_storage_conflicts(plan_response, repo_summary)
    duplicate_tasks = _detect_duplicate_tasks(plan_response)
    return _PlanValidationReport(conflicts=conflicts, duplicate_tasks=duplicate_tasks)


def _apply_plan_validation_annotations(
    plan_response: PlannerResponse,
    report: _PlanValidationReport,
) -> None:
    """Inject validation findings back into the planner response."""
    if report.conflicts:
        for index, conflict in enumerate(report.conflicts, start=1):
            title = conflict.get("issue", "repository contract conflict")
            content = conflict.get("details") or ""
            plan_response.decisions.append(
                PlannerDecision(
                    id=None,
                    title=f"Conflict Detected: {title}",
                    content=content or "Detected conflict with repository contracts. Resolve before execution.",
                    kind="conflict",
                )
            )

    if report.duplicate_tasks:
        lines = []
        for entry in report.duplicate_tasks:
            current = entry.get("task_id") or entry.get("title") or "(unknown task)"
            other = entry.get("conflicts_with") or "(duplicate)"
            lines.append(f"- {current} duplicates {other}")
        content = "\n".join(lines) if lines else "Duplicate tasks detected in plan output."
        plan_response.decisions.append(
            PlannerDecision(
                id=None,
                title="Duplicate tasks detected",
                content=content,
                kind="quality",
            )
        )


def _detect_duplicate_tasks(plan_response: PlannerResponse) -> list[dict[str, Any]]:
    """Identify planner tasks that repeat the same title or summary."""
    duplicates: list[dict[str, Any]] = []
    seen_titles: dict[str, str] = {}
    seen_summaries: dict[str, str] = {}

    for task in plan_response.tasks:
        task_label = task.id or task.title
        title_key = task.title.strip().lower()
        if title_key:
            if title_key in seen_titles:
                duplicates.append(
                    {
                        "task_id": task_label,
                        "title": task.title.strip(),
                        "conflicts_with": seen_titles[title_key],
                        "reason": "duplicate_title",
                    }
                )
            else:
                seen_titles[title_key] = task_label

        summary_key = " ".join(task.summary.split()).lower()
        if summary_key:
            if summary_key in seen_summaries:
                duplicates.append(
                    {
                        "task_id": task_label,
                        "title": task.title.strip(),
                        "conflicts_with": seen_summaries[summary_key],
                        "reason": "duplicate_summary",
                    }
                )
            else:
                seen_summaries[summary_key] = task_label

    return duplicates


def _detect_storage_conflicts(
    plan_response: PlannerResponse,
    repo_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Highlight mismatches between planner storage references and contracts."""
    workspace_contracts = repo_summary.get("workspace_contracts")
    storage_contract = None
    if isinstance(workspace_contracts, Mapping):
        storage_section = workspace_contracts.get("storage_entry_points")
        if isinstance(storage_section, Mapping):
            storage_contract = storage_section.get("memory_store")

    canonical_db_path: str | None = None
    if isinstance(storage_contract, Mapping):
        candidate = storage_contract.get("default_db_path")
        if isinstance(candidate, str) and candidate.strip():
            canonical_db_path = candidate.strip()

    sqlite_paths: set[str] = set()
    for entry in _gather_plan_text_entries(plan_response):
        for match in _DATA_PATH_RE.finditer(entry):
            path = match.group(0)
            if path.endswith(".sqlite"):
                sqlite_paths.add(path)

    if not sqlite_paths:
        return []

    conflicts: list[dict[str, Any]] = []
    if canonical_db_path:
        extras = sorted(path for path in sqlite_paths if path != canonical_db_path)
        if extras or canonical_db_path not in sqlite_paths:
            mentioned = sorted(sqlite_paths)
            conflicts.append(
                {
                    "issue": "storage_path_conflict",
                    "details": (
                        f"Planner output references sqlite paths {mentioned}, "
                        f"but repository canonical database lives at '{canonical_db_path}'. "
                        "Reconcile storage references to avoid divergent vaults."
                    ),
                }
            )
    elif len(sqlite_paths) > 1:
        conflicts.append(
            {
                "issue": "multiple_sqlite_targets",
                "details": (
                    f"Planner output references multiple sqlite paths {sorted(sqlite_paths)}. "
                    "Clarify which datastore should be used for the current goal."
                ),
            }
        )

    return conflicts


def _gather_plan_text_entries(plan_response: PlannerResponse) -> list[str]:
    """Collect free-form text fields from the planner response for analysis."""
    entries: list[str] = []
    for task in plan_response.tasks:
        entries.extend(
            [
                getattr(task, "title", ""),
                getattr(task, "summary", ""),
            ]
        )
        for collection_name in ("constraints", "deliverables", "optional_deliverables", "notes"):
            for value in getattr(task, collection_name, []):
                if isinstance(value, str):
                    entries.append(value)
    for decision in plan_response.decisions:
        entries.append(getattr(decision, "content", ""))
    return [entry for entry in entries if isinstance(entry, str) and entry]


def _resolve_clean_workspace_paths(repo_root: Path, module_index: Mapping[str, str]) -> set[str]:
    """Determine which module paths are clean in git and safe to skip."""
    module_paths = {Path(path).as_posix() for path in module_index.values() if isinstance(path, str)}
    if not module_paths:
        return set()

    try:
        repo = GitRepository(repo_root)
    except GitError:
        repo = None

    if repo is None:
        clean: set[str] = set()
        for path in module_paths:
            candidate = (repo_root / path).resolve()
            if candidate.exists():
                clean.add(path)
        return clean

    tracked_paths = {tracked.as_posix() for tracked in repo.list_tracked_paths()}
    dirty_paths = _collect_dirty_paths(repo)

    clean_paths: set[str] = set()
    for path in module_paths:
        if path in tracked_paths and path not in dirty_paths:
            clean_paths.add(path)
    return clean_paths


def _collect_dirty_paths(repo: GitRepository) -> set[str]:
    """Return the set of paths currently modified or indexed in the repository."""
    result = repo.git("status", "--short", check=False)
    if result.returncode != 0:
        return set()

    dirty_paths: set[str] = set()
    for raw_line in result.stdout.splitlines():
        line = raw_line.rstrip()
        if not line or len(line) < 4:
            continue
        fragment = line[3:]
        if " -> " in fragment:
            fragment = fragment.split(" -> ", 1)[1]
        fragment = fragment.strip()
        if not fragment:
            continue
        if fragment.startswith('"') and fragment.endswith('"'):
            fragment = fragment[1:-1]
        normalized = Path(fragment).as_posix()
        dirty_paths.add(normalized)
    return dirty_paths


def _extract_targets_from_text(
    text: str,
    module_index: Mapping[str, str],
    module_paths: set[str],
) -> set[str]:
    """Extract module or file targets referenced in free-form planner text."""
    if not text:
        return set()

    lowered = text.lower()
    targets: set[str] = set()

    for module_name, rel_path in module_index.items():
        candidate_token = module_name.lower()
        if candidate_token and candidate_token in lowered:
            targets.add(Path(rel_path).as_posix())
            continue
        dotted = module_name.replace(".", "/").lower()
        if dotted and dotted in lowered:
            targets.add(Path(rel_path).as_posix())

    for match in _TARGET_PATH_RE.finditer(text):
        candidate = match.group(0)
        candidate = candidate.split("::", 1)[0]
        candidate = candidate.split(":", 1)[0]
        cleaned = candidate.strip().strip("./")
        if not cleaned:
            continue
        normalized = Path(cleaned).as_posix()
        if normalized in module_paths:
            targets.add(normalized)
            continue
        prefixed = f"src/{normalized}"
        if prefixed in module_paths:
            targets.add(prefixed)

    return targets


def _extract_package_layout(repo_root: Path) -> dict[str, Any]:
    """Parse ``pyproject.toml`` to understand package and source directory layout."""
    if tomllib is None:
        return {}

    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.exists():
        return {}

    try:
        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, TOMLDecodeError):
        return {}

    layout: dict[str, Any] = {"pyproject": pyproject_path.relative_to(repo_root).as_posix()}
    source_roots: set[str] = set()
    package_dir_map: dict[str, str] = {}
    project_packages: set[str] = set()

    project_cfg = data.get("project")
    if isinstance(project_cfg, Mapping):
        packages_section = project_cfg.get("packages")
        if isinstance(packages_section, Sequence):
            for entry in packages_section:
                if isinstance(entry, str) and entry.strip():
                    project_packages.add(entry.strip())

    tool_cfg = data.get("tool")
    if isinstance(tool_cfg, Mapping):
        setuptools_cfg = tool_cfg.get("setuptools")
        if isinstance(setuptools_cfg, Mapping):
            pkg_dir_cfg = setuptools_cfg.get("package-dir")
            if isinstance(pkg_dir_cfg, Mapping):
                for key, value in pkg_dir_cfg.items():
                    if not isinstance(value, str):
                        continue
                    cleaned = value.strip().strip("/")
                    if not cleaned:
                        continue
                    package_dir_map[str(key or "")] = cleaned
                    source_roots.add(cleaned.split("/", 1)[0])
            packages_cfg = setuptools_cfg.get("packages")
            if isinstance(packages_cfg, Mapping):
                find_cfg = packages_cfg.get("find")
                if isinstance(find_cfg, Mapping):
                    where = find_cfg.get("where")
                    if isinstance(where, Sequence):
                        for entry in where:
                            if isinstance(entry, str) and entry.strip():
                                cleaned = entry.strip().strip("/")
                                if cleaned:
                                    source_roots.add(cleaned.split("/", 1)[0])
        poetry_cfg = tool_cfg.get("poetry")
        if isinstance(poetry_cfg, Mapping):
            packages = poetry_cfg.get("packages")
            if isinstance(packages, Sequence):
                for package_entry in packages:
                    if not isinstance(package_entry, Mapping):
                        continue
                    include = package_entry.get("include")
                    source_from = package_entry.get("from")
                    if isinstance(source_from, str) and source_from.strip():
                        cleaned_root = source_from.strip().strip("/")
                        source_roots.add(cleaned_root.split("/", 1)[0])
                        if isinstance(include, str) and include.strip():
                            project_packages.add(f"{cleaned_root}/{include.strip()}".strip("/"))

    if package_dir_map:
        layout["package_dir_map"] = dict(sorted(package_dir_map.items()))
    if project_packages:
        layout["project_packages"] = sorted(project_packages)
    if source_roots:
        layout["source_roots"] = sorted(source_roots)

    if len(layout) == 1:  # only pyproject path present
        return {}
    return layout


def _build_workspace_contracts(repo_root: Path) -> dict[str, Any]:
    """
    Assemble a compact view of the Python workspace so the planning prompt can
    reference concrete modules and the imports exercised by tests. This helps
    downstream tasks avoid conflicting guidance about where functionality
    belongs.
    """

    src_root = repo_root / "src"
    contracts: dict[str, Any] = {}

    module_index = _collect_python_module_index(repo_root, src_root)
    if module_index:
        sorted_items = sorted(module_index.items())
        trimmed = sorted_items[:120]
        contracts["module_index"] = {name: path for name, path in trimmed}

    test_imports = _collect_test_imports(repo_root / "tests")
    if test_imports:
        trimmed_imports = test_imports[:120]
        contracts["test_imports"] = trimmed_imports
        if module_index:
            mapped = {
                name: module_index[name]
                for name in trimmed_imports
                if name in module_index
            }
            if mapped:
                contracts["test_module_paths"] = mapped

    storage_contract = _collect_storage_entry_points(repo_root)
    if storage_contract:
        contracts["storage_entry_points"] = storage_contract

    cli_entry_points = _collect_cli_entry_points(repo_root)
    if cli_entry_points:
        contracts["cli_entry_points"] = cli_entry_points

    layout_rules: list[str] = []
    memory_store = None
    context_builder = None
    if isinstance(storage_contract, Mapping):
        memory_store = storage_contract.get("memory_store")
        context_builder = storage_contract.get("context_builder")
    if isinstance(memory_store, Mapping):
        default_db = memory_store.get("default_db_path")
        if isinstance(default_db, str) and default_db.strip():
            layout_rules.append(f"Persist plan state in '{default_db.strip()}' via MemoryStore.")
    if isinstance(context_builder, Mapping):
        data_root = context_builder.get("default_data_root")
        index_root = context_builder.get("default_index_root")
        logs_root = context_builder.get("default_logs_root")
        if all(isinstance(item, str) and item.strip() for item in (data_root, index_root, logs_root)):
            layout_rules.append(
                f"Data root defaults to '{data_root}'; reuse '{index_root}' for indexes and '{logs_root}' for logs."
            )
    if cli_entry_points:
        command_names = ", ".join(entry["command"] for entry in cli_entry_points[:5] if "command" in entry)
        if command_names:
            layout_rules.append(
                f"Extend CLI commands in 'ae.cli' ({command_names}) rather than creating ad-hoc entry points."
            )
    if layout_rules:
        contracts["layout_rules"] = layout_rules

    return contracts


def _collect_test_api_hints(
    repo_root: Path,
    config: Mapping[str, Any],
    contracts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Surface function/class signatures that tests import for API awareness."""
    module_index = contracts.get("module_index")
    test_imports = contracts.get("test_imports")
    if not isinstance(module_index, Mapping) or not isinstance(test_imports, Sequence):
        return []

    module_map: dict[str, str] = {
        str(name): str(path)
        for name, path in module_index.items()
        if isinstance(name, str) and isinstance(path, str)
    }
    if not module_map:
        return []

    relevant_modules = [
        name for name in test_imports if isinstance(name, str) and name in module_map
    ]
    if not relevant_modules:
        return []

    data_root = ContextBuilder._infer_data_root(config, repo_root)  # type: ignore[attr-defined]
    symbol_index_path = (data_root / "index" / "symbols.json").resolve()
    if not symbol_index_path.exists():
        return []

    try:
        symbol_index = SymbolIndex(symbol_index_path)
    except Exception:
        return []

    hints: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for module_name in relevant_modules:
        rel_path = module_map[module_name]
        try:
            records = symbol_index.symbols_for_path(rel_path)
        except Exception:
            continue
        for record in records:
            if record.kind not in {"function", "class"}:
                continue
            signature_key = (record.qualified_name, record.signature)
            if signature_key in seen:
                continue
            seen.add(signature_key)
            hint = {
                "module": module_name,
                "path": rel_path,
                "qualified_name": record.qualified_name,
                "symbol": record.name,
                "kind": record.kind,
                "signature": record.signature,
            }
            hints.append(hint)
            if len(hints) >= 120:
                return hints[:120]
    return hints[:120]


def _collect_python_module_index(repo_root: Path, src_root: Path) -> dict[str, str]:
    """Generate a mapping of module names to file paths under ``src``."""
    if not src_root.exists():
        return {}

    module_index: dict[str, str] = {}
    for path in sorted(src_root.rglob("*.py")):
        if not path.is_file():
            continue
        module_name = _module_name_from_path(path, src_root)
        if not module_name:
            continue
        rel_path = path.relative_to(repo_root).as_posix()
        module_index.setdefault(module_name, rel_path)
    return module_index


def _module_name_from_path(path: Path, src_root: Path) -> str | None:
    """Derive a dotted module name for a file inside ``src``."""
    try:
        relative = path.relative_to(src_root)
    except ValueError:
        return None

    parts = list(relative.parts)
    if not parts:
        return None

    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        last = parts[-1]
        if not last.endswith(".py"):
            return None
        parts[-1] = last[:-3]

    if not parts:
        return None
    return ".".join(parts)


def _collect_test_imports(tests_root: Path) -> list[str]:
    """Aggregate modules imported across the test suite."""
    if not tests_root.exists():
        return []

    modules: set[str] = set()
    for path in sorted(tests_root.rglob("*.py")):
        if not path.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.strip()
                    if name:
                        modules.add(name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base = node.module.strip()
                    if base:
                        modules.add(base)
                        for alias in node.names:
                            target = alias.name.strip()
                            if target and target != "*":
                                modules.add(f"{base}.{target}")
    return sorted(modules)


def _collect_cli_entry_points(repo_root: Path) -> list[dict[str, Any]]:
    """Inspect the CLI module for Typer command registrations."""
    cli_path = repo_root / "src" / "ae" / "cli.py"
    if not cli_path.exists():
        return []

    try:
        source = cli_path.read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    commands: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        explicit_name: str | None = None
        matched = False
        for decorator in node.decorator_list:
            marker = _extract_cli_command_name(decorator)
            if marker is None:
                continue
            matched = True
            if marker:
                explicit_name = marker
            break
        if not matched:
            continue

        command_token = explicit_name or node.name
        entry: dict[str, Any] = {
            "command": command_token,
            "callable": f"ae.cli.{node.name}",
            "python_function": node.name,
            "source_path": cli_path.relative_to(repo_root).as_posix(),
        }
        docstring = ast.get_docstring(node)
        if docstring:
            summary = docstring.strip()
            if summary:
                entry["summary"] = summary
        invocation = command_token.replace("_", "-")
        if invocation != command_token:
            entry["invocation"] = invocation
        commands.append(entry)

    commands.sort(key=lambda item: item["command"])
    return commands[:40]


def _extract_cli_command_name(decorator: ast.AST) -> str | None:
    """Return the explicit command name from a Typer decorator if provided."""
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr != "command":
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "app":
        return None

    explicit: str | None = None
    if decorator.args:
        first = decorator.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            candidate = first.value.strip()
            if candidate:
                explicit = candidate
    for keyword in decorator.keywords or ():
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    explicit = candidate
    return explicit or ""


def _collect_storage_entry_points(repo_root: Path) -> dict[str, Any]:
    """Document memory store and context builder entry points in the repo."""
    storage: dict[str, Any] = {}

    store_path = repo_root / "src" / "ae" / "memory" / "store.py"
    if store_path.exists():
        entry: dict[str, Any] = {
            "class": "MemoryStore",
            "module": "ae.memory.store",
            "path": store_path.relative_to(repo_root).as_posix(),
        }
        default_db = _extract_memory_store_db_path(store_path)
        if default_db:
            entry["default_db_path"] = default_db
        storage["memory_store"] = entry

    context_builder_path = repo_root / "src" / "ae" / "context_builder.py"
    if context_builder_path.exists():
        storage["context_builder"] = {
            "class": "ContextBuilder",
            "module": "ae.context_builder",
            "path": context_builder_path.relative_to(repo_root).as_posix(),
            "default_data_root": "data",
            "default_index_root": "data/index",
            "default_logs_root": "data/logs",
        }

    return storage


def _extract_memory_store_db_path(store_path: Path) -> str | None:
    """Read the default database path constant from ``memory.store``."""
    try:
        source = store_path.read_text(encoding="utf-8")
    except OSError:
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "DEFAULT_DB_PATH":
                literal = _extract_path_literal(node.value)
                if literal:
                    return literal
    return None


def _extract_path_literal(node: ast.AST) -> str | None:
    """Extract string literals representing paths from AST nodes."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "Path" and node.args:
            candidate = node.args[0]
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                text = candidate.value.strip()
                if text:
                    return text
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value.strip()
        if text:
            return text
    return None


def _infer_source_layout_constraints(known_context: Mapping[str, Any]) -> list[str]:
    """Generate constraint strings that reinforce the existing source layout."""
    roots: set[str] = set()
    package_paths: set[str] = set()

    layout = known_context.get("package_layout")
    if isinstance(layout, Mapping):
        source_roots = layout.get("source_roots")
        if isinstance(source_roots, Sequence):
            for root in source_roots:
                if isinstance(root, str) and root.strip():
                    cleaned = root.strip().strip("/")
                    if cleaned:
                        roots.add(cleaned.split("/", 1)[0])
        package_dir_map = layout.get("package_dir_map")
        if isinstance(package_dir_map, Mapping):
            for value in package_dir_map.values():
                if isinstance(value, str) and value.strip():
                    cleaned = value.strip().strip("/")
                    package_paths.add(cleaned)
                    roots.add(cleaned.split("/", 1)[0])
        project_packages = layout.get("project_packages")
        if isinstance(project_packages, Sequence):
            for pkg in project_packages:
                if isinstance(pkg, str) and pkg.strip():
                    cleaned = pkg.strip().strip("/")
                    package_paths.add(cleaned)
                    if "/" in cleaned:
                        roots.add(cleaned.split("/", 1)[0])

    src_packages = known_context.get("src_packages")
    if isinstance(src_packages, Sequence):
        for entry in src_packages:
            if isinstance(entry, str) and entry.strip():
                cleaned = entry.strip().strip("/")
                package_paths.add(cleaned)
                if "/" in cleaned:
                    roots.add(cleaned.split("/", 1)[0])
                else:
                    roots.add(cleaned)

    roots.discard(".")
    package_paths.discard(".")

    messages: list[str] = []
    if roots:
        formatted_roots = ", ".join(f"`{root}/`" for root in sorted(roots))
        messages.append(f"Use the existing src-layout: add Python modules inside {formatted_roots}.")
    if package_paths:
        formatted_paths = ", ".join(f"`{path}`" for path in sorted(package_paths))
        messages.append(f"Prefer extending the existing package directories: {formatted_paths}.")
    return messages


def _persist_llm_exchange(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    plan_id: str,
    goal: str,
    request_payload: Mapping[str, Any],
    context_package: ContextPackage,
    raw_response: Any,
    parsed_response: Mapping[str, Any],
    model_name: str,
) -> None:
    """Persist the full request/response payload for plan-level LLM calls."""
    try:
        logs_dir = _resolve_logs_dir(config, repo_root)
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "timestamp": timestamp,
        "phase": PhaseName.PLAN.value,
        "model": model_name,
        "plan_id": plan_id,
        "goal": goal,
        "request": {
            "system_prompt": context_package.system_prompt,
            "user_prompt": context_package.user_prompt,
            "metadata": context_package.metadata,
            "payload": request_payload,
        },
        "response": {
            "raw": raw_response,
            "parsed": parsed_response,
        },
    }

    file_name = f"plan_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    log_path = logs_dir / file_name
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            json.dump(entry, handle, indent=2, sort_keys=True, default=str)
    except OSError:
        return


def _resolve_logs_dir(config: Mapping[str, Any], repo_root: Path) -> Path:
    """Determine where to store LLM exchange logs for planning."""
    paths_cfg = config.get("paths")
    logs_candidate: Any = None
    if isinstance(paths_cfg, Mapping):
        logs_candidate = paths_cfg.get("logs")

    if isinstance(logs_candidate, str) and logs_candidate.strip():
        candidate_path = Path(logs_candidate.strip())
        if not candidate_path.is_absolute():
            candidate_path = (repo_root / candidate_path).resolve()
        return candidate_path

    data_root = ContextBuilder._infer_data_root(config, repo_root)  # type: ignore[attr-defined]
    return (data_root / "logs").resolve()


def _derive_plan_name(candidate: str, project_name: str) -> str:
    """Choose a readable plan name, falling back to the project name."""
    stripped = candidate.strip()
    if stripped:
        return stripped
    return f"{project_name} execution plan"


def _register_alias(alias_map: dict[str, str], alias: str | None, task_id: str) -> None:
    """Record alternate identifiers that resolve to a task ID."""
    if not alias:
        return
    normalized = _normalize_alias(alias)
    if normalized not in alias_map:
        alias_map[normalized] = task_id
    alias_clean = alias.strip()
    if alias_clean and alias_clean not in alias_map:
        alias_map[alias_clean] = task_id


def _resolve_dependency(reference: str | None, alias_map: Mapping[str, str]) -> str | None:
    """Resolve planner dependency references using the alias map."""
    if not reference:
        return None
    candidates = [
        reference,
        reference.strip(),
        reference.strip().lower(),
        _normalize_alias(reference),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = alias_map.get(candidate)
        if resolved:
            return resolved
    return None


def _normalize_alias(value: str) -> str:
    """Normalise alias strings before inserting into the alias map."""
    return _slugify(value).lower()


def _scoped_identifier(plan_id: str, candidate: str, existing: Iterable[str]) -> str:
    """Generate a unique identifier scoped to ``plan_id``."""
    base = candidate.strip() or "item"
    scoped = base if base.startswith(f"{plan_id}::") else f"{plan_id}::{_slugify(base)}"
    if scoped not in existing:
        return scoped
    suffix = 2
    while True:
        candidate_id = f"{scoped}-{suffix}"
        if candidate_id not in existing:
            return candidate_id
        suffix += 1


def _normalise_plan_id(candidate: str, *, fallback: str) -> str:
    """Normalise plan identifiers, returning a fallback when needed."""
    cleaned = candidate.strip()
    if not cleaned:
        return fallback
    slugged = _slugify(cleaned)
    return slugged or fallback


def _default_plan_id(project_name: str) -> str:
    """Construct a timestamped plan identifier derived from the project name."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    slug = _slugify(project_name)
    return f"{slug}-plan-{timestamp}"


def _slugify(value: str) -> str:
    """Convert arbitrary strings into lowercase slug tokens."""
    lowered = value.lower()
    mapped = ["-" if not ch.isalnum() else ch for ch in lowered]
    slug = "".join(mapped)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "plan"
