"""Plan Adjust phase: respond to blockers and refine the active plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from ..context_builder import ContextBuilder
from ..models.llm_client import LLMClient, LLMClientError
from ..structured import StructuredEditOperation, StructuredFileArtifact
from .base import invoke_phase
from .local import LocalPhaseLogic, supports_local_logic


@dataclass(slots=True)
class PlanAdjustRequest:
    """Input payload for the Plan-Adjust phase."""

    plan_id: str
    task_id: str
    reason: str
    suggested_changes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    suspect_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlanAdjustResponse:
    """Structured result returned by the Plan-Adjust phase."""

    adjustments: list["PlanAdjustment"]
    new_tasks: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlanAdjustmentItem:
    """Detailed adjustment entry returned by the Plan-Adjust phase."""

    action: str | None = None
    summary: str | None = None
    details: str | None = None
    rationale: str | list[str] | None = None
    priority: str | None = None
    id: str | None = None
    notes: list[str] | str | None = None
    files: list[StructuredFileArtifact] = field(default_factory=list)
    edits: list[StructuredEditOperation] = field(default_factory=list)

    def render(self) -> str:
        headline = self.action or self.summary or self.details or ""
        headline = headline.strip()

        extras: list[str] = []
        if self.summary and self.summary.strip() and self.summary.strip() != headline:
            extras.append(self.summary.strip())
        if self.details and self.details.strip() and self.details.strip() not in {headline, *(extras)}:
            extras.append(self.details.strip())
        if self.rationale:
            if isinstance(self.rationale, str):
                rationale_values = [self.rationale]
            else:
                rationale_values = [value for value in self.rationale if isinstance(value, str) and value]
            for value in rationale_values:
                cleaned = value.strip()
                if cleaned and cleaned not in {headline, *extras}:
                    extras.append(cleaned)
        if self.priority and self.priority.strip():
            extras.append(f"priority: {self.priority.strip()}")
        if self.id and self.id.strip():
            extras.append(f"id: {self.id.strip()}")
        if self.notes:
            if isinstance(self.notes, str):
                notes_values = [self.notes]
            else:
                notes_values = [note for note in self.notes if note]
            joined_notes = "; ".join(note.strip() for note in notes_values if note and note.strip())
            if joined_notes:
                extras.append(f"notes: {joined_notes}")
        if self.files or self.edits:
            extra_bits: list[str] = []
            if self.files:
                extra_bits.append(f"{len(self.files)} file{'s' if len(self.files) != 1 else ''}")
            if self.edits:
                extra_bits.append(f"{len(self.edits)} edit{'s' if len(self.edits) != 1 else ''}")
            extras.append(f"updates: {', '.join(extra_bits)}")

        if not headline and extras:
            headline = extras.pop(0)

        if extras:
            return f"{headline} ({'; '.join(extras)})".strip()
        return headline


PlanAdjustment = Union[PlanAdjustmentItem, str]


def run(
    request: PlanAdjustRequest,
    *,
    client: LLMClient,
    context_builder: ContextBuilder,
) -> PlanAdjustResponse:
    """Execute the Plan-Adjust phase via the shared LLM client."""
    if supports_local_logic(client):
        logic = LocalPhaseLogic(context_builder)
        return logic.plan_adjust(request)
    try:
        return invoke_phase(
            "plan_adjust",
            request,
            PlanAdjustResponse,
            client=client,
            context_builder=context_builder,
        )
    except LLMClientError:
        logic = LocalPhaseLogic(context_builder)
        return logic.plan_adjust(request)
