"""Planning phase: transform goals and constraints into actionable tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..context_builder import ContextBuilder
from ..models.llm_client import LLMClient, LLMClientError
from ..planning.schemas import PlannerResponse
from .base import invoke_phase
from .local import LocalPhaseLogic, supports_local_logic


@dataclass(slots=True)
class PlanRequest:
    """Input payload for the planning phase."""

    goal: str
    constraints: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    deadline: str | None = None
    notes: list[str] = field(default_factory=list)
    known_context: Mapping[str, Any] = field(default_factory=dict)


PlanResponse = PlannerResponse


def run(
    request: PlanRequest,
    *,
    client: LLMClient,
    context_builder: ContextBuilder,
) -> PlanResponse:
    """Execute the plan phase via the shared LLM client."""
    if supports_local_logic(client):
        logic = LocalPhaseLogic(context_builder)
        return logic.plan(request)
    try:
        return invoke_phase(
            "plan",
            request,
            PlannerResponse,
            client=client,
            context_builder=context_builder,
        )
    except LLMClientError:
        logic = LocalPhaseLogic(context_builder)
        return logic.plan(request)
