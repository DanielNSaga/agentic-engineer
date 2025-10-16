"""Design phase: refine interfaces and plan implementation tactics."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..context_builder import ContextBuilder
from ..models.llm_client import LLMClient, LLMClientError
from .base import invoke_phase
from .local import LocalPhaseLogic, supports_local_logic


@dataclass(slots=True)
class DesignRequest:
    """Input payload for the Design phase."""

    task_id: str
    goal: str
    proposed_interfaces: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DesignResponse:
    """Structured result returned by the Design phase."""

    design_summary: str
    interface_changes: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    validation_plan: list[str] = field(default_factory=list)


def run(request: DesignRequest, *, client: LLMClient, context_builder: ContextBuilder) -> DesignResponse:
    """Execute the Design phase via the shared LLM client."""
    logic = LocalPhaseLogic(context_builder)
    if supports_local_logic(client):
        return logic.design(request)
    try:
        return invoke_phase(
            "design",
            request,
            DesignResponse,
            client=client,
            context_builder=context_builder,
        )
    except LLMClientError:
        return logic.design(request)
