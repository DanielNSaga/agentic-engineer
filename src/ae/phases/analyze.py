"""Analyze phase: understand the task goal and outline next steps."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..context_builder import ContextBuilder
from ..models.llm_client import LLMClient, LLMClientError
from .base import invoke_phase
from .local import LocalPhaseLogic, supports_local_logic


@dataclass(slots=True)
class AnalyzeRequest:
    """Input payload for the Analyze phase."""

    task_id: str
    goal: str
    context: str = ""
    constraints: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalyzeResponse:
    """Structured result returned by the Analyze phase."""

    summary: str
    plan_steps: list[str]
    risks: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


def run(request: AnalyzeRequest, *, client: LLMClient, context_builder: ContextBuilder) -> AnalyzeResponse:
    """Execute the Analyze phase via the shared LLM client."""
    logic = LocalPhaseLogic(context_builder)
    if supports_local_logic(client):
        return logic.analyze(request)
    try:
        return invoke_phase(
            "analyze",
            request,
            AnalyzeResponse,
            client=client,
            context_builder=context_builder,
        )
    except LLMClientError:
        return logic.analyze(request)
