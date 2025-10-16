"""Implement phase: produce code changes and supporting guidance."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..context_builder import ContextBuilder
from ..models.llm_client import LLMClient, LLMClientError
from ..structured import StructuredEditOperation, StructuredFileArtifact
from ..tools.snippets import Snippet, SnippetRequest, StaticFinding
from .base import invoke_phase
from .local import LocalPhaseLogic, supports_local_logic


@dataclass(slots=True)
class ImplementRequest:
    """Input payload for the Implement phase."""

    task_id: str
    diff_goal: str
    touched_files: list[str] = field(default_factory=list)
    test_plan: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    snippets: list[Snippet] = field(default_factory=list)
    code_requests: list[SnippetRequest] = field(default_factory=list)
    static_findings: list[StaticFinding] = field(default_factory=list)
    structured_edits_only: bool = False


@dataclass(slots=True)
class ImplementResponse:
    """Structured result returned by the Implement phase."""

    summary: str
    diff: str = ""
    no_op_reason: str = ""
    test_commands: list[str] = field(default_factory=list)
    follow_up: list[str] = field(default_factory=list)
    code_requests: list[SnippetRequest] = field(default_factory=list)
    files: list[StructuredFileArtifact] = field(default_factory=list)
    edits: list[StructuredEditOperation] = field(default_factory=list)


def run(
    request: ImplementRequest,
    *,
    client: LLMClient,
    context_builder: ContextBuilder,
) -> ImplementResponse:
    """Execute the Implement phase via the shared LLM client."""
    logic = LocalPhaseLogic(context_builder)
    if supports_local_logic(client):
        response = logic.implement(request)
        if response is not None:
            return response
    try:
        return invoke_phase(
            "implement",
            request,
            ImplementResponse,
            client=client,
            context_builder=context_builder,
        )
    except LLMClientError:
        response = logic.implement(request)
        if response is not None:
            return response
        raise
