"""Diagnose phase: isolate failure causes and propose fixes."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..context_builder import ContextBuilder
from ..models.llm_client import LLMClient, LLMClientError
from ..structured import StructuredEditOperation, StructuredFileArtifact
from ..tools.snippets import Snippet, SnippetRequest
from .base import invoke_phase
from .local import LocalPhaseLogic, supports_local_logic


@dataclass(slots=True)
class DiagnoseRequest:
    """Input payload for the Diagnose phase."""

    task_id: str
    failing_tests: list[str]
    logs: str = ""
    recent_changes: list[str] = field(default_factory=list)
    snippets: list[Snippet] = field(default_factory=list)
    code_requests: list[SnippetRequest] = field(default_factory=list)
    attempt_history: list[str] = field(default_factory=list)
    iteration_guidance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DiagnoseResponse:
    """Structured result returned by the Diagnose phase."""

    suspected_causes: list[str]
    recommended_fixes: list[str]
    patch: str = ""
    no_op_reason: str = ""
    additional_tests: list[str] = field(default_factory=list)
    confidence: float = 0.5
    files: list[StructuredFileArtifact] = field(default_factory=list)
    edits: list[StructuredEditOperation] = field(default_factory=list)
    code_requests: list[SnippetRequest] = field(default_factory=list)
    restart_iteration: bool = False
    restart_summary: str = ""
    iteration_lessons: list[str] = field(default_factory=list)


def run(request: DiagnoseRequest, *, client: LLMClient, context_builder: ContextBuilder) -> DiagnoseResponse:
    """Execute the Diagnose phase via the shared LLM client."""
    logic = LocalPhaseLogic(context_builder)
    if supports_local_logic(client):
        response = logic.diagnose(request)
        if response is not None:
            return response
    try:
        return invoke_phase(
            "diagnose",
            request,
            DiagnoseResponse,
            client=client,
            context_builder=context_builder,
        )
    except LLMClientError:
        response = logic.diagnose(request)
        if response is not None:
            return response
        raise
