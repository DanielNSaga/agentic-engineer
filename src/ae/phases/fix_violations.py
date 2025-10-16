"""Fix Violations phase: remediate policy issues and regenerate patches."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..context_builder import ContextBuilder
from ..models.llm_client import LLMClient, LLMClientError
from ..structured import StructuredEditOperation, StructuredFileArtifact
from ..tools.snippets import StaticFinding
from .base import invoke_phase
from .local import LocalPhaseLogic, supports_local_logic


@dataclass(slots=True)
class FixViolationsRequest:
    """Input payload for the Fix Violations phase."""

    task_id: str
    violations: list[str]
    current_diff: str = ""
    attempts: int = 0
    suspect_files: list[str] = field(default_factory=list)
    static_findings: list[StaticFinding] = field(default_factory=list)


@dataclass(slots=True)
class FixViolationsResponse:
    """Structured result returned by the Fix Violations phase."""

    reason: str = ""
    touched_files: tuple[str, ...] = ()
    patch: str = ""
    no_op_reason: str = ""
    rationale: list[str] = field(default_factory=list)
    follow_up: list[str] = field(default_factory=list)
    files: list[StructuredFileArtifact] = field(default_factory=list)
    edits: list[StructuredEditOperation] = field(default_factory=list)

    def __post_init__(self) -> None:
        touched: list[str] = list(self.touched_files or ())
        if not touched:
            touched.extend(
                artifact.path
                for artifact in self.files
                if getattr(artifact, "path", None)
            )
            touched.extend(
                edit.path
                for edit in self.edits
                if getattr(edit, "path", None)
            )
        cleaned_paths: list[str] = []
        for entry in touched:
            text = str(entry).strip()
            if not text:
                continue
            cleaned = text.replace("\\", "/")
            cleaned_paths.append(cleaned)
        deduped = list(dict.fromkeys(cleaned_paths))
        object.__setattr__(self, "touched_files", tuple(deduped))

        reason = str(self.reason).strip()
        if not reason:
            for candidate in self.rationale:
                candidate_text = str(candidate).strip()
                if candidate_text:
                    reason = candidate_text
                    break
        if not reason:
            reason = "Resolve static gate failure."
        object.__setattr__(self, "reason", reason)


def run(
    request: FixViolationsRequest,
    *,
    client: LLMClient,
    context_builder: ContextBuilder,
) -> FixViolationsResponse:
    """Execute the Fix Violations phase via the shared LLM client."""
    logic = LocalPhaseLogic(context_builder)
    if supports_local_logic(client):
        response = logic.fix_violations(request)
        if response is not None:
            return response
    try:
        return invoke_phase(
            "fix_violations",
            request,
            FixViolationsResponse,
            client=client,
            context_builder=context_builder,
        )
    except LLMClientError:
        response = logic.fix_violations(request)
        if response is not None:
            return response
        raise
