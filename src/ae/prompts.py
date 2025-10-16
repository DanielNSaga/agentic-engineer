"""Prompt templates and helpers shared across Agentic Engineer phases."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

JSON_RESPONSE_INSTRUCTION = (
    "Return only JSON. Emit a single JSON object that satisfies the documented response schema. "
    "Do not include markdown fences, explanations, or trailing text. "
    "Use double-quoted keys and strings, and omit fields that are not required."
)


def render_json_instruction(repo_root: Path) -> str:
    """Render the standard JSON-only instruction block with repository context."""
    workspace_line = (
        f"Operate on the live workspace at {repo_root.as_posix()}; inspect files via `ls`, "
        "`rg --files`, `rg <pattern>`, or `cat` rather than git history commands such as `git show` when reviewing the workspace."
    )
    return f"{JSON_RESPONSE_INSTRUCTION} {workspace_line}"


def render_phase_brief(phase: str) -> str:
    """Return the canonical phase brief used across all agent prompts."""
    return (
        "## Phase Brief\n"
        f"You are executing the `{phase}` phase. Review the provided context, follow the policy capsule in the system prompt, "
        "and return structured JSON that matches the expected response schema."
    )


def render_project_guidance(guidance: Sequence[str]) -> str:
    """Format project guidance strings as a single bullet list block."""
    if not guidance:
        return ""
    body = "\n".join(f"- {line.strip()}" for line in guidance if line.strip())
    if not body:
        return ""
    return f"## Project Guidance\n{body}"


__all__ = [
    "JSON_RESPONSE_INSTRUCTION",
    "render_json_instruction",
    "render_phase_brief",
    "render_project_guidance",
]
