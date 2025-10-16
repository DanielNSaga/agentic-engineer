"""On-demand source snippet utilities used by the orchestrator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SnippetRequest",
    "Snippet",
    "StaticFinding",
    "collect_snippets",
    "normalize_static_findings",
    "build_requests_from_findings",
]


@dataclass(slots=True)
class SnippetRequest:
    """Instruction describing which portion of a file should be surfaced."""

    path: str
    start_line: int | None = None
    end_line: int | None = None
    surround: int | None = None
    reason: str | None = None


@dataclass(slots=True)
class Snippet:
    """Captured slice of a repository file."""

    path: str
    start_line: int
    end_line: int
    content: str
    reason: str | None = None


@dataclass(slots=True)
class StaticFinding:
    """Structured representation of a static analysis finding."""

    path: str
    line_start: int
    line_end: int
    message: str


_DEFAULT_FINDING_CONTEXT = 10


def normalize_static_findings(
    findings: Iterable[StaticFinding | Mapping[str, Any]],
) -> list[StaticFinding]:
    """Coerce raw finding payloads into :class:`StaticFinding` objects."""

    normalized: list[StaticFinding] = []
    for entry in findings:
        if isinstance(entry, StaticFinding):
            candidate = entry
        elif isinstance(entry, Mapping):
            path = entry.get("path")
            start = entry.get("line_start")
            end = entry.get("line_end", start)
            message = entry.get("message")
            if not isinstance(path, str):
                continue
            if not isinstance(start, int) or start <= 0:
                continue
            if not isinstance(end, int) or end <= 0:
                continue
            if not isinstance(message, str) or not message.strip():
                continue
            candidate = StaticFinding(
                path=path.strip(),
                line_start=start,
                line_end=end,
                message=message.strip(),
            )
        else:
            continue
        if candidate.line_end < candidate.line_start:
            candidate = StaticFinding(
                path=candidate.path,
                line_start=candidate.line_end,
                line_end=candidate.line_start,
                message=candidate.message,
            )
        normalized.append(candidate)
    return normalized


def collect_snippets(
    repo_root: Path,
    requests: Sequence[SnippetRequest] | Iterable[SnippetRequest],
    *,
    default_window: int = 120,
    max_lines: int = 400,
    static_findings: Iterable[StaticFinding | Mapping[str, Any]] | None = None,
    finding_context: int = _DEFAULT_FINDING_CONTEXT,
) -> list[Snippet]:
    """Materialise code snippets for the requested files.

    Each request is clamped to the available file length and the configured
    ``max_lines`` to prevent prompt bloat.
    """

    snippets: list[Snippet] = []
    repo_root = repo_root.resolve()

    combined_requests: list[SnippetRequest] = []
    seen_keys: set[tuple[str, int | None, int | None]] = set()

    def _add_request(request: SnippetRequest) -> None:
        path = request.path.strip()
        start = request.start_line if isinstance(request.start_line, int) and request.start_line > 0 else None
        end = request.end_line if isinstance(request.end_line, int) and request.end_line > 0 else None
        key = (path, start, end)
        if key in seen_keys:
            return
        seen_keys.add(key)
        combined_requests.append(
            SnippetRequest(
                path=path,
                start_line=start,
                end_line=end,
                surround=request.surround,
                reason=request.reason,
            )
        )

    for request in requests:
        if isinstance(request, SnippetRequest):
            _add_request(request)

    if static_findings:
        for finding in build_requests_from_findings(static_findings, context=finding_context):
            _add_request(finding)

    for request in combined_requests:
        resolved = _resolve_repo_path(repo_root, request.path)
        if resolved is None or not resolved.exists():
            message = (
                f"[agentic-engineer] Requested file '{request.path}' does not exist. "
                "Create the file to satisfy this request."
            )
            snippets.append(
                Snippet(
                    path=request.path,
                    start_line=1,
                    end_line=1,
                    content=message,
                    reason=request.reason.strip() if isinstance(request.reason, str) and request.reason.strip() else None,
                )
            )
            continue
        try:
            lines = resolved.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        total_lines = len(lines)
        if total_lines == 0:
            continue

        start = request.start_line if isinstance(request.start_line, int) and request.start_line > 0 else 1
        end = request.end_line if isinstance(request.end_line, int) and request.end_line > 0 else total_lines

        if request.surround:
            padding = max(int(request.surround), 0)
            start = max(1, start - padding)
            end = min(total_lines, end + padding)

        if start > end:
            start, end = end, start

        window = max(default_window, 1)
        span = end - start + 1
        if span > max_lines:
            end = start + max_lines - 1
        elif span < window and span < max_lines:
            # Expand small slices to provide additional context where possible.
            extra = window - span
            extend_before = min(extra // 2, start - 1)
            extend_after = min(extra - extend_before, total_lines - end)
            start -= extend_before
            end += extend_after

        selected_lines = lines[start - 1 : end]
        if not selected_lines:
            continue

        snippet_text = "\n".join(selected_lines)
        snippets.append(
            Snippet(
                path=_relative_key(repo_root, resolved),
                start_line=start,
                end_line=end,
                content=snippet_text,
                reason=request.reason.strip() if isinstance(request.reason, str) and request.reason.strip() else None,
            )
        )

    return snippets


def _resolve_repo_path(repo_root: Path, requested: str) -> Path | None:
    """Resolve a requested snippet path within the repository, preserving fallbacks."""
    fallback: Path | None = None
    for variant in _candidate_repo_paths(requested):
        resolved = _resolve_single_repo_path(repo_root, variant)
        if resolved is None:
            continue
        if resolved.exists():
            return resolved
        if fallback is None:
            fallback = resolved
    return fallback


def _resolve_single_repo_path(repo_root: Path, candidate_path: str) -> Path | None:
    """Return an absolute path inside ``repo_root`` when the candidate is valid."""
    candidate = Path(candidate_path)
    try:
        if candidate.is_absolute():
            candidate = candidate.resolve()
        else:
            candidate = (repo_root / candidate).resolve()
        candidate.relative_to(repo_root)
    except ValueError:
        return None
    return candidate


def _candidate_repo_paths(requested: str) -> list[str]:
    """Generate path variants to compensate for relative or normalised forms."""
    trimmed = requested.strip()
    if not trimmed:
        return []

    variants: list[str] = []
    seen: set[str] = set()

    def _add(entry: str) -> None:
        key = entry.strip()
        if not key or key in seen:
            return
        seen.add(key)
        variants.append(key)

    _add(trimmed)

    normalised = trimmed.replace("\\", "/")
    _add(normalised)

    cleaned = normalised
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    _add(cleaned)

    parts = [part for part in cleaned.split("/") if part]
    for index in range(1, len(parts)):
        candidate = "/".join(parts[index:])
        _add(candidate)

    return variants


def _relative_key(repo_root: Path, path: Path) -> str:
    """Produce a repository-relative key suitable for snippet deduplication."""
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return path.name
    return relative.as_posix()


def build_requests_from_findings(
    findings: Iterable[StaticFinding | Mapping[str, Any]],
    *,
    context: int,
) -> list[SnippetRequest]:
    """Transform static analysis findings into snippet requests with padding."""
    normalized = normalize_static_findings(findings)
    requests: list[SnippetRequest] = []
    for finding in normalized:
        start = max(finding.line_start, 1)
        end = max(finding.line_end, 1)
        if context > 0:
            start = max(start - context, 1)
            end = end + context
        requests.append(
            SnippetRequest(
                path=finding.path,
                start_line=start,
                end_line=end,
                reason=finding.message,
            )
        )
    return requests
