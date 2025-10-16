"""Unified diff helpers with guard rails for automated patching."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, Set, Tuple

import yaml


class PatchError(RuntimeError):
    """Raised when a patch fails validation or application."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = dict(details or {})


@dataclass(slots=True)
class PatchResult:
    """Outcome of applying a patch to the repository."""

    command: Tuple[str, ...]
    paths: Tuple[Path, ...]
    stdout: str
    stderr: str
    telemetry: "PatchTelemetry | None" = None

    @property
    def touched_paths(self) -> Tuple[Path, ...]:
        return self.paths


@dataclass(slots=True)
class _StructuredFileState:
    """In-memory representation of a structured edit target."""

    original: str
    existed: bool
    content: str
    encoding: str | None = None
    executable: bool | None = None


TELEMETRY_LOGGER = logging.getLogger("ae.telemetry")
_DEFAULT_MAX_PATCH_BYTES = 200_000

_GUIDANCE_PATH_PATTERN = re.compile(r"(?:src|tests|docs|scripts|data)/[A-Za-z0-9_.\\/-]+")
_GUIDANCE_FILE_PATTERN = re.compile(r"\b[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+\b")
_HARD_NEGATIVE_PHRASES = ("do not", "don't", "never", "must not", "cannot", "can't", "forbidden", "forbid")
_SOFT_NEGATIVE_PHRASES = ("avoid", "try not to", "prefer not to")
_SOFTENING_HINTS = ("unless", "except", "if needed", "if necessary", "when needed", "when necessary", "prefer", "ideally")
_PATCH_FAILED_RE = re.compile(r"error: patch failed: (?P<path>.+?)(?::(?P<line>\d+))?$")
_PATCH_DOES_NOT_APPLY_RE = re.compile(r"error: (?P<path>.+?): patch does not apply")
_HUNK_FAILED_RE = re.compile(r"error: (?P<path>.+?): hunk #(?P<hunk>\d+) failed at (?P<line>-?\d+)")


@dataclass(slots=True)
class _GuidanceRules:
    """Normalised project guidance for patch validation."""

    allowed_prefixes: Tuple[Path, ...] = ()
    blocked_paths: Tuple[Path, ...] = ()
    soft_blocked_paths: Tuple[Path, ...] = ()
    max_patch_bytes: int = _DEFAULT_MAX_PATCH_BYTES
    enforce_lf: bool = True
    enforce_utf8: bool = True


@dataclass(slots=True)
class PatchTelemetry:
    """Structured telemetry for patch validation and application."""

    patch_path: Path | None = None
    patch_bytes: int = 0
    patch_lines: int = 0
    check_returncode: int | None = None
    check_stdout: str = ""
    check_stderr: str = ""
    failing_hunks: Tuple[Mapping[str, Any], ...] = ()
    adjustments: Tuple[str, ...] = ()
    guidance_violations: Tuple[str, ...] = ()
    guidance_warnings: Tuple[str, ...] = ()
    touched_paths: Tuple[Path, ...] = ()
    allowed_prefixes: Tuple[str, ...] = ()
    blocked_paths: Tuple[str, ...] = ()
    soft_blocked_paths: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_path": self.patch_path.as_posix() if self.patch_path else None,
            "patch_bytes": self.patch_bytes,
            "patch_lines": self.patch_lines,
            "check": {
                "returncode": self.check_returncode,
                "stdout": self.check_stdout,
                "stderr": self.check_stderr,
            },
            "failing_hunks": [dict(item) for item in self.failing_hunks],
            "adjustments": list(self.adjustments),
            "guidance_violations": list(self.guidance_violations),
            "guidance_warnings": list(self.guidance_warnings),
            "touched_paths": [path.as_posix() for path in self.touched_paths],
            "allowed_prefixes": list(self.allowed_prefixes),
            "blocked_paths": list(self.blocked_paths),
            "soft_blocked_paths": list(self.soft_blocked_paths),
        }


@dataclass(slots=True)
class _PatchOperation:
    """Single file operation represented within a diff."""

    path: Path
    change_type: str  # "add", "modify", or "delete"


@dataclass(slots=True)
class _PatchPreparation:
    """Prepared patch metadata prior to git apply."""

    patch: str
    paths: Set[Path]
    backup_paths: Set[Path]
    telemetry: PatchTelemetry


def _serialise_event_value(value: Any) -> Any:
    """Convert telemetry payload values into JSON-friendly representations."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple, set)):
        return [_serialise_event_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialise_event_value(child) for key, child in value.items()}
    return str(value)


def _emit_patch_event(event: str, **fields: Any) -> None:
    """Log structured telemetry events when applying patches."""
    if not TELEMETRY_LOGGER:
        return
    payload = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat()}
    for key, value in fields.items():
        payload[key] = _serialise_event_value(value)
    try:
        message = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError):
        fallback = {key: _serialise_event_value(value) for key, value in payload.items()}
        message = json.dumps(fallback, separators=(",", ":"), ensure_ascii=True)
    TELEMETRY_LOGGER.info(message)


def _parse_git_apply_failures(output: str) -> Tuple[Mapping[str, Any], ...]:
    """Parse git apply stderr for failing hunk metadata."""
    if not output:
        return ()
    entries: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _HUNK_FAILED_RE.match(line)
        if match:
            entries.append(
                {
                    "path": match.group("path"),
                    "hunk": int(match.group("hunk")),
                    "line": int(match.group("line")),
                    "reason": "hunk_failed",
                }
            )
            continue
        match = _PATCH_FAILED_RE.match(line)
        if match:
            line_text = match.group("line")
            entries.append(
                {
                    "path": match.group("path"),
                    "line": int(line_text) if line_text is not None else None,
                    "reason": "patch_failed",
                }
            )
            continue
        match = _PATCH_DOES_NOT_APPLY_RE.match(line)
        if match:
            entries.append(
                {
                    "path": match.group("path"),
                    "reason": "does_not_apply",
                }
            )
            continue
    return tuple(entries)


def _path_matches_prefix(path: Path, prefix: Path) -> bool:
    """Return True when ``path`` resides under ``prefix``."""
    try:
        return path.is_relative_to(prefix)
    except AttributeError:
        try:
            path.relative_to(prefix)
            return True
        except ValueError:
            return False
    except ValueError:
        return False


def _derive_allowed_candidates(raw: str) -> Set[Path]:
    """Expand guidance tokens into candidate paths that may be edited."""
    candidate = raw.strip().strip("/")
    if not candidate:
        return set()
    path = Path(candidate)
    derived: Set[Path] = {path}
    if path.parts and path.parts[0] == "src" and len(path.parts) > 1:
        tail = Path(*path.parts[1:])
        derived.add(Path("tests") / tail)
    return {entry for entry in derived if str(entry).strip()}


def _infer_package_name(config: Mapping[str, Any]) -> str | None:
    """Guess the primary package name from configuration metadata."""
    project_section = config.get("project")
    candidate: str | None = None
    if isinstance(project_section, Mapping):
        potential = project_section.get("package") or project_section.get("name")
        if isinstance(potential, str) and potential.strip():
            candidate = potential.strip()
    if not candidate:
        return None
    normalised = re.sub(r"[^A-Za-z0-9_]+", "_", candidate)
    normalised = re.sub(r"_+", "_", normalised).strip("_")
    return normalised.lower() or None


def _expand_guidance_token(token: str, package_name: str | None) -> Set[str]:
    """Resolve package-name placeholders inside guidance tokens."""
    results: Set[str] = {token}
    if not package_name:
        return results

    placeholders = {
        "<package_name>",
        "<package-name>",
        "{package_name}",
        "{package-name}",
    }
    for placeholder in placeholders:
        if placeholder in token:
            results.add(token.replace(placeholder, package_name))

    return {entry for entry in results if entry}


def _load_project_config(repo_root: Path, config_path: Path | None) -> Mapping[str, Any]:
    """Load the agent configuration file associated with ``repo_root``."""
    if config_path is None:
        candidate = repo_root / "config.yaml"
    else:
        candidate = Path(config_path)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, Mapping):
                return loaded
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return {}


def _normalise_guidance_rules(
    config: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> _GuidanceRules:
    """Interpret project guidance into structured allow/deny path rules."""
    env_mapping = env or os.environ
    allowed: Set[Path] = set()
    blocked: Set[Path] = set()
    soft_blocked: Set[Path] = set()

    context_section = config.get("context")
    guidance_entries: list[str] = []
    if isinstance(context_section, Mapping):
        guidance_raw = context_section.get("guidance")
        if isinstance(guidance_raw, str):
            trimmed = guidance_raw.strip()
            if trimmed:
                guidance_entries.append(trimmed)
        elif isinstance(guidance_raw, Sequence):
            for entry in guidance_raw:
                if isinstance(entry, str):
                    trimmed = entry.strip()
                    if trimmed:
                        guidance_entries.append(trimmed)

    package_name = _infer_package_name(config)

    for entry in guidance_entries:
        lowered = entry.lower()
        path_tokens = {match.strip().strip("/") for match in _GUIDANCE_PATH_PATTERN.findall(entry)}
        file_tokens = {match.strip().strip("., ") for match in _GUIDANCE_FILE_PATTERN.findall(entry)}
        negative_hard = any(phrase in lowered for phrase in _HARD_NEGATIVE_PHRASES)
        negative_soft = any(phrase in lowered for phrase in _SOFT_NEGATIVE_PHRASES)
        softened = any(hint in lowered for hint in _SOFTENING_HINTS)
        if negative_hard or negative_soft:
            target_set = blocked if negative_hard and not softened else soft_blocked
            other_set = soft_blocked if target_set is blocked else blocked
            for token in path_tokens.union(file_tokens):
                if not token:
                    continue
                expanded_tokens = _expand_guidance_token(token, package_name)
                if not expanded_tokens:
                    candidate = Path(token)
                    target_set.add(candidate)
                    other_set.discard(candidate)
                    continue
                for expanded in expanded_tokens:
                    cleaned = expanded.strip().strip("/")
                    if not cleaned:
                        continue
                    candidate = Path(cleaned)
                    target_set.add(candidate)
                    other_set.discard(candidate)
            continue
        for token in path_tokens:
            if token:
                for expanded in _expand_guidance_token(token, package_name):
                    allowed.update(_derive_allowed_candidates(expanded))

    max_patch_bytes = _DEFAULT_MAX_PATCH_BYTES
    env_limit = env_mapping.get("AE_MAX_PATCH_BYTES")
    if env_limit is not None:
        try:
            parsed = int(str(env_limit).strip())
            if parsed > 0:
                max_patch_bytes = parsed
        except ValueError:
            pass

    iteration_section = config.get("iteration")
    if isinstance(iteration_section, Mapping):
        candidate = iteration_section.get("max_patch_bytes")
        if isinstance(candidate, int) and candidate > 0:
            max_patch_bytes = candidate
        elif isinstance(candidate, str):
            try:
                parsed = int(candidate.strip())
                if parsed > 0:
                    max_patch_bytes = parsed
            except ValueError:
                pass

    return _GuidanceRules(
        allowed_prefixes=tuple(sorted(allowed, key=lambda item: item.as_posix())),
        blocked_paths=tuple(sorted(blocked, key=lambda item: item.as_posix())),
        soft_blocked_paths=tuple(sorted(soft_blocked, key=lambda item: item.as_posix())),
        max_patch_bytes=max_patch_bytes,
        enforce_lf=True,
        enforce_utf8=True,
    )


def _collect_patch_operations(patch: str) -> list[_PatchOperation]:
    """Summarise each file operation described within a unified diff."""
    operations: list[_PatchOperation] = []
    lines = patch.splitlines()
    sections = _split_diff_sections(lines)
    for start, end in sections:
        section_lines = lines[start:end]
        if not section_lines:
            continue
        header = section_lines[0]
        match = _DIFF_HEADER.match(header)
        if not match:
            continue
        left = _normalise_diff_path(match.group(1))
        right = _normalise_diff_path(match.group(2))
        path = right or left
        if path is None:
            continue
        change_type = "modify"
        for candidate in section_lines[1:]:
            if candidate.startswith("new file mode"):
                change_type = "add"
                break
            if candidate.startswith("deleted file mode"):
                change_type = "delete"
                break
        operations.append(_PatchOperation(path=path, change_type=change_type))
    return operations


@dataclass(slots=True)
class _PatchApplicationBuilder:
    """Prepare and validate patches prior to git apply."""

    repo_root: Path
    patch_text: str
    config_path: Path | None = None
    _config: Mapping[str, Any] = field(init=False, repr=False)
    _guidance: _GuidanceRules = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self._config = _load_project_config(self.repo_root, self.config_path)
        self._guidance = _normalise_guidance_rules(self._config)

    def prepare(self) -> _PatchPreparation:
        telemetry = PatchTelemetry()
        telemetry.allowed_prefixes = tuple(prefix.as_posix() for prefix in self._guidance.allowed_prefixes)
        telemetry.blocked_paths = tuple(path.as_posix() for path in self._guidance.blocked_paths)
        telemetry.soft_blocked_paths = tuple(path.as_posix() for path in self._guidance.soft_blocked_paths)

        raw_patch = self.patch_text or ""
        telemetry.patch_lines = raw_patch.count("\n")

        if self._guidance.enforce_lf and "\r" in raw_patch:
            telemetry.guidance_violations = ("Patch must use LF line endings; carriage returns detected.",)
            telemetry.patch_bytes = len(raw_patch.encode("utf-8", errors="ignore"))
            raise PatchError(
                "Patch must use LF line endings (CR characters detected).",
                details={"telemetry": telemetry.to_dict()},
            )

        patch = _convert_apply_patch_format(raw_patch)
        patch = _trim_dangling_hunks(patch)
        telemetry.patch_lines = patch.count("\n")

        if not patch or not patch.strip():
            raise PatchError("Patch is empty.", details={"telemetry": telemetry.to_dict()})
        if "GIT binary patch" in patch:
            raise PatchError("Binary patches are not supported.", details={"telemetry": telemetry.to_dict()})

        if not (self.repo_root / ".git").exists():
            raise PatchError(f"Not a git repository: {self.repo_root}", details={"telemetry": telemetry.to_dict()})

        patch, backup_paths, _ = _handle_existing_new_files(patch, self.repo_root)
        telemetry.patch_lines = patch.count("\n")

        if not patch.strip():
            telemetry.patch_bytes = 0
            telemetry.touched_paths = tuple(sorted(backup_paths, key=lambda item: item.as_posix()))
            return _PatchPreparation(
                patch="",
                paths=set(),
                backup_paths=backup_paths,
                telemetry=telemetry,
            )

        patch, adjustments = _normalise_hunks(patch)
        telemetry.adjustments = tuple(adjustments)
        telemetry.patch_lines = patch.count("\n")

        try:
            encoded_patch = patch.encode("utf-8")
        except UnicodeEncodeError as error:
            telemetry.guidance_violations = ("Patch is not valid UTF-8.",)
            raise PatchError("Patch is not valid UTF-8.", details={"telemetry": telemetry.to_dict()}) from error
        telemetry.patch_bytes = len(encoded_patch)

        if self._guidance.enforce_lf and "\r" in patch:
            telemetry.guidance_violations = ("Patch must use LF line endings; carriage returns detected.",)
            raise PatchError("Patch must use LF line endings (CR characters detected).", details={"telemetry": telemetry.to_dict()})

        if self._guidance.max_patch_bytes > 0 and telemetry.patch_bytes > self._guidance.max_patch_bytes:
            telemetry.guidance_violations = (
                f"Patch size {telemetry.patch_bytes} bytes exceeds limit of {self._guidance.max_patch_bytes} bytes.",
            )
            raise PatchError(
                f"Patch exceeds maximum size of {self._guidance.max_patch_bytes} bytes.",
                details={"telemetry": telemetry.to_dict()},
            )

        paths = _extract_paths(patch)
        if not paths:
            if backup_paths:
                telemetry.touched_paths = tuple(sorted(backup_paths, key=lambda item: item.as_posix()))
                return _PatchPreparation(
                    patch="",
                    paths=set(),
                    backup_paths=backup_paths,
                    telemetry=telemetry,
                )
            raise PatchError("Patch does not describe any file changes.", details={"telemetry": telemetry.to_dict()})

        violations: list[str] = []
        if self._guidance.allowed_prefixes:
            operations = _collect_patch_operations(patch)
            for operation in operations:
                if operation.change_type != "add":
                    continue
                if any(
                    operation.path == prefix or _path_matches_prefix(operation.path, prefix)
                    for prefix in self._guidance.allowed_prefixes
                ):
                    continue
                violations.append(
                    f"New file {operation.path.as_posix()} is outside allowed paths: "
                    f"{', '.join(prefix.as_posix() for prefix in self._guidance.allowed_prefixes)}."
                )

        if self._guidance.blocked_paths:
            for path in paths:
                if any(path == blocked or _path_matches_prefix(path, blocked) for blocked in self._guidance.blocked_paths):
                    violations.append(f"{path.as_posix()} is blocked by project guidance.")

        if self._guidance.soft_blocked_paths:
            soft_hits: set[Path] = set()
            for path in paths:
                if any(path == soft or _path_matches_prefix(path, soft) for soft in self._guidance.soft_blocked_paths):
                    soft_hits.add(path)
            if soft_hits:
                telemetry.guidance_warnings = tuple(
                    sorted(f"{path.as_posix()} intersects soft guidance; review before applying." for path in soft_hits)
                )

        if violations:
            telemetry.guidance_violations = tuple(violations)
            raise PatchError("Patch violates project guidance.", details={"telemetry": telemetry.to_dict()})

        try:
            _validate_paths(paths)
        except PatchError as error:
            telemetry.guidance_violations = telemetry.guidance_violations + (str(error),)
            raise PatchError(str(error), details={"telemetry": telemetry.to_dict()}) from error

        try:
            _validate_hunks(patch)
        except PatchError as error:
            telemetry.guidance_violations = telemetry.guidance_violations + (str(error),)
            raise PatchError(str(error), details={"telemetry": telemetry.to_dict()}) from error

        telemetry.touched_paths = tuple(sorted(paths, key=lambda item: item.as_posix()))

        return _PatchPreparation(
            patch=patch,
            paths=paths,
            backup_paths=backup_paths,
            telemetry=telemetry,
        )


def _run_git(args: Sequence[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command and optionally raise `PatchError` on failure."""
    command = ["git", *args]
    process = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=False,
        check=False,
    )
    stdout = process.stdout.decode("utf-8", errors="replace") if process.stdout else ""
    stderr = process.stderr.decode("utf-8", errors="replace") if process.stderr else ""
    result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    if check and result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        raise PatchError(f"git {' '.join(args)} failed: {message}")
    return result


@contextmanager
def _temporary_worktree(repo_root: Path) -> Iterator[Path]:
    """Create a disposable worktree for safely staging patch operations."""
    base_dir = tempfile.TemporaryDirectory(prefix="ae-patch-")
    try:
        worktree_root = Path(base_dir.name) / "worktree"
        add = _run_git(
            ["worktree", "add", "--detach", str(worktree_root), "HEAD"],
            cwd=repo_root,
            check=False,
        )
        if add.returncode != 0:
            message = add.stderr.strip() or add.stdout.strip() or "unable to create temporary worktree"
            raise PatchError(f"Failed to create temporary worktree: {message}")
        try:
            yield worktree_root
        finally:
            _run_git(
                ["worktree", "remove", "--force", str(worktree_root)],
                cwd=repo_root,
                check=False,
            )
    finally:
        base_dir.cleanup()


def _normalise_structured_path(raw: Any) -> str | None:
    """Normalise structured file paths to trimmed strings."""
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    return candidate or None


def _normalise_line_endings(text: str) -> str:
    """Convert CRLF/CR sequences to LF for deterministic diffs."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalise_structured_content(content: Any) -> str:
    """Serialise structured content into newline-terminated text."""
    if content is None:
        return ""
    normalised = _normalise_line_endings(str(content))
    if normalised and not normalised.endswith("\n"):
        normalised += "\n"
    return normalised


def _split_lines(content: str | None) -> list[str]:
    """Split text into normalised lines, guarding against ``None``."""
    if not content:
        return []
    return _normalise_line_endings(content).splitlines()


def _join_lines(lines: list[str]) -> str:
    """Join lines back into text, ensuring a trailing newline."""
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _apply_structured_operation(base_text: str, operation: Any) -> str:
    """Apply a structured edit operation to the provided text."""
    action = (getattr(operation, "action", "") or "").lower()
    if action not in {"replace", "insert", "delete"}:
        return base_text

    lines = _split_lines(base_text)

    if action == "replace":
        start_line = getattr(operation, "start_line", None) or 1
        end_line = getattr(operation, "end_line", None)
        replacement_lines = _split_lines(getattr(operation, "content", None))
        start_index = max(start_line - 1, 0)
        if start_index > len(lines):
            start_index = len(lines)
        if isinstance(end_line, int) and end_line > 0:
            end_bound = end_line
        else:
            end_bound = len(lines)
        slice_end = min(max(end_bound, start_line), len(lines))
        if slice_end < start_index:
            slice_end = start_index
        lines[start_index:slice_end] = replacement_lines
    elif action == "insert":
        insertion_lines = _split_lines(getattr(operation, "content", None))
        if not insertion_lines:
            return base_text
        start_line = getattr(operation, "start_line", None)
        index = start_line - 1 if isinstance(start_line, int) and start_line > 0 else len(lines)
        index = max(0, min(index, len(lines)))
        lines[index:index] = insertion_lines
    elif action == "delete":
        start_line = getattr(operation, "start_line", None)
        end_line = getattr(operation, "end_line", None)
        if start_line is None and end_line is None:
            lines = []
        else:
            start = start_line if isinstance(start_line, int) and start_line > 0 else 1
            end_bound = end_line if isinstance(end_line, int) and end_line > 0 else start
            start_index = max(start - 1, 0)
            slice_end = min(max(end_bound, start), len(lines))
            del lines[start_index:slice_end]

    return _join_lines(lines)


def _load_original_text(repo_root: Path, path: str) -> tuple[str, bool]:
    """Read the current on-disk contents for ``path`` within the repo."""
    fs_path = (repo_root / path).resolve()
    if not fs_path.exists():
        return "", False
    try:
        original_text = fs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            original_text = fs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            original_text = ""
    except OSError:
        original_text = ""
    return _normalise_line_endings(original_text), True


def _collect_structured_updates(
    repo_root: Path,
    files: Sequence[Any],
    edits: Sequence[Any],
) -> Mapping[str, _StructuredFileState]:
    """Materialise file states that should be written into the worktree."""
    states: dict[str, _StructuredFileState] = {}

    def ensure_state(path: str) -> _StructuredFileState:
        if path not in states:
            original_text, existed = _load_original_text(repo_root, path)
            states[path] = _StructuredFileState(
                original=original_text,
                existed=existed,
                content=original_text,
            )
        return states[path]

    for artifact in files:
        path = _normalise_structured_path(getattr(artifact, "path", None))
        if not path:
            continue
        state = ensure_state(path)
        state.content = _normalise_structured_content(getattr(artifact, "content", None))
        encoding = getattr(artifact, "encoding", None)
        if isinstance(encoding, str) and encoding.strip():
            state.encoding = encoding.strip()
        executable = getattr(artifact, "executable", False)
        if executable is not None:
            state.executable = bool(executable)

    for edit in edits:
        path = _normalise_structured_path(getattr(edit, "path", None))
        if not path:
            continue
        state = ensure_state(path)
        state.content = _apply_structured_operation(state.content, edit)

    updates: dict[str, _StructuredFileState] = {}
    for path, state in states.items():
        if state.content == state.original:
            continue
        updates[path] = state
    return updates


def _mirror_worktree_state(source_root: Path, target_root: Path) -> None:
    """Synchronise indexed changes from the source repository into a worktree."""
    diff_result = _run_git(
        ["diff", "--unified=3"],
        cwd=source_root,
        check=False,
    )
    diff_text = diff_result.stdout
    if not diff_text or not diff_text.strip():
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(diff_text)
        handle.flush()
        temp_path = Path(handle.name)
    try:
        apply_result = _run_git(["apply", str(temp_path)], cwd=target_root, check=False)
        if apply_result.returncode != 0:
            message = apply_result.stderr.strip() or apply_result.stdout.strip() or "unable to mirror worktree state"
            raise PatchError(f"Failed to mirror worktree state: {message}")
    finally:
        temp_path.unlink(missing_ok=True)
    status = _run_git(["status", "--porcelain"], cwd=target_root, check=False).stdout
    if status.strip():
        _run_git(["config", "--local", "user.name", "Agentic Engineer"], cwd=target_root, check=False)
        _run_git(["config", "--local", "user.email", "agent@example.com"], cwd=target_root, check=False)
        _run_git(["add", "--all"], cwd=target_root)
        _run_git(["commit", "--allow-empty", "-m", "ae: mirror worktree state"], cwd=target_root, check=False)


def _write_structured_updates(worktree_root: Path, updates: Mapping[str, _StructuredFileState]) -> None:
    """Write updated file contents and metadata into the temporary worktree."""
    root = worktree_root.resolve()
    for path, state in updates.items():
        target = (root / Path(path)).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise PatchError(f"Structured edit escaped worktree: {path}") from None
        target.parent.mkdir(parents=True, exist_ok=True)
        encoding = state.encoding or "utf-8"
        data = state.content.encode(encoding, errors="replace")
        target.write_bytes(data)
        if state.executable is True:
            try:
                target.chmod(0o755)
            except (OSError, PermissionError):
                pass
        elif state.executable is False:
            try:
                target.chmod(0o644)
            except (OSError, PermissionError):
                pass


def _restore_diff_prefixes(diff: str) -> str:
    """Ensure diff headers include git-style ``a/`` and ``b/`` prefixes."""
    if not diff:
        return diff
    lines = diff.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                left = parts[2]
                right = parts[3]
                if not left.startswith(("a/", "b/", "/")):
                    left = f"a/{left}"
                if not right.startswith(("a/", "b/", "/")):
                    right = f"b/{right}"
                parts[2] = left
                parts[3] = right
                lines[index] = " ".join(parts)
        elif line.startswith("--- "):
            operand = line[4:]
            if operand not in {"/dev/null"} and not operand.startswith(("a/", "b/")):
                lines[index] = f"--- a/{operand}"
        elif line.startswith("+++ "):
            operand = line[4:]
            if operand not in {"/dev/null"} and not operand.startswith(("a/", "b/")):
                lines[index] = f"+++ b/{operand}"
    updated = "\n".join(lines)
    if diff.endswith("\n") and not updated.endswith("\n"):
        updated += "\n"
    return updated


def _collect_worktree_diff(worktree_root: Path) -> str:
    """Capture the index diff from the temporary worktree."""
    _run_git(["add", "--all"], cwd=worktree_root)
    diff = _run_git(
        ["diff", "--cached", "--no-prefix", "--unified=3"],
        cwd=worktree_root,
    ).stdout
    diff = diff.replace("\r\n", "\n")
    diff = _restore_diff_prefixes(diff)
    if diff and not diff.endswith("\n"):
        diff += "\n"
    return diff


def build_structured_patch(
    *,
    repo_root: Path | str,
    files: Sequence[Any],
    edits: Sequence[Any],
) -> str:
    """Construct a unified diff from structured file artifacts and edits."""
    repo_path = Path(repo_root).resolve()
    if not (repo_path / ".git").exists():
        raise PatchError(f"Not a git repository: {repo_path}")
    updates = _collect_structured_updates(repo_path, files, edits)
    if not updates:
        return ""
    with _temporary_worktree(repo_path) as worktree_root:
        _mirror_worktree_state(repo_path, worktree_root)
        _write_structured_updates(worktree_root, updates)
        diff = _collect_worktree_diff(worktree_root)
    return diff


def canonicalise_unified_diff(
    patch: str,
    *,
    repo_root: Path | str,
) -> str:
    """Normalise diffs (including apply_patch format) against repository state."""
    cleaned = _convert_apply_patch_format(patch or "")
    cleaned = _trim_dangling_hunks(cleaned)
    if not cleaned.strip():
        return ""
    cleaned, _ = _normalise_hunks(cleaned)
    repo_path = Path(repo_root).resolve()
    if not (repo_path / ".git").exists():
        raise PatchError(f"Not a git repository: {repo_path}")
    with _temporary_worktree(repo_path) as worktree_root:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(cleaned)
            handle.flush()
            patch_path = Path(handle.name)
        try:
            _run_git(["apply", "--check", str(patch_path)], cwd=worktree_root)
            _run_git(["apply", str(patch_path)], cwd=worktree_root)
            diff = _collect_worktree_diff(worktree_root)
        finally:
            patch_path.unlink(missing_ok=True)
    return diff


_DIFF_HEADER = re.compile(r"^diff --git (\S+) (\S+)$", re.MULTILINE)
_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)
_APPLY_PATCH_HEADER = re.compile(r"^\*\*\* (Begin|End) Patch", re.MULTILINE)


def _normalise_diff_path(entry: str) -> Path | None:
    """Translate diff header operands into repository-relative paths."""
    if entry == "/dev/null":
        return None
    if entry.startswith("a/") or entry.startswith("b/"):
        entry = entry[2:]
    entry = entry.strip()
    if not entry:
        return None
    return Path(entry)


def _extract_paths(patch: str) -> Set[Path]:
    """Collect all file paths referenced by a unified diff."""
    paths: Set[Path] = set()
    for match in _DIFF_HEADER.finditer(patch):
        left = _normalise_diff_path(match.group(1))
        right = _normalise_diff_path(match.group(2))
        for candidate in (left, right):
            if candidate is None:
                continue
            paths.add(candidate)
    return paths


def _validate_paths(paths: Iterable[Path]) -> None:
    """Enforce path safety rules for diff entries."""
    for path in paths:
        if path.is_absolute():
            raise PatchError(f"Absolute paths are not permitted in patches: {path}")
        parts = list(path.parts)
        if any(part == ".." for part in parts):
            raise PatchError(f"Path escaping detected in patch: {path}")
        if parts and parts[0] == ".git":
            raise PatchError("Patches may not target the .git directory.")


def _default_count(value: str | None) -> int:
    """Return the number of lines represented in a hunk header."""
    return int(value) if value is not None else 1


def _format_range(start: str, original_count: str | None, actual: int) -> str:
    """Format an ``@@`` range using actual line counts."""
    if original_count is None and actual == 1:
        return start
    return f"{start},{actual}"


def _split_diff_sections(lines: list[str]) -> list[tuple[int, int]]:
    """Identify line ranges corresponding to individual diff sections."""
    sections: list[tuple[int, int]] = []
    start: int | None = None
    for index, line in enumerate(lines):
        if line.startswith("diff --git "):
            if start is not None:
                sections.append((start, index))
            start = index
    if start is not None:
        sections.append((start, len(lines)))
    return sections


def _reconstruct_new_file_bytes(lines: list[str]) -> bytes:
    """Rebuild file contents from a new-file diff hunk."""
    capture = False
    newline_at_end = True
    body: list[str] = []

    for line in lines:
        if line.startswith("@@ "):
            capture = True
            continue
        if not capture:
            continue
        if line.startswith("\\ No newline at end of file"):
            newline_at_end = False
            continue
        prefix = line[:1]
        if prefix in {"+", " "}:
            body.append(line[1:])

    if not capture:
        return b""

    text = "\n".join(body)
    if newline_at_end and (capture and (body or lines)):
        text += "\n"
    return text.encode("utf-8")


def _candidate_backup_path(path: Path) -> Path:
    """Suggest a unique backup location for an overwritten new-file target."""
    suffix = ".pre_patch"
    candidate = path.with_suffix(path.suffix + suffix)
    counter = 1
    while candidate.exists():
        candidate = path.with_suffix(path.suffix + f"{suffix}{counter}")
        counter += 1
    return candidate


def _handle_existing_new_files(patch: str, repo_root: Path) -> tuple[str, Set[Path], bool]:
    """Relocate existing files when a patch adds them as new files."""
    lines = patch.splitlines()
    if not lines:
        return patch, set(), False

    sections = _split_diff_sections(lines)
    if not sections:
        return patch, set(), False

    updated_lines: list[str] = []
    backup_paths: Set[Path] = set()
    removed_any = False

    for start, end in sections:
        section_lines = lines[start:end]
        header = section_lines[0]
        match = _DIFF_HEADER.match(header)
        if not match:
            updated_lines.extend(section_lines)
            continue

        target = _normalise_diff_path(match.group(2))
        if target is None:
            updated_lines.extend(section_lines)
            continue

        is_new_file = any(line.startswith("new file mode") for line in section_lines)
        if not is_new_file:
            updated_lines.extend(section_lines)
            continue

        target_path = repo_root / target
        if not target_path.exists():
            updated_lines.extend(section_lines)
            continue

        desired_bytes = _reconstruct_new_file_bytes(section_lines)

        try:
            current_bytes = target_path.read_bytes()
        except OSError:
            current_bytes = b""

        if current_bytes == desired_bytes:
            removed_any = True
            continue

        backup_path = _candidate_backup_path(target_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.rename(backup_path)
        backup_paths.add(backup_path.relative_to(repo_root))
        updated_lines.extend(section_lines)

    if not updated_lines:
        return "", backup_paths, removed_any

    new_patch = "\n".join(updated_lines)
    if new_patch and not new_patch.endswith("\n"):
        new_patch += "\n"
    return new_patch, backup_paths, removed_any


def _build_diff_header(operation: str, path: str) -> list[str]:
    """Render diff headers for add/update/delete operations."""
    clean_path = path.strip()
    if not clean_path:
        raise PatchError("Patch block missing target path.")
    lines: list[str] = []
    if operation == "update":
        lines.append(f"diff --git a/{clean_path} b/{clean_path}")
        lines.append(f"--- a/{clean_path}")
        lines.append(f"+++ b/{clean_path}")
    elif operation == "add":
        lines.append(f"diff --git a/{clean_path} b/{clean_path}")
        lines.append("new file mode 100644")
        lines.append("index 0000000..1111111")
        lines.append("--- /dev/null")
        lines.append(f"+++ b/{clean_path}")
    elif operation == "delete":
        lines.append(f"diff --git a/{clean_path} b/{clean_path}")
        lines.append("deleted file mode 100644")
        lines.append("index 1111111..0000000")
        lines.append(f"--- a/{clean_path}")
        lines.append("+++ /dev/null")
    else:
        raise PatchError(f"Unknown patch operation: {operation}")
    return lines


def _convert_apply_patch_format(patch: str) -> str:
    """Translate apply_patch DSL into standard unified diff lines."""
    if _APPLY_PATCH_HEADER.search(patch) is None:
        return patch

    lines = patch.splitlines()
    converted: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("*** Begin Patch"):
            index += 1
            continue

        index += 1
        if index >= len(lines):
            break

        header = lines[index]
        if header.startswith("*** Update File: "):
            operation = "update"
            target = header.split(": ", 1)[1].strip()
        elif header.startswith("*** Add File: "):
            operation = "add"
            target = header.split(": ", 1)[1].strip()
        elif header.startswith("*** Delete File: "):
            operation = "delete"
            target = header.split(": ", 1)[1].strip()
        else:
            raise PatchError(f"Unsupported apply_patch header: {header}")

        index += 1
        body: list[str] = []
        while index < len(lines):
            candidate = lines[index]
            if candidate.startswith("*** End Patch"):
                break
            body.append(candidate)
            index += 1

        while index < len(lines) and not lines[index].startswith("*** End Patch"):
            index += 1
        if index < len(lines) and lines[index].startswith("*** End Patch"):
            index += 1

        converted.extend(_build_diff_header(operation, target))
        converted.extend(body)

    converted_text = "\n".join(converted)
    if converted_text and not converted_text.endswith("\n"):
        converted_text += "\n"
    return converted_text or patch


def _trim_dangling_hunks(patch: str) -> str:
    """Drop empty hunks that only contain headers but no body."""
    lines = patch.splitlines()
    if not lines:
        return patch

    keep = [True] * len(lines)
    for idx, line in enumerate(lines):
        if not line.startswith("@@"):
            continue
        j = idx + 1
        has_body = False
        while j < len(lines):
            follower = lines[j]
            if follower.startswith("diff --git "):
                break
            if follower.startswith("@@"):
                break
            if follower.startswith(("+", "-", " ", "\\ No newline")):
                has_body = True
                break
            if follower.strip():
                has_body = True
                break
            j += 1
        if not has_body:
            keep[idx] = False

    trimmed = "\n".join(line for flag, line in zip(keep, lines) if flag)
    if trimmed and not trimmed.endswith("\n"):
        trimmed += "\n"
    return trimmed


def _sync_src_layout(repo_root: Path) -> Set[Path]:
    """Ensure top-level package copies merge into src/ layout."""

    top_level = repo_root / "password_manager"
    src_root = repo_root / "src"
    touched: Set[Path] = set()

    if not top_level.exists() or not top_level.is_dir():
        return touched

    target_pkg = src_root / "password_manager"
    target_pkg.parent.mkdir(parents=True, exist_ok=True)

    if not target_pkg.exists():
        shutil.move(str(top_level), str(target_pkg))
        touched.add(target_pkg.relative_to(repo_root))
        return touched

    # Merge contents into existing src/password_manager
    for path in sorted(top_level.rglob("*")):
        relative = path.relative_to(top_level)
        destination = target_pkg / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(path), str(destination))
        touched.add(destination.relative_to(repo_root))

    shutil.rmtree(top_level, ignore_errors=True)
    return touched


def _normalise_hunks(patch: str) -> tuple[str, list[str]]:
    """Return a patch with corrected hunk line counts where needed."""

    lines = patch.splitlines()
    normalised: list[str] = []
    adjustments: list[str] = []

    index = 0
    current_path: Path | None = None

    while index < len(lines):
        line = lines[index]

        if line.startswith("diff --git "):
            normalised.append(line)
            match = _DIFF_HEADER.match(line)
            current_path = None
            if match:
                current_path = _normalise_diff_path(match.group(2))
            index += 1
            continue

        if line.startswith("@@ "):
            match = _HUNK_HEADER.match(line)
            if not match:
                raise PatchError(f"Malformed hunk header: {line}")

            header_position = len(normalised)
            normalised.append(line)  # placeholder, may be replaced later

            expected_removed = _default_count(match.group("old_count"))
            expected_added = _default_count(match.group("new_count"))
            seen_removed = 0
            seen_added = 0

            index += 1
            while index < len(lines):
                candidate = lines[index]
                if candidate.startswith("diff --git ") or candidate.startswith("@@ "):
                    break
                if candidate.startswith("\\"):
                    normalised.append(candidate)
                    index += 1
                    continue

                prefix = candidate[:1]
                if prefix == "+":
                    seen_added += 1
                elif prefix == "-":
                    seen_removed += 1
                else:
                    seen_added += 1
                    seen_removed += 1

                normalised.append(candidate)
                index += 1

            if seen_removed != expected_removed or seen_added != expected_added:
                location = current_path.as_posix() if current_path else "<unknown>"
                adjustments.append(
                    f"{location}: adjusted hunk counts "
                    f"(-{expected_removed}/+{expected_added} -> "
                    f"-{seen_removed}/+{seen_added})"
                )
                new_header = (
                    f"@@ -{_format_range(match.group('old_start'), match.group('old_count'), seen_removed)} "
                    f"+{_format_range(match.group('new_start'), match.group('new_count'), seen_added)} @@"
                )
                normalised[header_position] = new_header
            else:
                normalised[header_position] = line
            continue

        normalised.append(line)
        index += 1

    result = "\n".join(normalised)
    if result and not result.endswith("\n"):
        result += "\n"
    return result, adjustments


def _validate_hunks(patch: str) -> None:
    """Ensure each hunk in the patch reports accurate line counts."""
    current_path: Path | None = None
    lines = patch.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            match = _DIFF_HEADER.match(line)
            current_path = None
            if match:
                target = _normalise_diff_path(match.group(2))
                current_path = target
            index += 1
            continue

        if line.startswith("@@ "):
            match = _HUNK_HEADER.match(line)
            if not match:
                raise PatchError(f"Malformed hunk header: {line}")

            expected_removed = _default_count(match.group("old_count"))
            expected_added = _default_count(match.group("new_count"))
            seen_removed = 0
            seen_added = 0

            index += 1
            while index < len(lines):
                candidate = lines[index]
                if candidate.startswith("diff --git ") or candidate.startswith("@@ "):
                    break
                if candidate.startswith("\\"):
                    index += 1
                    continue

                prefix = candidate[:1]
                if prefix == "+":
                    seen_added += 1
                elif prefix == "-":
                    seen_removed += 1
                else:
                    seen_added += 1
                    seen_removed += 1
                index += 1

            if seen_removed != expected_removed or seen_added != expected_added:
                location = current_path.as_posix() if current_path else "<unknown>"
                raise PatchError(
                    "Patch hunk line count mismatch for "
                    f"{location}: expected -{expected_removed}/+{expected_added} "
                    f"but saw -{seen_removed}/+{seen_added}."
                )
            continue

        index += 1


def _is_docstring_only_module(source: str) -> bool:
    """Return True when a module body consists solely of docstring/pass nodes."""
    try:
        module = ast.parse(source)
    except SyntaxError:
        return False
    if not module.body:
        return False

    for node in module.body:
        if isinstance(node, ast.Expr):
            value = getattr(node, "value", None)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                continue
        if isinstance(node, ast.Pass):
            continue
        return False
    return True


def _read_head_file(repo_root: Path, path: Path) -> str | None:
    """Read the contents of ``path`` from ``HEAD`` without raising on failure."""
    rel_path = path.as_posix()
    result = _run_git(["show", f"HEAD:{rel_path}"], cwd=repo_root, check=False)
    if result.returncode != 0:
        return None
    return result.stdout


def _collect_docstring_map(source: str | None) -> dict[str, str]:
    """Return a mapping of module/class/function docstring contents."""
    if not source:
        return {}
    try:
        module = ast.parse(source)
    except SyntaxError:
        return {}

    docstrings: dict[str, str] = {}
    module_doc = ast.get_docstring(module, clean=False)
    if module_doc is not None:
        docstrings["module"] = module_doc

    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualname = ".".join(parents + [child.name])
                doc = ast.get_docstring(child, clean=False)
                if doc is not None:
                    docstrings[f"class:{qualname}"] = doc
                visit(child, parents + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join(parents + [child.name])
                doc = ast.get_docstring(child, clean=False)
                if doc is not None:
                    docstrings[f"function:{qualname}"] = doc
                visit(child, parents + [child.name])
            elif isinstance(child, ast.Assign):
                continue
            else:
                visit(child, parents)

    visit(module, [])
    return docstrings


def _format_docstring_label(raw: str) -> str:
    """Render human-friendly labels for docstring map keys."""
    if raw == "module":
        return "<module>"
    if ":" not in raw:
        return raw
    kind, name = raw.split(":", 1)
    return f"{kind} {name}"


def apply_patch(
    patch: str,
    *,
    repo_root: Path | str = ".",
    check: bool = True,
    config_path: Path | str | None = None,
) -> PatchResult:
    """Apply a unified diff ``patch`` to ``repo_root`` with hardened validation."""

    repo_root_path = Path(repo_root).resolve()
    config_path_obj = Path(config_path) if config_path is not None else None

    builder = _PatchApplicationBuilder(
        repo_root=repo_root_path,
        patch_text=patch,
        config_path=config_path_obj,
    )
    try:
        preparation = builder.prepare()
    except PatchError as error:
        telemetry_payload = error.details.get("telemetry")
        if telemetry_payload:
            _emit_patch_event(
                "patch_validation_failed",
                stage="prepare",
                telemetry=telemetry_payload,
            )
        raise

    telemetry = preparation.telemetry

    if not preparation.patch.strip():
        ordered_paths = tuple(sorted(preparation.backup_paths, key=lambda item: item.as_posix()))
        telemetry.touched_paths = ordered_paths
        _emit_patch_event(
            "patch_skipped",
            reason="no-op",
            telemetry=telemetry.to_dict(),
        )
        return PatchResult(
            command=(),
            paths=ordered_paths,
            stdout="",
            stderr="",
            telemetry=telemetry,
        )

    command: Tuple[str, ...] = ("git", "apply")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(preparation.patch)
        handle.flush()
        temp_path = Path(handle.name)

    telemetry.patch_path = temp_path

    try:
        if check:
            dry_run = subprocess.run(
                ["git", "apply", "--check", str(temp_path)],
                cwd=repo_root_path,
                capture_output=True,
                text=True,
                check=False,
            )
            telemetry.check_returncode = dry_run.returncode
            telemetry.check_stdout = dry_run.stdout.strip()
            telemetry.check_stderr = dry_run.stderr.strip()
            combined_output = "\n".join(
                part for part in (dry_run.stderr, dry_run.stdout) if part
            )
            telemetry.failing_hunks = _parse_git_apply_failures(combined_output)

            if dry_run.returncode != 0:
                message = telemetry.check_stderr or telemetry.check_stdout or "unknown error"
                telemetry.guidance_violations = telemetry.guidance_violations + (f"git apply --check failed: {message}",)
                payload = telemetry.to_dict()
                _emit_patch_event(
                    "patch_validation_failed",
                    stage="git-apply-check",
                    telemetry=payload,
                )
                raise PatchError(
                    f"Patch failed validation: {message}",
                    details={"telemetry": payload},
                )

            _emit_patch_event(
                "patch_validation_passed",
                telemetry=telemetry.to_dict(),
            )

        result = subprocess.run(
            ["git", "apply", str(temp_path)],
            cwd=repo_root_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "unknown error"
            telemetry.guidance_violations = telemetry.guidance_violations + (f"git apply failed: {message}",)
            payload = telemetry.to_dict()
            _emit_patch_event(
                "patch_apply_failed",
                telemetry=payload,
            )
            raise PatchError(
                f"Patch failed to apply: {message}",
                details={"telemetry": payload},
            )


        combined_paths = set(preparation.paths)
        combined_paths.update(preparation.backup_paths)
        combined_paths.update(_sync_src_layout(repo_root_path))

        ordered_paths = tuple(sorted(combined_paths, key=lambda item: item.as_posix()))
        telemetry.touched_paths = ordered_paths

        payload = telemetry.to_dict()
        _emit_patch_event(
            "patch_apply_succeeded",
            telemetry=payload,
        )

        return PatchResult(
            command=command,
            paths=ordered_paths,
            stdout=result.stdout,
            stderr=result.stderr,
            telemetry=telemetry,
        )
    finally:
        temp_path.unlink(missing_ok=True)


__all__ = [
    "PatchError",
    "PatchResult",
    "PatchTelemetry",
    "apply_patch",
    "build_structured_patch",
    "canonicalise_unified_diff",
]
