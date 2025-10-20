"""Capture and replay lightweight snapshots of the git workspace state."""

from __future__ import annotations

import base64
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .vcs import GitError, GitRepository

__all__ = [
    "MAX_DIFF_BYTES",
    "MAX_UNTRACKED_BYTES",
    "WorkspaceState",
    "capture_workspace_state",
    "apply_workspace_state",
]

# Limit diff/untracked payloads recorded alongside phase logs.
MAX_DIFF_BYTES = 400_000
MAX_UNTRACKED_BYTES = 250_000


@dataclass(slots=True)
class WorkspaceState:
    """Serialisable snapshot of the workspace relative to a git commit."""

    format_version: int = 1
    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    repo_root: str | None = None
    checkpoint_label: str | None = None
    head: str | None = None
    diff: str | None = None
    diff_truncated: bool = False
    untracked: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` representation suitable for JSON logs."""
        return {
            "format_version": self.format_version,
            "captured_at": self.captured_at,
            "repo_root": self.repo_root,
            "checkpoint_label": self.checkpoint_label,
            "head": self.head,
            "diff": self.diff,
            "diff_truncated": self.diff_truncated,
            "untracked": [dict(entry) for entry in self.untracked],
        }


def capture_workspace_state(
    repo: GitRepository,
    *,
    checkpoint_label: str | None = None,
    max_diff_bytes: int = MAX_DIFF_BYTES,
    max_untracked_bytes: int = MAX_UNTRACKED_BYTES,
) -> WorkspaceState:
    """Capture a best-effort snapshot of the workspace for later replay."""

    state = WorkspaceState(
        repo_root=repo.root.as_posix(),
        checkpoint_label=checkpoint_label,
    )

    # Record the current HEAD commit if available.
    head_probe = repo.git("rev-parse", "--verify", "HEAD", check=False)
    if head_probe.returncode == 0:
        head = head_probe.stdout.strip()
        if head:
            state.head = head

    # Capture the working tree diff relative to HEAD for tracked paths.
    diff_result = repo.git("diff", "--binary", "HEAD", check=False)
    if diff_result.returncode == 0:
        diff_payload = diff_result.stdout or ""
        if diff_payload:
            encoded = diff_payload if isinstance(diff_payload, str) else diff_payload.decode("utf-8", errors="replace")
            if len(encoded.encode("utf-8")) > max_diff_bytes:
                state.diff = encoded.encode("utf-8")[:max_diff_bytes].decode("utf-8", errors="ignore")
                state.diff_truncated = True
            else:
                state.diff = encoded

    # Persist lightweight payloads for untracked files so replays can restore them.
    for relative in repo.untracked_files():
        # Skip directories/symlinks for now; replays can recreate empty dirs on demand.
        absolute = (repo.root / relative).resolve()
        if not absolute.is_file():
            continue
        try:
            data = absolute.read_bytes()
        except OSError:
            continue
        if not data:
            encoded_payload = ""
            truncated_flag = False
        elif len(data) > max_untracked_bytes:
            encoded_payload = base64.b64encode(data[:max_untracked_bytes]).decode("ascii")
            truncated_flag = True
        else:
            encoded_payload = base64.b64encode(data).decode("ascii")
            truncated_flag = False
        entry: dict[str, Any] = {
            "path": relative.as_posix(),
            "encoding": "base64",
            "content": encoded_payload,
            "size": _safe_size(absolute, fallback=len(data)),
            "truncated": truncated_flag,
            "mode": _file_mode(absolute),
        }
        state.untracked.append(entry)

    return state


def apply_workspace_state(repo: GitRepository, snapshot: Mapping[str, Any]) -> None:
    """Restore the workspace to the snapshot best effort without touching commits.

    This helper resets tracked files to the recorded commit (when present),
    applies the recorded diff, and re-materialises untracked file payloads.
    """

    head = snapshot.get("head")
    if isinstance(head, str) and head.strip():
        repo.git("reset", "--hard", head.strip(), check=True)

    diff_payload = snapshot.get("diff")
    diff_truncated = snapshot.get("diff_truncated")
    if isinstance(diff_truncated, bool) and diff_truncated:
        diff_payload = None
    if isinstance(diff_payload, str) and diff_payload.strip():
        process = subprocess.run(
            ["git", "apply", "--whitespace=nowarn"],
            cwd=repo.root,
            input=diff_payload,
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            message = process.stderr.strip() or process.stdout.strip() or "unable to apply workspace diff"
            raise GitError(message)

    untracked_payload = snapshot.get("untracked")
    if isinstance(untracked_payload, Iterable):
        for entry in untracked_payload:
            if not isinstance(entry, Mapping):
                continue
            path = entry.get("path")
            content = entry.get("content")
            encoding = entry.get("encoding")
            if not isinstance(path, str) or not isinstance(content, str):
                continue
            target = (repo.root / path).resolve()
            try:
                target.relative_to(repo.root)
            except ValueError:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = base64.b64decode(content) if encoding == "base64" else content.encode("utf-8")
            except ValueError:
                data = content.encode("utf-8", errors="ignore")
            with target.open("wb") as handle:
                handle.write(data)
            mode = entry.get("mode")
            if isinstance(mode, str):
                try:
                    os.chmod(target, int(mode, 8))
                except OSError:
                    continue


def _file_mode(path: Path) -> str | None:
    try:
        return oct(path.stat().st_mode & 0o777)
    except OSError:
        return None


def _safe_size(path: Path, *, fallback: int) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return fallback
