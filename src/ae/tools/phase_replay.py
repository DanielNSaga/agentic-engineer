"""Helpers for constructing replay workspaces from stored phase logs."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from .phase_logs import PhaseLogEntry, load_phase_log
from .vcs import GitError, GitRepository
from .workspace_state import apply_workspace_state

__all__ = [
    "PhaseReplayConfig",
    "PhaseReplayWorkspace",
    "prepare_replay_workspace",
    "load_phase_log",
]


@dataclass(slots=True)
class PhaseReplayConfig:
    """Configuration for constructing a replay workspace."""

    repo: GitRepository
    data_root: Path
    keep_existing: bool = False
    identifier: str | None = None

    def workspace_root(self) -> Path:
        base = (self.data_root / "replay-workspaces").resolve()
        base.mkdir(parents=True, exist_ok=True)
        slug = self.identifier or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        name = f"replay-{slug}"
        return (base / name).resolve()


@dataclass(slots=True)
class PhaseReplayWorkspace:
    """Workspace materialised for replaying a stored phase log."""

    log: PhaseLogEntry
    repo: GitRepository
    root: Path
    snapshot_applied: bool


def prepare_replay_workspace(
    config: PhaseReplayConfig,
    log: PhaseLogEntry,
    *,
    apply_snapshot: bool = True,
    allow_partial: bool = True,
) -> PhaseReplayWorkspace:
    """
    Clone the repository into a new replay workspace and, if requested, apply
    the logged workspace snapshot so file contents reflect the original failure.
    """

    workspace_root = config.workspace_root()
    if workspace_root.exists():
        if config.keep_existing:
            raise FileExistsError(f"Replay workspace already exists: {workspace_root}")
        shutil.rmtree(workspace_root, ignore_errors=True)

    command = [
        "git",
        "clone",
        "--local",
        "--no-hardlinks",
        config.repo.root.as_posix(),
        workspace_root.as_posix(),
    ]
    process = subprocess.run(
        command,
        cwd=workspace_root.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip() or "unable to clone repository"
        raise GitError(message)

    workspace_repo = GitRepository(workspace_root)
    snapshot_applied = False

    if apply_snapshot:
        snapshot = log.workspace_state
        if snapshot:
            try:
                apply_workspace_state(workspace_repo, snapshot)
                snapshot_applied = True
            except GitError:
                if not allow_partial:
                    raise

    return PhaseReplayWorkspace(
        log=log,
        repo=workspace_repo,
        root=workspace_root,
        snapshot_applied=snapshot_applied,
    )
