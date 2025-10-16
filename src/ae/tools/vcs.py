"""Minimal git helpers
The helpers below provide just enough structure to snapshot the
working tree, list pending changes, and roll back to a known-good checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set

import shutil
import subprocess
import time


class GitError(RuntimeError):
    """Raised when a git command fails or the repository cannot be used."""


@dataclass(slots=True)
class GitCheckpoint:
    """Snapshot of the working tree at a point in time.

    The checkpoint records the current ``HEAD`` (if any) and the set of
    pre-existing untracked paths.  Rolling back restores tracked files to the
    recorded commit and removes only the untracked files that appeared after
    the checkpoint was taken.
    """

    repo: "GitRepository"
    label: str
    head: str | None
    baseline_untracked: tuple[str, ...]
    created_at: float

    def rollback(self) -> None:
        """Restore the repository to the checkpoint."""

        self.repo.restore_checkpoint(self)

    def dirty_paths(self, *, include_untracked: bool = True) -> List[Path]:
        """Return the current working tree changes relative to the checkpoint."""

        return self.repo.working_tree_changes(include_untracked=include_untracked)


class GitRepository:
    """Lightweight wrapper around ``git`` commands."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        if not (self.root / ".git").exists():
            raise GitError(f"Not a git repository: {self.root}")

    @classmethod
    def discover(cls, start: Path | str | None = None) -> "GitRepository":
        """Locate the nearest git repository starting from ``start``."""

        path = Path(start or Path.cwd()).resolve()
        for candidate in (path, *path.parents):
            if (candidate / ".git").exists():
                return cls(candidate)
        raise GitError(f"Unable to locate a git repository from {path}")

    @classmethod
    def initialise(cls, root: Path | str) -> "GitRepository":
        """Initialise a new git repository at ``root`` with an initial commit."""

        path = Path(root).resolve()
        path.mkdir(parents=True, exist_ok=True)
        git_dir = path / ".git"
        if git_dir.exists():
            for item in git_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            git_dir.rmdir()

        def _run(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
            command = ["git", *args]
            process = subprocess.run(
                command,
                cwd=path,
                capture_output=True,
                text=False,
                check=False,
            )
            stdout = process.stdout.decode("utf-8", errors="replace") if process.stdout else ""
            stderr = process.stderr.decode("utf-8", errors="replace") if process.stderr else ""
            result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
            if check and result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip() or "unknown git error"
                raise GitError(f"git {' '.join(args)} failed: {message}")
            return result

        _run(["init"])

        def _ensure_config(key: str, value: str) -> None:
            probe = _run(["config", "--get", key], check=False)
            if probe.returncode != 0 or not probe.stdout.strip():
                _run(["config", key, value])

        _ensure_config("user.email", "agent@example.com")
        _ensure_config("user.name", "Agentic Engineer")

        _run(["add", "."])
        _run(["commit", "--allow-empty", "-m", "Initial commit"])

        return cls(path)

    # ------------------------------------------------------------------ git IO
    def _run_git(self, args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = ["git", *args]
        process = subprocess.run(
            command,
            cwd=self.root,
            capture_output=True,
            text=False,
            check=False,
        )
        stdout = process.stdout.decode("utf-8", errors="replace") if process.stdout else ""
        stderr = process.stderr.decode("utf-8", errors="replace") if process.stderr else ""
        result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
        if check and result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "unknown git error"
            raise GitError(f"git {' '.join(args)} failed: {message}")
        return result

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Execute ``git`` with ``args`` relative to the repository root."""

        return self._run_git(list(args), check=check)

    def list_tracked_paths(self, *patterns: str) -> List[Path]:
        """Return tracked paths that match the supplied git pathspec patterns.

        When no patterns are supplied the entire tracked file list is returned.
        Paths are reported relative to the repository root.
        """

        args: List[str] = ["ls-files", "-z"]
        if patterns:
            args.extend(["--", *patterns])

        result = self._run_git(args, check=False)
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "unable to list tracked paths"
            raise GitError(f"git ls-files failed: {message}")

        payload = result.stdout
        if not payload:
            return []

        entries = [entry for entry in payload.split("\0") if entry]
        return [Path(entry) for entry in entries]

    # -------------------------------------------------------------- branches
    def current_branch(self) -> str | None:
        """Return the current branch name or ``None`` when detached."""

        result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], check=False)
        if result.returncode != 0:
            return None
        branch = result.stdout.strip()
        if not branch or branch == "HEAD":
            return None
        return branch

    # ------------------------------------------------------------------- stash
    def stash_push(
        self,
        *,
        message: str | None = None,
        include_untracked: bool = False,
    ) -> str | None:
        """Stash pending changes and return the created reference."""

        args: List[str] = ["stash", "push"]
        if include_untracked:
            args.append("-u")
        if message:
            args.extend(["-m", message])
        result = self._run_git(args, check=False)
        combined = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode != 0:
            if "No local changes to save" in combined:
                return None
            message_text = combined or "unknown git error"
            raise GitError(f"git {' '.join(args)} failed: {message_text}")
        if "No local changes to save" in combined:
            return None
        return "stash@{0}"

    def stash_apply(self, ref: str = "stash@{0}") -> None:
        """Apply the specified stash without dropping it."""

        result = self._run_git(["stash", "apply", ref], check=False)
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "unknown git error"
            raise GitError(f"git stash apply {ref} failed: {message}")

    def stash_drop(self, ref: str = "stash@{0}") -> None:
        """Drop the specified stash entry."""

        result = self._run_git(["stash", "drop", ref], check=False)
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "unknown git error"
            raise GitError(f"git stash drop {ref} failed: {message}")

    # ------------------------------------------------------------- repo status
    def _status_entries(self) -> List[tuple[str, Path]]:
        result = self._run_git(["status", "--porcelain"], check=True)
        entries: List[tuple[str, Path]] = []
        for line in result.stdout.splitlines():
            if not line:
                continue
            status = line[:2]
            raw_path = line[3:]
            if status[0] in {"R", "C"} and " -> " in raw_path:
                raw_path = raw_path.split(" -> ", 1)[1]
            status_clean = status.strip() or status
            entries.append((status_clean, Path(raw_path.strip())))
        return entries

    def working_tree_changes(self, *, include_untracked: bool = True) -> List[Path]:
        """Return the set of paths with pending modifications."""

        entries = self._status_entries()
        paths: Set[Path] = set()
        for status, path in entries:
            if status == "??" and not include_untracked:
                continue
            paths.add(path)
        return sorted(paths, key=lambda item: item.as_posix())

    def status_entries(self) -> List[tuple[str, Path]]:
        """Return raw porcelain status entries as ``(status, path)`` pairs."""

        return self._status_entries()

    def untracked_files(self) -> List[Path]:
        """Return the list of untracked files/directories."""

        return [
            path
            for status, path in self._status_entries()
            if status == "??"
        ]

    def has_changes(self, *, include_untracked: bool = True) -> bool:
        """Return ``True`` when there are working tree changes."""

        return bool(self.working_tree_changes(include_untracked=include_untracked))

    def is_clean(self, *, include_untracked: bool = True) -> bool:
        """Return ``True`` when the working tree has no pending changes."""

        return not self.working_tree_changes(include_untracked=include_untracked)

    def ensure_clean(self, *, include_untracked: bool = True) -> None:
        """Raise :class:`GitError` if the working tree is not clean."""

        if not self.is_clean(include_untracked=include_untracked):
            raise GitError("Working tree has pending changes.")

    # ------------------------------------------------------------- checkpoints
    def _current_head(self) -> str | None:
        result = self._run_git(["rev-parse", "--verify", "HEAD"], check=False)
        if result.returncode != 0:
            return None
        head = result.stdout.strip()
        return head or None

    def create_checkpoint(self, label: str | None = None) -> GitCheckpoint:
        """Record the current ``HEAD`` and untracked files."""

        head = self._current_head()
        baseline_untracked = tuple(sorted(path.as_posix() for path in self.untracked_files()))
        checkpoint_label = label or head or "working-tree"
        return GitCheckpoint(
            repo=self,
            label=checkpoint_label,
            head=head,
            baseline_untracked=baseline_untracked,
            created_at=time.time(),
        )

    def restore_checkpoint(self, checkpoint: GitCheckpoint) -> None:
        """Restore the repository to the state captured by ``checkpoint``."""

        if checkpoint.repo is not self:
            raise GitError("Checkpoint does not belong to this repository.")

        restore_args: List[str] = ["restore", "--worktree", "--staged"]
        if checkpoint.head:
            restore_args.extend(["--source", checkpoint.head])
        restore_args.extend(["--", "."])

        self._run_git(restore_args, check=True)

        baseline = {Path(entry) for entry in checkpoint.baseline_untracked}
        current_untracked = set(self.untracked_files())
        extra = sorted(
            (path for path in current_untracked if path not in baseline),
            key=lambda item: len(item.parts),
            reverse=True,
        )

        for relative in extra:
            target = self.root / relative
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists() or target.is_symlink():
                target.unlink(missing_ok=True)

    # ----------------------------------------------------------- diff helpers
    def diff(self, *paths: str) -> str:
        """Return the unified diff for ``paths`` (defaults to the whole repo)."""

        args: List[str] = ["diff"]
        args.extend(paths)
        result = self._run_git(args, check=True)
        return result.stdout

    # -------------------------------------------------------------- remotes
    def push(
        self,
        remote: str,
        branch: str,
        *,
        set_upstream: bool = False,
        force: bool = False,
    ) -> None:
        """Push ``branch`` to ``remote`` applying requested flags."""

        args: List[str] = ["push"]
        if set_upstream:
            args.append("-u")
        if force:
            args.append("--force-with-lease")
        args.extend([remote, branch])
        self._run_git(args, check=True)

    def commit_all(self, message: str, *, allow_empty: bool = False) -> str | None:
        """Add all changes to the index and create a commit.

        Returns the new commit SHA when a commit was created.  Returns ``None`` when
        there were no changes to commit (and ``allow_empty`` is ``False``).
        """

        self._run_git(["add", "--all"], check=True)

        commit_args: List[str] = ["commit", "-m", message]
        if allow_empty:
            commit_args.append("--allow-empty")

        commit = self._run_git(commit_args, check=False)
        if commit.returncode != 0:
            output = commit.stderr.strip() or commit.stdout.strip() or ""
            if "nothing to commit" in output.lower():
                return None
            raise GitError(f"git commit failed: {output}")

        rev = self._run_git(["rev-parse", "HEAD"], check=True)
        return rev.stdout.strip()


__all__ = ["GitCheckpoint", "GitError", "GitRepository"]
