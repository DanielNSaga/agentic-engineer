"""Workspace hygiene helpers to keep repositories patch-friendly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .vcs import GitError, GitRepository


_BYTECODE_PATTERNS: tuple[str, ...] = ("*.pyc", "*.pyo", "*.pyd", "*$py.class")
_CHUNK_SIZE = 64


@dataclass(slots=True)
class HygieneResult:
    """Report emitted after enforcing workspace hygiene rules."""

    removed: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def ensure_workspace_hygiene(repo: GitRepository) -> HygieneResult:
    """Ensure common compiled artifacts are not tracked by git.

    The agent frequently runs into ``git apply`` failures when ``*.pyc`` files
    or ``__pycache__`` directories are committed.  Prior to applying a patch we
    inspect the index and untrack any compiled Python artifacts so subsequent
    patches operate on source files only.  The files remain on disk, but git no
    longer considers them part of the index.
    """

    try:
        tracked = repo.list_tracked_paths(*_BYTECODE_PATTERNS)
    except GitError as error:
        return HygieneResult(errors=(f"Unable to inspect tracked bytecode artifacts: {error}",))

    if not tracked:
        return HygieneResult()

    tracked_sorted = tuple(sorted(tracked, key=lambda path: path.as_posix()))
    removed: list[Path] = []
    for chunk in _chunked(tracked_sorted, _CHUNK_SIZE):
        if not chunk:
            continue
        args = ["rm", "--cached", "--ignore-unmatch", "-q", "--", *[path.as_posix() for path in chunk]]
        try:
            repo.git(*args)
        except GitError as error:
            details = f"Failed to untrack compiled artifacts: {error}"
            return HygieneResult(removed=tuple(removed), errors=(details,))
        removed.extend(chunk)

    warning = "Removed compiled Python artifacts from git index to unblock patch application."
    return HygieneResult(removed=tuple(removed), warnings=(warning,))


def _chunked(sequence: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    """Yield sequential slices of ``sequence`` with at most ``size`` items."""
    step = max(1, size)
    for index in range(0, len(sequence), step):
        yield sequence[index : index + step]
