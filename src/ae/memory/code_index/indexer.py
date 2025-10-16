"""Coordinate the symbol, text, embedding, and graph indices for source files."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Set

from .embeddings import EmbeddingsIndex
from .graph_index import GraphIndex
from .symbol_index import SymbolIndex
from .text_index import TextIndex


class CodeIndexer:
    """Coordinates the individual code-memory indices."""

    def __init__(self, repo_root: Path, data_root: Path) -> None:
        self.repo_root = repo_root
        self.data_root = data_root
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.index_root = self.data_root / "index"
        self.index_root.mkdir(parents=True, exist_ok=True)

        self.symbol_index = SymbolIndex(self.index_root / "symbols.json")
        self.text_index = TextIndex(self.index_root / "text.sqlite")
        self.embedding_index = EmbeddingsIndex(self.index_root / "embeddings.json")
        self.graph_index = GraphIndex(self.index_root / "graph.json")

    def reindex(self, paths: Optional[Sequence[Path]] = None) -> List[Path]:
        changed = list(paths) if paths is not None else self._detect_changed_files()
        if not changed and paths is None and self.symbol_index.is_empty:
            changed = self._list_all_python_files()

        processed: List[Path] = []
        for path in changed:
            normalized = self._normalize_path(path)
            if normalized.suffix != ".py":
                continue
            absolute = self.repo_root / normalized
            if self.index_root in absolute.parents:
                continue
            if absolute.exists():
                try:
                    source = absolute.read_text(encoding="utf-8")
                except OSError:
                    continue
                self.symbol_index.index_file(normalized, source)
                self.text_index.index_document(normalized, source)
                self.embedding_index.index_document(normalized, source)
                self.graph_index.index_file(normalized, source)
            else:
                self.symbol_index.remove(normalized)
                self.text_index.remove(normalized)
                self.embedding_index.remove(normalized)
                self.graph_index.remove(normalized)
            processed.append(normalized)
        return processed

    def _normalize_path(self, path: Path | str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            candidate = candidate.relative_to(self.repo_root)
        return candidate

    def _detect_changed_files(self) -> List[Path]:
        files: Set[Path] = set()
        if self._git_has_head():
            files.update(self._git_paths(["git", "diff", "--name-only", "HEAD"]))
            files.update(self._git_paths(["git", "diff", "--name-only", "--cached", "HEAD"]))
        else:
            files.update(self._git_paths(["git", "ls-files"]))
        files.update(self._git_paths(["git", "ls-files", "--others", "--exclude-standard"]))
        return sorted(files)

    def _git_paths(self, command: Sequence[str]) -> Set[Path]:
        try:
            output = subprocess.check_output(
                command,
                cwd=self.repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return set()
        paths: Set[Path] = set()
        for line in output.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            try:
                normalized = self._normalize_path(candidate)
            except ValueError:
                continue
            if (self.repo_root / normalized).suffix != ".py":
                continue
            paths.add(normalized)
        return paths

    def _list_all_python_files(self) -> List[Path]:
        python_files: List[Path] = []
        for path in self.repo_root.rglob("*.py"):
            try:
                relative = path.relative_to(self.repo_root)
            except ValueError:
                continue
            if self.index_root in path.parents:
                continue
            if ".git" in path.parts:
                continue
            python_files.append(relative)
        return sorted(python_files)

    def _git_has_head(self) -> bool:
        try:
            subprocess.check_output(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=self.repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @classmethod
    def from_config(cls, config: Mapping[str, object], config_path: Optional[Path] = None) -> "CodeIndexer":
        project = config.get("project")
        paths_section = config.get("paths")

        repo_root_value: str = "."
        data_root_value: str = "data"

        if isinstance(project, Mapping):
            candidate = project.get("repo_root")
            if isinstance(candidate, str):
                repo_root_value = candidate

        if isinstance(paths_section, Mapping):
            candidate = paths_section.get("data")
            if isinstance(candidate, str):
                data_root_value = candidate

        base_path = config_path.parent if config_path is not None else Path.cwd()

        repo_root = Path(repo_root_value)
        if not repo_root.is_absolute():
            repo_root = (base_path / repo_root).resolve()

        data_root = Path(data_root_value)
        if not data_root.is_absolute():
            data_root = (repo_root / data_root).resolve()

        return cls(repo_root=repo_root, data_root=data_root)
