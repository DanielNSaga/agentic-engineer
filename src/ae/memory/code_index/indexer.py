"""Coordinate the symbol, text, embedding, and graph indices for source files."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

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
        self._metadata_path = self.index_root / "metadata.json"
        self._metadata: Dict[str, Any] = self._load_metadata()

        self.symbol_index = SymbolIndex(self.index_root / "symbols.json")
        self.text_index = TextIndex(self.index_root / "text.sqlite")
        self.embedding_index = EmbeddingsIndex(self.index_root / "embeddings.json")
        self.graph_index = GraphIndex(self.index_root / "graph.json")

    def reindex(self, paths: Optional[Sequence[Path]] = None) -> List[Path]:
        current_head = self._current_head()
        last_indexed = self._last_indexed_commit()
        explicit_paths = paths is not None

        changed: Set[Path] = set()
        if paths is not None:
            changed.update(Path(path) for path in paths)

        fallback_full_scan = False
        if not explicit_paths:
            changed.update(self._detect_changed_files())
            commit_changes, commit_diff_ok = self._collect_commit_changes(
                last_indexed, current_head
            )
            changed.update(commit_changes)
            if (
                not commit_diff_ok
                and current_head is not None
                and current_head != last_indexed
            ):
                fallback_full_scan = True

        removed_disallowed = self._purge_disallowed_entries()

        if (not changed and not explicit_paths) and (
            self.symbol_index.is_empty or fallback_full_scan
        ):
            changed.update(self._list_all_python_files())

        if not changed:
            if (
                not explicit_paths
                and current_head is not None
                and current_head != last_indexed
            ):
                self._update_last_indexed_commit(current_head)
            return []

        processed: List[Path] = list(removed_disallowed)
        seen: Set[Path] = set()
        for path in sorted(changed, key=lambda entry: str(entry)):
            try:
                normalized = self._normalize_path(path)
            except ValueError:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            if not self._should_index_path(normalized):
                self.symbol_index.remove(normalized)
                self.text_index.remove(normalized)
                self.embedding_index.remove(normalized)
                self.graph_index.remove(normalized)
                processed.append(normalized)
                continue
            absolute = self.repo_root / normalized
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

        if not explicit_paths and current_head is not None:
            self._update_last_indexed_commit(current_head)
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
            files.update(
                self._git_paths(["git", "diff", "--name-only", "--cached", "HEAD"])
            )
        else:
            files.update(self._git_paths(["git", "ls-files"]))
        files.update(
            self._git_paths(["git", "ls-files", "--others", "--exclude-standard"])
        )
        return sorted(files)

    def _git_paths(self, command: Sequence[str]) -> Set[Path]:
        paths, _ = self._run_git_path_command(command)
        return paths

    def _run_git_path_command(self, command: Sequence[str]) -> Tuple[Set[Path], bool]:
        try:
            output = subprocess.check_output(
                command,
                cwd=self.repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return set(), False
        return self._parse_git_path_output(output), True

    def _parse_git_path_output(self, output: str) -> Set[Path]:
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
        collected: Set[Path] = set()
        tracked, tracked_ok = self._run_git_path_command(["git", "ls-files"])
        if tracked_ok:
            collected.update(tracked)
        untracked, untracked_ok = self._run_git_path_command(
            ["git", "ls-files", "--others", "--exclude-standard"]
        )
        if untracked_ok:
            collected.update(untracked)
        if collected:
            return sorted(collected)

        python_files: List[Path] = []
        for path in self.repo_root.rglob("*.py"):
            try:
                relative = path.relative_to(self.repo_root)
            except ValueError:
                continue
            if not self._should_index_path(relative):
                continue
            python_files.append(relative)
        return sorted(python_files)

    def _path_under_data_root(self, path: Path) -> bool:
        try:
            data_root = self.data_root.resolve()
        except OSError:
            data_root = self.data_root
        try:
            candidate = path.resolve()
        except OSError:
            candidate = path
        try:
            candidate.relative_to(data_root)
        except ValueError:
            return False
        return True

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

    def _collect_commit_changes(
        self,
        last_commit: Optional[str],
        current_head: Optional[str],
    ) -> Tuple[Set[Path], bool]:
        if current_head is None or not current_head.strip():
            return set(), True
        if last_commit is not None and last_commit.strip() == current_head.strip():
            return set(), True
        if not last_commit:
            return set(), False
        return self._run_git_path_command(
            ["git", "diff", "--name-only", f"{last_commit}..{current_head}"]
        )

    def _current_head(self) -> Optional[str]:
        try:
            output = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        commit = output.strip()
        return commit or None

    def _last_indexed_commit(self) -> Optional[str]:
        value = self._metadata.get("last_indexed_commit")
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return trimmed
        return None

    def _update_last_indexed_commit(self, commit: str) -> None:
        trimmed = commit.strip()
        if not trimmed:
            return
        if self._metadata.get("last_indexed_commit") == trimmed:
            return
        self._metadata["last_indexed_commit"] = trimmed
        self._save_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        if not self._metadata_path.exists():
            return {}
        try:
            data = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(data, dict):
            return dict(data)
        return {}

    def _save_metadata(self) -> None:
        try:
            self._metadata_path.write_text(
                json.dumps(self._metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _should_index_path(self, normalized: Path) -> bool:
        if normalized.suffix != ".py":
            return False
        if ".git" in normalized.parts:
            return False
        absolute = self.repo_root / normalized
        if self.index_root in absolute.parents:
            return False
        if self._path_under_data_root(absolute):
            return False
        return True

    def _purge_disallowed_entries(self) -> List[Path]:
        removed: List[Path] = []
        for path_key in self.text_index.list_paths():
            normalized = Path(path_key)
            absolute = self.repo_root / normalized
            if self._should_index_path(normalized) and absolute.exists():
                continue
            self.symbol_index.remove(normalized)
            self.text_index.remove(normalized)
            self.embedding_index.remove(normalized)
            self.graph_index.remove(normalized)
            removed.append(normalized)
        return removed

    @classmethod
    def from_config(
        cls, config: Mapping[str, object], config_path: Optional[Path] = None
    ) -> "CodeIndexer":
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
