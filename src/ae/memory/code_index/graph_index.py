"""Import graph index tracking module-level dependencies."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set
from importlib.util import resolve_name


@dataclass(frozen=True)
class GraphRecord:
    """Single indexed module with its import dependencies."""

    path: str
    module: str
    imports: List[str]


class GraphIndex:
    """Basic import graph index keyed by source path."""

    def __init__(self, storage_path: Path) -> None:
        self._storage_path = storage_path
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, GraphRecord] = {}
        self._load()

    def index_file(self, path: Path, source: str) -> None:
        path_key = self._path_key(path)
        module_name = self._module_name(path)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            self._records.pop(path_key, None)
            self._persist()
            return

        imports: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                target = self._resolve_import_from(module_name, path, node)
                if target:
                    imports.add(target)
        record = GraphRecord(
            path=path_key,
            module=module_name,
            imports=sorted(imports),
        )
        self._records[path_key] = record
        self._persist()

    def remove(self, path: Path) -> None:
        path_key = self._path_key(path)
        if path_key in self._records:
            del self._records[path_key]
            self._persist()

    def get_imports(self, path: Path | str) -> List[str]:
        path_key = path if isinstance(path, str) else self._path_key(path)
        record = self._records.get(path_key)
        if record is None:
            return []
        return list(record.imports)

    def modules(self) -> Dict[str, GraphRecord]:
        return dict(self._records)

    def _resolve_import_from(self, module_name: str, path: Path, node: ast.ImportFrom) -> str:
        level = node.level or 0
        module_fragment = node.module or ""
        dotted = "." * level + module_fragment
        if not dotted:
            return ""
        package = self._package_for_path(module_name, path)
        if package:
            try:
                return resolve_name(dotted, package)
            except ValueError:
                return dotted.lstrip(".")
        return dotted.lstrip(".")

    def _module_name(self, path: Path) -> str:
        relative = path.with_suffix("")
        parts = [
            part
            for part in relative.parts
            if part not in {"__pycache__", ""}
        ]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _path_key(self, path: Path) -> str:
        return path.as_posix()

    def _package_for_path(self, module_name: str, path: Path) -> str:
        if not module_name:
            return ""
        if path.name == "__init__.py":
            return module_name
        parts = module_name.split(".")
        if len(parts) <= 1:
            return parts[0]
        return ".".join(parts[:-1])

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        for path, payload in data.items():
            self._records[path] = GraphRecord(
                path=payload["path"],
                module=payload["module"],
                imports=list(payload["imports"]),
            )

    def _persist(self) -> None:
        payload = {
            path: asdict(record)
            for path, record in sorted(self._records.items())
        }
        self._storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
