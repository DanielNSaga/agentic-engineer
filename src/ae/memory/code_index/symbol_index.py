"""Symbol index built from libcst metadata for quick code navigation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import libcst as cst
from libcst import metadata


@dataclass(frozen=True)
class SymbolRecord:
    """Persistable representation of a Python symbol."""

    name: str
    qualified_name: str
    kind: str
    path: str
    signature: str
    start: int
    end: int


class ByteOffsetCalculator:
    """Utility that converts libcst line/column data into byte offsets."""

    def __init__(self, source: str) -> None:
        self._char_offsets: List[int] = [0]
        running = 0
        for line in source.splitlines(keepends=True):
            running += len(line)
            self._char_offsets.append(running)

        self._byte_offsets: List[int] = [0] * (len(source) + 1)
        running_bytes = 0
        for index, char in enumerate(source):
            running_bytes += len(char.encode("utf-8"))
            self._byte_offsets[index + 1] = running_bytes

    def offset(self, line: int, column: int) -> int:
        if line < 1 or line > len(self._char_offsets):
            raise ValueError(f"Line out of range: {line}")
        char_index = self._char_offsets[line - 1] + column
        if char_index < 0 or char_index >= len(self._byte_offsets):
            raise ValueError(f"Column out of range: {column}")
        return self._byte_offsets[char_index]


class _SymbolCollector(cst.CSTVisitor):
    """Collect symbol definitions along with byte offsets and signatures."""

    METADATA_DEPENDENCIES = (
        metadata.PositionProvider,
        metadata.QualifiedNameProvider,
    )

    def __init__(
        self,
        path_key: str,
        byte_calculator: ByteOffsetCalculator,
        module: cst.Module,
    ) -> None:
        self._path_key = path_key
        self._byte_calculator = byte_calculator
        self._module = module
        self._class_stack: List[str] = []
        self.records: List[SymbolRecord] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        qualified = self._qualified_name(node)
        if not qualified:
            return
        start, end = self._node_byte_range(node)
        bases = [
            self._module.code_for_node(base.value)
            for base in node.bases
        ]
        signature = f"class {node.name.value}"
        if bases:
            signature = f"{signature}({', '.join(bases)})"
        record = SymbolRecord(
            name=node.name.value,
            qualified_name=qualified,
            kind="class",
            path=self._path_key,
            signature=signature,
            start=start,
            end=end,
        )
        self.records.append(record)
        self._class_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        if self._class_stack:
            self._class_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        qualified = self._qualified_name(node)
        if not qualified:
            return
        start, end = self._node_byte_range(node)
        params = self._module.code_for_node(node.params)
        signature = f"{node.name.value}{params}"
        if node.returns is not None:
            annotation = self._module.code_for_node(node.returns.annotation)
            signature = f"{signature} -> {annotation}"
        kind = "method" if self._class_stack else "function"
        display_name = node.name.value if not self._class_stack else ".".join(
            [*self._class_stack, node.name.value]
        )
        record = SymbolRecord(
            name=display_name,
            qualified_name=qualified,
            kind=kind,
            path=self._path_key,
            signature=signature,
            start=start,
            end=end,
        )
        self.records.append(record)

    def _node_byte_range(self, node: cst.CSTNode) -> tuple[int, int]:
        code_range = self.get_metadata(metadata.PositionProvider, node)
        start = self._byte_calculator.offset(code_range.start.line, code_range.start.column)
        end = self._byte_calculator.offset(code_range.end.line, code_range.end.column)
        return start, end

    def _qualified_name(self, node: cst.CSTNode) -> Optional[str]:
        qualified_names = self.get_metadata(metadata.QualifiedNameProvider, node, default=None)
        if not qualified_names:
            return None
        for qualified in qualified_names:
            if qualified.source is metadata.QualifiedNameSource.LOCAL:
                return qualified.name
        return next(iter(qualified_names)).name


class SymbolIndex:
    """Filesystem-backed index for Python symbols extracted with libcst."""

    def __init__(self, storage_path: Path) -> None:
        self._storage_path = storage_path
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._records_by_path: Dict[str, List[SymbolRecord]] = {}
        self._load()

    @property
    def is_empty(self) -> bool:
        return not self._records_by_path

    def index_file(self, path: Path, source: str) -> None:
        path_key = self._path_key(path)
        try:
            module = cst.parse_module(source)
        except cst.ParserSyntaxError:
            self._records_by_path.pop(path_key, None)
            self._persist()
            return

        wrapper = metadata.MetadataWrapper(module)
        collector = _SymbolCollector(
            path_key=path_key,
            byte_calculator=ByteOffsetCalculator(source),
            module=module,
        )
        wrapper.visit(collector)
        self._records_by_path[path_key] = collector.records
        self._persist()

    def remove(self, path: Path) -> None:
        path_key = self._path_key(path)
        if path_key in self._records_by_path:
            del self._records_by_path[path_key]
            self._persist()

    def query(self, symbol_name: str) -> List[SymbolRecord]:
        results: List[SymbolRecord] = []
        for records in self._records_by_path.values():
            for record in records:
                if record.name == symbol_name or record.qualified_name.endswith(f".{symbol_name}"):
                    results.append(record)
        return results

    def symbols_for_path(self, path: Path | str) -> List[SymbolRecord]:
        """Return the indexed symbols for a given source path."""
        path_key = self._path_key(path if isinstance(path, Path) else Path(path))
        return list(self._records_by_path.get(path_key, []))

    def _path_key(self, path: Path) -> str:
        return path.as_posix()

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        for path_key, entries in data.items():
            self._records_by_path[path_key] = [
                SymbolRecord(**entry) for entry in entries
            ]

    def _persist(self) -> None:
        payload = {
            path: [asdict(record) for record in records]
            for path, records in sorted(self._records_by_path.items())
        }
        self._storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
