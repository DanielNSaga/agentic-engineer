"""Lightweight coverage mapping utilities.

The eventual agent loop will consume precise coverage data to select focused
tests after a code change. This module provides a scaffold that can store and
query a mapping between source files and the tests that exercise them. A simple
JSON serialisation format keeps the footprint small while enabling future
integration with real coverage tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set

import json


@dataclass(slots=True)
class CoverageMap:
    """Bidirectional mapping between tests and the files they cover."""

    _data: Dict[str, Set[str]] = field(default_factory=dict)

    def record(self, test_nodeid: str, files: Iterable[str | Path]) -> None:
        """Associate ``test_nodeid`` with ``files``.

        Existing entries are merged rather than replaced so the call can be
        repeated as additional coverage chunks arrive.
        """

        normalised = {self._normalise_path(path) for path in files if path}
        if not normalised:
            return

        bucket = self._data.setdefault(test_nodeid, set())
        bucket.update(normalised)

    def touched_by(self, file_path: str | Path) -> Set[str]:
        """Return the set of tests that exercise ``file_path``."""

        target = self._normalise_path(file_path)
        return {test for test, files in self._data.items() if target in files}

    def affected_tests(self, changed_files: Iterable[str | Path]) -> Set[str]:
        """Return tests that touch any of ``changed_files``."""

        targets = {self._normalise_path(path) for path in changed_files}
        if not targets:
            return set()

        return {
            test
            for test, files in self._data.items()
            if files.intersection(targets)
        }

    def to_dict(self) -> Dict[str, List[str]]:
        """Serialise to a JSON-friendly mapping."""

        return {test: sorted(files) for test, files in self._data.items()}

    def dump(self, path: Path) -> None:
        """Write the map to ``path`` as JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "CoverageMap":
        """Load a coverage map from ``path`` if it exists."""

        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle) or {}

        data: Dict[str, Set[str]] = {}
        for test, files in raw.items():
            data[str(test)] = {cls._normalise_path_static(entry) for entry in files}

        return cls(_data=data)

    @staticmethod
    def _normalise_path(path: str | Path) -> str:
        return CoverageMap._normalise_path_static(path)

    @staticmethod
    def _normalise_path_static(path: str | Path) -> str:
        if isinstance(path, Path):
            return path.as_posix()
        return str(path)


__all__ = ["CoverageMap"]
