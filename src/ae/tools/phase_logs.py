"""Utilities for loading and inspecting structured phase execution logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = ["PhaseLogEntry", "load_phase_log"]


@dataclass(slots=True)
class PhaseLogEntry:
    """In-memory representation of a stored phase execution log."""

    path: Path
    phase: str
    payload: Mapping[str, Any]

    @property
    def request(self) -> Mapping[str, Any]:
        value = self.payload.get("request")
        if isinstance(value, Mapping):
            return value
        return {}

    @property
    def context(self) -> Mapping[str, Any]:
        value = self.payload.get("context")
        if isinstance(value, Mapping):
            return value
        return {}

    @property
    def metadata(self) -> Mapping[str, Any]:
        context = self.context
        value = context.get("metadata")
        if isinstance(value, Mapping):
            return value
        return {}

    @property
    def workspace_state(self) -> Mapping[str, Any] | None:
        """Return the recorded workspace snapshot if present."""
        state = self.metadata.get("workspace_state")
        if isinstance(state, Mapping) and state:
            return state
        return None

    @property
    def task_id(self) -> str | None:
        candidate = self.request.get("task_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return None

    @property
    def plan_id(self) -> str | None:
        candidate = self.request.get("plan_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return None


def load_phase_log(path: Path | str) -> PhaseLogEntry:
    """Load a structured phase log from disk."""
    log_path = Path(path).resolve()
    with log_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    phase = str(payload.get("phase") or "").strip()
    return PhaseLogEntry(path=log_path, phase=phase, payload=payload)
