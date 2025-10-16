"""Typed payloads that describe structured edits emitted by phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class StructuredFileArtifact:
    """Complete file payload emitted by an agent phase."""

    path: str
    content: str
    executable: bool = False
    encoding: str | None = None


@dataclass(slots=True)
class StructuredEditOperation:
    """Line-oriented edit instruction emitted by an agent phase."""

    path: str
    action: Literal["replace", "insert", "delete"]
    start_line: int | None = None
    end_line: int | None = None
    content: str | None = None
    note: str | None = None
