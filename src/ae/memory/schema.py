"""Typed records tracked by the Agentic Engineer memory store."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class RecordModel(BaseModel):
    """Base Pydantic model with strict field handling."""

    model_config = ConfigDict(extra="forbid", frozen=False)


class PlanStatus(str, Enum):
    """Lifecycle states for a plan."""

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"


class TaskStatus(str, Enum):
    """Lifecycle states for a task."""

    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    DONE = "DONE"
    FAILED = "FAILED"


class IncidentSeverity(str, Enum):
    """Severity classification for incidents."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TestStatus(str, Enum):
    """Outcome of an automated or manual test execution."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNKNOWN = "UNKNOWN"


class Plan(RecordModel):
    """Top-level execution plan maintained by the agent."""

    id: str
    name: str
    goal: str
    status: PlanStatus = PlanStatus.DRAFT
    summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Task(RecordModel):
    """Single actionable task within a plan."""

    id: str
    plan_id: str
    title: str
    status: TaskStatus = TaskStatus.READY
    summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)
    priority: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Decision(RecordModel):
    """Captured decision or assumption recorded during execution."""

    id: str
    plan_id: str
    task_id: Optional[str] = None
    title: str
    content: str
    kind: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class Incident(RecordModel):
    """Operational issue encountered during an iteration."""

    id: str
    plan_id: str
    task_id: Optional[str] = None
    severity: IncidentSeverity = IncidentSeverity.INFO
    summary: str
    details: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TestRun(RecordModel):
    """Recorded outcome of a verification step."""

    id: str
    plan_id: str
    task_id: Optional[str] = None
    name: str
    status: TestStatus = TestStatus.UNKNOWN
    command: str = ""
    output: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class Checkpoint(RecordModel):
    """Serialized snapshot of repository or agent state."""

    id: str
    plan_id: str
    task_id: Optional[str] = None
    label: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class Reflection(RecordModel):
    """Higher-level insight derived from prior work."""

    id: str
    scope: str
    content: str
    context: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)
