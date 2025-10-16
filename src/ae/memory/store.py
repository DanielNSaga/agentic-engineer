"""Durable storage layer for plans, tasks, and iteration results."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Mapping, Optional, Sequence

from .reflections import ReflectionStore
from .schema import (
    Checkpoint,
    Decision,
    Incident,
    IncidentSeverity,
    Plan,
    Task,
    TaskStatus,
    TestRun,
    TestStatus,
    utc_now,
)

DEFAULT_DB_PATH = Path("data/ae.sqlite")
LOGGER = logging.getLogger(__name__)


def _as_iso(timestamp: datetime) -> str:
    """Serialise a timestamp to a timezone-aware ISO 8601 string."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat()


def _from_iso(value: str) -> datetime:
    """Parse an ISO 8601 timestamp produced by `_as_iso`."""
    return datetime.fromisoformat(value)


def _dump_json(data: Any, *, default: Any) -> str:
    """Convert arbitrary JSON-like payloads into a persisted string."""
    if data is None:
        serialisable = default
    else:
        if isinstance(data, set):
            serialisable = list(data)
        else:
            serialisable = data
    return json.dumps(serialisable)


def _load_json(value: Optional[str], *, default: Any) -> Any:
    """Decode JSON columns while falling back to the provided default."""
    if not value:
        return default
    data = json.loads(value)
    if data is None:
        return default
    return data


class MemoryStore:
    """SQLite-backed persistence for the Agentic Engineer runtime."""

    @staticmethod
    def _is_writable(path: Path) -> bool:
        parent = path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        if path.exists():
            return os.access(path, os.W_OK)
        return os.access(parent, os.W_OK)

    @staticmethod
    def _fallback_db_path(source: Path) -> Path:
        digest = hashlib.sha1(source.as_posix().encode("utf-8")).hexdigest()[:12]
        fallback_dir = Path(tempfile.gettempdir()) / "agentic-engineer" / "db" / digest
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir / source.name

    @classmethod
    def _resolve_db_path(cls, requested: Path) -> Path:
        resolved = requested.resolve()
        if cls._is_writable(resolved):
            return resolved
        fallback = cls._fallback_db_path(resolved)
        if not fallback.exists():
            if resolved.exists() and os.access(resolved, os.R_OK):
                try:
                    shutil.copy2(resolved, fallback)
                except OSError:
                    fallback.touch(exist_ok=True)
            else:
                fallback.touch(exist_ok=True)
        try:
            fallback.chmod(0o600)
        except OSError:
            pass
        if not cls._is_writable(fallback):
            raise OSError(f"Unable to locate writable database path (attempted {requested})")
        return fallback

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        requested_path = Path(db_path)
        self.db_path = self._resolve_db_path(requested_path)
        if self.db_path != requested_path.resolve():
            LOGGER.warning(
                "Database path %s is not writable; using fallback %s",
                requested_path,
                self.db_path,
            )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._open_connection()
        self._bootstrap()
        self.reflections = ReflectionStore(self._conn)

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def reopen(self) -> None:
        """Close and reopen the underlying SQLite connection."""

        self.close()
        self._conn = self._open_connection()
        self._bootstrap()
        self.reflections = ReflectionStore(self._conn)

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MemoryStore":
        paths = config.get("paths") or {}
        db_path = paths.get("db_path")
        if db_path:
            return cls(Path(db_path))

        data_path = paths.get("data") or "data"
        return cls(Path(data_path) / "ae.sqlite")

    def _bootstrap(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                goal TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                metadata TEXT NOT NULL,
                depends_on TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_plan_status
                ON tasks(plan_id, status);
            CREATE INDEX IF NOT EXISTS idx_tasks_status_priority
                ON tasks(status, priority);

            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                task_id TEXT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                kind TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE,
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                task_id TEXT,
                severity TEXT NOT NULL,
                summary TEXT NOT NULL,
                details TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE,
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS test_runs (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                task_id TEXT,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                command TEXT NOT NULL,
                output TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE,
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                task_id TEXT,
                label TEXT NOT NULL,
                payload TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE,
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_checkpoints_plan
                ON checkpoints(plan_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS reflections (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reflections_scope
                ON reflections(scope);
            """
        )
        self._conn.commit()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # Plan operations -----------------------------------------------------------------
    def create_plan(self, plan: Plan) -> None:
        record = plan.model_copy(update={"updated_at": utc_now()})
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO plans (id, name, goal, status, summary, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    goal = excluded.goal,
                    status = excluded.status,
                    summary = excluded.summary,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    record.id,
                    record.name,
                    record.goal,
                    record.status.value,
                    record.summary,
                    _dump_json(record.metadata, default={}),
                    _as_iso(record.created_at),
                    _as_iso(record.updated_at),
                ),
            )

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        cursor = self._conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return Plan(
            id=row["id"],
            name=row["name"],
            goal=row["goal"],
            status=row["status"],
            summary=row["summary"],
            metadata=_load_json(row["metadata"], default={}),
            created_at=_from_iso(row["created_at"]),
            updated_at=_from_iso(row["updated_at"]),
        )

    def list_plans(self) -> List[Plan]:
        cursor = self._conn.execute("SELECT * FROM plans ORDER BY created_at ASC")
        return [
            Plan(
                id=row["id"],
                name=row["name"],
                goal=row["goal"],
                status=row["status"],
                summary=row["summary"],
                metadata=_load_json(row["metadata"], default={}),
                created_at=_from_iso(row["created_at"]),
                updated_at=_from_iso(row["updated_at"]),
            )
            for row in cursor.fetchall()
        ]

    # Task operations -----------------------------------------------------------------
    def save_task(self, task: Task) -> None:
        record = task.model_copy(update={"updated_at": utc_now()})
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO tasks (
                    id, plan_id, title, status, summary, metadata, depends_on,
                    priority, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    plan_id = excluded.plan_id,
                    title = excluded.title,
                    status = excluded.status,
                    summary = excluded.summary,
                    metadata = excluded.metadata,
                    depends_on = excluded.depends_on,
                    priority = excluded.priority,
                    updated_at = excluded.updated_at
                """,
                (
                    record.id,
                    record.plan_id,
                    record.title,
                    record.status.value,
                    record.summary,
                    _dump_json(record.metadata, default={}),
                    _dump_json(record.depends_on, default=[]),
                    record.priority,
                    _as_iso(record.created_at),
                    _as_iso(record.updated_at),
                ),
            )

    def get_task(self, task_id: str) -> Optional[Task]:
        cursor = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_task(row)

    def list_tasks(
        self,
        *,
        plan_id: Optional[str] = None,
        statuses: Optional[Sequence[TaskStatus]] = None,
    ) -> List[Task]:
        query = "SELECT * FROM tasks"
        clauses = []
        params: List[Any] = []
        if plan_id:
            clauses.append("plan_id = ?")
            params.append(plan_id)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status.value for status in statuses)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY priority DESC, created_at ASC"

        cursor = self._conn.execute(query, params)
        return [self._row_to_task(row) for row in cursor.fetchall()]

    def list_ready_tasks(self, plan_id: Optional[str] = None) -> List[Task]:
        return self.list_tasks(
            plan_id=plan_id,
            statuses=[TaskStatus.READY],
        )

    def get_ready_task(self, plan_id: Optional[str] = None) -> Optional[Task]:
        rows = self.list_ready_tasks(plan_id=plan_id)
        return rows[0] if rows else None

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        timestamp = _as_iso(utc_now())
        with self._transaction():
            self._conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, timestamp, task_id),
            )

    # Decision operations --------------------------------------------------------------
    def record_decision(self, decision: Decision) -> None:
        record = decision.model_copy()
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO decisions (
                    id, plan_id, task_id, title, content, kind, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    plan_id = excluded.plan_id,
                    task_id = excluded.task_id,
                    title = excluded.title,
                    content = excluded.content,
                    kind = excluded.kind,
                    metadata = excluded.metadata,
                    created_at = excluded.created_at
                """,
                (
                    record.id,
                    record.plan_id,
                    record.task_id,
                    record.title,
                    record.content,
                    record.kind,
                    _dump_json(record.metadata, default={}),
                    _as_iso(record.created_at),
                ),
            )

    def list_decisions(self, plan_id: Optional[str] = None) -> List[Decision]:
        query = "SELECT * FROM decisions"
        params: List[Any] = []
        if plan_id:
            query += " WHERE plan_id = ?"
            params.append(plan_id)
        query += " ORDER BY created_at ASC"
        cursor = self._conn.execute(query, params)
        return [
            Decision(
                id=row["id"],
                plan_id=row["plan_id"],
                task_id=row["task_id"],
                title=row["title"],
                content=row["content"],
                kind=row["kind"],
                metadata=_load_json(row["metadata"], default={}),
                created_at=_from_iso(row["created_at"]),
            )
            for row in cursor.fetchall()
        ]

    # Incident operations --------------------------------------------------------------
    def log_incident(self, incident: Incident) -> None:
        record = incident.model_copy()
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO incidents (
                    id, plan_id, task_id, severity, summary, details, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    plan_id = excluded.plan_id,
                    task_id = excluded.task_id,
                    severity = excluded.severity,
                    summary = excluded.summary,
                    details = excluded.details,
                    metadata = excluded.metadata,
                    created_at = excluded.created_at
                """,
                (
                    record.id,
                    record.plan_id,
                    record.task_id,
                    record.severity.value,
                    record.summary,
                    record.details,
                    _dump_json(record.metadata, default={}),
                    _as_iso(record.created_at),
                ),
            )

    def list_incidents(self, plan_id: Optional[str] = None) -> List[Incident]:
        query = "SELECT * FROM incidents"
        params: List[Any] = []
        if plan_id:
            query += " WHERE plan_id = ?"
            params.append(plan_id)
        query += " ORDER BY created_at DESC"
        cursor = self._conn.execute(query, params)
        return [
            Incident(
                id=row["id"],
                plan_id=row["plan_id"],
                task_id=row["task_id"],
                severity=IncidentSeverity(row["severity"]),
                summary=row["summary"],
                details=row["details"],
                metadata=_load_json(row["metadata"], default={}),
                created_at=_from_iso(row["created_at"]),
            )
            for row in cursor.fetchall()
        ]

    # Test run operations --------------------------------------------------------------
    def record_test_run(self, test_run: TestRun) -> None:
        record = test_run.model_copy()
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO test_runs (
                    id, plan_id, task_id, name, status, command, output, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    plan_id = excluded.plan_id,
                    task_id = excluded.task_id,
                    name = excluded.name,
                    status = excluded.status,
                    command = excluded.command,
                    output = excluded.output,
                    metadata = excluded.metadata,
                    created_at = excluded.created_at
                """,
                (
                    record.id,
                    record.plan_id,
                    record.task_id,
                    record.name,
                    record.status.value,
                    record.command,
                    record.output,
                    _dump_json(record.metadata, default={}),
                    _as_iso(record.created_at),
                ),
            )

    def list_test_runs(self, plan_id: Optional[str] = None) -> List[TestRun]:
        query = "SELECT * FROM test_runs"
        params: List[Any] = []
        if plan_id:
            query += " WHERE plan_id = ?"
            params.append(plan_id)
        query += " ORDER BY created_at DESC"
        cursor = self._conn.execute(query, params)
        return [
            TestRun(
                id=row["id"],
                plan_id=row["plan_id"],
                task_id=row["task_id"],
                name=row["name"],
                status=TestStatus(row["status"]),
                command=row["command"],
                output=row["output"],
                metadata=_load_json(row["metadata"], default={}),
                created_at=_from_iso(row["created_at"]),
            )
            for row in cursor.fetchall()
        ]

    # Checkpoint operations ------------------------------------------------------------
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        record = checkpoint.model_copy()
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO checkpoints (
                    id, plan_id, task_id, label, payload, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    plan_id = excluded.plan_id,
                    task_id = excluded.task_id,
                    label = excluded.label,
                    payload = excluded.payload,
                    metadata = excluded.metadata,
                    created_at = excluded.created_at
                """,
                (
                    record.id,
                    record.plan_id,
                    record.task_id,
                    record.label,
                    _dump_json(record.payload, default={}),
                    _dump_json(record.metadata, default={}),
                    _as_iso(record.created_at),
                ),
            )

    def list_checkpoints(self, plan_id: Optional[str] = None, limit: Optional[int] = None) -> List[Checkpoint]:
        query = "SELECT * FROM checkpoints"
        params: List[Any] = []
        if plan_id:
            query += " WHERE plan_id = ?"
            params.append(plan_id)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        cursor = self._conn.execute(query, params)
        return [
            Checkpoint(
                id=row["id"],
                plan_id=row["plan_id"],
                task_id=row["task_id"],
                label=row["label"],
                payload=_load_json(row["payload"], default={}),
                metadata=_load_json(row["metadata"], default={}),
                created_at=_from_iso(row["created_at"]),
            )
            for row in cursor.fetchall()
        ]

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            plan_id=row["plan_id"],
            title=row["title"],
            status=TaskStatus(row["status"]),
            summary=row["summary"],
            metadata=_load_json(row["metadata"], default={}),
            depends_on=_load_json(row["depends_on"], default=[]),
            priority=row["priority"],
            created_at=_from_iso(row["created_at"]),
            updated_at=_from_iso(row["updated_at"]),
        )
