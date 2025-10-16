"""Reflection storage helpers used to surface guidance during planning."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, List, Optional

from .schema import Reflection


def _as_iso(timestamp: datetime) -> str:
    """Normalise timestamps for storage in the reflections database."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat()


def _load_json(value: Optional[str]) -> dict:
    """Decode optional JSON blobs, returning a dictionary by default."""
    if not value:
        return {}
    data = json.loads(value)
    if isinstance(data, dict):
        return data
    return {}


PLANNER_PREFLIGHT_SCOPE = "planner.preflight"

_PLANNER_PREFLIGHT_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "seed::planner-preflight::typed-wrappers",
        "content": (
            "When adding new CLI or service entrypoints, ensure typed wrappers (or stubs) mirror the "
            "runtime API so downstream consumers retain type clarity even when implementations lean on dynamic calls."
        ),
        "score": 0.85,
        "context": {
            "tags": ["typing", "wrappers", "preflight"],
            "task": {
                "id": "plan::preflight-typed-wrappers",
                "title": "Audit typed wrappers",
                "summary": "Confirm new or modified entrypoints expose typed wrappers or stubs to preserve typing parity.",
                "acceptance_criteria": [
                    "Identify public entrypoints lacking typed coverage.",
                    "Outline wrapper or stub updates needed before implementation.",
                ],
                "notes": [
                    "Check for `.pyi` stubs or wrapper modules matching CLI surfaces.",
                    "Capture missing wrappers so they can be implemented ahead of runtime changes.",
                ],
            },
            "applies_when": {"missing_typed_wrappers": True},
        },
    },
    {
        "id": "seed::planner-preflight::cli-command-routing",
        "content": (
            "Argparse subparsers terminate the process before custom handlers run. Plan CLI routing without relying on "
            "`ArgumentParser.add_subparsers` so unknown commands can surface friendly messages."
        ),
        "score": 0.88,
        "context": {
            "tags": ["cli", "argparse", "error-handling", "preflight"],
            "task": {
                "id": "plan::preflight-cli-routing",
                "title": "Design CLI command routing without argparse subparsers",
                "summary": "Document how the CLI will dispatch commands while keeping control in our own code path for custom errors.",
                "acceptance_criteria": [
                    "Describe the dispatch approach (e.g., manual command map or custom parser) that avoids `add_subparsers`.",
                    "Explain how the chosen flow emits the project-specific 'unknown command' messaging without argparse exiting early.",
                ],
                "notes": [
                    "Argparse's built-in error handling calls `sys.exit`; the plan must retain control to format graceful error messages.",
                ],
            },
            "applies_when": {"goal_mentions_cli": True},
        },
    },
    {
        "id": "seed::planner-preflight::__all__-exports",
        "content": (
            "Keep `__all__` exports in sync for packages so lint and import surfaces stay predictable "
            "after adding modules."
        ),
        "score": 0.8,
        "context": {
            "tags": ["exports", "__all__", "preflight"],
            "task": {
                "id": "plan::preflight-update-exports",
                "title": "Review package exports",
                "summary": "List packages that need `__all__` updates before adding new modules.",
                "acceptance_criteria": [
                    "Enumerate packages missing `__all__` declarations.",
                    "Plan the export list updates required by the forthcoming changes.",
                ],
                "notes": ["Prefer deterministic export ordering and include new modules explicitly."],
            },
            "applies_when": {"missing_package_exports": True},
        },
    },
    {
        "id": "seed::planner-preflight::docstrings",
        "content": "Flag modules that lack module-level docstrings so new work adds documentation up front.",
        "score": 0.75,
        "context": {
            "tags": ["docstrings", "quality", "preflight"],
            "task": {
                "id": "plan::preflight-docstrings",
                "title": "Document touched modules",
                "summary": "Identify modules that need docstrings or docstring updates before implementing changes.",
                "acceptance_criteria": [
                    "List modules missing docstrings that will be touched.",
                    "Describe the docstring updates required for compliance.",
                ],
                "notes": ["Align docstrings with repository style and include parameter annotations where needed."],
            },
            "applies_when": {"missing_module_docstrings": True},
        },
    },
    {
        "id": "seed::planner-preflight::test-scaffolding",
        "content": "Call out gaps in test scaffolding so implementations do not skip regression coverage.",
        "score": 0.74,
        "context": {
            "tags": ["tests", "coverage", "preflight"],
            "task": {
                "id": "plan::preflight-test-scaffolding",
                "title": "Scaffold regression tests",
                "summary": "Ensure the repository has test stubs or harnesses ready for new feature work.",
                "acceptance_criteria": [
                    "Identify missing or incomplete test modules relevant to the goal.",
                    "Outline the fixtures or harnesses needed before writing the implementation.",
                ],
                "notes": ["Prefer pytest patterns already used in the repo and document new fixtures."],
            },
            "applies_when": {"missing_test_scaffolding": True},
        },
    },
)

_PLANNER_PREFLIGHT_SEED: tuple[Reflection, ...] = tuple(
    Reflection(
        id=entry["id"],
        scope=entry.get("scope", PLANNER_PREFLIGHT_SCOPE),
        content=entry["content"],
        context=dict(entry.get("context") or {}),
        score=float(entry.get("score", 0.7)),
    )
    for entry in _PLANNER_PREFLIGHT_RULES
)


def get_planner_preflight_seed() -> tuple[Reflection, ...]:
    """Return the default planner preflight reflection seed."""
    return tuple(item.model_copy(deep=True) for item in _PLANNER_PREFLIGHT_SEED)


class ReflectionStore:
    """Manages reflective insights persisted in SQLite."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._seed_defaults()

    def add_reflection(self, reflection: Reflection) -> None:
        record = reflection.model_copy()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO reflections (id, scope, content, context, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    scope = excluded.scope,
                    content = excluded.content,
                    context = excluded.context,
                    score = excluded.score,
                    created_at = excluded.created_at
                """,
                (
                    record.id,
                    record.scope,
                    record.content,
                    json.dumps(record.context or {}),
                    record.score,
                    _as_iso(record.created_at),
                ),
            )

    def get_top_reflections(self, scope: str, limit: int = 5) -> List[Reflection]:
        limit = max(0, limit)
        cursor = self._conn.execute(
            """
            SELECT id, scope, content, context, score, created_at
              FROM reflections
             WHERE scope = ?
             ORDER BY score DESC, created_at DESC
             LIMIT ?
            """,
            (scope, limit),
        )
        rows = cursor.fetchall()
        return [
            Reflection(
                id=row["id"],
                scope=row["scope"],
                content=row["content"],
                context=_load_json(row["context"]),
                score=row["score"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def _seed_defaults(self) -> None:
        reflections = get_planner_preflight_seed()
        if not reflections:
            return
        with self._conn:
            for reflection in reflections:
                self._conn.execute(
                    """
                    INSERT INTO reflections (id, scope, content, context, score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        scope = excluded.scope,
                        content = excluded.content,
                        context = excluded.context,
                        score = excluded.score
                    """,
                    (
                        reflection.id,
                        reflection.scope,
                        reflection.content,
                        json.dumps(reflection.context or {}),
                        reflection.score,
                        _as_iso(reflection.created_at),
                    ),
                )
