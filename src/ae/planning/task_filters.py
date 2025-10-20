"""Shared helpers for classifying planner tasks by their intent."""

from __future__ import annotations

import re
from typing import Any

from .schemas import PlannerTask

TEST_EXECUTION_PHRASES: tuple[str, ...] = (
    "run tests",
    "run the tests",
    "run unit tests",
    "run integration tests",
    "run pytest",
    "run the test suite",
    "run test suite",
    "execute tests",
    "execute the tests",
    "execute unit tests",
    "execute integration tests",
    "re-run tests",
    "rerun tests",
    "re run tests",
    "verify tests pass",
    "ensure tests pass",
    "confirm tests pass",
    "validate tests pass",
    "kick off tests",
    "trigger tests",
    "trigger the tests",
)

TEST_COMMAND_PREFIXES: tuple[str, ...] = (
    "pytest",
    "python -m pytest",
    "npm test",
    "pnpm test",
    "yarn test",
    "go test",
    "cargo test",
    "dotnet test",
    "mvn test",
    "gradle test",
    "./mvnw test",
    "./gradlew test",
)

RUN_TESTS_PATTERN = re.compile(
    r"\b(run|re[- ]?run|execute|trigger|kick[- ]?off|launch|start)\b[^\n\r]{0,80}\btests?\b"
)


def looks_like_test_execution(value: Any) -> bool:
    """Return True when a free-form string appears to describe running tests."""
    if not isinstance(value, str):
        return False
    lowered = value.strip().lower()
    if not lowered:
        return False
    for phrase in TEST_EXECUTION_PHRASES:
        if phrase in lowered:
            return True
    if RUN_TESTS_PATTERN.search(lowered):
        return True
    if "pytest" in lowered and any(token in lowered for token in ("run", "re-run", "rerun", "execute", "trigger")):
        return True
    for prefix in TEST_COMMAND_PREFIXES:
        if lowered.startswith(prefix):
            return True
    return False


def task_is_test_execution(task: PlannerTask) -> bool:
    """Return True when a planner task is solely focused on executing tests."""
    for field in (task.title, task.summary):
        if looks_like_test_execution(field):
            return True
    return False
