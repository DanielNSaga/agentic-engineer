from __future__ import annotations

from pathlib import Path

import pytest

from ae.context_builder import ContextBuilder
from ae.phases.diagnose import DiagnoseRequest
from ae.phases.local import LocalPhaseLogic


@pytest.fixture
def local_logic(tmp_path: Path) -> tuple[LocalPhaseLogic, Path]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = repo_root / "data"
    data_root.mkdir()
    builder = ContextBuilder(
        repo_root=repo_root,
        data_root=data_root,
    )
    return LocalPhaseLogic(builder), repo_root


def test_parse_assertion_pairs_handles_equality(local_logic: tuple[LocalPhaseLogic, Path]) -> None:
    logic, _ = local_logic
    pairs = logic._parse_assertion_pairs("AssertionError: assert 'hi' == 'hello'")
    assert ("hi", "hello") in pairs


def test_parse_assertion_pairs_handles_membership(local_logic: tuple[LocalPhaseLogic, Path]) -> None:
    logic, _ = local_logic
    pairs = logic._parse_assertion_pairs("AssertionError: assert 'needle' in ''")
    assert ("", "needle") in pairs


def test_diagnose_missing_export_generates_actionable_fix(
    local_logic: tuple[LocalPhaseLogic, Path]
) -> None:
    logic, repo_root = local_logic

    package_init = repo_root / "src" / "package" / "__init__.py"
    package_init.parent.mkdir(parents=True, exist_ok=True)
    package_init.write_text("__all__: list[str] = []\n", encoding="utf-8")

    logs = (
        f"ImportError: cannot import name 'solution_to_ascii' from 'package' ({package_init})"
    )
    request = DiagnoseRequest(
        task_id="diag",
        failing_tests=["tests/test_ascii.py::test_solution"],
        logs=logs,
        recent_changes=["src/package/__init__.py"],
    )

    response = logic.diagnose(request)

    assert any("solution_to_ascii" in cause for cause in response.suspected_causes)
    assert any("solution_to_ascii" in fix for fix in response.recommended_fixes)
    assert any(req.path == "src/package/__init__.py" for req in response.code_requests)
    assert response.confidence > 0.7
    assert any("solution_to_ascii" in lesson for lesson in response.iteration_lessons)
