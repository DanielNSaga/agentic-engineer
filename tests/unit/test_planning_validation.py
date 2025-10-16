from __future__ import annotations

from pathlib import Path

from ae.planning.bootstrap import (
    _apply_plan_validation_annotations,
    _collect_repo_summary,
    _prune_completed_goal_inputs,
    _split_optional_deliverables,
    _validate_plan_against_contracts,
)
from ae.planning.schemas import PlannerResponse, PlannerTask
from ae.planning.executor import PlanExecutor
from ae.tools.vcs import GitRepository


def test_prune_completed_inputs_removes_tracked_modules(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "src" / "pkg"
    source_path.mkdir(parents=True, exist_ok=True)
    (repo_root / "tests").mkdir(parents=True, exist_ok=True)
    module_path = source_path / "__init__.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    GitRepository.initialise(repo_root)

    summary = _collect_repo_summary(repo_root, {"project": {}, "iteration": {}})

    result = _prune_completed_goal_inputs(
        repo_root=repo_root,
        repo_summary=summary,
        constraints=["Create pkg module at src/pkg/__init__.py"],
        deliverables=["src/pkg/__init__.py"],
    )

    assert result.constraints == []
    assert result.deliverables == []
    assert result.pruned_constraints
    assert result.pruned_constraints[0]["targets"] == ["src/pkg/__init__.py"]


def test_split_optional_deliverables_detects_optional_markers() -> None:
    planner_task = PlannerTask(
        title="Document CLI",
        summary="Write docs.",
        deliverables=["Ship CLI docs (optional)", "Publish help output"],
        optional_deliverables=["Update README"],
    )

    required, optional = _split_optional_deliverables(planner_task)
    assert required == ["Publish help output"]
    assert "Ship CLI docs" in optional
    assert "Update README" in optional


def test_storage_conflict_detection_creates_conflict_decision() -> None:
    plan = PlannerResponse(
        tasks=[
            PlannerTask(
                title="Write vault",
                summary="Store secrets in data/vault.sqlite for persistence.",
            )
        ],
    )
    repo_summary = {
        "workspace_contracts": {
            "storage_entry_points": {
                "memory_store": {"default_db_path": "data/ae.sqlite"},
            }
        }
    }

    report = _validate_plan_against_contracts(plan, repo_summary)
    assert report.conflicts

    initial_decision_count = len(plan.decisions)
    _apply_plan_validation_annotations(plan, report)
    assert len(plan.decisions) == initial_decision_count + 1
    assert any(decision.kind == "conflict" for decision in plan.decisions)


def test_duplicate_tasks_are_flagged_by_validator() -> None:
    plan = PlannerResponse(
        tasks=[
            PlannerTask(id="alpha", title="Refactor CLI", summary="Tidy CLI layout."),
            PlannerTask(id="beta", title="Refactor CLI", summary="Tidy CLI layout."),
        ]
    )

    report = _validate_plan_against_contracts(plan, {})
    assert any(entry["reason"] == "duplicate_title" for entry in report.duplicate_tasks)


def test_extract_touched_files_ignores_bare_init_metadata() -> None:
    planner_task = PlannerTask(
        title="Update package exports",
        summary="Ensure init exposes symbols.",
        metadata={"touched_files": ["__init__.py", "src/pkg/__init__.py"]},
    )

    paths = PlanExecutor._extract_touched_files(planner_task)

    assert "src/pkg/__init__.py" in paths
    assert "__init__.py" not in paths


def test_extract_touched_files_ignores_bare_init_deliverables() -> None:
    planner_task = PlannerTask(
        title="Document package",
        summary="Adjust package markers.",
        deliverables=["Review __init__.py", "Update src/pkg/__init__.py"],
    )

    paths = PlanExecutor._extract_touched_files(planner_task)

    assert "src/pkg/__init__.py" in paths
    assert "__init__.py" not in paths
