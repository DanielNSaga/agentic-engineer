from __future__ import annotations

import json
from unittest.mock import MagicMock

from ae.memory.schema import Plan, PlanStatus, Task, TaskStatus
from ae.memory.store import MemoryStore
from ae.orchestrator import CodingIterationResult, GateRunSummary, PatchApplicationResult
from ae.planning.executor import PlanExecutor
from ae.planning.schemas import PlannerTask
from ae.phases.implement import ImplementResponse
from ae.tools.pytest_runner import PytestResult


def _build_executor(tmp_path, *, config: dict | None = None):
    db_path = tmp_path / "ae.sqlite"
    config_path = tmp_path / "config.yaml"
    config_data = dict(config or {})
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    store = MemoryStore(db_path)
    executor = PlanExecutor(
        store=store,
        orchestrator=MagicMock(),
        config_path=config_path,
        config=config_data,
    )
    return executor, store


def test_select_test_plan_prioritises_metadata_coverage_and_analysis(tmp_path) -> None:
    executor, store = _build_executor(tmp_path)
    try:
        executor._coverage_map.record("tests/unit/test_example.py::test_case", ["src/foo.py"])
        planner_task = PlannerTask(
            title="Modify foo",
            summary="Apply fix",
            metadata={
                "touched_files": ["src/foo.py"],
                "validation_commands": ["pytest tests/unit/test_direct.py"],
            },
        )
        plan = Plan(
            id="plan-1",
            name="Plan",
            goal="Improve foo",
            metadata={
                "planner": {
                    "analysis": {
                        "components": [
                            {
                                "primary_paths": ["src/foo.py"],
                                "related_tests": ["tests/unit/test_analysis.py::test_component"],
                            }
                        ]
                    }
                }
            },
        )

        commands = executor._select_test_plan(planner_task, plan)

        assert commands == [
            "pytest tests/unit/test_direct.py",
            "pytest tests/unit/test_example.py::test_case",
            "pytest tests/unit/test_analysis.py::test_component",
        ]
    finally:
        store.close()


def test_select_test_plan_falls_back_to_default(tmp_path) -> None:
    executor, store = _build_executor(tmp_path)
    try:
        planner_task = PlannerTask(title="Document", summary="Write docs")
        plan = Plan(id="plan-2", name="Plan", goal="Document")

        commands = executor._select_test_plan(planner_task, plan)

        assert commands == ["pytest -q"]
    finally:
        store.close()


def test_update_coverage_map_records_selectors(tmp_path) -> None:
    executor, store = _build_executor(tmp_path)
    try:
        source_path = tmp_path / "src"
        source_path.mkdir(parents=True, exist_ok=True)
        touched_file = source_path / "foo.py"
        touched_file.write_text("# stub\n", encoding="utf-8")

        pytest_result = PytestResult(
            command=("pytest", "tests/unit/test_example.py::test_case"),
            cwd=tmp_path,
            exit_code=0,
            status="passed",
            collected=1,
            stdout="",
            stderr="",
        )
        result = CodingIterationResult(
            tests=pytest_result,
            patch=PatchApplicationResult(touched_paths=(touched_file,)),
        )
        planner_task = PlannerTask(
            title="Modify foo",
            summary="Apply fix",
            metadata={"touched_files": ["src/foo.py"]},
        )

        executor._update_coverage_map(planner_task, result)

        affected = executor._coverage_map.affected_tests(["src/foo.py"])
        assert affected == {"tests/unit/test_example.py::test_case"}
    finally:
        store.close()


def test_coverage_map_not_persisted_by_default(tmp_path) -> None:
    executor, store = _build_executor(tmp_path)
    try:
        assert executor._coverage_map_path is None
        pytest_result = PytestResult(
            command=("pytest", "tests/unit/test_example.py::test_case"),
            cwd=tmp_path,
            exit_code=0,
            status="passed",
            collected=1,
            stdout="",
            stderr="",
        )
        result = CodingIterationResult(
            tests=pytest_result,
            patch=PatchApplicationResult(touched_paths=(tmp_path / "src/foo.py",)),
        )
        planner_task = PlannerTask(
            title="Modify foo",
            summary="Apply fix",
            metadata={"touched_files": ["src/foo.py"]},
        )

        executor._update_coverage_map(planner_task, result)

        assert not any(
            child.name == "coverage-map.json" for child in tmp_path.glob("**/coverage-map.json")
        )
    finally:
        store.close()


def test_coverage_map_persistence_can_be_enabled(tmp_path) -> None:
    executor, store = _build_executor(
        tmp_path,
        config={"iteration": {"persist_coverage_map": True}},
    )
    try:
        assert executor._coverage_map_path is not None
        pytest_result = PytestResult(
            command=("pytest", "tests/unit/test_example.py::test_case"),
            cwd=tmp_path,
            exit_code=0,
            status="passed",
            collected=1,
            stdout="",
            stderr="",
        )
        touched = tmp_path / "src"
        touched.mkdir(parents=True, exist_ok=True)
        target = touched / "foo.py"
        target.write_text("# stub\n", encoding="utf-8")
        result = CodingIterationResult(
            tests=pytest_result,
            patch=PatchApplicationResult(touched_paths=(target,)),
        )
        planner_task = PlannerTask(
            title="Modify foo",
            summary="Apply fix",
            metadata={"touched_files": ["src/foo.py"]},
        )

        executor._update_coverage_map(planner_task, result)

        assert executor._coverage_map_path.exists()
    finally:
        store.close()


def test_build_iteration_plan_sets_structured_edits(tmp_path) -> None:
    executor, store = _build_executor(tmp_path)
    try:
        plan = Plan(id="plan-structured", name="Plan", goal="Improve")
        task = Task(id="task-1", plan_id=plan.id, title="Implement update", summary="Apply change")
        planner_task = PlannerTask(
            title="Implement update",
            summary="Apply change",
            metadata={"execution": {"structured_edits_only": "true"}},
            notes=["Existing reminder"],
        )

        iteration_plan = executor._build_iteration_plan(plan, task, planner_task)

        assert iteration_plan.implement.structured_edits_only is True
        assert iteration_plan.implement.notes == ["Existing reminder"]
    finally:
        store.close()


def test_build_iteration_plan_respects_prioritised_deliverables(tmp_path) -> None:
    executor, store = _build_executor(tmp_path)
    try:
        plan = Plan(id="plan-deliverables", name="Plan", goal="Deliver feature")
        task = Task(id="task-2", plan_id=plan.id, title="Update API", summary="Update interface")
        planner_task = PlannerTask(
            title="Update API",
            summary="Update interface",
            deliverables=["Implement endpoint", "Document behaviour"],
            optional_deliverables=["Add changelog entry"],
            metadata={
                "prioritised_deliverables": [
                    {"name": "Document behaviour"},
                    {"title": "Implement endpoint"},
                ],
                "implement_notes": ["Coordinate release notes"],
            },
            notes=["Mind backwards compatibility"],
        )

        iteration_plan = executor._build_iteration_plan(plan, task, planner_task)

        assert iteration_plan.design.proposed_interfaces == [
            "Document behaviour",
            "Implement endpoint",
            "Add changelog entry",
        ]
        assert iteration_plan.implement.notes == [
            "Mind backwards compatibility",
            "Coordinate release notes",
        ]
    finally:
        store.close()


def test_execute_refreshes_code_index_before_each_task(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-index", name="Plan", goal="Exercise index refresh")
        store.create_plan(plan)

        task_one = Task(
            id="task-a",
            plan_id=plan.id,
            title="First task",
            summary="Do first thing",
            status=TaskStatus.READY,
        )
        task_two = Task(
            id="task-b",
            plan_id=plan.id,
            title="Second task",
            summary="Do second thing",
            status=TaskStatus.READY,
        )
        store.save_task(task_one)
        store.save_task(task_two)

        code_indexer = MagicMock()
        code_indexer.reindex.return_value = []

        orchestrator = MagicMock()
        orchestrator.run_coding_iteration.return_value = CodingIterationResult(
            gates=GateRunSummary(ok=True)
        )

        executor = PlanExecutor(
            store=store,
            orchestrator=orchestrator,
            config_path=config_path,
            config={},
            code_indexer=code_indexer,
            revert_on_exit=False,
        )
        executor._maybe_schedule_code_review_iteration = MagicMock(return_value=False)
        executor._maybe_schedule_readme_iteration = MagicMock(return_value=False)

        summary = executor.execute(plan_id=plan.id)

        assert summary is not None
        assert code_indexer.reindex.call_count == 4
        assert orchestrator.run_coding_iteration.call_count == 2


def test_maybe_schedule_readme_iteration_creates_task_when_work_complete(tmp_path) -> None:
    executor, store = _build_executor(tmp_path)
    try:
        plan = Plan(id="plan-readme", name="Plan", goal="Ship feature")
        store.create_plan(plan)
        core_task = Task(
            id="task-core",
            plan_id=plan.id,
            title="Implement feature",
            summary="Add behaviour",
            status=TaskStatus.DONE,
        )
        review_task = Task(
            id="task-review",
            plan_id=plan.id,
            title="Auto review",
            summary="Review work",
            status=TaskStatus.DONE,
            metadata={"auto_generated": "code_review"},
        )
        store.save_task(core_task)
        store.save_task(review_task)

        scheduled = executor._maybe_schedule_readme_iteration(plan)

        assert scheduled is True
        tasks = store.list_tasks(plan_id=plan.id)
        readme_tasks = [
            task for task in tasks if task.metadata.get("auto_generated") == "readme_polish"
        ]
        assert len(readme_tasks) == 1
        readme_task = readme_tasks[0]
        assert readme_task.status == TaskStatus.READY
        assert set(readme_task.depends_on) == {core_task.id, review_task.id}
        assert executor._maybe_schedule_readme_iteration(plan) is False
    finally:
        store.close()


def test_maybe_schedule_readme_iteration_skips_when_pending_work(tmp_path) -> None:
    executor, store = _build_executor(tmp_path)
    try:
        plan = Plan(id="plan-readme-pending", name="Plan", goal="Ship feature")
        store.create_plan(plan)
        core_task = Task(
            id="task-core",
            plan_id=plan.id,
            title="Implement feature",
            summary="Add behaviour",
            status=TaskStatus.DONE,
        )
        pending_followup = Task(
            id="task-followup",
            plan_id=plan.id,
            title="Address review",
            summary="Fix issue",
            status=TaskStatus.READY,
        )
        review_task = Task(
            id="task-review",
            plan_id=plan.id,
            title="Auto review",
            summary="Review work",
            status=TaskStatus.DONE,
            metadata={"auto_generated": "code_review"},
        )
        store.save_task(core_task)
        store.save_task(pending_followup)
        store.save_task(review_task)

        scheduled = executor._maybe_schedule_readme_iteration(plan)

        assert scheduled is False
        tasks = store.list_tasks(plan_id=plan.id)
        assert all(task.metadata.get("auto_generated") != "readme_polish" for task in tasks)
    finally:
        store.close()


def test_handle_iteration_readme_auto_schedules_after_clean_review(tmp_path) -> None:
    executor, store = _build_executor(tmp_path)
    try:
        plan = Plan(id="plan-readme-auto", name="Plan", goal="Ship feature")
        store.create_plan(plan)
        core_task = Task(
            id="plan-readme-auto::core",
            plan_id=plan.id,
            title="Core task",
            status=TaskStatus.DONE,
        )
        review_task = Task(
            id="plan-readme-auto::code-review",
            plan_id=plan.id,
            title="Perform targeted code review",
            status=TaskStatus.READY,
            metadata={"auto_generated": "code_review"},
            depends_on=[core_task.id],
        )
        store.save_task(core_task)
        store.save_task(review_task)

        result = CodingIterationResult()
        result.implement = ImplementResponse(summary="All good", no_op_reason="Review-only pass")
        result.gates = GateRunSummary(ok=True)
        result.patch = PatchApplicationResult(attempted=False, applied=True, no_op_reason="Review-only pass")

        adjustment, followups = executor._handle_iteration_result(plan, review_task, result)
        assert adjustment is None
        assert followups == []

        tasks = store.list_tasks(plan_id=plan.id)
        readme_tasks = [
            task for task in tasks if task.metadata.get("auto_generated") == "readme_polish"
        ]
        assert len(readme_tasks) == 1
        readme_task = readme_tasks[0]
        assert readme_task.status == TaskStatus.READY
        assert set(readme_task.depends_on) == {core_task.id, review_task.id}
    finally:
        store.close()


def test_saving_ready_task_reactivates_plan(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(
            id="plan-reactivate",
            name="Plan",
            goal="Ship",
            status=PlanStatus.COMPLETED,
        )
        store.create_plan(plan)

        task = Task(
            id="task-new",
            plan_id=plan.id,
            title="Follow-up",
            summary="Handle remaining work",
            status=TaskStatus.READY,
        )

        store.save_task(task)

        refreshed = store.get_plan(plan.id)
        assert refreshed is not None
        assert refreshed.status == PlanStatus.ACTIVE


def test_final_iteration_mode_adds_guidance(tmp_path) -> None:
    executor, store = _build_executor(tmp_path)
    try:
        plan = Plan(id="plan-final-mode", name="Plan", goal="Deliver baseline")
        store.create_plan(plan)
        task = Task(id="task-final", plan_id=plan.id, title="Implement core", summary="Build feature")
        store.save_task(task)

        result = CodingIterationResult()
        result.gates.ok = True

        for _ in range(PlanExecutor._FINAL_ITERATION_THRESHOLD):
            plan = executor._record_iteration_metrics(plan, task, result)

        assert executor._final_iteration_mode_active(plan) is True

        planner_task = PlannerTask(title="Wrap up", summary="Prepare final delivery")
        implement_request = executor._build_implement_request(
            plan=plan,
            task=task,
            planner_task=planner_task,
            structured_only=False,
            extra_notes=[],
            product_spec="",
        )

        assert any("Final delivery mode" in constraint for constraint in implement_request.constraints)
        assert any("Final push mode is active" in note for note in implement_request.notes)
    finally:
        store.close()
