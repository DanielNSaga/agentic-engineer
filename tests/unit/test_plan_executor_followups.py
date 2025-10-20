from __future__ import annotations

from unittest.mock import MagicMock

from ae.memory.schema import Plan, Task, TaskStatus
from ae.memory.store import MemoryStore
from ae.phases.implement import ImplementResponse
from ae.phases.plan_adjust import PlanAdjustResponse
from ae.planning.executor import PlanExecutor
from ae.planning.schemas import PlannerResponse, PlannerTask
from ae.orchestrator import CodingIterationResult, FailureCategory, PatchApplicationResult


def _build_executor(store: MemoryStore) -> PlanExecutor:
    orchestrator = MagicMock()
    config_path = store.db_path.parent / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    return PlanExecutor(store=store, orchestrator=orchestrator, config_path=config_path, config={})


def test_materialize_followups_queues_new_tasks(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-queue", name="Queue Plan", goal="Ensure sequential tasks")
        store.create_plan(plan)

        origin_task = Task(
            id="task-origin",
            plan_id=plan.id,
            title="Initial task",
            priority=3,
        )
        store.save_task(origin_task)

        executor = _build_executor(store)
        adjustment = PlanAdjustResponse(
            adjustments=[],
            new_tasks=[
                "Run pytest",
                "Resolve failing tests",
                "Add regression coverage",
                "Clean up temporary scaffolding",
            ],
        )

        followups = executor._materialize_followups(plan, origin_task, adjustment)

        assert len(followups) == 3
        assert followups[0].status == TaskStatus.READY
        assert followups[0].depends_on == []
        assert followups[1].status == TaskStatus.BLOCKED
        assert followups[1].depends_on == [followups[0].id]
        assert followups[2].status == TaskStatus.BLOCKED
        assert followups[2].depends_on == [followups[0].id, followups[1].id]

        stored_second = store.get_task(followups[1].id)
        stored_third = store.get_task(followups[2].id)
        assert stored_second is not None and stored_second.status == TaskStatus.BLOCKED
        assert stored_third is not None and stored_third.depends_on == followups[2].depends_on

        decisions = store.list_decisions(plan.id)
        assert decisions
        decision = decisions[-1]
        assert "ignored_new_tasks" in decision.metadata
        assert decision.metadata["ignored_new_tasks"] == ["Run pytest"]
        assert decision.metadata.get("duplicate_new_tasks") == []
        assert decision.metadata.get("capped_new_tasks") == []


def test_materialize_followups_respects_existing_dependencies(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-existing", name="Existing blockers", goal="Respect outstanding work")
        store.create_plan(plan)

        origin_task = Task(
            id="task-origin",
            plan_id=plan.id,
            title="Initial task",
            priority=2,
        )
        store.save_task(origin_task)

        pre_existing = Task(
            id="task-prior",
            plan_id=plan.id,
            title="Prior follow-up",
            status=TaskStatus.READY,
            metadata={"origin_task_id": origin_task.id},
        )
        store.save_task(pre_existing)

        unrelated = Task(
            id="task-unrelated",
            plan_id=plan.id,
            title="Other task",
            status=TaskStatus.READY,
            metadata={"origin_task_id": "different"},
        )
        store.save_task(unrelated)

        executor = _build_executor(store)
        adjustment = PlanAdjustResponse(adjustments=[], new_tasks=["Tighten error handling"])

        followups = executor._materialize_followups(plan, origin_task, adjustment)

        assert len(followups) == 1
        followup = followups[0]
        assert followup.status == TaskStatus.BLOCKED
        assert followup.depends_on == [pre_existing.id]

        stored = store.get_task(followup.id)
        assert stored is not None
        assert stored.status == TaskStatus.BLOCKED
        assert stored.depends_on == [pre_existing.id]

def test_materialize_followups_skips_duplicates(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-dedupe", name="Deduplicate", goal="Avoid redundant tasks")
        store.create_plan(plan)

        origin_task = Task(
            id="task-origin",
            plan_id=plan.id,
            title="Initial task",
            priority=2,
        )
        store.save_task(origin_task)

        existing = Task(
            id="task-existing",
            plan_id=plan.id,
            title="Tighten error handling",
            summary="Tighten error handling.",
            status=TaskStatus.READY,
            metadata={"origin_task_id": origin_task.id},
        )
        store.save_task(existing)

        executor = _build_executor(store)
        adjustment = PlanAdjustResponse(adjustments=[], new_tasks=["Tighten error handling", "Improve logging"])

        followups = executor._materialize_followups(plan, origin_task, adjustment)

        assert len(followups) == 1
        assert followups[0].summary == "Improve logging"

        decisions = store.list_decisions(plan.id)
        assert decisions
        metadata = decisions[-1].metadata
        assert metadata.get("duplicate_new_tasks") == ["Tighten error handling"]
        assert metadata.get("capped_new_tasks") == []


def test_materialize_followups_enforces_capacity_limit(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-capacity", name="Capacity", goal="Limit pending backlog")
        store.create_plan(plan)

        origin_task = Task(
            id="task-origin",
            plan_id=plan.id,
            title="Initial task",
            priority=1,
        )
        store.save_task(origin_task)

        for index in range(3):
            pending = Task(
                id=f"task-pending-{index}",
                plan_id=plan.id,
                title=f"Follow-up {index}",
                summary=f"Resolve issue {index}",
                status=TaskStatus.BLOCKED,
                metadata={"origin_task_id": origin_task.id},
            )
            store.save_task(pending)

        executor = _build_executor(store)
        adjustment = PlanAdjustResponse(adjustments=[], new_tasks=["Add fallback logic", "Improve docs"])

        followups = executor._materialize_followups(plan, origin_task, adjustment)

        assert followups == []
        decisions = store.list_decisions(plan.id)
        assert decisions
        metadata = decisions[-1].metadata
        assert metadata.get("capped_new_tasks") == ["Add fallback logic", "Improve docs"]
        assert metadata.get("duplicate_new_tasks") == []


def test_plan_review_followups_ignore_test_runs(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-review", name="Review Plan", goal="Address review findings")
        store.create_plan(plan)

        review_task = Task(
            id="task-review",
            plan_id=plan.id,
            title="Conduct code review",
            status=TaskStatus.DONE,
        )
        store.save_task(review_task)

        executor = _build_executor(store)

        test_only_task = PlannerTask(
            id="task-run-tests",
            title="Run pytest suite",
            summary="Run pytest to ensure all tests pass.",
        )
        remediation_task = PlannerTask(
            id="task-fix",
            title="Patch configuration edge case",
            summary="Update config loader to handle missing keys gracefully.",
            deliverables=["Updated config loader"],
        )
        planner_response = PlannerResponse(tasks=[test_only_task, remediation_task])

        executor._invoke_review_followup_planner = MagicMock(return_value=planner_response)

        findings = [("Fix configuration bug", "deadbeef"), ("Add missing validation", "feedface")]
        result = CodingIterationResult()

        followups = executor._plan_review_followups(
            plan=plan,
            review_task=review_task,
            result=result,
            findings=findings,
            existing_tasks=store.list_tasks(plan_id=plan.id),
        )

        assert len(followups) == 1
        assert followups[0].title == remediation_task.title
        assert followups[0].summary == remediation_task.summary
        executor._invoke_review_followup_planner.assert_called_once()

        created_tasks = [task for task in store.list_tasks(plan_id=plan.id) if task.id not in {review_task.id}]
        assert len(created_tasks) == 1
        assert created_tasks[0].title == remediation_task.title

        decisions = store.list_decisions(plan.id)
        assert decisions
        decision_metadata = decisions[-1].metadata
        ignored_entries = decision_metadata.get("ignored_test_only_tasks")
        assert ignored_entries and ignored_entries[0]["title"] == test_only_task.title


def test_code_review_task_includes_recent_touched_files(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-review-files", name="Review Files Plan", goal="Review implementation")
        store.create_plan(plan)

        completed_planner_task = PlannerTask(
            id="task-impl",
            title="Implement feature",
            summary="Add feature code",
            deliverables=["src/service/core.py updated", "tests/test_core.py refreshed"],
            metadata={"touched_files": ["src/service/core.py", "tests/test_core.py"]},
        )
        completed_task = Task(
            id="task-impl",
            plan_id=plan.id,
            title="Implement feature",
            status=TaskStatus.DONE,
            metadata={"planner_task": completed_planner_task.model_dump(mode="python")},
            priority=4,
        )
        store.save_task(completed_task)

        executor = _build_executor(store)
        review_task = executor._create_code_review_task(plan, [completed_task])

        planner_payload = review_task.metadata["planner_task"]
        touched_files = planner_payload["metadata"].get("touched_files", [])
        assert touched_files
        assert touched_files[0] == "src/service/core.py"
        assert "tests/test_core.py" in touched_files

        review_planner = executor._load_planner_task(review_task)
        iteration_plan = executor._build_iteration_plan(plan, review_task, review_planner)
        assert iteration_plan is not None
        implement_request = iteration_plan.implement
        assert implement_request.touched_files
        assert implement_request.touched_files[0] == "src/service/core.py"
        assert "tests/test_core.py" in implement_request.touched_files


def test_final_iteration_mode_skips_auto_review_iteration(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(
            id="plan-final-review",
            name="Review Plan",
            goal="Ship baseline",
            metadata={"final_iteration_mode": {"active": True}},
        )
        store.create_plan(plan)

        completed = Task(
            id="task-core",
            plan_id=plan.id,
            title="Implement feature",
            status=TaskStatus.DONE,
        )
        store.save_task(completed)

        executor = _build_executor(store)
        scheduled = executor._maybe_schedule_code_review_iteration(plan)
        assert scheduled is False

        tasks = store.list_tasks(plan_id=plan.id)
        assert len(tasks) == 1


def test_auto_review_followups_stop_after_limit(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-review-limit", name="Review Plan", goal="Constrain followups")
        store.create_plan(plan)

        review_task = Task(
            id="task-review",
            plan_id=plan.id,
            title="Conduct review",
            status=TaskStatus.DONE,
            priority=3,
        )
        store.save_task(review_task)

        executor = _build_executor(store)
        executor._invoke_review_followup_planner = MagicMock(return_value=None)

        first_result = CodingIterationResult(
            implement=ImplementResponse(
                summary="Initial review",
                follow_up=["Fix bug :: ensure new config path is validated"],
            ),
            patch=PatchApplicationResult(),
        )

        followups_one = executor._spawn_review_followups(plan, review_task, first_result)
        assert followups_one
        assert followups_one[0].metadata.get("source") == "auto_code_review_followup"

        plan = store.get_plan(plan.id) or plan

        second_result = CodingIterationResult(
            implement=ImplementResponse(
                summary="Second review",
                follow_up=["Add docs :: update README with usage example"],
            ),
            patch=PatchApplicationResult(),
        )

        followups_two = executor._spawn_review_followups(plan, review_task, second_result)
        assert followups_two == []

        decisions = store.list_decisions(plan.id)
        suppressed = [decision for decision in decisions if decision.kind == "auto_review_followup_suppressed"]
        assert suppressed
        assert suppressed[-1].metadata.get("reason") == "followup_limit_reached"

        state = (store.get_plan(plan.id) or plan).metadata.get("auto_review_state", {})
        assert state.get("followup_plan_rounds") == 1
        assert state.get("suppressed_rounds") == 1


def test_auto_review_iteration_without_edits_marks_task_done(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-review-no-edits", name="Review Plan", goal="Assess implementation")
        store.create_plan(plan)

        completed_task = Task(
            id="task-impl",
            plan_id=plan.id,
            title="Implement feature",
            status=TaskStatus.DONE,
        )
        store.save_task(completed_task)

        executor = _build_executor(store)
        review_task = executor._create_code_review_task(plan, [completed_task])
        store.save_task(review_task)

        result = CodingIterationResult(
            implement=ImplementResponse(summary="Review completed"),
            patch=PatchApplicationResult(
                attempted=False,
                applied=False,
                error="Implement response did not include any updates.",
            ),
        )
        result.errors.append("Implement response did not include any updates.")

        adjustment, followups = executor._handle_iteration_result(plan, review_task, result)

        assert adjustment is None
        assert followups == []

        stored_review = store.get_task(review_task.id)
        assert stored_review is not None
        assert stored_review.status == TaskStatus.DONE
        assert "Implement response did not include any updates." not in result.errors


def test_iteration_no_patch_marks_task_failed_without_followups(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-no-diff", name="No Diff Plan", goal="Handle empty patches")
        store.create_plan(plan)

        origin_task = Task(
            id="task-no-diff",
            plan_id=plan.id,
            title="Implement fix",
            status=TaskStatus.READY,
        )
        store.save_task(origin_task)

        executor = _build_executor(store)

        result = CodingIterationResult(
            patch=PatchApplicationResult(
                attempted=False,
                applied=False,
                error="Implement response did not include any updates.",
            )
        )
        result.errors.append("Implement response did not include any updates.")
        result.failure_category = FailureCategory.NO_PATCH

        adjustment, followups = executor._handle_iteration_result(plan, origin_task, result)

        assert adjustment is None
        assert followups == []

        stored_task = store.get_task(origin_task.id)
        assert stored_task is not None
        assert stored_task.status == TaskStatus.FAILED
        executor._orchestrator.run_phase.assert_not_called()


def test_iteration_safety_stop_marks_task_failed(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-safety", name="Safety Plan", goal="Handle safety rails")
        store.create_plan(plan)

        origin_task = Task(
            id="task-safety",
            plan_id=plan.id,
            title="Implement feature",
            status=TaskStatus.READY,
        )
        store.save_task(origin_task)

        executor = _build_executor(store)

        result = CodingIterationResult()
        result.errors.append("Auto-loop exceeded 20 cycles without success.")
        result.failure_category = FailureCategory.SAFETY_STOP

        adjustment, followups = executor._handle_iteration_result(plan, origin_task, result)

        assert adjustment is None
        assert followups == []

        stored_task = store.get_task(origin_task.id)
        assert stored_task is not None
        assert stored_task.status == TaskStatus.FAILED
        executor._orchestrator.run_phase.assert_not_called()
