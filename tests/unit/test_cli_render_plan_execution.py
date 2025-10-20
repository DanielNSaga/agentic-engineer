from __future__ import annotations

import textwrap

from ae.cli import _render_plan_execution
from ae.memory.schema import Plan, PlanStatus, Task, TaskStatus
from ae.orchestrator import CodingIterationResult, GateRunSummary
from ae.phases.plan_adjust import PlanAdjustResponse
from ae.planning.executor import IterationOutcome, PlanExecutionSummary


def test_render_plan_execution_includes_task_status_details(capsys) -> None:
    plan = Plan(id="plan-123", name="Demo", goal="Ship feature", status=PlanStatus.ACTIVE)

    origin_task = Task(
        id="task-origin",
        plan_id=plan.id,
        title="Fix regression",
        status=TaskStatus.DONE,
    )
    follow_ready = Task(
        id="task-follow-ready",
        plan_id=plan.id,
        title="Address failing test",
        status=TaskStatus.READY,
        metadata={"origin_task_id": origin_task.id},
    )
    follow_blocked = Task(
        id="task-follow-blocked",
        plan_id=plan.id,
        title="Polish docs",
        status=TaskStatus.BLOCKED,
        depends_on=[follow_ready.id],
        metadata={"origin_task_id": origin_task.id},
    )

    result = CodingIterationResult()
    result.gates = GateRunSummary(ok=True)

    adjustment = PlanAdjustResponse(adjustments=[], new_tasks=[])
    iteration = IterationOutcome(
        task=origin_task,
        result=result,
        plan_adjustment=adjustment,
        follow_up_tasks=[follow_ready, follow_blocked],
    )
    summary = PlanExecutionSummary(
        plan=plan,
        iterations=[iteration],
        tasks=[origin_task, follow_ready, follow_blocked],
    )

    _render_plan_execution(summary)
    output = textwrap.dedent(capsys.readouterr().out)

    assert "Task backlog:" in output
    assert "[READY] task-follow-ready" in output
    assert "[BLOCKED] task-follow-blocked" in output
    assert "(waits on task-follow-ready)" in output
    assert "new tasks:" in output
    assert "[READY] task-follow-ready" in output
