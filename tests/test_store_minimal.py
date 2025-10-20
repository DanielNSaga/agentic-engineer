from __future__ import annotations

import os

from ae.memory.schema import Checkpoint, Plan, PlanStatus, Reflection, Task, TaskStatus
from ae.memory.store import MemoryStore


def test_plan_task_checkpoint_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(id="plan-1", name="Demo Plan", goal="Validate memory layer")
        store.create_plan(plan)

        loaded_plan = store.get_plan(plan.id)
        assert loaded_plan is not None
        assert loaded_plan.name == plan.name

        task = Task(id="task-1", plan_id=plan.id, title="Scaffold memory store")
        store.save_task(task)

        ready_task = store.get_ready_task()
        assert ready_task is not None
        assert ready_task.id == task.id

        store.update_task_status(task.id, TaskStatus.DONE)
        updated_task = store.get_task(task.id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.DONE

        checkpoint = Checkpoint(
            id="ckpt-1",
            plan_id=plan.id,
            task_id=task.id,
            label="post-task",
            payload={"note": "completed"},
        )
        store.save_checkpoint(checkpoint)
        checkpoints = store.list_checkpoints(plan_id=plan.id)
        assert checkpoints
        assert checkpoints[0].label == "post-task"

        reflection = Reflection(
            id="refl-1",
            scope=f"plan:{plan.id}",
            content="Tasks complete smoothly.",
            score=0.8,
        )
        store.reflections.add_reflection(reflection)
        top_scope = store.reflections.get_top_reflections(reflection.scope, limit=1)
        assert top_scope
        assert top_scope[0].id == reflection.id


def test_plan_status_auto_updates(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        plan = Plan(
            id="auto-plan",
            name="Auto completion plan",
            goal="Ensure plans transition automatically",
            status=PlanStatus.ACTIVE,
        )
        store.create_plan(plan)

        first = Task(id="task-a", plan_id=plan.id, title="First task")
        second = Task(id="task-b", plan_id=plan.id, title="Second task")
        store.save_task(first)
        store.save_task(second)

        store.update_task_status(first.id, TaskStatus.DONE)
        after_first = store.get_plan(plan.id)
        assert after_first is not None
        assert after_first.status == PlanStatus.ACTIVE

        store.update_task_status(second.id, TaskStatus.DONE)
        after_second = store.get_plan(plan.id)
        assert after_second is not None
        assert after_second.status == PlanStatus.COMPLETED

        store.update_task_status(second.id, TaskStatus.READY)
        reopened = store.get_plan(plan.id)
        assert reopened is not None
        assert reopened.status == PlanStatus.ACTIVE


def test_memory_store_falls_back_when_db_path_readonly(tmp_path) -> None:
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    db_path = readonly_dir / "ae.sqlite"
    db_path.write_text("", encoding="utf-8")
    db_path.chmod(0o444)

    with MemoryStore(db_path) as store:
        fallback_path = store.db_path
        plan = Plan(id="fallback-plan", name="Fallback", goal="Ensure fallback works")
        store.create_plan(plan)
        assert store.get_plan(plan.id) is not None

    with MemoryStore(db_path) as store_again:
        assert store_again.db_path == fallback_path
        assert store_again.get_plan("fallback-plan") is not None

    assert fallback_path != db_path.resolve()
    assert os.access(fallback_path, os.W_OK)
    if fallback_path.exists():
        fallback_path.unlink()
        try:
            fallback_path.parent.rmdir()
        except OSError:
            pass
