from __future__ import annotations

import textwrap

import yaml
from typer.testing import CliRunner

from ae.cli import app
from ae.memory.schema import PlanStatus, TaskStatus
from ae.memory.store import MemoryStore


def test_init_plan_bootstraps_iteration_zero(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = repo_root / "data"
    data_dir.mkdir()

    policy_path = repo_root / "policy.txt"
    policy_path.write_text("Policy capsule placeholder.\n", encoding="utf-8")

    db_path = data_dir / "ae.sqlite"
    config_path = repo_root / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project:
              name: Sample Project
              repo_root: .
            iteration:
              current: 0
              goal: bootstrap
              plan_id: demo-plan
            policy:
              capsule_path: "{policy_path.as_posix()}"
            paths:
              config: "{config_path.as_posix()}"
              data: "{data_dir.as_posix()}"
              db_path: "{db_path.as_posix()}"
            models:
              default: gpt-5-offline
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["init", "--plan", "--config", str(config_path)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "Created plan" in result.output

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    with MemoryStore.from_config(config_data) as store:
        plan = store.get_plan("demo-plan")
        assert plan is not None
        assert plan.status == PlanStatus.ACTIVE
        tasks = store.list_tasks(plan_id=plan.id)
        assert tasks, "Expected tasks to be persisted."
        ready_tasks = [task for task in tasks if task.status == TaskStatus.READY]
        assert ready_tasks, "Expected at least one READY task."
        checkpoint_labels = [cp.label for cp in store.list_checkpoints(plan_id=plan.id)]
        assert "checkpoint-0" in checkpoint_labels
        decision_kinds = {decision.kind for decision in store.list_decisions(plan_id=plan.id)}
        assert {"architecture", "contract", "risk"}.issubset(decision_kinds)
