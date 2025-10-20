from __future__ import annotations

import json
from pathlib import Path

from ae.orchestrator import CodingIterationResult, Orchestrator
from ae.models.llm_client import LLMClient
from ae.phases.base import _log_llm_output, _slug


class _DummyClient(LLMClient):
    def __init__(self) -> None:
        super().__init__("dummy")

    def _raw_invoke(self, payload):
        raise NotImplementedError("Dummy client does not support invocation.")


def test_iteration_artifact_handles_long_identifiers(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    config = {"paths": {"data": data_dir.as_posix(), "logs": logs_dir.as_posix()}}
    orchestrator = Orchestrator(
        client=_DummyClient(),
        config=config,
        repo_root=tmp_path,
    )

    long_plan_id = (
        "n-queens-problem-visualizer-program-plan-20251018001330::"
        "add-ascii-render-and-solutions-to-ascii-implementations-to-src-"
        "n-queens-problem-visualizer-program-visualizer-py"
    )
    long_task_id = (
        "ae:n-queens-problem-visualizer-program-plan-20251018001330::"
        "add-ascii-render-and-solutions-to-ascii-implementations-to-src-"
        "n-queens-problem-visualizer-program-visualizer-py"
    )

    result = CodingIterationResult()
    artifact_path = orchestrator._write_iteration_artifact(
        plan_id=long_plan_id,
        task_id=long_task_id,
        goal="Test goal",
        iteration_result=result,
        revert_on_exit=False,
    )

    assert artifact_path.exists()
    assert artifact_path.parent.name == "iterations"
    assert len(artifact_path.name) <= 200

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["plan_id"] == long_plan_id
    assert payload["task_id"] == long_task_id


def test_phase_log_slug_truncates_long_identifiers() -> None:
    long_value = "x" * 200
    slug = _slug(long_value)
    assert len(slug) <= 80
    assert slug.startswith("x")


def test_llm_output_logs_to_primary_data_root(tmp_path: Path) -> None:
    primary_data_root = tmp_path / "data"
    workspace_data_root = primary_data_root / "workspaces" / "ae-workspace-1" / "data"
    workspace_data_root.mkdir(parents=True)

    stub_builder = type("StubBuilder", (), {})()
    stub_builder.data_root = workspace_data_root

    _log_llm_output(
        stub_builder,  # type: ignore[arg-type]
        "implement",
        raw="raw response payload",
        parsed={"foo": "bar"},
        error=None,
        attempt=1,
        plan_id="plan-123",
        task_id="task-456",
    )

    output_dir = primary_data_root / "llm_inputs"
    files = list(output_dir.glob("output__*.txt"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "Raw Response:" in content
    assert "raw response payload" in content
