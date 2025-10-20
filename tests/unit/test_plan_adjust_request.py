from __future__ import annotations

from pathlib import Path

from ae.orchestrator import (
    CodingIterationResult,
    CycleAdjustment,
    GateRunSummary,
    IterationCycle,
    PatchApplicationResult,
)
from ae.planning.executor import build_plan_adjust_request


def test_plan_adjust_request_includes_touched_paths_and_adjustments() -> None:
    patch_result = PatchApplicationResult(
        touched_paths=(Path("src/core.py"), Path("tests/test_core.py")),
    )
    gate_summary = GateRunSummary(
        suspect_files=["tests/test_core.py"],
        violations=["policy::failure"],
    )
    cycle = IterationCycle(
        cycle_index=1,
        reason="tests failed",
        gates_ok=False,
        tests_ok=False,
        adjustments=[
            CycleAdjustment(
                source="auto_fix",
                description="normalize inputs",
                touched_paths=(Path("src/utils.py"),),
            )
        ],
        errors=["pytest failed"],
    )
    result = CodingIterationResult(
        patch=patch_result,
        gates=gate_summary,
        cycles=[cycle],
        errors=["Tests failed"],
    )

    request = build_plan_adjust_request("plan-123", "task-456", result)

    assert request.suspect_files[0] == "tests/test_core.py"
    assert "src/core.py" in request.suspect_files
    assert "src/utils.py" in request.suspect_files
    assert request.suspect_files.count("tests/test_core.py") == 1
