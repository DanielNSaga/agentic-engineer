"""Ensure plan-adjust union coercion preserves structured adjustments."""

from __future__ import annotations

from pydantic import TypeAdapter

from ae.models.llm_client import _coerce_value
from ae.phases.plan_adjust import PlanAdjustment, PlanAdjustmentItem


def test_plan_adjustment_coercion_prefers_structured_items() -> None:
    payload = {
        "action": "Mirror workspace edits into src package",
        "summary": "Reconcile top-level edits",
        "details": "Copy edits into the installed package.",
        "rationale": ["Ensure tests exercise updated code"],
        "priority": "high",
        "id": "A1",
        "notes": ["Keep changes small."],
        "files": [],
        "edits": [],
    }

    coerced = _coerce_value(PlanAdjustment, payload)
    result = TypeAdapter(PlanAdjustment).validate_python(coerced)

    assert isinstance(result, PlanAdjustmentItem)
    assert result.action == payload["action"]
