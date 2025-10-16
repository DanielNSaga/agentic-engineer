"""Shared phase enumerations and execution ordering."""

from __future__ import annotations

from enum import Enum


class PhaseName(str, Enum):
    """Enumeration of the supported orchestration phases."""

    PLAN = "plan"
    ANALYZE = "analyze"
    DESIGN = "design"
    IMPLEMENT = "implement"
    DIAGNOSE = "diagnose"
    FIX_VIOLATIONS = "fix_violations"
    PLAN_ADJUST = "plan_adjust"


PHASE_SEQUENCE = [
    PhaseName.PLAN,
    PhaseName.ANALYZE,
    PhaseName.DESIGN,
    PhaseName.IMPLEMENT,
    PhaseName.DIAGNOSE,
    PhaseName.FIX_VIOLATIONS,
    PhaseName.PLAN_ADJUST,
]


__all__ = ["PHASE_SEQUENCE", "PhaseName"]
