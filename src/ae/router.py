"""Routing logic that maps phase requests to their concrete implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

from pydantic import ValidationError
from pydantic.type_adapter import TypeAdapter

from .context_builder import ContextBuilder
from .models.llm_client import LLMClient
from .phases import PhaseName
from .phases.analyze import AnalyzeRequest, AnalyzeResponse, run as run_analyze
from .phases.design import DesignRequest, DesignResponse, run as run_design
from .phases.diagnose import DiagnoseRequest, DiagnoseResponse, run as run_diagnose
from .phases.fix_violations import (
    FixViolationsRequest,
    FixViolationsResponse,
    run as run_fix_violations,
)
from .phases.implement import ImplementRequest, ImplementResponse, run as run_implement
from .phases.plan import PlanRequest, PlanResponse, run as run_plan
from .phases.plan_adjust import PlanAdjustRequest, PlanAdjustResponse, run as run_plan_adjust


PhaseRunner = Callable[..., Any]


@dataclass(slots=True)
class PhaseEntry:
    """Metadata describing how to execute a single phase."""

    request_model: type[Any]
    response_model: type[Any]
    runner: PhaseRunner


class PhaseRouter:
    """Dispatch table mapping phase names to their concrete handlers."""

    def __init__(self, *, client: LLMClient, context_builder: ContextBuilder) -> None:
        self._client = client
        self._context_builder = context_builder
        self._registry: Dict[PhaseName, PhaseEntry] = {
            PhaseName.PLAN: PhaseEntry(PlanRequest, PlanResponse, run_plan),
            PhaseName.ANALYZE: PhaseEntry(AnalyzeRequest, AnalyzeResponse, run_analyze),
            PhaseName.DESIGN: PhaseEntry(DesignRequest, DesignResponse, run_design),
            PhaseName.IMPLEMENT: PhaseEntry(ImplementRequest, ImplementResponse, run_implement),
            PhaseName.DIAGNOSE: PhaseEntry(DiagnoseRequest, DiagnoseResponse, run_diagnose),
            PhaseName.FIX_VIOLATIONS: PhaseEntry(
                FixViolationsRequest,
                FixViolationsResponse,
                run_fix_violations,
            ),
            PhaseName.PLAN_ADJUST: PhaseEntry(PlanAdjustRequest, PlanAdjustResponse, run_plan_adjust),
        }

    def dispatch(self, phase: PhaseName | str, payload: Any) -> Any:
        """Coerce the payload into the expected request type and execute the phase."""
        phase_name = self._normalize_phase(phase)
        entry = self._registry[phase_name]
        request = self._coerce_payload(payload, entry.request_model)
        return entry.runner(
            request,
            client=self._client,
            context_builder=self._context_builder,
        )

    def available_phases(self) -> Iterable[PhaseName]:
        """Return the phases currently registered with the router."""
        return self._registry.keys()

    @staticmethod
    def _normalize_phase(phase: PhaseName | str) -> PhaseName:
        """Resolve ``phase`` into a concrete ``PhaseName`` enum member."""
        if isinstance(phase, PhaseName):
            return phase
        try:
            return PhaseName(phase)
        except ValueError as error:
            valid = ", ".join(item.value for item in PhaseName)
            raise KeyError(f"Unknown phase '{phase}'. Expected one of: {valid}") from error

    @staticmethod
    def _coerce_payload(payload: Any, request_type: type[Any]) -> Any:
        """Validate or convert ``payload`` into the ``request_type`` instance."""
        if isinstance(payload, request_type):
            return payload

        adapter = TypeAdapter(request_type)
        try:
            return adapter.validate_python(payload)
        except ValidationError as error:
            raise ValueError(
                f"Payload for {request_type.__name__} did not validate: {error}"
            ) from error
