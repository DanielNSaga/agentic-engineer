"""Shared helpers for invoking phases and emitting structured logs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from ..context_builder import ContextBuilder
from ..models.llm_client import LLMClient, LLMClientError, LLMRequest

T = TypeVar("T")


def invoke_phase(
    phase: str,
    request: Any,
    response_model: type[T],
    *,
    client: LLMClient,
    context_builder: ContextBuilder,
) -> T:
    """Common helper used by the phase modules to call the LLM client."""
    package = context_builder.build(phase, request)
    llm_request = LLMRequest(
        prompt=package.user_prompt,
        system_prompt=package.system_prompt,
        response_model=response_model,
        metadata=package.metadata,
    )
    attempts: list[dict[str, Any]] = []

    def _attempt_logger(
        payload: dict[str, Any],
        raw: str | None,
        parsed: Any,
        error: Exception | None,
        attempt: int,
    ) -> None:
        attempts.append(
            {
                "attempt": attempt,
                "payload": _json_safe(payload),
                "raw": raw,
                "parsed": _json_safe(parsed),
                "error": str(error) if error else None,
            }
        )

    try:
        result, raw_payload = client.invoke_structured(llm_request, logger=_attempt_logger)
    except LLMClientError as error:
        _write_phase_log(
            context_builder,
            phase,
            request,
            llm_request,
            attempts,
            error=error,
        )
        raise

    _write_phase_log(
        context_builder,
        phase,
        request,
        llm_request,
        attempts,
        result=result,
    )
    return result


def _write_phase_log(
    context_builder: ContextBuilder,
    phase: str,
    request: Any,
    llm_request: LLMRequest[Any],
    attempts: list[dict[str, Any]],
    *,
    result: Any | None = None,
    error: Exception | None = None,
) -> None:
    """Persist a structured phase execution log for later debugging."""
    logs_root = context_builder.logs_root / "phases"
    try:
        logs_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    request_payload = _json_safe(request)
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "request": request_payload,
        "context": {
            "system_prompt": llm_request.system_prompt,
            "user_prompt": llm_request.prompt,
            "metadata": _json_safe(llm_request.metadata),
        },
        "attempts": attempts,
    }
    if result is not None:
        entry["result"] = _json_safe(result)
    if error is not None:
        entry["error"] = str(error)

    plan_id = ""
    task_id = ""
    if isinstance(request_payload, dict):
        plan_id = str(request_payload.get("plan_id") or "")
        task_id = str(request_payload.get("task_id") or "")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parts = ["phase", phase]
    if plan_id:
        parts.append(_slug(plan_id))
    if task_id:
        parts.append(_slug(task_id))
    parts.append(timestamp)
    file_name = "__".join(filter(None, parts)) + ".json"
    log_path = logs_root / file_name
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            json.dump(entry, handle, indent=2, sort_keys=True, ensure_ascii=False)
    except OSError:
        return


def _json_safe(value: Any) -> Any:
    """Coerce complex objects into JSON-serialisable representations."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except TypeError:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _slug(value: str) -> str:
    """Normalise identifiers for use in log filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return cleaned or "item"
