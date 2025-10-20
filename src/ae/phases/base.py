"""Shared helpers for invoking phases and emitting structured logs."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
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
    request_payload = _json_safe(request)
    plan_id, task_id = _extract_request_identifiers(request_payload)

    def _attempt_logger(
        payload: dict[str, Any],
        raw: str | None,
        parsed: Any,
        error: Exception | None,
        attempt: int,
    ) -> None:
        _log_llm_input(
            context_builder,
            phase,
            payload,
            attempt=attempt,
            plan_id=plan_id,
            task_id=task_id,
        )
        _log_llm_output(
            context_builder,
            phase,
            raw,
            parsed,
            error,
            attempt=attempt,
            plan_id=plan_id,
            task_id=task_id,
        )
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

    plan_id, task_id = _extract_request_identifiers(request_payload)

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


def _log_llm_input(
    context_builder: ContextBuilder,
    phase: str,
    payload: dict[str, Any] | None,
    *,
    attempt: int,
    plan_id: str,
    task_id: str,
) -> None:
    """Persist the raw prompt text for each LLM invocation attempt."""
    if not isinstance(payload, dict):
        return

    inputs_root = _resolve_llm_inputs_root(context_builder)
    try:
        inputs_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    timestamp = datetime.now(timezone.utc)
    file_parts = [
        "input",
        _slug(phase, fallback="phase"),
    ]
    if plan_id:
        file_parts.append(_slug(plan_id))
    if task_id:
        file_parts.append(_slug(task_id))
    file_parts.append(f"attempt-{attempt}")
    file_parts.append(timestamp.strftime("%Y%m%dT%H%M%S%fZ"))
    file_parts.append(uuid.uuid4().hex[:8])
    file_name = "__".join(file_parts) + ".txt"

    log_path = inputs_root / file_name
    lines = [
        f"Timestamp: {timestamp.isoformat()}",
        f"Phase: {phase}",
        f"Attempt: {attempt}",
    ]
    if plan_id:
        lines.append(f"Plan ID: {plan_id}")
    if task_id:
        lines.append(f"Task ID: {task_id}")
    model_name = payload.get("model")
    if isinstance(model_name, str) and model_name:
        lines.append(f"Model: {model_name}")

    metadata = payload.get("metadata")
    if metadata:
        lines.append("")
        lines.append("Metadata:")
        try:
            metadata_text = json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            metadata_text = str(metadata)
        lines.append(metadata_text)

    prompt_sections = _format_prompt_sections(payload.get("input"))
    if prompt_sections:
        lines.append("")
        lines.extend(prompt_sections)

    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        return


def _log_llm_output(
    context_builder: ContextBuilder,
    phase: str,
    raw: str | None,
    parsed: Any,
    error: Exception | None,
    *,
    attempt: int,
    plan_id: str,
    task_id: str,
) -> None:
    """Persist the raw LLM response for each invocation attempt."""
    if raw is None:
        return

    outputs_root = _resolve_llm_inputs_root(context_builder)
    try:
        outputs_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    timestamp = datetime.now(timezone.utc)
    file_parts = [
        "output",
        _slug(phase, fallback="phase"),
    ]
    if plan_id:
        file_parts.append(_slug(plan_id))
    if task_id:
        file_parts.append(_slug(task_id))
    file_parts.append(f"attempt-{attempt}")
    file_parts.append(timestamp.strftime("%Y%m%dT%H%M%S%fZ"))
    file_parts.append(uuid.uuid4().hex[:8])
    file_name = "__".join(file_parts) + ".txt"

    log_path = outputs_root / file_name
    lines = [
        f"Timestamp: {timestamp.isoformat()}",
        f"Phase: {phase}",
        f"Attempt: {attempt}",
    ]
    if plan_id:
        lines.append(f"Plan ID: {plan_id}")
    if task_id:
        lines.append(f"Task ID: {task_id}")

    if error is not None:
        lines.append(f"Error: {error}")

    if parsed is not None:
        safe_parsed = _json_safe(parsed)
        try:
            parsed_text = json.dumps(safe_parsed, indent=2, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            parsed_text = str(safe_parsed)
    else:
        parsed_text = None

    lines.append("")
    lines.append("Raw Response:")
    lines.append(raw)

    if parsed_text:
        lines.append("")
        lines.append("Parsed Response:")
        lines.append(parsed_text)

    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        return


def _format_prompt_sections(messages: Any) -> list[str]:
    """Return formatted prompt sections extracted from the request payload."""
    sections: list[str] = []
    if not isinstance(messages, list):
        return sections

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role:
            heading = f"{role.title()} Prompt:"
        else:
            heading = "Prompt:"

        content_items = message.get("content")
        if not isinstance(content_items, list):
            continue

        text_fragments: list[str] = []
        for item in content_items:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_fragments.append(text.strip())

        if not text_fragments:
            continue

        body = "\n\n".join(text_fragments)
        sections.append(f"{heading}\n{body}")
    return sections


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


def _extract_request_identifiers(request_payload: Any) -> tuple[str, str]:
    """Return plan/task identifiers when available."""
    plan_id = ""
    task_id = ""
    if isinstance(request_payload, dict):
        raw_plan = request_payload.get("plan_id")
        if raw_plan:
            plan_id = str(raw_plan)
        raw_task = request_payload.get("task_id")
        if raw_task:
            task_id = str(raw_task)
    return plan_id, task_id


def _resolve_llm_inputs_root(context_builder: ContextBuilder) -> Path:
    """Return the stable llm_inputs directory anchored to the primary data root."""
    data_root = context_builder.data_root
    try:
        resolved = data_root.resolve()
    except OSError:
        resolved = data_root

    parts = resolved.parts
    if "workspaces" in parts:
        workspace_index = next((idx for idx, part in enumerate(parts) if part == "workspaces"), None)
        if workspace_index is not None and workspace_index > 0:
            base = Path(parts[0])
            for part in parts[1:workspace_index]:
                base /= part
            return base / "llm_inputs"
    return resolved / "llm_inputs"


def _slug(value: str, *, fallback: str = "item", max_length: int = 80) -> str:
    """Normalise identifiers for use in log filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    slug = cleaned or fallback
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:8]
    prefix_length = max(max_length - len(digest) - 1, 1)
    prefix = slug[:prefix_length].rstrip("-") or slug[:prefix_length]
    return f"{prefix}-{digest}"
