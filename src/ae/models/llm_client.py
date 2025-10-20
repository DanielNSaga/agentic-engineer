"""Typed client base class shared by all language-model integrations."""

from __future__ import annotations

import json
import time
import copy
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union, get_args, get_origin, get_type_hints, Literal

import ast
import re

from pydantic import ValidationError
from pydantic.type_adapter import TypeAdapter

__all__ = [
    "LLMClient",
    "LLMClientError",
    "LLMRequest",
    "LLMResponseFormatError",
    "LLMRetryError",
    "LLMTransportError",
]


T = TypeVar("T")


def _close_schema(value: Any) -> Any:
    """Recursively tighten JSON Schema objects to disallow unknown keys."""
    if isinstance(value, dict):
        schema_type = value.get("type")
        if schema_type == "object":
            value["additionalProperties"] = False
            properties = value.get("properties")
            if isinstance(properties, dict):
                required = value.get("required")
                all_keys = list(properties.keys())
                if not isinstance(required, list):
                    required = all_keys
                else:
                    missing = [key for key in all_keys if key not in required]
                    if missing:
                        required.extend(missing)
                value["required"] = required
                for key, child in list(properties.items()):
                    properties[key] = _close_schema(child)
        for key, child in list(value.items()):
            if key == "properties":
                continue
            value[key] = _close_schema(child)
    elif isinstance(value, list):
        return [_close_schema(item) for item in value]
    return value


def _planner_response_schema() -> Dict[str, Any]:
    """Hand-authored JSON schema used when the planner model needs overrides."""
    string = {"type": "string"}
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    string_array = {"type": "array", "items": {"type": "string"}}

    task_schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": nullable_string,
            "title": string,
            "summary": string,
            "depends_on": string_array,
            "constraints": string_array,
            "deliverables": string_array,
            "acceptance_criteria": string_array,
        },
        "required": [
            "id",
            "title",
            "summary",
            "depends_on",
            "constraints",
            "deliverables",
            "acceptance_criteria",
        ],
    }

    decision_schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": nullable_string,
            "title": string,
            "content": string,
            "kind": string,
        },
        "required": ["id", "title", "content", "kind"],
    }

    risk_schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": nullable_string,
            "description": string,
            "mitigation": nullable_string,
            "impact": nullable_string,
            "likelihood": nullable_string,
        },
        "required": ["id", "description", "mitigation", "impact", "likelihood"],
    }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "plan_id": nullable_string,
            "plan_name": string,
            "plan_summary": string,
            "tasks": {"type": "array", "items": task_schema},
            "decisions": {"type": "array", "items": decision_schema},
            "risks": {"type": "array", "items": risk_schema},
        },
        "required": [
            "plan_id",
            "plan_name",
            "plan_summary",
            "tasks",
            "decisions",
            "risks",
        ],
    }


def _schema_override_for_model(model: Type[Any]) -> Optional[Dict[str, Any]]:
    """Return schema overrides for specific response models when required."""
    name = getattr(model, "__name__", "")
    if name == "PlannerResponse":
        return _planner_response_schema()
    return None


class LLMClientError(RuntimeError):
    """Base error raised for structured LLM client failures."""


class LLMTransportError(LLMClientError):
    """Raised when the underlying transport fails to return a response."""


class LLMResponseFormatError(LLMClientError):
    """Raised when the model returns payload that is not valid JSON."""


class LLMRetryError(LLMClientError):
    """Raised after exhausting retries due to repeated validation failures."""


@dataclass(slots=True)
class LLMRequest(Generic[T]):
    """Typed request payload sent to an LLM."""

    prompt: str
    response_model: Type[T]
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    max_attempts: Optional[int] = None

    def to_payload(self, default_model: str) -> Dict[str, Any]:
        """Render a transport-ready payload for the JSON responses API."""
        def _message(role: str, text: str) -> Dict[str, Any]:
            return {
                "role": role,
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            }

        messages: list[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append(_message("system", self.system_prompt))
        messages.append(_message("user", self.prompt))

        schema_name = getattr(self.response_model, "__name__", "ae_response")
        schema_override = _schema_override_for_model(self.response_model)
        if schema_override is not None:
            schema = schema_override
        else:
            try:
                adapter = TypeAdapter(self.response_model)
                schema = adapter.json_schema()
            except Exception:  # pragma: no cover - defensive guard
                schema = {"type": "object"}
            schema = _close_schema(schema)

        payload: Dict[str, Any] = {
            "model": self.model or default_model,
            "input": messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        if self.temperature not in (None, 0.0):
            payload["temperature"] = self.temperature
        if self.metadata:
            max_metadata_len = 512
            serialised_metadata: Dict[str, Any] = {}
            for key, value in self.metadata.items():
                formatted: str
                if isinstance(value, str):
                    formatted = value
                else:
                    formatted = json.dumps(value, separators=(",", ":"), sort_keys=True)
                if len(formatted) > max_metadata_len:
                    formatted = f"{formatted[: max_metadata_len - 3]}..."
                serialised_metadata[key] = formatted
            payload["metadata"] = serialised_metadata
        return payload


class LLMClient:
    """High-level helper that enforces JSON responses and schema validation."""

    def __init__(self, model: str, *, max_attempts: int = 5, retry_delay: float = 0.5) -> None:
        self._model = model
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay

    @property
    def model(self) -> str:
        """Return the default model name configured for this client."""
        return self._model

    def invoke(self, request: LLMRequest[T]) -> T:
        """Invoke the underlying model and return a validated response."""
        result, _ = self.invoke_structured(request)
        return result

    def invoke_structured(
        self,
        request: LLMRequest[T],
        *,
        logger: Optional[
            Callable[[Dict[str, Any], Optional[str], Optional[Any], Optional[Exception], int], None]
        ] = None,
    ) -> tuple[T, Any]:
        """Invoke the model and return both the structured response and raw payload."""
        attempts = request.max_attempts or self._max_attempts
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                payload = request.to_payload(self._model)
                raw: Optional[str] = None
                data: Optional[Any] = None
                raw = self._raw_invoke(payload)
                data = self._parse_json(raw)
                data = _coerce_to_model_schema(request.response_model, data)
                data = _hydrate_response_payload(request.response_model, data)
                adapter = TypeAdapter(request.response_model)
                validated = adapter.validate_python(data)
                if logger:
                    logger(payload, raw, data, None, attempt)
                return validated, data
            except (LLMResponseFormatError, ValidationError, LLMTransportError) as error:
                if isinstance(error, ValidationError):
                    hydrated = _hydrate_response_payload(request.response_model, data)
                    coerced = _coerce_to_model_schema(request.response_model, hydrated)
                    if coerced is not data:
                        try:
                            adapter = TypeAdapter(request.response_model)
                            validated = adapter.validate_python(coerced)
                        except ValidationError:
                            pass
                        else:
                            if logger:
                                logger(payload, raw, coerced, None, attempt)
                            return validated, coerced
                last_error = error
                if logger:
                    logger(payload, raw, data, error, attempt)
                if attempt >= attempts:
                    break
                time.sleep(self._retry_delay)

        error_message = (
            f"Failed to produce schema-valid JSON after {attempts} attempt(s) for model "
            f"{request.model or self._model}"
        )
        raise LLMRetryError(error_message) from last_error

    def _raw_invoke(self, payload: Dict[str, Any]) -> str:
        """Perform the transport call. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement _raw_invoke().")

    @staticmethod
    def _parse_json(raw_response: str) -> Any:
        """Parse JSON payloads and normalize errors."""
        text = raw_response.strip()
        if not text:
            raise LLMResponseFormatError("Model returned an empty response.")

        text = _normalise_json_string(text)
        candidates = [text]
        repaired = _repair_json_payload(text)
        if repaired and repaired not in candidates:
            candidates.append(_normalise_json_string(repaired))

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pythonic = _coerce_python_literal(candidate)
                if pythonic is not None:
                    return pythonic

        snippet = text[:200]
        raise LLMResponseFormatError(f"Model returned invalid JSON: {snippet}")


def _strip_code_fence(payload: str) -> str:
    """Remove Markdown-style code fences that wrap JSON payloads."""
    if not payload.startswith("```"):
        return payload
    fence_header_match = re.match(r"```(?:json)?", payload[:10], re.IGNORECASE)
    if not fence_header_match:
        return payload
    fence_end = payload.find("```", len(fence_header_match.group(0)))
    if fence_end == -1:
        return payload
    content_start = payload.find("\n", len(fence_header_match.group(0)))
    if content_start == -1:
        return payload
    return payload[content_start + 1 : fence_end].strip()


def _normalise_json_string(payload: str) -> str:
    """Normalise common non-JSON characters emitted by models."""
    if not payload:
        return payload
    translation = {
        0x201C: '"',
        0x201D: '"',
        0x2018: "'",
        0x2019: "'",
        0xFF07: "'",
        0x2014: "-",
        0x2013: "-",
        0x2026: "...",
        0x00A0: " ",
        0xFEFF: "",
    }
    return payload.translate(str.maketrans(translation))


def _strip_trailing_commas(payload: str) -> str:
    """Remove trailing commas before closing braces/brackets."""
    if not payload:
        return payload
    return re.sub(r",(\s*[}\]])", r"\1", payload)


def _repair_json_payload(raw: str) -> str | None:
    """Attempt to salvage JSON objects embedded in noisy output."""
    stripped = _strip_code_fence(raw.strip())
    if stripped == raw and "```" in raw:
        stripped = _strip_code_fence(raw)
    if not stripped:
        return None

    # When the payload already parses, honour it.
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        return stripped

    opening_idx = None
    expected: list[str] = []
    for index, char in enumerate(stripped):
        if char in "{[":
            if opening_idx is None:
                opening_idx = index
            expected.append("}" if char == "{" else "]")
        elif expected and char == expected[-1]:
            expected.pop()
            if not expected and opening_idx is not None:
                candidate = stripped[opening_idx : index + 1]
                return _strip_trailing_commas(candidate.strip())
    return None


def _coerce_python_literal(candidate: str) -> Any | None:
    """Fall back to Python literal parsing when JSON decoding fails."""
    try:
        literal = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        return None
    return _normalise_literal(literal)


def _normalise_literal(value: Any) -> Any:
    """Convert Python literals into JSON-compatible structures recursively."""
    if isinstance(value, dict):
        return {str(key): _normalise_literal(sub) for key, sub in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalise_literal(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _hydrate_response_payload(model: Type[Any], payload: Any) -> Any:
    """Populate missing dataclass fields with defaults during coercion."""
    if not isinstance(payload, dict):
        return payload
    if not is_dataclass(model):
        return payload
    updated = dict(payload)
    for field_info in fields(model):
        if field_info.name in updated:
            continue
        if field_info.default is not MISSING:
            updated[field_info.name] = field_info.default
            continue
        if field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
            updated[field_info.name] = field_info.default_factory()  # type: ignore[misc]
            continue
        annotation = field_info.type
        if _type_allows_none(annotation):
            updated[field_info.name] = None
    return updated


def _type_allows_none(annotation: Any) -> bool:
    """Return True when the annotation admits ``None`` as a valid value."""
    origin = get_origin(annotation)
    if origin is None:
        return annotation in (Any, type(None))
    return any(arg is type(None) for arg in get_args(annotation))


def _coerce_to_model_schema(model: Type[Any], value: Any) -> Any:
    """Coerce raw mappings into the schema expected by the dataclass model."""
    if not is_dataclass(model):
        return value
    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    hints = _cached_type_hints(model)
    cleaned: dict[str, Any] = {}
    for field_info in fields(model):
        name = field_info.name
        field_type = hints.get(name, field_info.type)
        if name in payload:
            cleaned[name] = _coerce_value(field_type, payload[name])
    return cleaned


def _coerce_value(annotation: Any, value: Any) -> Any:
    """Recursively coerce nested values to match the annotated structure."""
    origin = get_origin(annotation)
    if is_dataclass_type(annotation):
        if not isinstance(value, Mapping):
            if value is None:
                return {}
            return {}
        return _coerce_to_model_schema(annotation, value)
    if origin in {list, Sequence}:
        item_type = _first_arg(annotation)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            if value is None:
                return []
            value = [value]
        return [_coerce_value(item_type, item) for item in value]
    if origin in (tuple,):
        item_type = _first_arg(annotation)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            if value is None:
                return ()
            value = (value,)
        return tuple(_coerce_value(item_type, item) for item in value)
    if origin in (set, frozenset):
        item_type = _first_arg(annotation)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            if value is None:
                return set()
            value = [value]
        return {_coerce_value(item_type, item) for item in value}
    if origin in (dict, Mapping):
        key_type, val_type = _dict_args(annotation)
        if not isinstance(value, Mapping):
            return {}
        return {
            _coerce_scalar(key_type, key): _coerce_value(val_type, item) for key, item in value.items()
        }
    if origin is Union:
        for candidate in get_args(annotation):
            coerced = _coerce_value(candidate, copy.deepcopy(value))
            try:
                _cached_type_adapter(candidate).validate_python(coerced)
            except ValidationError:
                continue
            else:
                return coerced
        return value
    return _coerce_scalar(annotation, value)


def _coerce_scalar(annotation: Any, value: Any) -> Any:
    """Coerce scalar-like values according to the provided annotation."""
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin is Literal:
        allowed = set(get_args(annotation))
        if value in allowed:
            return value
        return next(iter(allowed), None)
    target = annotation
    if target in {Any, object}:
        return value
    if target in {str}:
        if isinstance(value, str):
            return value
        return str(value)
    if target in {int}:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    if target in {float}:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    if target in {bool}:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(value)
    return value


def is_dataclass_type(tp: Any) -> bool:
    """Return True when ``tp`` refers to a dataclass type."""
    try:
        return is_dataclass(tp)  # type: ignore[arg-type]
    except TypeError:
        return False


def _first_arg(annotation: Any) -> Any:
    """Return the first generic parameter for a typing annotation."""
    args = get_args(annotation)
    if not args:
        return Any
    return args[0]


def _dict_args(annotation: Any) -> tuple[Any, Any]:
    """Return key/value annotations for mapping types."""
    args = get_args(annotation)
    if len(args) != 2:
        return (Any, Any)
    return args[0], args[1]


@lru_cache(maxsize=None)
def _cached_type_hints(model: type[Any]) -> dict[str, Any]:
    """Cache `get_type_hints` lookups to avoid repeated reflection cost."""
    try:
        return get_type_hints(model, include_extras=True)
    except Exception:
        return {field.name: field.type for field in fields(model)}


@lru_cache(maxsize=None)
def _cached_type_adapter(annotation: Any) -> TypeAdapter:
    """Reuse `TypeAdapter` instances required during coercion."""
    return TypeAdapter(annotation)
