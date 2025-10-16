"""Production GPT-5 client that speaks the JSON Responses API."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional

from .llm_client import LLMClient, LLMResponseFormatError, LLMTransportError

__all__ = ["GPT5Client"]


Transport = Callable[[Dict[str, Any]], str]


class GPT5Client(LLMClient):
    """Thin adapter around the GPT-5 JSON Responses API."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1/responses",
        model: str = "gpt-5",
        transport: Optional[Transport] = None,
        timeout: float = 60.0,
        max_attempts: int = 3,
        retry_delay: float = 0.5,
    ) -> None:
        super().__init__(model=model, max_attempts=max_attempts, retry_delay=retry_delay)
        self._api_key = api_key or os.getenv("GPT5_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url
        timeout_override = os.getenv("GPT5_TIMEOUT")
        if timeout_override:
            try:
                parsed = float(timeout_override)
                if parsed > 0:
                    timeout = parsed
            except ValueError:
                pass
        self._timeout = timeout
        self._transport = transport or self._http_transport

        if transport is None and not self._api_key:
            raise ValueError("An API key is required when using the default transport.")

    def _raw_invoke(self, payload: Dict[str, Any]) -> str:
        """Send the request over the configured transport."""
        try:
            raw_response = self._transport(payload)
        except LLMTransportError:
            raise
        except Exception as error:  # pragma: no cover - defensive path
            raise LLMTransportError(f"Transport rejected the request: {error}") from error

        normalised = self._extract_model_payload(raw_response)
        if normalised is None:
            raise LLMResponseFormatError("GPT-5 response did not contain JSON output text.")
        return normalised

    def _http_transport(self, payload: Dict[str, Any]) -> str:
        """Default HTTP transport that targets the OpenAI Responses API."""
        import urllib.error
        import urllib.request

        if os.getenv("AE_DEBUG_GPT5_PAYLOAD"):
            pretty = json.dumps(payload, indent=2, sort_keys=True)
            print("[GPT5] request payload:")
            print(pretty)

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._base_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "X-OpenAI-Client": "agentic-engineer/0.1",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
                status = getattr(response, "status", 200)
        except TimeoutError as error:  # pragma: no cover - network-dependent
            raise LLMTransportError("GPT-5 response timed out.") from error
        except urllib.error.HTTPError as error:  # pragma: no cover - network-dependent
            message = error.read().decode("utf-8", errors="ignore")
            raise LLMTransportError(f"HTTP {error.code}: {message}") from error
        except urllib.error.URLError as error:  # pragma: no cover - network-dependent
            raise LLMTransportError(f"Failed to reach GPT-5 endpoint: {error.reason}") from error

        if status >= 400:
            raise LLMTransportError(f"Unexpected HTTP status {status}")

        return raw.decode("utf-8")

    def _extract_model_payload(self, raw_response: str) -> Optional[str]:
        """Extract the JSON content returned by the Responses API."""
        if not raw_response:
            return None

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            return raw_response

        if isinstance(data, dict):
            # Responses API uses `output` for ordered events.
            output_events = data.get("output") or data.get("outputs")
            text_payload = self._first_text_content(output_events)
            if text_payload:
                return text_payload

            response_container = data.get("response")
            if isinstance(response_container, dict):
                text_payload = self._first_text_content(
                    response_container.get("output") or response_container.get("outputs")
                )
                if text_payload:
                    return text_payload

            # Some variants use `response.output[0].content[0].text` or similar.
            candidate = data.get("content") or data.get("choices")
            text_payload = self._first_text_content(candidate)
            if text_payload:
                return text_payload

        # Fallback to original string when we cannot detect the JSON payload.
        return raw_response

    @staticmethod
    def _first_text_content(container: Any) -> Optional[str]:
        """Return the first text field found within the responses container."""
        if not container:
            return None

        if isinstance(container, dict):
            container = [container]

        for item in container:
            if not isinstance(item, dict):
                continue

            # Structured message entry as returned by the Responses API.
            contents = item.get("content")
            if isinstance(contents, list):
                for content_item in contents:
                    if isinstance(content_item, dict):
                        json_payload = content_item.get("json")
                        if isinstance(json_payload, (dict, list)):
                            try:
                                return json.dumps(json_payload)
                            except (TypeError, ValueError):
                                pass

                        text = content_item.get("text")
                        if isinstance(text, str) and text.strip():
                            return text

            # Reasoning entries may embed text directly.
            text_value = item.get("text")
            if isinstance(text_value, str) and text_value.strip():
                return text_value

            # Choices-like payloads.
            message = item.get("message") if isinstance(item.get("message"), dict) else None
            if message:
                text = message.get("content") or message.get("text")
                if isinstance(text, str) and text.strip():
                    return text

        return None
