from __future__ import annotations

import json

from ae.models.gpt5 import GPT5Client
from ae.models.llm_client import LLMRequest
from ae.planning.schemas import PlannerResponse


def _make_response_payload(plan_summary: str) -> str:
    payload = {
        "plan_summary": plan_summary,
        "tasks": [
            {
                "id": "t1",
                "title": "Example task",
                "summary": "Do something",
                "depends_on": [],
                "constraints": [],
                "deliverables": [],
                "acceptance_criteria": [],
            }
        ],
        "decisions": [],
        "risks": [],
        "metadata": {},
    }
    response = {
        "id": "resp_mock",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(payload),
                    }
                ],
            }
        ],
    }
    return json.dumps(response)


def test_gpt5_client_extracts_json_from_responses_api() -> None:
    def transport(_: dict[str, str]) -> str:
        return _make_response_payload("Plan summary")

    client = GPT5Client(model="gpt-5-mini", transport=transport)
    request = LLMRequest(
        prompt="{}",
        response_model=PlannerResponse,
    )

    result = client.invoke(request)
    assert result.plan_summary == "Plan summary"
    assert len(result.tasks) == 1


def test_gpt5_client_extracts_from_output_json_block() -> None:
    payload = {
        "plan_summary": "Alt summary",
        "tasks": [],
        "decisions": [],
        "risks": [],
        "metadata": {},
    }

    def transport(_: dict[str, str]) -> str:
        response = {
            "output": [
                {
                    "id": "msg_json",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_json",
                            "json": payload,
                        }
                    ],
                }
            ]
        }
        return json.dumps(response)

    client = GPT5Client(model="gpt-5-mini", transport=transport)
    request = LLMRequest(
        prompt="{}",
        response_model=PlannerResponse,
    )

    result = client.invoke(request)
    assert result.plan_summary == "Alt summary"
