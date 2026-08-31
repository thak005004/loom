"""Tests for the NL parser. A FakeLLMClient returns scripted raw
responses keyed by substring match against the prompt (which always
embeds the original message verbatim) — this tests the real logic that
matters offline: JSON extraction from realistically messy LLM output,
and strict validation against the known vocabulary. It does not test
whether a live model would extract the *right* fields from novel text —
that's not something an offline suite can verify.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.parsing.nl_parser import parse_demand_event, parse_nl_job


class FakeLLMClient:
    def __init__(self, responses: Dict[str, str]):
        self.responses = responses
        self.prompts_seen: List[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts_seen.append(prompt)
        for key, response in self.responses.items():
            if key in prompt:
                return response
        raise AssertionError(f"no canned response configured for prompt:\n{prompt}")


TEXT_A = "urgent camera check on line 3, needs GPU"
TEXT_B = "can someone run inference on the batch from this morning, not time sensitive"
TEXT_C = "need a firmware update pushed to all NPU units before end of day"


def test_parses_clean_fenced_json():
    llm = FakeLLMClient({TEXT_A: '```json\n{"job_type": "camera_check", "priority": "urgent", "requires": "gpu"}\n```'})
    result = parse_nl_job(TEXT_A, llm)
    assert result == {"job_type": "camera_check", "priority": "urgent", "requires": "gpu"}


def test_parses_json_embedded_in_surrounding_prose():
    llm = FakeLLMClient(
        {
            TEXT_B: (
                "Sure, here's the classification:\n"
                '{"job_type": "inference_batch", "priority": "normal", "requires": null}\n'
                "Let me know if you need anything else."
            )
        }
    )
    result = parse_nl_job(TEXT_B, llm)
    assert result == {"job_type": "inference_batch", "priority": "normal", "requires": None}


def test_parses_plain_json_with_no_wrapping():
    llm = FakeLLMClient({TEXT_C: '{"job_type": "firmware_update", "priority": "urgent", "requires": "npu"}'})
    result = parse_nl_job(TEXT_C, llm)
    assert result == {"job_type": "firmware_update", "priority": "urgent", "requires": "npu"}


def test_unparseable_response_does_not_produce_a_bad_structured_event():
    llm = FakeLLMClient({"check the thing": "I'm not totally sure what device this needs."})
    assert parse_nl_job("check the thing", llm) is None


def test_response_with_out_of_vocabulary_job_type_is_rejected():
    llm = FakeLLMClient({"make coffee": '{"job_type": "make_coffee", "priority": "normal", "requires": null}'})
    assert parse_nl_job("make coffee", llm) is None


def test_response_missing_a_required_field_is_rejected():
    llm = FakeLLMClient({"urgent thing": '{"job_type": "camera_check", "priority": "urgent"}'})  # no "requires"
    assert parse_nl_job("urgent thing", llm) is None


def test_response_with_invalid_priority_is_rejected():
    llm = FakeLLMClient({"whenever": '{"job_type": "camera_check", "priority": "whenever", "requires": "gpu"}'})
    assert parse_nl_job("whenever", llm) is None


def test_parse_demand_event_produces_a_valid_structured_added_event():
    llm = FakeLLMClient({TEXT_A: '{"job_type": "camera_check", "priority": "urgent", "requires": "gpu"}'})
    raw_event = Event(
        type=EventType.DEMAND_CHANGED,
        change_kind=ChangeKind.ADDED,
        source="demand",
        payload={"job_id": "job-0007", "structured": False, "text": TEXT_A},
    )

    parsed_event = parse_demand_event(raw_event, llm)

    assert parsed_event is not None
    assert parsed_event.type == EventType.DEMAND_CHANGED
    assert parsed_event.change_kind == ChangeKind.ADDED
    assert parsed_event.payload["job_id"] == "job-0007"
    assert parsed_event.payload["structured"] is True
    assert parsed_event.payload["job_type"] == "camera_check"
    assert parsed_event.payload["priority"] == "urgent"
    assert parsed_event.payload["requires"] == "gpu"
    assert parsed_event.payload["text"] == TEXT_A  # original message preserved


def test_parse_demand_event_returns_none_when_parsing_fails():
    llm = FakeLLMClient({"check the thing": "not json at all"})
    raw_event = Event(
        type=EventType.DEMAND_CHANGED,
        change_kind=ChangeKind.ADDED,
        source="demand",
        payload={"job_id": "job-0008", "structured": False, "text": "check the thing"},
    )
    assert parse_demand_event(raw_event, llm) is None


def test_parse_demand_event_ignores_already_structured_events():
    llm = FakeLLMClient({})  # should never be called
    raw_event = Event(
        type=EventType.DEMAND_CHANGED,
        change_kind=ChangeKind.ADDED,
        source="demand",
        payload={"job_id": "job-0009", "structured": True, "job_type": "camera_check", "priority": "normal", "requires": None},
    )
    assert parse_demand_event(raw_event, llm) is None
    assert llm.prompts_seen == []
