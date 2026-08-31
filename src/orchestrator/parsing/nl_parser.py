"""Language parser (Section 3c): turns the messy NL job requests
demand_stream already passes through untouched
(`{"structured": False, "text": ...}`) into the structured
job_type/priority/requires fields the solver and WorldState expect.

Validation is strict and un-guessing on purpose: if the LLM's response
isn't parseable JSON, or any field is missing or outside the known
vocabulary, `parse_nl_job` returns None rather than filling in a
default. A job that fails to parse simply stays `structured: False` —
which is already exactly how the solver treats it (no `requires` key
-> `_is_schedulable` excludes it, left unassigned rather than guessed
at). Nothing new needs to be invented for the "flag as unparsed" case;
doing nothing *is* that flag.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.llm.client import LLMClient
from orchestrator.streams.demand_stream import JOB_TYPES

_VALID_PRIORITIES = {"normal", "urgent"}
_VALID_REQUIRES = {"cpu", "gpu", "npu", None}

_PROMPT_TEMPLATE = """You are extracting a structured job request from a short, messy operator message.

Message: "{text}"

If the message clearly describes a job matching one of the job types below, respond with ONLY a JSON object with exactly these three fields:
  - job_type: one of {job_types}
  - priority: one of "normal", "urgent"
  - requires: one of "cpu", "gpu", "npu", or null if no specific device type is mentioned

If the message is too vague, unrelated, or nonsensical to confidently classify, respond with exactly the word UNPARSEABLE and nothing else — do not guess a job_type just to produce valid-looking JSON.

Do not include any text outside the JSON object (or the literal word UNPARSEABLE)."""


def _build_prompt(text: str) -> str:
    return _PROMPT_TEMPLATE.format(text=text, job_types=list(JOB_TYPES))


def _extract_json(raw: str) -> Optional[Any]:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Real LLM output is often wrapped in a markdown fence or a
    # sentence of prose either side of the object — pull out the
    # outermost {...} span and try again rather than giving up.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _validate(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    # Check key presence explicitly, not just `.get(...)`'s value: None
    # is a legitimate *explicit* value for "requires" (no specific
    # device needed), but a *missing* key means the response came back
    # incomplete — those two cases must not be conflated, or "don't
    # guess on an incomplete response" silently stops applying to the
    # one field where the valid value and the missing-key default
    # happen to coincide.
    if not all(key in data for key in ("job_type", "priority", "requires")):
        return None
    job_type = data["job_type"]
    priority = data["priority"]
    requires = data["requires"]
    if job_type not in JOB_TYPES:
        return None
    if priority not in _VALID_PRIORITIES:
        return None
    if requires not in _VALID_REQUIRES:
        return None
    return {"job_type": job_type, "priority": priority, "requires": requires}


def parse_nl_job(text: str, llm: LLMClient) -> Optional[Dict[str, Any]]:
    raw = llm.complete(_build_prompt(text))
    return _validate(_extract_json(raw))


def parse_demand_event(event: Event, llm: LLMClient) -> Optional[Event]:
    """Given a raw demand_changed/ADDED event carrying an unparsed NL
    request, return a new event with the structured fields filled in —
    or None if it isn't that kind of event, or parsing didn't produce a
    valid result. Never mutates the original event."""
    if event.type != EventType.DEMAND_CHANGED or event.change_kind != ChangeKind.ADDED:
        return None
    if event.payload.get("structured") is not False:
        return None
    text = event.payload.get("text")
    if not text:
        return None

    parsed = parse_nl_job(text, llm)
    if parsed is None:
        return None

    payload = {
        "job_id": event.payload["job_id"],
        "structured": True,
        "text": text,  # keep the original message around for the explainer
        **parsed,
    }
    return Event(type=EventType.DEMAND_CHANGED, change_kind=ChangeKind.ADDED, source=event.source, payload=payload)
