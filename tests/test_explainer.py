"""Tests for the explainer agent. Whether a live model faithfully
mentions the facts it was given isn't something an offline suite can
check — so instead we use an EchoLLMClient that returns the prompt
verbatim, which turns "does the explanation reference real state" into
a directly checkable claim: does explainer.py's prompt construction
actually embed the real battery/priority/weight values, not
placeholder or made-up ones. That's the property that's actually under
this module's control.
"""

from __future__ import annotations

from orchestrator.agents.explainer import explain_assignment, explain_no_replan
from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.state import WorldState


class EchoLLMClient:
    def __init__(self):
        self.prompts_seen = []

    def complete(self, prompt: str) -> str:
        self.prompts_seen.append(prompt)
        return prompt


def _resource_event(device_id, change_kind, **fields) -> Event:
    return Event(type=EventType.RESOURCE_CHANGED, change_kind=change_kind, source="test", payload={"device_id": device_id, **fields})


def _demand_event(job_id, change_kind, **fields) -> Event:
    return Event(type=EventType.DEMAND_CHANGED, change_kind=change_kind, source="test", payload={"job_id": job_id, **fields})


def test_explain_assignment_references_real_device_and_job_values():
    world = WorldState()
    world.apply_event(_resource_event("dev-7", ChangeKind.ADDED, kind="gpu", battery=63.5, load=0.42, connected=True))
    world.apply_event(_demand_event("job-9", ChangeKind.ADDED, structured=True, job_type="camera_check", priority="urgent", requires="gpu"))
    assignments = {"job-9": "dev-7"}
    weights = {"priority_weight": 2.0, "load_penalty_scale": 1.0, "battery_bonus_scale": 0.5, "urgent_unassigned_penalty": 1000.0}
    llm = EchoLLMClient()

    explanation = explain_assignment("job-9", assignments, world, weights, llm)

    assert "job-9" in explanation
    assert "dev-7" in explanation
    assert "63.5" in explanation  # the device's actual battery level
    assert "urgent" in explanation  # the job's actual priority
    assert "gpu" in explanation
    assert "2.0" in explanation  # the active policy's actual priority_weight
    assert len(llm.prompts_seen) == 1


def test_explain_assignment_for_unassigned_job_does_not_invent_a_device():
    world = WorldState()
    world.apply_event(_demand_event("job-5", ChangeKind.ADDED, structured=True, job_type="quality_scan", priority="normal", requires="npu"))
    llm = EchoLLMClient()

    explanation = explain_assignment("job-5", assignments={}, world=world, weights={}, llm=llm)

    assert "job-5" in explanation
    assert "no assignment" in explanation or "no device" in explanation


def test_explain_no_replan_references_actual_device_state_and_the_routine_tick_rule():
    world = WorldState()
    world.apply_event(_resource_event("dev-3", ChangeKind.ADDED, kind="cpu", battery=41.0, load=0.18, connected=True))
    event = _resource_event("dev-3", ChangeKind.CHANGED, kind="cpu", battery=41.0, load=0.18, connected=True)
    llm = EchoLLMClient()

    explanation = explain_no_replan(event, world, llm)

    assert "dev-3" in explanation
    assert "41.0" in explanation  # the device's actual battery
    assert "did not trigger a re-plan" in explanation
    assert "routine" in explanation
