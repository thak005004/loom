"""Explainer agent (Section 3c): answers "why is this job on this
device" and "why wasn't this job reassigned", grounded in the actual
solver inputs — never a free-form guess.

Grounding here means literal: every fact the explanation could cite
(the job's own priority/requires, the device's own battery/load/kind,
the bandit's currently active weight profile) is assembled into the
prompt *before* the LLM ever sees the question, and the prompt
explicitly tells it not to invent anything beyond those facts. The LLM
is only ever asked to phrase an explanation of numbers we already
looked up — never asked to infer or recall them itself.
"""

from __future__ import annotations

from typing import Any, Dict

from orchestrator.events.types import Event
from orchestrator.fleet.state import WorldState
from orchestrator.llm.client import LLMClient
from orchestrator.scheduling.whatif import WhatIfResult


def explain_assignment(job_id: str, assignments: Dict[str, str], world: WorldState, weights: Dict[str, Any], llm: LLMClient) -> str:
    job = world.jobs.get(job_id)
    if job is None:
        return f"No record of job {job_id} in the current world state."

    device_id = assignments.get(job_id)
    if device_id is None:
        return _explain_unassigned(job_id, job, llm)

    device = world.devices.get(device_id)
    if device is None:
        return f"Job {job_id} is recorded as assigned to {device_id}, but that device is no longer in the fleet."

    prompt = (
        f"Job {job_id} (priority={job.get('priority')}, requires={job.get('requires')}) "
        f"was assigned to device {device_id} (kind={device.get('kind')}, "
        f"battery={device.get('battery')}, load={device.get('load')}). "
        f"The scheduler's active weights at the time were {dict(weights)}. "
        "In 1-2 plain-English sentences, explain why this assignment makes sense. "
        "Ground your answer only in these facts — do not invent any other reason."
    )
    return llm.complete(prompt)


def _explain_unassigned(job_id: str, job: Dict[str, Any], llm: LLMClient) -> str:
    prompt = (
        f"Job {job_id} (priority={job.get('priority')}, requires={job.get('requires')}) "
        "currently has no device assigned. In 1-2 plain-English sentences, explain why not, "
        "grounded only in the fact that it currently has no assignment — do not invent a reason "
        "not supported by that fact."
    )
    return llm.complete(prompt)


def explain_no_replan(event: Event, world: WorldState, llm: LLMClient) -> str:
    """Explains a RESOURCE_CHANGED/CHANGED event that the re-planner
    deliberately treated as a no-op (see replanner.py's own docstring on
    why routine telemetry ticks don't trigger a re-plan)."""
    device_id = event.payload.get("device_id")
    device = world.devices.get(device_id, {})
    prompt = (
        f"A {event.type.value}/{event.change_kind.value} event arrived for device {device_id} "
        f"(battery={device.get('battery')}, load={device.get('load')}, connected={device.get('connected')}). "
        "This did not trigger a re-plan. In 1-2 plain-English sentences, explain why not, grounded in the fact "
        "that routine resource telemetry updates (change_kind=changed) are informational updates to the fleet's "
        "known state and don't by themselves invalidate any existing job assignment — only a device being added, "
        "removed, or gaining/losing a capability, or any demand or rule change, triggers a scoped re-plan. "
        "Do not invent any other reason."
    )
    return llm.complete(prompt)


def explain_what_if(result: WhatIfResult, llm: LLMClient) -> str:
    """Narrates a hypothetical device-failure simulation (see
    scheduling/whatif.py) — grounded the same way as every other
    explanation here: the diff is computed first, and the LLM is only
    ever asked to phrase facts it's already been handed."""
    if not result.device_existed:
        return f"Device {result.device_id} isn't in the current fleet, so there's nothing to simulate."
    if not result.orphaned_job_ids:
        return f"Device {result.device_id} has no jobs assigned to it right now, so its failure wouldn't move any work."

    move_descriptions = []
    for job_id in result.orphaned_job_ids:
        new_device = result.moves.get(job_id)
        if new_device:
            move_descriptions.append(f"{job_id} would move to {new_device}")
        else:
            move_descriptions.append(f"{job_id} would become unassigned")
    moves_text = "; ".join(move_descriptions)

    prompt = (
        f"If device {result.device_id} failed right now, the following jobs currently assigned to it "
        f"would be rescheduled by the same scheduler and the same active policy weights: {moves_text}. "
        "In 1-2 plain-English sentences, summarize what this hypothetical failure would mean for the fleet. "
        "Ground your answer only in these facts — do not invent any other reason or outcome."
    )
    return llm.complete(prompt)
