"""The one seam both Day 2 LLM consumers (nl_parser, explainer) go
through. `LLMClient` is a structural Protocol — a fake test double just
needs a `.complete(prompt) -> str` method, no inheritance required —
so the parser and explainer's actual logic can be tested deterministically
and offline, without a network call or an API key. `AnthropicLLMClient`
is the real implementation and is not exercised by the test suite.
`MockLLMClient` is the offline fallback the dashboard uses automatically
when no ANTHROPIC_API_KEY is set, so the app works with zero setup.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Optional, Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class AnthropicLLMClient:
    """Thin wrapper around the real Claude API. Requires ANTHROPIC_API_KEY
    in the environment; not covered by tests for that reason."""

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 300) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))


# -- MockLLMClient: a real rule-based fallback, not a test stub --
#
# This is deliberately *not* a general-purpose LLM stand-in — it only
# handles the exact prompt shapes nl_parser.py and explainer.py send
# (there are exactly four), by pattern-matching the fixed, known
# vocabulary those two callers work with. `complete(prompt) -> str` is
# the only interface LLMClient offers, so "cite the real grounding
# data" necessarily means parsing it back out of the prompt text itself
# — the prompt already contains every fact worth citing, since that's
# the whole point of grounding it before the call. Each regex has a
# graceful fallback if the prompt wording ever drifts, rather than
# raising: a less specific answer beats a crash in offline demo mode.

_JOB_TYPE_KEYWORDS: Dict[str, str] = {
    "camera": "camera_check",
    "inference": "inference_batch",
    "batch": "inference_batch",
    "firmware": "firmware_update",
    "quality": "quality_scan",
    "scan": "quality_scan",
    "sensor": "sensor_diagnostic",
    "diagnostic": "sensor_diagnostic",
}
_URGENCY_KEYWORDS = ("urgent", "asap", "immediately")
_REQUIRES_KEYWORDS: Dict[str, str] = {"gpu": "gpu", "npu": "npu", "cpu": "cpu"}

_WEIGHT_LABELS: Dict[str, str] = {
    "priority_weight": "prioritizing urgent work",
    "load_penalty_scale": "spreading load evenly across devices",
    "battery_bonus_scale": "preserving battery on low devices",
    "urgent_unassigned_penalty": "avoiding stranded urgent jobs",
}
# Rough normalization so four differently-scaled weights are comparable
# enough to pick a "dominant" one — not scientifically precise, just
# enough to make a plausible, grounded-sounding pick from real numbers.
_WEIGHT_SCALE: Dict[str, float] = {
    "priority_weight": 1.0,
    "load_penalty_scale": 2.0,
    "battery_bonus_scale": 1.0,
    "urgent_unassigned_penalty": 500.0,
}

_MESSAGE_RE = re.compile(r'Message: "(.*)"')
_ASSIGNMENT_RE = re.compile(
    r"Job (\S+) \(priority=(\w+), requires=(\w+|None)\) "
    r"was assigned to device (\S+) \(kind=(\w+), "
    r"battery=([\d.]+), load=([\d.]+)\)\. "
    r"The scheduler's active weights at the time were (\{.*?\})\."
)
_UNASSIGNED_RE = re.compile(r"Job (\S+) \(priority=(\w+), requires=(\w+|None)\) currently has no device assigned\.")
_NO_REPLAN_RE = re.compile(
    r"A (\S+)/(\S+) event arrived for device (\S+) " r"\(battery=([\d.]+|None), load=([\d.]+|None), connected=(\w+)\)\."
)
_WHATIF_RE = re.compile(
    r"If device (\S+) failed right now, the following jobs currently assigned to it "
    r"would be rescheduled by the same scheduler and the same active policy weights: (.*?)\. "
    r"In 1-2 plain-English"
)


def _classify_text(text: str) -> Optional[Dict[str, Any]]:
    lowered = text.lower()

    job_type = None
    for keyword, jt in _JOB_TYPE_KEYWORDS.items():
        if keyword in lowered:
            job_type = jt
            break
    if job_type is None:
        return None  # nothing in the known vocabulary — abstain, don't guess

    priority = "urgent" if any(k in lowered for k in _URGENCY_KEYWORDS) else "normal"

    requires = None
    for keyword, kind in _REQUIRES_KEYWORDS.items():
        if keyword in lowered:
            requires = kind
            break

    return {"job_type": job_type, "priority": priority, "requires": requires}


def _parse_weights(weights_str: str) -> Dict[str, float]:
    try:
        return ast.literal_eval(weights_str)
    except (ValueError, SyntaxError):
        return {}


def _dominant_weight_phrase(weights: Dict[str, float]) -> str:
    if not weights:
        return "a balanced set of priorities"
    normalized = {key: weights.get(key, 0.0) / scale for key, scale in _WEIGHT_SCALE.items() if key in weights}
    if not normalized:
        return "a balanced set of priorities"
    top_key = max(normalized, key=normalized.get)
    return _WEIGHT_LABELS.get(top_key, "the active policy weights")


def _requires_phrase(requires: str) -> str:
    return "no specific device type" if requires == "None" else f"a {requires.upper()}-class device"


def _round_str(value: str, digits: int) -> str:
    """Battery/load are extracted as raw f-string output, which can
    carry full float precision (e.g. 99.09857745984947). A real model
    would naturally round this when paraphrasing; the mock should too,
    or the "readable" half of "grounded and readable" doesn't hold."""
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


class MockLLMClient:
    """Offline fallback for both LLM consumers in this codebase. See the
    module-level comment above for the design constraint that shapes
    this whole class: prompt-text pattern matching is the only option
    given LLMClient's single-method interface."""

    def complete(self, prompt: str) -> str:
        if "UNPARSEABLE" in prompt:
            return self._parse_job_request(prompt)
        if "would be rescheduled by the same scheduler" in prompt:
            return self._explain_what_if(prompt)
        if "currently has no device assigned" in prompt:
            return self._explain_unassigned(prompt)
        if "did not trigger a re-plan" in prompt:
            return self._explain_no_replan(prompt)
        if "was assigned to device" in prompt:
            return self._explain_assignment(prompt)
        return "This is running in offline demo mode without a live model, so no explanation is available for this prompt shape."

    def _parse_job_request(self, prompt: str) -> str:
        match = _MESSAGE_RE.search(prompt)
        text = match.group(1) if match else ""
        parsed = _classify_text(text)
        if parsed is None:
            return "UNPARSEABLE"
        return json.dumps(parsed)

    def _explain_assignment(self, prompt: str) -> str:
        match = _ASSIGNMENT_RE.search(prompt)
        if not match:
            return "This assignment is grounded in the job's requirements and the device's current battery and load, per the active policy weights."
        job_id, priority, requires, device_id, kind, battery, load, weights_str = match.groups()
        weights = _parse_weights(weights_str)
        dominant = _dominant_weight_phrase(weights)
        return (
            f"Job {job_id} (priority: {priority}) was assigned to {device_id} because it requires "
            f"{_requires_phrase(requires)} and {device_id} is a {kind.upper()}-type device. {device_id} had "
            f"{_round_str(battery, 1)}% battery and {_round_str(load, 2)} load at the time, and the active "
            f"policy is currently {dominant}."
        )

    def _explain_unassigned(self, prompt: str) -> str:
        match = _UNASSIGNED_RE.search(prompt)
        if not match:
            return "This job currently has no device assigned."
        job_id, priority, requires = match.groups()
        return (
            f"Job {job_id} (priority: {priority}, requires: {_requires_phrase(requires)}) currently has no "
            "device assigned. No device in the fleet right now satisfies its requirements with spare capacity "
            "at the same time."
        )

    def _explain_no_replan(self, prompt: str) -> str:
        match = _NO_REPLAN_RE.search(prompt)
        if not match:
            return "This was a routine telemetry update, which does not by itself invalidate any existing assignment, so no re-plan was triggered."
        event_type, change_kind, device_id, battery, load, connected = match.groups()
        return (
            f"This was a routine {event_type}/{change_kind} update for {device_id} (battery {_round_str(battery, 1)}%, "
            f"load {_round_str(load, 2)}, connected {connected}), which just refreshes the fleet's known state. "
            "Routine telemetry changes don't invalidate existing assignments, so no re-plan was triggered."
        )

    def _explain_what_if(self, prompt: str) -> str:
        match = _WHATIF_RE.search(prompt)
        if not match:
            return (
                "If that device failed, the jobs it's currently holding would be rescheduled by the same "
                "solver under the same active policy weights, exactly like a real device failure."
            )
        device_id, moves_text = match.groups()
        return (
            f"If {device_id} failed right now: {moves_text}, the same as a real device failure would trigger, "
            "under the policy's current active weights."
        )
