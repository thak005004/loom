"""Tests for MockLLMClient — the offline fallback used automatically
when no ANTHROPIC_API_KEY is set. Unlike EchoLLMClient (which just
echoes the prompt back, useful for checking prompt construction),
MockLLMClient does real work: rule-based classification for the
parser, and templated-but-grounded explanations for the explainer. It
was verified against the real Anthropic API earlier in this project
that a live model, given the same prompts, extracts these exact five
templates correctly and abstains on the same kind of nonsense — this
suite confirms the offline fallback matches that behavior using its
own (much simpler) method.
"""

from __future__ import annotations

from orchestrator.agents.explainer import explain_assignment, explain_no_replan
from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.state import WorldState
from orchestrator.llm.client import MockLLMClient
from orchestrator.parsing.nl_parser import parse_nl_job
from orchestrator.streams.demand_stream import NL_TEMPLATES


def _resource_event(device_id, change_kind, **fields) -> Event:
    return Event(type=EventType.RESOURCE_CHANGED, change_kind=change_kind, source="test", payload={"device_id": device_id, **fields})


def _demand_event(job_id, change_kind, **fields) -> Event:
    return Event(type=EventType.DEMAND_CHANGED, change_kind=change_kind, source="test", payload={"job_id": job_id, **fields})


# -- parser: the real demand_stream templates --


def test_mock_client_parses_urgent_gpu_camera_check():
    mock = MockLLMClient()
    result = parse_nl_job("urgent camera check on line 3, needs GPU", mock)
    assert result == {"job_type": "camera_check", "priority": "urgent", "requires": "gpu"}


def test_mock_client_parses_inference_batch_request():
    mock = MockLLMClient()
    result = parse_nl_job("can someone run inference on the batch from this morning, not time sensitive", mock)
    assert result == {"job_type": "inference_batch", "priority": "normal", "requires": None}


def test_mock_client_parses_sensor_diagnostic_request():
    mock = MockLLMClient()
    result = parse_nl_job("line 7 sensor's been acting weird, check it out when free", mock)
    assert result == {"job_type": "sensor_diagnostic", "priority": "normal", "requires": None}


def test_mock_client_parses_npu_firmware_update():
    mock = MockLLMClient()
    result = parse_nl_job("need a firmware update pushed to all NPU units before end of day", mock)
    assert result == {"job_type": "firmware_update", "priority": "normal", "requires": "npu"}


def test_mock_client_parses_quality_scan_request():
    mock = MockLLMClient()
    result = parse_nl_job("quality scan backlog is piling up on the east line, can we get to it soon", mock)
    assert result == {"job_type": "quality_scan", "priority": "normal", "requires": None}


def test_mock_client_handles_every_real_nl_template_without_abstaining():
    """All 5 templates demand_stream actually generates, run through the
    mock in one pass — not just the hand-picked ones above."""
    mock = MockLLMClient()
    for text in NL_TEMPLATES:
        result = parse_nl_job(text, mock)
        assert result is not None, f"mock abstained on a real template: {text!r}"
        assert result["job_type"] in {"camera_check", "inference_batch", "firmware_update", "quality_scan", "sensor_diagnostic"}


# -- parser: abstention on genuinely ambiguous input --


def test_mock_client_abstains_on_gibberish():
    mock = MockLLMClient()
    assert parse_nl_job("asdkjaslkdj", mock) is None


def test_mock_client_abstains_on_unrelated_sentence():
    mock = MockLLMClient()
    assert parse_nl_job("what time is the meeting tomorrow", mock) is None


# -- explainer: grounded facts actually match real state --


def test_mock_client_explanation_references_real_device_and_job_values():
    world = WorldState()
    world.apply_event(_resource_event("dev-7", ChangeKind.ADDED, kind="gpu", battery=63.5, load=0.42, connected=True))
    world.apply_event(_demand_event("job-9", ChangeKind.ADDED, structured=True, job_type="camera_check", priority="urgent", requires="gpu"))
    assignments = {"job-9": "dev-7"}
    weights = {"priority_weight": 2.0, "load_penalty_scale": 1.0, "battery_bonus_scale": 0.5, "urgent_unassigned_penalty": 1000.0}
    mock = MockLLMClient()

    explanation = explain_assignment("job-9", assignments, world, weights, mock)

    assert "job-9" in explanation
    assert "dev-7" in explanation
    assert "63.5" in explanation  # the device's actual battery level
    assert "urgent" in explanation  # the job's actual priority
    assert "0.42" in explanation  # the device's actual load
    assert "GPU" in explanation.upper()


def test_mock_client_names_the_dominant_weight_from_real_weights():
    """priority_weight is by far the largest normalized value here, so
    the mock should describe prioritizing urgent work specifically —
    proving it actually read the weights dict, not just echoed a
    generic phrase regardless of the numbers."""
    world = WorldState()
    world.apply_event(_resource_event("dev-1", ChangeKind.ADDED, kind="cpu", battery=50.0, load=0.1, connected=True))
    world.apply_event(_demand_event("job-1", ChangeKind.ADDED, structured=True, job_type="camera_check", priority="normal", requires="cpu"))
    weights = {"priority_weight": 5.0, "load_penalty_scale": 0.1, "battery_bonus_scale": 0.1, "urgent_unassigned_penalty": 1.0}
    mock = MockLLMClient()

    explanation = explain_assignment("job-1", {"job-1": "dev-1"}, world, weights, mock)

    assert "prioritizing urgent work" in explanation


def test_mock_client_explanation_for_unassigned_job_does_not_invent_a_device():
    world = WorldState()
    world.apply_event(_demand_event("job-5", ChangeKind.ADDED, structured=True, job_type="quality_scan", priority="normal", requires="npu"))
    mock = MockLLMClient()

    explanation = explain_assignment("job-5", assignments={}, world=world, weights={}, llm=mock)

    assert "job-5" in explanation
    assert "no device" in explanation.lower() or "no assignment" in explanation.lower()


def test_mock_client_explain_no_replan_references_actual_device_state():
    world = WorldState()
    world.apply_event(_resource_event("dev-3", ChangeKind.ADDED, kind="cpu", battery=41.0, load=0.18, connected=True))
    event = _resource_event("dev-3", ChangeKind.CHANGED, kind="cpu", battery=41.0, load=0.18, connected=True)
    mock = MockLLMClient()

    explanation = explain_no_replan(event, world, mock)

    assert "dev-3" in explanation
    assert "41.0" in explanation  # the device's actual battery
    assert "no re-plan was triggered" in explanation.lower()
