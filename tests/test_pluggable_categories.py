"""Proves resource/task categories are as pluggable as data streams:
introducing a device kind and a job requirement that have never
appeared anywhere in this codebase (not cpu/gpu/npu) requires zero
changes to the solver, replanner, policy, or feasibility layer — it's
just a new string value flowing through code that was never written
against a fixed set of kinds.

Direct analog of Day 2 Step 2's live-stream-registration test, applied
to categories instead of data sources: the proof there was "a fourth
stream registers mid-run with zero other code touched"; the proof here
is "a fourth *kind* is scheduled correctly with zero other code
touched."
"""

from __future__ import annotations

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.feasibility import feasible_categories
from orchestrator.fleet.state import WorldState
from orchestrator.scheduling.replanner import RePlanner

NOVEL_KIND = "quantum"  # never appears anywhere else in src/ — see test below


def _resource_event(device_id, change_kind, **fields) -> Event:
    return Event(
        type=EventType.RESOURCE_CHANGED,
        change_kind=change_kind,
        source="test",
        payload={"device_id": device_id, **fields},
    )


def _demand_event(job_id, change_kind, **fields) -> Event:
    return Event(
        type=EventType.DEMAND_CHANGED,
        change_kind=change_kind,
        source="test",
        payload={"job_id": job_id, **fields},
    )


def _seed_device(world, device_id, kind, battery=90.0, load=0.0, connected=True):
    world.apply_event(_resource_event(device_id, ChangeKind.ADDED, kind=kind, battery=battery, load=load, connected=connected))


def test_novel_kind_does_not_appear_anywhere_else_in_the_codebase():
    """Guards the premise: if a future change happens to special-case
    "quantum" somewhere, this stops being a valid proof of open
    pluggability and should fail loudly rather than pass by accident."""
    import pathlib

    src_root = pathlib.Path(__file__).resolve().parent.parent / "src"
    hits = [
        path
        for path in src_root.rglob("*.py")
        if NOVEL_KIND in path.read_text()
    ]
    assert hits == [], f"{NOVEL_KIND!r} already appears in {hits} — pick a different novel kind"


def test_a_brand_new_device_kind_is_scheduled_by_the_real_solver_and_replanner_unmodified():
    world = WorldState()
    _seed_device(world, "dev-q1", NOVEL_KIND)
    replanner = RePlanner(world)  # no policy — same default-weights path every other RePlanner test uses

    result = replanner.on_event(_demand_event("job-q1", ChangeKind.ADDED, priority="urgent", requires=NOVEL_KIND))

    assert replanner.assignments.get("job-q1") == "dev-q1"
    assert {a.job_id for a in result.assignments} == {"job-q1"}


def test_a_regular_kind_job_does_not_wrongly_match_the_novel_kind_device():
    world = WorldState()
    _seed_device(world, "dev-q1", NOVEL_KIND)
    replanner = RePlanner(world)

    replanner.on_event(_demand_event("job-cpu", ChangeKind.ADDED, priority="normal", requires="cpu"))

    assert "job-cpu" not in replanner.assignments


def test_feasibility_layer_reports_the_novel_category_feasible_once_the_device_exists():
    world = WorldState()
    world.apply_event(_demand_event("job-q1", ChangeKind.ADDED, priority="normal", requires=NOVEL_KIND))
    assert feasible_categories(world)[NOVEL_KIND] is False

    _seed_device(world, "dev-q1", NOVEL_KIND)
    assert feasible_categories(world)[NOVEL_KIND] is True


def test_feasibility_layer_reports_the_novel_category_infeasible_once_the_device_is_removed():
    world = WorldState()
    _seed_device(world, "dev-q1", NOVEL_KIND)
    world.apply_event(_demand_event("job-q1", ChangeKind.ADDED, priority="normal", requires=NOVEL_KIND))
    assert feasible_categories(world)[NOVEL_KIND] is True

    world.apply_event(_resource_event("dev-q1", ChangeKind.REMOVED))

    assert feasible_categories(world)[NOVEL_KIND] is False
