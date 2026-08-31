"""Confirms the Day-1 pieces actually fit together end to end:
streams -> WorldState -> bandit context -> solve_world() with the
bandit's chosen weights -> reward -> bandit.update(). Not testing any
one piece's behavior in depth (that's what the other test files are
for) — just that the interfaces line up and nothing throws when wired
for real.
"""

from __future__ import annotations

import random

from orchestrator.events.bus import EventBus
from orchestrator.fleet.state import WorldState
from orchestrator.policy.bandit_policy import BanditPolicy, context_from_world
from orchestrator.policy.reward import compute_reward
from orchestrator.scheduling.solver import solve_world
from orchestrator.streams.demand_stream import DemandStreamAdapter
from orchestrator.streams.registry import StreamRegistry
from orchestrator.streams.telemetry_stream import TelemetryStreamAdapter


def test_full_loop_from_streams_through_bandit_update():
    bus = EventBus()
    registry = StreamRegistry(bus)
    world = WorldState()
    world.attach(bus)

    telemetry = TelemetryStreamAdapter(num_devices=10, rng=random.Random(1))
    demand = DemandStreamAdapter(rng=random.Random(2))
    registry.register_stream(telemetry)
    registry.register_stream(demand)

    for _ in range(60):
        telemetry.emit(telemetry.next_raw())
    for _ in range(20):
        demand.emit(demand.next_raw())

    assert world.available_devices()
    assert world.open_jobs()

    context = context_from_world(world)
    policy = BanditPolicy(rng=random.Random(3))
    weights = policy.select_weights(context)
    assert policy.last_arm_index is not None

    result = solve_world(world, **weights)
    assert result.status in ("OPTIMAL", "FEASIBLE")

    reward = compute_reward(result, world.open_jobs(), world.available_devices())
    assert 0.0 <= reward <= 1.0

    policy.update(reward)  # closes the loop; must not raise
