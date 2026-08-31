"""Ad hoc check: does load_distribution_stdev (Section 11's fairness
metric) actually trend down over a long steady-state run — organic
stream traffic + continuous bandit adaptation, no manual disruptions —
or was the 0.381 -> 0.412 result seen in the dashboard's short,
disruption-heavy session the only data point? Reports the honest
number either way; this is a verification script, not a demo.
"""

from __future__ import annotations

import random
import statistics

from orchestrator.events.bus import EventBus
from orchestrator.fleet.state import WorldState
from orchestrator.policy.bandit_policy import BanditPolicy, context_from_world
from orchestrator.policy.fairness import load_distribution_stdev
from orchestrator.policy.reward import compute_reward
from orchestrator.scheduling.replanner import RePlanner
from orchestrator.scheduling.solver import Assignment, ScheduleResult
from orchestrator.streams.context_stream import ContextStreamAdapter
from orchestrator.streams.demand_stream import DemandStreamAdapter
from orchestrator.streams.registry import StreamRegistry
from orchestrator.streams.telemetry_stream import TelemetryStreamAdapter

FLEET_SIZE = 25
ROUNDS = 300
# Roughly matches the historical log's realistic stream mix (telemetry
# dominates, context is rare) rather than picking uniformly at random.
STREAM_WEIGHTS = [0.6, 0.35, 0.05]


def _current_result(replanner: RePlanner, world: WorldState) -> ScheduleResult:
    assignments = [Assignment(job_id=jid, device_id=did) for jid, did in replanner.assignments.items()]
    assigned_ids = set(replanner.assignments)
    unassigned = [j["job_id"] for j in world.open_jobs() if j["job_id"] not in assigned_ids]
    return ScheduleResult(status="CURRENT", assignments=assignments, unassigned_job_ids=unassigned)


def run(seed: int, rounds: int = ROUNDS) -> list:
    rng = random.Random(seed)
    bus = EventBus()
    registry = StreamRegistry(bus)
    world = WorldState()
    world.attach(bus)
    replanner = RePlanner(world)
    replanner.attach(bus)
    policy = BanditPolicy(rng=random.Random(seed))

    telemetry = TelemetryStreamAdapter(num_devices=FLEET_SIZE, rng=random.Random(seed + 1))
    demand = DemandStreamAdapter(rng=random.Random(seed + 2))
    context = ContextStreamAdapter(rng=random.Random(seed + 3))
    registry.register_stream(telemetry)
    registry.register_stream(demand)
    registry.register_stream(context)

    for _ in range(FLEET_SIZE * 3):
        telemetry.emit(telemetry.next_raw())
    for _ in range(FLEET_SIZE):
        demand.emit(demand.next_raw())
    for _ in range(3):
        context.emit(context.next_raw())

    weights = policy.select_weights(context_from_world(world))
    replanner.weights = weights
    replanner.full_solve()
    reward = compute_reward(_current_result(replanner, world), world.open_jobs(), world.available_devices())
    policy.update(reward)
    fairness = [load_distribution_stdev(world.available_devices(), replanner.assignments)]

    streams = [telemetry, demand, context]
    for _ in range(rounds):
        adapter = rng.choices(streams, weights=STREAM_WEIGHTS, k=1)[0]
        w = policy.select_weights(context_from_world(world))
        replanner.weights = w
        adapter.emit(adapter.next_raw())
        reward = compute_reward(_current_result(replanner, world), world.open_jobs(), world.available_devices())
        policy.update(reward)
        fairness.append(load_distribution_stdev(world.available_devices(), replanner.assignments))

    return fairness


def report(seed: int) -> None:
    series = run(seed)
    n = len(series)
    first_20 = series[:20]
    last_20 = series[-20:]

    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(series) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, series))
    var = sum((x - mean_x) ** 2 for x in xs)
    slope = cov / var if var else 0.0

    print(f"seed={seed}  rounds={n - 1}")
    print(f"  round 0:            {series[0]:.4f}")
    print(f"  final round:        {series[-1]:.4f}")
    print(f"  mean, first 20:     {statistics.mean(first_20):.4f}")
    print(f"  mean, last 20:      {statistics.mean(last_20):.4f}")
    print(f"  min / max / mean:   {min(series):.4f} / {max(series):.4f} / {statistics.mean(series):.4f}")
    print(f"  linear trend slope: {slope:+.6f} per round  ", end="")
    print("(trending DOWN = more fair over time)" if slope < -1e-5 else ("(flat)" if abs(slope) <= 1e-5 else "(trending UP = less fair over time)"))
    print()


if __name__ == "__main__":
    for seed in (1, 2, 3):
        report(seed)
