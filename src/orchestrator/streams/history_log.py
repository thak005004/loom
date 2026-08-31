"""Generates a large synthetic historical event log for the Section 11
throughput benchmark (events/sec at different dataset sizes).

Not wired into the live bus — these adapters are never registered
anywhere. `parse()` is called directly on each raw record, which is
enough to produce well-formed Events without a bus to publish to.
Written one line at a time (a JSON Line per Event) rather than building
one giant list in memory, consistent with Section 8's "process as a
stream, don't load a giant list upfront."

Run directly to generate the default 50,000-event log:
    python -m orchestrator.streams.history_log
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from orchestrator.events.types import Event
from orchestrator.streams.context_stream import ContextStreamAdapter
from orchestrator.streams.demand_stream import DemandStreamAdapter
from orchestrator.streams.telemetry_stream import TelemetryStreamAdapter

# Telemetry dominates a real fleet (frequent small updates); context is
# the rarest (shift/policy changes don't happen often).
_STREAM_WEIGHTS = {"telemetry": 0.6, "demand": 0.35, "context": 0.05}

# Arbitrary fixed epoch (not "now") so historical timestamps are spread
# out realistically but the whole log is reproducible byte-for-byte
# given the same seed, independent of when it's generated.
_HISTORY_START_TS = 1_700_000_000.0
_TICK_INTERVAL_RANGE = (0.01, 2.0)  # seconds between synthetic events


def event_to_json(event: Event) -> str:
    return json.dumps(
        {
            "type": event.type.value,
            "change_kind": event.change_kind.value,
            "source": event.source,
            "payload": event.payload,
            "timestamp": event.timestamp,
        }
    )


def iter_history_log(total_events: int, seed: int = 42) -> Iterator[Event]:
    rng = random.Random(seed)
    ts_rng = random.Random(seed + 1000)
    adapters = {
        "telemetry": TelemetryStreamAdapter(rng=random.Random(seed + 1)),
        "demand": DemandStreamAdapter(rng=random.Random(seed + 2)),
        "context": ContextStreamAdapter(rng=random.Random(seed + 3)),
    }
    names = list(adapters)
    weights = [_STREAM_WEIGHTS[n] for n in names]
    t = _HISTORY_START_TS
    for _ in range(total_events):
        adapter = adapters[rng.choices(names, weights=weights, k=1)[0]]
        event = adapter.parse(adapter.next_raw())
        t += ts_rng.uniform(*_TICK_INTERVAL_RANGE)
        yield replace(event, timestamp=t)


def generate_history_log(total_events: int, out_path: Path, seed: int = 42) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w") as f:
        for event in iter_history_log(total_events, seed=seed):
            f.write(event_to_json(event) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--out", type=Path, default=Path("data/history_log.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    written = generate_history_log(args.n, args.out, seed=args.seed)
    print(f"wrote {written} events to {args.out}")
