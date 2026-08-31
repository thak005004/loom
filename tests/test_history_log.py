"""Small-scale test of the historical-log generator used later for the
Section 11 throughput benchmark. Only exercises a few hundred events
here — the full tens-of-thousands-event file is generated separately
via `python -m orchestrator.streams.history_log`, not on every test run.
"""

from __future__ import annotations

import json

from orchestrator.streams.history_log import generate_history_log


def test_generate_history_log_writes_well_formed_jsonl(tmp_path):
    out_path = tmp_path / "history_log.jsonl"
    count = generate_history_log(500, out_path, seed=1)

    assert count == 500
    lines = out_path.read_text().splitlines()
    assert len(lines) == 500

    seen_types = set()
    for line in lines:
        record = json.loads(line)
        assert record["type"] in {"resource_changed", "demand_changed", "rule_changed"}
        assert record["change_kind"] in {"added", "removed", "changed", "capability_changed"}
        assert record["source"] in {"telemetry", "demand", "context"}
        assert isinstance(record["payload"], dict)
        seen_types.add(record["type"])

    # with 500 events across three weighted streams, expect to see all three
    assert seen_types == {"resource_changed", "demand_changed", "rule_changed"}


def test_generate_history_log_is_deterministic_given_a_seed(tmp_path):
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    generate_history_log(200, out_a, seed=7)
    generate_history_log(200, out_b, seed=7)

    assert out_a.read_text() == out_b.read_text()
