from __future__ import annotations

from orchestrator.policy.fairness import load_distribution_stdev


def _device(device_id, kind="gpu", connected=True):
    return {"device_id": device_id, "kind": kind, "connected": connected}


def test_perfectly_balanced_load_has_zero_stdev():
    devices = [_device("dev-a"), _device("dev-b")]
    assignments = {"job-1": "dev-a", "job-2": "dev-b"}
    assert load_distribution_stdev(devices, assignments) == 0.0


def test_no_open_jobs_has_zero_stdev():
    devices = [_device("dev-a"), _device("dev-b")]
    assert load_distribution_stdev(devices, {}) == 0.0


def test_lopsided_load_has_higher_stdev_than_balanced():
    devices = [_device("dev-a"), _device("dev-b")]
    balanced = {"job-1": "dev-a", "job-2": "dev-b"}
    lopsided = {"job-1": "dev-a", "job-2": "dev-a"}  # both on dev-a (capacity 2, so 100% vs 0%)

    assert load_distribution_stdev(devices, lopsided) > load_distribution_stdev(devices, balanced)


def test_disconnected_devices_are_excluded_from_the_calculation():
    devices = [_device("dev-a"), _device("dev-b", connected=False)]
    assignments = {"job-1": "dev-a"}
    # only dev-a counts; a single connected device is trivially "balanced"
    assert load_distribution_stdev(devices, assignments) == 0.0


def test_no_connected_devices_has_zero_stdev():
    devices = [_device("dev-a", connected=False)]
    assert load_distribution_stdev(devices, {"job-1": "dev-a"}) == 0.0
