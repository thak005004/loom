"""Live demo dashboard (Section 9/10/12). A thin Streamlit shell over
the already-built system — every button below reuses an existing
adapter's parse()/emit(), the existing RePlanner, and the existing
bandit policy/reward loop. Nothing here re-implements scheduling logic;
it only looks up real field values to construct raw records and wires
button clicks to calls already covered by the test suite.

RePlanner is constructed with `policy=` here, so it sources weights and
feeds reward back into the bandit *itself*, on every solve — this file
no longer does that bookkeeping by hand (it used to, and that was the
gap: only whatever a UI layer remembered to wire up ever trained the
bandit; see RePlanner's own module docstring for the full story).

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from orchestrator.agents.explainer import explain_assignment
from orchestrator.events.bus import EventBus
from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.state import WorldState
from orchestrator.llm.client import AnthropicLLMClient
from orchestrator.parsing.nl_parser import parse_demand_event
from orchestrator.policy.bandit_policy import ARM_NAMES, BanditPolicy
from orchestrator.policy.fairness import load_distribution_stdev
from orchestrator.scheduling.replanner import RePlanner
from orchestrator.scheduling.solver import solve_world
from orchestrator.streams.context_stream import RULES, ContextStreamAdapter
from orchestrator.streams.demand_stream import DemandStreamAdapter
from orchestrator.streams.maintenance_stream import MaintenanceStreamAdapter
from orchestrator.streams.override_stream import OverrideStreamAdapter
from orchestrator.streams.registry import StreamRegistry
from orchestrator.streams.telemetry_stream import TelemetryStreamAdapter

FLEET_SIZE = 25


# -- setup, once per session --


def init_state() -> None:
    if st.session_state.get("initialized"):
        return

    bus = EventBus()
    registry = StreamRegistry(bus)
    world = WorldState()
    world.attach(bus)
    policy = BanditPolicy(rng=random.Random())
    replanner = RePlanner(world, policy=policy)
    replanner.attach(bus)

    # Deliberately just the original 3 streams at startup (Section 12,
    # demo beat 1) — maintenance/override are registered live from the
    # UI below, which is the actual point being demonstrated.
    telemetry = TelemetryStreamAdapter(num_devices=FLEET_SIZE, rng=random.Random())
    demand = DemandStreamAdapter(rng=random.Random())
    context = ContextStreamAdapter(rng=random.Random())
    registry.register_stream(telemetry)
    registry.register_stream(demand)
    registry.register_stream(context)

    # Every incremental re-plan triggered during this seeding traffic
    # now sources real (if early/untrained) bandit weights and trains
    # the policy on the outcome — RePlanner does that itself now, not
    # this loop.
    for _ in range(FLEET_SIZE * 3):
        telemetry.emit(telemetry.next_raw())
    for _ in range(FLEET_SIZE):
        demand.emit(demand.next_raw())
    for _ in range(3):
        context.emit(context.next_raw())

    replanner.full_solve()

    st.session_state.update(
        initialized=True,
        bus=bus,
        registry=registry,
        world=world,
        replanner=replanner,
        policy=policy,
        streams={"telemetry": telemetry, "demand": demand, "context": context},
        weight_history=[],
        last_timing=None,
        chat_log=[],
        parse_log=[],
    )
    _record_round(policy, replanner, world)


@st.cache_resource
def get_llm_client() -> Optional[AnthropicLLMClient]:
    # anthropic.Anthropic() does NOT raise at construction time even
    # with no key configured — it only fails on the first real API
    # call — so a bare try/except here would never actually catch the
    # "no key" case and the explainer would show as ready, then crash
    # on first use. Check for the key explicitly instead.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        return AnthropicLLMClient()
    except Exception:
        return None


# -- shared helpers (reused by every disruption button, no per-button logic) --


def _time_full_solve(world: WorldState, weights: Dict[str, Any]) -> float:
    """Times solve_world() for the Section 11 comparison only — the
    result is discarded, replanner.assignments is never touched by it."""
    start = time.perf_counter()
    solve_world(world, **weights)
    return time.perf_counter() - start


def _record_round(policy: BanditPolicy, replanner: RePlanner, world: WorldState) -> None:
    """Appends one round to the weight-history the Adaptive Policy tab
    charts — including the fairness metric (Section 11: "workload
    distribution across devices, before vs. after the policy adapts"),
    tracked every round so that trend is directly visible, not just
    inferable from the reward. Only call this right after a *real*
    solve (check replanner.solve_count first) — policy.last_arm_index
    and replanner.last_reward reflect whatever RePlanner's own internal
    policy loop most recently did, which is only meaningful then."""
    st.session_state.weight_history.append(
        {
            "round": len(st.session_state.weight_history),
            "arm": ARM_NAMES[policy.last_arm_index],
            "reward": replanner.last_reward,
            "fairness": load_distribution_stdev(world.available_devices(), replanner.assignments),
            **dict(policy.arms[policy.last_arm_index]),
        }
    )


def run_disruption(emit_fn: Callable[[], Optional[Event]]) -> None:
    world = st.session_state.world
    replanner = st.session_state.replanner
    policy = st.session_state.policy

    solves_before = replanner.solve_count
    event = emit_fn()
    if event is None:
        st.warning("Nothing to disrupt right now.")
        return

    if replanner.solve_count == solves_before:
        # RePlanner published/applied the event but its dispatcher found
        # nothing to re-solve (e.g. a rule fired but no matching jobs
        # existed) — no fresh weights were selected, nothing to score.
        st.session_state.last_timing = {
            "event_type": event.type.value,
            "change_kind": event.change_kind.value,
            "replan_seconds": replanner.last_replan_seconds,
            "full_solve_seconds": None,
        }
        st.rerun()
        return

    # The weights RePlanner just used internally for this round, read
    # back from the policy rather than re-selected here — re-selecting
    # would both waste an exploration draw and risk scoring a *different*
    # round than the one that actually just happened.
    weights = dict(policy.arms[policy.last_arm_index])
    st.session_state.last_timing = {
        "event_type": event.type.value,
        "change_kind": event.change_kind.value,
        "replan_seconds": replanner.last_replan_seconds,
        "full_solve_seconds": _time_full_solve(world, weights),
    }
    _record_round(policy, replanner, world)
    # Streamlit renders top-to-bottom in one pass per interaction, and
    # the state overview (metrics/tables) is drawn *before* this handler
    # runs — without forcing an immediate second pass, the visible
    # tables would lag one click behind whatever this action just did.
    st.rerun()


def parse_pending_requests() -> None:
    """Runs the NL parser (Section 3c) over every currently-unparsed
    job. Each job's outcome — parsed, or abstained with a reason — is
    logged and shown in the UI, so an abstention is visible with an
    explanation rather than the job just silently continuing to sit
    unassigned with no indication why."""
    world = st.session_state.world
    bus = st.session_state.bus
    replanner = st.session_state.replanner
    policy = st.session_state.policy

    llm = get_llm_client()
    if llm is None:
        st.session_state.parse_log = ["Set ANTHROPIC_API_KEY (and restart) to enable the parser."]
        return

    pending = [j for j in world.open_jobs() if j.get("structured") is False]
    if not pending:
        st.session_state.parse_log = ["No unparsed requests right now."]
        return

    solves_before = replanner.solve_count
    log: List[str] = []
    for job in pending:
        raw_event = Event(type=EventType.DEMAND_CHANGED, change_kind=ChangeKind.ADDED, source="demand", payload=dict(job))
        try:
            parsed_event = parse_demand_event(raw_event, llm)
        except Exception as exc:  # network/rate-limit error mid-demo — don't crash the page
            log.append(f"⚠️ `{job['job_id']}`: parser call failed ({exc}) — left unparsed.")
            continue
        if parsed_event is None:
            text = job.get("text", "")
            log.append(f"🚫 `{job['job_id']}`: couldn't confidently classify \"{text}\" — left unparsed, not guessed at.")
            continue
        # Publishing (not adapter.emit()) is deliberate here: parse_demand_event
        # already produced a finished Event, so this goes straight to the
        # bus — RePlanner picks it up via its own subscription exactly
        # like any stream-produced event, sourcing weights and scoring
        # the outcome itself, same as every other event in this file.
        bus.publish(parsed_event)
        log.append(f"✅ `{job['job_id']}`: parsed as **{parsed_event.payload['job_type']} / {parsed_event.payload['priority']}**.")

    st.session_state.parse_log = log

    # Each successfully-parsed job triggered its own real re-plan (and
    # its own policy round) inside RePlanner already; this just records
    # one chart point reflecting the *last* of those rounds, for the UI.
    if replanner.solve_count > solves_before:
        _record_round(policy, replanner, world)
    st.rerun()


# -- disruption actions: each reuses an existing adapter's parse()/emit(), nothing new --


def kill_random_device() -> Optional[Event]:
    telemetry = st.session_state.streams["telemetry"]
    world = st.session_state.world
    replanner = st.session_state.replanner
    connected = [d for d, dev in world.devices.items() if dev.get("connected", True)]
    if not connected:
        return None
    # Prefer a device that's actually holding a job. A random pick would
    # usually land on an idle one (most devices are idle at any given
    # moment), which reassigns nothing — the re-plan timing would then
    # show a trivial near-zero no-op instead of a real reassignment,
    # which is a misleading number to show right under a "look how fast
    # incremental re-planning is" readout.
    busy = [d for d in connected if d in replanner.assignments.values()]
    device_id = random.choice(busy) if busy else random.choice(connected)
    device = world.devices[device_id]
    raw = {
        "device_id": device_id,
        "kind": device.get("kind"),
        "battery": 0.0,
        "connected": False,
        "load": device.get("load", 0.0),
        "change_kind": "removed",
    }
    return telemetry.emit(raw)  # reuses telemetry_stream's own parse()


def inject_urgent_job() -> Optional[Event]:
    demand = st.session_state.streams["demand"]
    raw = demand.next_raw()
    attempts = 0
    while not (raw.get("change_kind") == "added" and raw.get("structured") is True) and attempts < 50:
        raw = demand.next_raw()
        attempts += 1
    raw["priority"] = "urgent"
    return demand.emit(raw)  # reuses demand_stream's own parse()


def trigger_rule_change() -> Optional[Event]:
    context = st.session_state.streams["context"]
    rule = next(r for r in RULES if r["rule"] == "quality_scan_requires_gpu")
    raw = {"rule": rule["rule"], "description": rule["description"], "active": True, "change_kind": "changed"}
    return context.emit(raw)  # reuses context_stream's own parse()


def register_maintenance_live() -> None:
    if "maintenance" in st.session_state.streams:
        return
    world = st.session_state.world
    maintenance = MaintenanceStreamAdapter(device_ids=list(world.devices.keys()), rng=random.Random())
    st.session_state.registry.register_stream(maintenance)
    st.session_state.streams["maintenance"] = maintenance
    st.rerun()  # see run_disruption()'s comment on why this is needed


def register_override_live() -> None:
    if "override" in st.session_state.streams:
        return
    world = st.session_state.world
    override = OverrideStreamAdapter(
        device_ids=list(world.devices.keys()),
        job_ids=[j["job_id"] for j in world.open_jobs()],
        rng=random.Random(),
    )
    st.session_state.registry.register_stream(override)
    st.session_state.streams["override"] = override
    st.rerun()  # see run_disruption()'s comment on why this is needed


# -- view helpers --


_KIND_ICON = {"cpu": "🖥️ CPU", "gpu": "🎮 GPU", "npu": "🧠 NPU"}


def _device_rows(world: WorldState, replanner: RePlanner) -> List[Dict[str, Any]]:
    jobs_by_device: Dict[str, List[str]] = {}
    for job_id, device_id in replanner.assignments.items():
        jobs_by_device.setdefault(device_id, []).append(job_id)
    rows = []
    for device_id, device in sorted(world.devices.items()):
        connected = device.get("connected", True)
        reliability = device.get("reliability", "nominal")
        rows.append(
            {
                "Device": device_id,
                "Type": _KIND_ICON.get(device.get("kind"), device.get("kind")),
                "Battery": round(device.get("battery", 0.0)),
                "Load": round(device.get("load", 0.0) * 100),
                "Status": "🟢 online" if connected else "🔴 offline",
                "Reliability": "✅ nominal" if reliability == "nominal" else "⚠️ degraded",
                "Assigned jobs": ", ".join(sorted(jobs_by_device.get(device_id, []))) or "—",
            }
        )
    return rows


_PRIORITY_LABEL = {"urgent": "🔴 urgent", "normal": "⚪ normal"}


def _job_rows(world: WorldState, replanner: RePlanner) -> List[Dict[str, Any]]:
    rows = []
    for job_id, job in sorted(world.jobs.items()):
        job_type = job.get("job_type")
        if job_type is None and job.get("structured") is False:
            job_type = "💬 unparsed request"
        priority = job.get("priority")
        assigned = replanner.assignments.get(job_id)
        rows.append(
            {
                "Job": job_id,
                "Type": job_type or "—",
                "Priority": _PRIORITY_LABEL.get(priority, priority or "—"),
                "Requires": _KIND_ICON.get(job.get("requires"), "any device"),
                "Assigned to": assigned or "⏳ unassigned",
            }
        )
    return rows


# -- page --

_DEVICE_COLUMN_CONFIG = {
    "Battery": st.column_config.ProgressColumn("Battery", min_value=0, max_value=100, format="%d%%"),
    "Load": st.column_config.ProgressColumn("Load", min_value=0, max_value=100, format="%d%%"),
}


def _hide_streamlit_chrome() -> None:
    # Cosmetic only: hides the default hamburger menu and footer so the
    # page reads as a finished dashboard rather than a bare dev preview.
    # Purely presentational — no app behavior depends on this.
    st.markdown("<style>#MainMenu, footer {visibility: hidden;}</style>", unsafe_allow_html=True)


def _render_state_overview(world: WorldState, replanner: RePlanner, registry: StreamRegistry) -> None:
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Registered streams",
        len(registry),
        help="Independent data sources currently feeding the system. New ones can be plugged in live, "
        "from the Disruptions tab, without restarting anything.",
    )
    m2.metric("Devices", len(world.devices), help="CPU/GPU/NPU devices currently known to the fleet.")
    m3.metric("Open jobs", len(world.jobs), help="Jobs currently pending assignment or already running.")
    st.caption("🔌 Streams online:  " + "  ".join(f"`{s}`" for s in sorted(st.session_state.streams)))

    st.subheader("Fleet")
    st.dataframe(_device_rows(world, replanner), width="stretch", hide_index=True, column_config=_DEVICE_COLUMN_CONFIG)

    st.subheader("Jobs")
    st.dataframe(_job_rows(world, replanner), width="stretch", hide_index=True)


def _render_disruptions_tab() -> None:
    st.caption(
        "Each button below fires a real event through the same pipeline the live streams use — nothing "
        "here is a separate simulation. Watch the re-plan latency at the bottom: only the affected slice "
        "of the plan gets re-solved, not the whole fleet."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💀 Kill a random device", width="stretch"):
            run_disruption(kill_random_device)
        st.caption("A device goes offline mid-job.")
    with c2:
        if st.button("🚨 Inject an urgent job", width="stretch"):
            run_disruption(inject_urgent_job)
        st.caption("A new high-priority request arrives.")
    with c3:
        if st.button("📋 Trigger a rule change", width="stretch"):
            run_disruption(trigger_rule_change)
        st.caption("A policy rule is introduced mid-run.")

    st.divider()
    st.markdown("**Register a new stream, live** — the pluggability claim, actually exercised, not asserted.")
    rc1, rc2 = st.columns(2)
    with rc1:
        if "maintenance" not in st.session_state.streams:
            if st.button("🔧 Register maintenance stream", width="stretch"):
                register_maintenance_live()
            st.caption("Adds a device-reliability feed the system has never seen before.")
        else:
            if st.button("⚠️ Simulate maintenance issue", width="stretch"):
                m = st.session_state.streams["maintenance"]
                run_disruption(lambda: m.emit(m.next_raw()))
            st.caption("✅ Registered — fire a reliability event from it.")
    with rc2:
        if "override" not in st.session_state.streams:
            if st.button("🧑 Register override stream", width="stretch"):
                register_override_live()
            st.caption("Adds a stand-in for a human operator's manual actions.")
        else:
            if st.button("🧑 Simulate operator override", width="stretch"):
                o = st.session_state.streams["override"]
                run_disruption(lambda: o.emit(o.next_raw()))
            st.caption("✅ Registered — fire an operator action from it.")

    if st.session_state.last_timing:
        t = st.session_state.last_timing
        st.divider()
        st.markdown(f"**Last disruption:** `{t['event_type']}` / `{t['change_kind']}`")
        if t["full_solve_seconds"] is None:
            st.caption(f"No re-plan was needed for this one (dispatch took {t['replan_seconds'] * 1000:.2f} ms) — nothing matched.")
        else:
            tc1, tc2 = st.columns(2)
            tc1.metric("Incremental re-plan", f"{t['replan_seconds'] * 1000:.2f} ms")
            tc2.metric("Full re-solve (for comparison)", f"{t['full_solve_seconds'] * 1000:.2f} ms")
    else:
        st.caption("Trigger a disruption above to see the re-plan latency comparison.")

    st.divider()
    st.markdown("**Parse pending natural-language requests**")
    st.caption(
        "demand_stream passes messy text through untouched — this runs the language parser (Section 3c) "
        "over whatever's still unparsed. A request the model can't confidently classify is left unparsed "
        "on purpose, with the reason shown below, not guessed at."
    )
    if st.button("🗣️ Parse pending NL requests"):
        parse_pending_requests()
    for line in st.session_state.parse_log:
        st.markdown(f"- {line}")


def _render_policy_tab(history: List[Dict[str, Any]]) -> None:
    st.caption(
        "The solver's objective isn't hand-tuned — a contextual bandit picks the weighting per round based "
        "on current fleet conditions, and learns from how well each round's plan actually worked out. "
        "Trigger a few disruptions to watch it move."
    )
    if len(history) > 1:
        st.markdown("**priority_weight · load_penalty_scale · battery_bonus_scale**")
        st.line_chart(
            {
                "priority_weight": [h["priority_weight"] for h in history],
                "load_penalty_scale": [h["load_penalty_scale"] for h in history],
                "battery_bonus_scale": [h["battery_bonus_scale"] for h in history],
            }
        )
        st.markdown("**urgent_unassigned_penalty**  _(shown separately — different scale)_")
        st.line_chart({"urgent_unassigned_penalty": [h["urgent_unassigned_penalty"] for h in history]})
    else:
        st.info("No disruptions yet — trigger one from the Disruptions tab to start seeing the policy adapt.")

    a1, a2 = st.columns(2)
    a1.metric("Active strategy", history[-1]["arm"], help="Which of the policy's fixed weight profiles is currently favored.")
    a2.metric("Last round's reward", f"{history[-1]['reward']:.2f}", help="How well the resulting plan scored: capacity respected, load balanced, urgent jobs served.")

    st.divider()
    st.markdown("**Fairness — workload distribution across devices**")
    st.caption(
        "Standard deviation of per-device utilization (assigned jobs ÷ capacity). Lower means work is "
        "spread more evenly across the fleet; 0 means perfectly even."
    )
    f1, f2 = st.columns(2)
    f1.metric("Before (round 0)", f"{history[0].get('fairness', 0.0):.3f}", help="Measured right after the initial full solve, before any adaptation.")
    f2.metric("Now", f"{history[-1].get('fairness', 0.0):.3f}", help="Same metric, after the policy has adapted across every round since.")
    if len(history) > 1:
        st.line_chart({"fairness (load stdev)": [h.get("fairness", 0.0) for h in history]})


def _render_explainer_tab(world: WorldState, replanner: RePlanner) -> None:
    st.caption(
        "Ask why a job ended up where it did. The answer is grounded in the real numbers behind that "
        "decision — the job's priority and requirements, the device's battery and load, the policy's "
        "active weights — not a free-form guess."
    )
    llm = get_llm_client()
    if llm is None:
        st.info("Set ANTHROPIC_API_KEY (and restart the app) to enable the explainer.")
        return

    job_ids = sorted(world.jobs)
    if job_ids:
        selected_job = st.selectbox("Job", job_ids)
        if st.button("Why is this job assigned this way?"):
            try:
                answer = explain_assignment(selected_job, replanner.assignments, world, replanner.weights, llm)
            except Exception as exc:  # e.g. invalid key, network/rate-limit error mid-demo
                answer = f"(explainer call failed: {exc})"
            st.session_state.chat_log.append((selected_job, answer))

    for job_id, answer in reversed(st.session_state.chat_log[-10:]):
        with st.chat_message("assistant"):
            st.markdown(f"**Why `{job_id}`?**\n\n{answer}")


def main() -> None:
    st.set_page_config(page_title="Heterogeneous AI Task Orchestrator", layout="wide", page_icon="🛰️")
    _hide_streamlit_chrome()
    init_state()

    world = st.session_state.world
    replanner = st.session_state.replanner
    registry = st.session_state.registry
    history = st.session_state.weight_history

    st.title("🛰️ Heterogeneous AI Task Orchestrator")
    st.caption(
        "A fleet of CPU/GPU/NPU devices fed by independent data streams, scheduled by a CP-SAT solver "
        "whose objective is chosen by a self-adapting policy, and re-planned incrementally — not from "
        "scratch — whenever something changes."
    )
    with st.expander("ℹ️  How to read this page"):
        st.markdown(
            "- **Fleet / Jobs** below are the live state: who's assigned to what, right now.\n"
            "- **⚡ Disruptions & Streams** tab: trigger a real event and watch the system re-plan in "
            "milliseconds — and add a brand-new data stream while everything keeps running.\n"
            "- **📈 Adaptive policy** tab: watch the scheduler's objective weights change on their own as "
            "outcomes come in.\n"
            "- **💬 Ask the explainer** tab: ask why any job landed where it did."
        )

    _render_state_overview(world, replanner, registry)

    st.divider()
    tab_disrupt, tab_policy, tab_explain = st.tabs(["⚡ Disruptions & Streams", "📈 Adaptive policy", "💬 Ask the explainer"])
    with tab_disrupt:
        _render_disruptions_tab()
    with tab_policy:
        _render_policy_tab(history)
    with tab_explain:
        _render_explainer_tab(world, replanner)

    st.sidebar.button("🔄 Reset simulation", on_click=lambda: st.session_state.clear(), width="stretch")
    st.sidebar.caption("Clears all state and starts over with a fresh fleet.")


main()
