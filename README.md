# Loom
### A dynamic AI task orchestrator that weaves multiple data streams into one constantly-adapting plan

A system that decides which device should do which job, learns from how those decisions turn out, reacts to change in milliseconds, and never has to be rewritten when a new kind of data source or disruption shows up.

---

## 1. What this is

A fleet of devices with mixed capabilities (battery, load, connectivity) gets a continuous stream of jobs. Most systems hardcode rules that never learn, or special-case each scenario, breaking when something unanticipated happens. Loom does neither: a policy that learns continuously from outcomes, and one mechanism for every kind of change, both backed by tests.

---

## 2. The three ideas that make Loom dynamic

### 2a. A policy that learns

A contextual bandit replaces the fixed weighting formula: given fleet load, battery, and recent failures, it picks weights, observes the outcome, and updates its beliefs, continuously, never freezing.

Proof, not assertion: two identical policies trained on the same input, one rewarded only for arm A, the other for arm B, deterministically chose the rewarded arm after training: same input, opposite behavior.

A gap we found and closed: incremental replans used default weights and never reported outcomes back. We refactored so every replan sources live weights and feeds its outcome back, and added the missing test.

### 2b. Reacting to any kind of change the same way

Every change resolves to one of three shapes, so nothing is special-cased:

- **Resource changed**: a device added, removed, or its capabilities changed
- **Demand changed**: a job added, removed, or its priority changed
- **Rule changed**: a constraint changed, e.g. "this job type now requires a GPU"

The replanner only re-decides the piece of the plan actually touched.

| What happened | Change type | What the system does |
|---|---|---|
| Device battery dies mid-job | Resource removed | Reassigns only the jobs that were on that device |
| New device joins the fleet | Resource added | Considers it for pending work immediately |
| Device reliability changes (failure history) | Resource capability changed | Re-evaluates what that device is eligible for |
| Urgent job arrives | Demand added | Replans only the devices that could take it |
| Job priority changes mid-flight | Demand changed | Re-solves only around that job |
| Job finishes or is cancelled | Demand removed | Dropped, no solver call needed |
| Rule changes (e.g. job type now needs a GPU) | Rule changed | Reassigns only the jobs that rule actually affects |
| Human operator manually reassigns something | Demand or resource changed | Handled identically to an automated change |

Full recompute time grows sharply with fleet size; the incremental path stays flat, since it only touches what changed (numbers in Section 8).

### 2c. Any number of data sources, pluggable at any time

Data arrives from many places, and the set grows over time. Any source implements one interface (parse its format into a resource, demand, or rule change) and registers itself.

Five feed the system: device telemetry, job requests, external context, maintenance history, and manual overrides.

Proof, not assertion: a test starts with three sources active, registers a fourth mid-run with zero other code touched. The count updates immediately, data merges correctly, and the replanner picks it up through a handler written before this source existed.

---

## 3. Language understanding and grounded explanations

Job requests sometimes arrive as plain sentences ("urgent camera check on line 3, needs GPU"). An LLM structures these and is allowed to abstain: nonsense input first produced confident wrong guesses, until we gave it permission to say "unparseable."

The same model explains any decision in plain language, grounded in the real numbers behind it, including why nothing happened for routine changes that don't trigger a replan.

---

## 4. How it fits together

```
                     Multiple data sources
                (devices, jobs, rules, more)
                              |
                              v
                          Event bus
              (normalizes every kind of change)
                              |
                              v
                         World state
                    (live devices and jobs)
                              |
                  +-----------+-----------+
                  v                       v
          Adaptive policy  <----->    Scheduler
        (learns from outcomes)   (assigns jobs to devices)
                  |                       |
                  +-----------+-----------+
                              v
                   Incremental re-planner
                    (reacts to any change)
                              |
                              v
                Explainer + live dashboard
              (shows state, answers "why")
```

Data flows down: sources feed the event bus, which keeps world state current. The policy and scheduler work together in a loop (the policy sets priorities, the scheduler assigns work, outcomes feed back to the policy). Any change triggers the re-planner, which only touches what's actually affected, and everything surfaces in the dashboard.

---

## 5. An honest result: what the system does and doesn't optimize for

We tested whether fairness (how evenly work spreads across devices) improves as the policy adapts. It doesn't, reliably: across multiple runs, load distribution trended slightly worse over time in steady-state operation, not better.

Why: the reward deliberately weights urgent-job-service far more than balance. This was a deliberate earlier fix; an evenly-weighted reward scored "doing nothing" almost as well as a real assignment, teaching the policy that inaction is nearly as good as action. The policy is correctly optimizing what it's actually told to. Fairness is measured but isn't part of the reward. A periodic full rebalancing solve would likely fix this, and is first on the roadmap.

The honest claim: the policy demonstrably adapts and improves at the objective it's actually given. Fairness is monitored, not yet optimized.

---

## 6. Architecture

```
Five pluggable data sources         Shared event language          Live current state
(telemetry, demand, context,   ->   (resource changed /       ->   (devices that exist now,
 maintenance, manual override)       demand changed /               jobs currently open)
                                      rule changed)
                                            |
                                            v
                        Learning policy picks priorities   ->   Scheduler assigns jobs to devices
                        (adjusts from real outcomes)             (respects capability and capacity)
                                            ^                              |
                                            |                              v
                                            +----- outcome fed back ---- Incremental replanner
                                                                          (reacts to any change,
                                                                           touches only what's affected)
                                                                                    |
                                                                                    v
                                                                    Explains any decision in plain
                                                                    language, grounded in real state
                                                                                    |
                                                                                    v
                                                                          Live dashboard
```

---

## 7. Fit with Intel's direction

The heterogeneous fleet mirrors where OpenVINO's roadmap is headed: task-based scheduling across CPU/GPU/NPU instead of targeting hardware manually. Parsing and matching run on-device via OpenVINO; the CP-SAT solves and learning loop are the steady, latency-sensitive workload Xeon and AMX are built for. The pluggable-services shape mirrors OPEA's reference architecture.

---

## 8. What's tested, and the real numbers

- **76 automated tests**: the scheduler never assigns a device to work it can't handle; the replanner scopes correctly to the affected slice for every kind of change; the policy's behavior measurably shifts based on real outcomes, for both full and incremental decisions; a new data source can be added while the system is running with zero other code touched; the parser abstains rather than guesses on input it can't confidently classify.
- **Incremental replan latency**: roughly 5 to 8.5ms, scaling with how many jobs were actually affected, not with fleet size.
- **Full recompute latency**: roughly 0.10s at 50 devices, growing past a second at 150 devices.
- **Event throughput**: roughly 80,000 to 86,000 events/sec, stable from 1,000 to 50,000 events.
- **Fairness**: measured and reported honestly (Section 5); currently monitored, not optimized.

---

## 9. What's next

Deliberately not built, each a real multi-day undertaking:

- Periodic rebalancing solves, for the fairness gap in Section 5
- A meta-controller choosing between a fast approximate match and a full solve, by system load
- A decentralized alternative where devices bid for jobs based on their own state
- A separate fairness-auditing check, independent of the main scheduling objective

---

## 10. Running it

```bash
pip install -e .
pytest
streamlit run dashboard/app.py
```

No API key needed: without `ANTHROPIC_API_KEY` set, the parser and explainer fall back to rule-based logic, with a banner saying so.

The dashboard shows live state, lets you trigger disruptions, register a new source while running, watch policy weights shift, and query the explainer.
