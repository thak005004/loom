# Loom
### A dynamic AI task orchestrator that weaves multiple data streams into one constantly-adapting plan

A system that decides which device should do which job, learns from how those decisions turn out, reacts to change in milliseconds, and never has to be rewritten when a new kind of data source or disruption shows up.

---

## 1. What this is

Imagine a fleet of devices with mixed capabilities and shifting conditions (battery, load, connectivity), receiving a continuous stream of jobs with varying priority and requirements. Something has to decide, continuously, which device does which job, and keep re-deciding as conditions change: a device drops, a battery runs low, an urgent job jumps the queue, a new rule gets introduced.

Most systems that do this fall into one of two traps:

- **Fixed rules that never improve.** A hand-written formula for what matters most that never learns from whether its decisions actually worked out.
- **Special-cased reactions.** Separate code paths for "handle a dead device," "handle an urgent job," "handle a new data source," meaning every new kind of change requires new code, and the system quietly breaks the moment reality does something nobody anticipated.

Loom avoids both. Two properties make it genuinely different, and both are backed by tests, not just claims:

1. It learns from its own outcomes, continuously. Not a one-time training run: a policy that keeps adjusting itself as it sees more results.
2. It treats every kind of change the same general way. A device dying, a person calling out, an ingredient running out, a brand-new data source appearing all get handled by one mechanism, so it isn't limited to the scenarios anyone thought to hardcode.

---

## 2. The three ideas that make Loom dynamic

### 2a. A policy that learns

Instead of a fixed formula for balancing priority, battery life, and fairness, the scheduler is guided by a contextual bandit. Given the current situation (fleet load, average battery, recent failure rate), it picks a weighting for what matters most right now, observes how that decision actually played out, and updates its beliefs. It never stops doing this; there's no train-once-freeze-forever step. As conditions drift over time, the policy adapts on its own.

Proof, not assertion: two identical policies were trained on the same input, one rewarded only for choosing arm A, the other only for arm B. After training, forcing pure exploitation, the first policy deterministically chose A and the second deterministically chose B: same input, opposite learned behavior, driven entirely by which pattern was rewarded.

A gap we found and closed: early on, the fast reactive path and the learning path were technically both real but not actually connected. The incremental replanner used default weights and never reported its outcomes back to the learner. We caught this, refactored so every replan (full or incremental) sources live learned weights and feeds its outcome back, and added the test that should have existed from the start to prove it. This is now one coherent loop, not two features that happen to demo well side by side.

### 2b. Reacting to any kind of change the same way

When something changes, the system doesn't recompute everything from scratch, and it doesn't have separate code per scenario. Every change resolves to one of three general shapes:

- **Resource changed** (something on the supply side: a device added, removed, or its capabilities changed)
- **Demand changed** (something on the task side: a job added, removed, or its priority changed)
- **Rule changed** (a constraint changed, e.g. "this job type now requires a GPU")

The replanner reacts to the shape of the change, not the real-world scenario behind it, so it only ever re-decides the piece of the plan actually touched, and it correctly handles scenarios nobody explicitly wrote code for.

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

Measured results: a full recompute over the whole fleet runs about 0.10s at 50 devices, growing past a second at 150 devices, since it reconsiders everything. The scoped, incremental replan stays in the single-digit milliseconds regardless of fleet size, since it only touches the jobs actually affected by whatever changed.

### 2c. Any number of data sources, pluggable at any time

Real systems get data from several places, in different formats, at different speeds, and the set of sources grows over time. Rather than writing the system against a fixed list of inputs, any data source implements one small interface (parse its raw format into a resource, demand, or rule change) and registers itself. Nothing downstream (event handling, replanner, learning policy) needs to know how many sources exist or where they come from.

Five sources currently feed the system this way:
- **Device telemetry**: battery, load, connectivity, frequent small updates
- **Job requests**: structured and messy natural-language requests, irregular arrival
- **External context**: lower-frequency schedule and priority-policy signals
- **Maintenance history**: failure records that adjust device reliability
- **Manual overrides**: a stand-in for a human operator stepping in directly

Proof, not assertion: a test starts the system with only the first three sources active, real traffic flowing, a real assignment already in place, then registers a fourth source while the system is running, with zero other code touched. Source count updates immediately, the new data merges correctly into the current state without corrupting anything, and the replanner picks up its events through a handler that was written and tested before this source existed. The dashboard has a live button that does exactly this.

---

## 3. Language understanding and grounded explanations

Job requests sometimes arrive as plain sentences ("urgent camera check on line 3, needs GPU"). An LLM turns these into the structured form the scheduler needs, and is explicitly allowed to abstain. We found, while stress-testing with nonsense input, that the model would confidently produce a plausible but wrong structured guess rather than admit it couldn't classify the text. Giving it explicit permission to say "unparseable" fixed this: real requests still parse correctly, and gibberish is now cleanly rejected instead of silently poisoning the system.

A separate use of the same model lets you ask, in plain language, why a specific decision was made ("why is this job on this device?"), grounded in the actual numbers that drove it (real priority, real battery/load, the policy's currently active weighting), not a plausible-sounding guess. It can also explain why nothing happened, for routine changes that correctly don't trigger a replan.

---

## 4. An honest result: what the system does and doesn't optimize for

We tested whether fairness (how evenly work is spread across devices) improves as the policy adapts over time. It doesn't, reliably. Across multiple runs, load distribution trended slightly worse over time in steady-state operation, not better.

We dug into why: the reward signal deliberately weights urgent-job-service far more heavily than balance across devices. This was a deliberate earlier fix; an evenly-weighted reward was found to score "doing nothing" almost as well as a real assignment, which would have taught the policy that inaction is nearly as good as action. The policy is correctly learning to optimize what it's actually told to optimize. Fairness is measured and displayed but isn't currently part of the reward. A periodic full rebalancing solve, rather than relying solely on incremental reactive replanning, would likely address this and is the first item on the roadmap.

The honest claim: the policy demonstrably adapts and improves at the objective it's actually given (serving high-priority work, respecting capacity). Fairness is monitored, not yet optimized.

---

## 5. Architecture

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

## 6. Fit with Intel's direction

The heterogeneous device fleet (mixed CPU/GPU/NPU, scheduled by task rather than by manually targeting specific hardware) mirrors where OpenVINO's roadmap is headed. The parsing and matching components are light enough to run on-device via OpenVINO on NPU-class hardware; the CP-SAT solves and the learning loop are the kind of steady, latency-sensitive workload Xeon and AMX are positioned for. The pluggable-services shape (independent stream adapters, scheduler, policy, explainer, each addable or swappable without touching the rest) mirrors OPEA's reference architecture for production AI systems rather than a monolith.

---

## 7. What's tested, and the real numbers

- **64 automated tests**, covering: the scheduler never assigns a device to work it can't handle; the replanner correctly scopes to the affected slice for every kind of change; the policy's behavior measurably and provably shifts based on real outcomes, for both full and incremental decisions; a new data source can be added while the system is running with zero other code touched; the parser abstains rather than guesses on input it can't confidently classify.
- **Incremental replan latency**: roughly 5 to 8.5ms, scaling with how many jobs were actually affected, not with fleet size.
- **Full recompute latency**: roughly 0.10s at 50 devices, growing past a second at 150 devices.
- **Event throughput**: roughly 80,000 to 86,000 events/sec, stable from 1,000 to 50,000 events.
- **Fairness**: measured and reported honestly (Section 4); currently monitored, not optimized.

---

## 8. What's next

Deliberately not built, each a real multi-day undertaking:

- Periodic rebalancing solves, to address the fairness gap in Section 4
- A meta-controller that learns to choose between a fast approximate match and a full solve, based on system load
- A decentralized alternative where devices bid for jobs based on their own state, for fleets that can't always reach a central decision-maker
- A separate fairness-auditing check, independent of the main scheduling objective

---

## 9. Running it

```bash
pip install -e .
pytest
streamlit run dashboard/app.py
```

The dashboard shows live device and job state, lets you trigger disruptions (device failure, urgent job, rule change), register a new data source while the system runs, watch the policy's weights shift over time, and query the explainer agent directly.
