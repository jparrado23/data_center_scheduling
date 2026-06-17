# Energy-Aware Scheduling for AI Data Centers

Progress briefing for the master thesis supervisor.

Purpose: summarize the problem definition, current Layer 0 MILP model, static and dynamic research directions, and roadmap toward QUBO/QAOA and hybrid methods.

---

## Slide 1: Thesis Motivation

- AI data centers run heterogeneous workloads such as training, fine-tuning, preprocessing, and inference.
- Some workloads are flexible: they can be shifted within a time window without changing the final service.
- Energy cost is time-dependent because grid prices, renewable availability, and peak demand charges vary across the day.
- The scheduling question is: when and where should flexible jobs run so compute demand is served at lower energy cost and lower grid stress?

Speaker note: Position the thesis as optimization for AI infrastructure under energy constraints, not only as a quantum computing exercise.

---

## Slide 2: Problem Definition

Inputs:

- Set of jobs with duration, power, start window, workload type, GPU demand, and cluster compatibility.
- Data-center clusters with power and GPU capacities.
- Hourly renewable availability and grid electricity price.
- Baseline non-flexible load, such as inference.

Outputs:

- Start time for each flexible job.
- Cluster assignment for each job.
- Hourly renewable and grid consumption.
- Peak load and peak charge above contracted power.

Goal:

```text
minimize renewable cost + grid cost + peak-excess cost
```

---

## Slide 3: Energy Considerations

- Renewable vs grid: the model can choose renewable or grid energy economically, subject to renewable availability.
- Grid prices: OMIE prices define hourly grid energy cost for representative days.
- Renewable price: a fixed renewable/PPA-like price parameter is used for renewable energy consumed.
- Peak penalty: contracted power is a soft billing threshold; exceeding it is allowed but charged.
- Baseline load: fixed inference or facility load is added to scheduled flexible workloads.

Core equations:

```text
renewable_t + grid_t = total_load_t
renewable_t <= renewable_available_t
peak_cost = peak_price * max(0, max_t(total_load_t) - contracted_power)
```

---

## Slide 4: Two Scheduling Approaches

Static scheduling:

- All jobs and energy inputs are known at the start of the day.
- A complete 24-hour schedule is optimized once.
- Status: implemented for Layer 0.
- Research role: baseline for MILP, QUBO, and quantum-inspired comparisons.

Dynamic scheduling:

- Jobs arrive or change during the day.
- The scheduler must update decisions online or in rolling horizons.
- Status: planned.
- Research role: future extension closer to real operations; candidate for hybrid and heuristic approaches.

Speaker note: Static first gives a controlled mathematical benchmark. Dynamic scheduling can be introduced after the baseline is validated.

---

## Slide 5: Layered Complexity Roadmap

Layer 0: Base scheduling

- Features: energy prices, renewable/grid split, peak penalty, cluster compatibility and capacities.
- Purpose: clean classical baseline.
- Status: current focus.

Layer 1: More realistic power/facility modeling

- Features: average vs peak power per job, and potentially PUE/facility-load adjustment.
- Purpose: more realistic power and energy accounting.
- Status: next branch.

Layer 2: Battery/storage

- Features: charge, discharge, state of charge.
- Purpose: study renewable shifting and grid peak reduction.
- Status: later.

Layer 3: Priority and tardiness

- Features: priority, soft deadlines, weighted tardiness.
- Purpose: reflect business importance of different workloads.
- Status: later.

Layer 4: Dynamic scheduling

- Features: rolling horizon or online scheduling.
- Purpose: move toward operational scheduling.
- Status: future research.

---

## Slide 6: Current Implementation Progress

Implemented:

- Repo-local job, solar, and OMIE price inputs.
- Gurobi MILP model for static Layer 0 scheduling.
- Notebook and CLI execution paths.
- Metrics for energy cost, grid cost, renewable cost, peak load, and curtailment.
- Optional GPU capacity constraints.
- Unit tests for importers, model behavior, metrics, and scenario builder.

Current scope boundaries:

- No battery yet.
- No PUE layer yet.
- No priority/tardiness yet.
- No dynamic scheduling yet.
- No QUBO/QAOA implementation yet.

Speaker note: Emphasize that we intentionally stabilized Layer 0 before adding complexity.

---

## Slide 7: Why We Use Gurobi For The Classical Baseline

- The current repository uses Gurobi for the MILP, not CBC.
- CBC notebooks from the collaborator were used to compare modeling ideas, not to replace the solver.
- Gurobi gives a stronger exact optimization baseline for proving optimality and diagnosing infeasibility.
- This is important because the thesis later compares classical exact optimization against QUBO/QAOA and hybrid methods.

Summary:

```text
Classical baseline = Gurobi MILP
Collaborator notebooks = modeling reference and layer roadmap
```

---

## Slide 8: Static Layer 0 Mathematical Model

Sets:

- `I`: flexible jobs.
- `K`: compute clusters.
- `T`: hourly time slots.
- `S_i`: feasible start times for job `i`.

Main binary decision variable:

```text
x[i,k,s] = 1 if job i starts on cluster k at hour s; 0 otherwise
```

Continuous variables:

- `R_t`: renewable energy consumed in hour `t`.
- `Q_t`: grid energy consumed in hour `t`.
- `P_peak`: maximum total load.
- `P_peak_excess`: peak load above contracted power.

---

## Slide 9: Layer 0 Constraints

Assignment:

```text
sum_{k,s} x[i,k,s] = 1 for every job i
```

Compatibility:

```text
x[i,k,s] exists only if job i is compatible with cluster k
```

Non-preemptive execution:

```text
active load is counted for every hour from start s to s + duration_i - 1
```

Cluster power capacity:

```text
cluster_load[k,t] <= cluster_capacity[k]
```

Optional GPU capacity:

```text
cluster_gpu_load[k,t] <= gpu_capacity[k]
```

Energy balance:

```text
R_t + Q_t = baseline_t + flexible_load_t
```

Renewable availability:

```text
R_t <= renewable_available_t
```

Peak excess:

```text
P_peak >= total_load_t
P_peak_excess >= P_peak - contracted_power
```

---

## Slide 10: Layer 0 Objective Function

The Layer 0 model minimizes total operating cost:

```text
minimize:
sum_t renewable_price * R_t
+ sum_t grid_price_t * Q_t
+ peak_price * P_peak_excess
```

Key interpretation:

- Renewable and grid consumption are chosen economically.
- If grid is cheaper than renewable in an hour, the model can choose grid.
- Contracted power is not a hard cap; exceeding it is allowed but charged.
- GPU constraints can be enabled or disabled depending on the experiment.

---

## Slide 11: Current Data And Scenarios

Workload pressure scenarios:

- `jobs_light`
- `jobs_tense`
- `jobs_limit`

Purpose:

- Represent increasing scheduling difficulty and tighter feasible windows.

Energy day scenarios:

- `clear_sky`
- `base`
- `overcast`

Purpose:

- Compare renewable-rich, representative, and renewable-poor days.

Solver settings:

- GPU constraints on or off.
- Renewable price.
- Peak price.
- Contracted power.

Purpose:

- Analyze sensitivity of the schedule and cost structure.

---

## Slide 12: Classical Execution Plan

1. Use Gurobi MILP to solve Layer 0 exactly for the representative instances.
2. Scale workload size, time granularity, and constraints to identify when exact MILP becomes difficult.
3. Record objective value, runtime, optimality gap, infeasibility cases, and schedule quality.
4. Use this as the classical benchmark for quantum and hybrid experiments.

Recommended wording:

```text
The goal is not to assume MILP fails.
The goal is to experimentally identify scaling limits and establish a rigorous baseline.
```

---

## Slide 13: Quantum Roadmap: QUBO And QAOA

- Translate a simplified version of Layer 0 into a QUBO.
- Start with small instances where the number of binary variables is manageable.
- Compare decoded QUBO/QAOA schedules against the Gurobi MILP optimum.
- Evaluate feasibility violations, objective quality, and runtime behavior.
- Use QAOA as proof-of-concept for small instances, not as an immediate replacement for Gurobi.

Research question:

```text
Can QUBO/QAOA reproduce good feasible schedules on small energy-aware scheduling instances?
```

---

## Slide 14: Hybrid Research Direction

- Use classical optimization for parts of the problem that are continuous or constraint-heavy.
- Use QUBO/quantum-inspired methods for selected binary assignment or timing subproblems.
- Investigate decomposition: clusters, time windows, workload groups, or rolling horizons.
- Evaluate whether hybrid methods can produce good feasible schedules faster or for larger instances.

Speaker note: Position hybrid methods as the research-oriented phase after the baseline and small QUBO experiments are credible.

---

## Slide 15: Proposed Thesis Workflow

Phase 1: Classical baseline

- Layer 0 Gurobi MILP.
- Scenario matrix.
- Scaling and runtime study.

Phase 2: Quantum proof of concept

- QUBO formulation.
- Small instances.
- QAOA comparison against MILP optimum.

Phase 3: Hybrid research

- Problem decomposition.
- Hybrid classical-quantum workflow.
- Quality, feasibility, and scalability comparison.

---

## Slide 16: Immediate Next Steps

1. Finalize and review the aligned Layer 0 model.
2. Run the full workload-by-energy scenario matrix with Gurobi.
3. Add support for `e_kw_avg` and `e_kw_peak` as the next modeling improvement.
4. Define the smallest QUBO-compatible test instances.
5. Prepare metrics for comparison: cost, feasibility, peak excess, renewable share, runtime, and optimality gap.
6. Only after this, add PUE, battery, priority, and dynamic scheduling layers.

---

## Slide 17: Key Points To Ask The Supervisor

- Is the static-first, dynamic-later structure appropriate for the thesis scope?
- Should contracted power be treated as a soft billing threshold, as currently modeled?
- Should renewable energy be economically dispatched against grid price?
- Is the roadmap from MILP baseline to QUBO/QAOA to hybrid methods academically coherent?
- Which future layer should be prioritized: average/peak power, PUE, battery, priority, or dynamic scheduling?

---

## Slide 18: Closing Message

The project has moved from general problem framing to an executable Layer 0 Gurobi MILP baseline with real job, solar, and price inputs.

The next milestone is to use this baseline to quantify classical performance and prepare the smallest valid QUBO/QAOA experiments.

Current thesis position:

```text
Build a rigorous classical baseline first.
Use it to define, test, and evaluate quantum and hybrid alternatives.
```

