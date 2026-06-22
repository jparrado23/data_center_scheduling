# Energy-Aware Scheduling for AI Data Centers

Progress briefing for the master thesis supervisor.

Purpose: summarize the current deterministic MILP baseline, the resource-aware
job/compute-partition model, energy features, and roadmap toward validation,
Alibaba-derived scenarios, and later QUBO/hybrid methods.

---

## Slide 1: Thesis Motivation

- AI data centers run heterogeneous workloads such as training, fine-tuning,
  preprocessing, batch inference, and online inference.
- Some workloads are flexible and can be shifted within a time window.
- Online inference is usually non-flexible and should be forecast, reserved, or
  represented as baseline load before scheduling flexible work.
- Energy cost depends on grid prices, renewable availability, facility overhead,
  battery operation, and grid-import peak charges.

Speaker note: Position the thesis as optimization for AI infrastructure under
energy constraints, not only as a quantum computing exercise.

---

## Slide 2: Current Problem Definition

Inputs:

- Flexible jobs with duration, IT power, start window, GPU type requirements,
  GPU count, CPU, memory, and workload family.
- Heterogeneous compute partitions with IT power capacity, GPU type/count,
  aggregate CPU and memory capacity, and operational role.
- Hourly renewable availability and grid electricity price.
- Optional fixed IT baseline load, such as forecast inference demand.
- PUE, which scales IT load into facility load.
- Optional battery capacity, power limit, efficiencies, and initial/final SOC.

Outputs:

- Start time for each flexible job.
- Compute partition assignment for each job.
- Hourly renewable and grid consumption.
- Optional battery charge, discharge, and state of charge.
- Grid-import peak and peak charge above contracted power.

Goal:

```text
minimize renewable cost + grid cost + grid-import peak-excess cost
```

---

## Slide 3: Modeling Position

The current branch is a deterministic aggregate MILP baseline.

It is not:

- node-level bin packing,
- individual GPU placement,
- rolling-horizon scheduling,
- stochastic optimization,
- a full production battery dispatch model.

It is:

- a static 24-hour benchmark,
- a resource-aware compute-partition scheduler,
- an energy-aware MILP with PUE and optional battery,
- a classical baseline for later QUBO/hybrid comparison.

---

## Slide 4: Job and Compute-Partition Compatibility

The model separates business interpretation from feasibility.

Business/reporting label:

```text
workload_family
```

Operational feasibility profile:

```text
gpu_type_required
gpu_count_required
cpu_required
memory_required_gb
power
duration
earliest_start
latest_start
```

Compatibility checks:

- required GPU count fits partition GPU capacity,
- required GPU type is allowed by the partition GPU type,
- CPU and memory fit aggregate partition capacities,
- online-inference reservations are respected,
- legacy `alpha_B`, `alpha_C`, `alpha_D` rules are respected when present.

Reference: `docs/RESOURCE_COMPATIBILITY.md`.

---

## Slide 5: Energy Model

The model distinguishes IT load from facility load.

```text
IT_load_t = baseline_IT_t + scheduled_flexible_IT_t
facility_load_t = PUE * IT_load_t
```

Energy balance without battery:

```text
renewable_t + grid_t = facility_load_t
```

Energy balance with battery:

```text
renewable_t + grid_t + battery_discharge_t
= facility_load_t + battery_charge_t
```

Renewable use is economic, not forced:

```text
renewable_t <= renewable_available_t
```

If grid is cheaper than renewable in an hour, the model may choose grid and
curtail renewable.

---

## Slide 6: Peak and Battery Interpretation

Contracted power is modeled as a soft billing threshold on grid import:

```text
P_peak >= grid_t
P_peak_excess >= P_peak - contracted_power
peak_cost = peak_price * P_peak_excess
```

It is not a hard facility power cap.

Battery state of charge:

```text
SOC_t = SOC_{t-1}
      + charge_efficiency * charge_t
      - discharge_t / discharge_efficiency
```

Battery is optional and disabled by default. It is enabled only when requested
with positive power and energy capacities.

Current simplifications:

- no battery degradation cost,
- no explicit binary constraint preventing same-hour charge and discharge,
- final SOC is optional and currently modeled as a minimum when provided.

---

## Slide 7: Current Implementation Progress

Implemented:

- Repo-local job, solar, and OMIE price inputs.
- Gurobi MILP model for static resource-aware scheduling.
- Aggregate power, GPU, CPU, and memory capacity constraints.
- PUE scaling from IT load to facility load.
- Optional battery variables and SOC constraints.
- Grid-import peak charge.
- CLI controls for PUE and battery.
- Synthetic feasible-by-construction instance generator.
- Alibaba 2023 trace EDA notebook for workload grouping exploration.
- Tiny/absurd instance validation plan.
- Unit tests for importers, resource compatibility, model behavior, metrics, and
  scenario builder.

Current scope boundaries:

- No node-level or individual-GPU placement.
- No rolling-horizon scheduling.
- No uncertainty/forecast-error modeling.
- No priority/tardiness layer yet.
- QUBO/QAOA has not been updated to the full resource/PUE/battery model.

---

## Slide 8: Why Gurobi First

- Gurobi gives an exact classical baseline for feasibility, objective value, and
  optimality gap.
- A correct MILP baseline is needed before QUBO, QAOA, or hybrid decomposition.
- Tiny and absurd instances are being used to validate the formulation before
  scaling experiments.

Summary:

```text
Validated classical baseline first.
Scaling and quantum/hybrid comparison second.
```

---

## Slide 9: Static MILP Variables

Binary scheduling variable:

```text
x[i,k,s] = 1 if job i starts on compute partition k at start time s
```

Continuous energy variables:

```text
R_t = renewable consumption
Q_t = grid import
P_peak = maximum grid import
P_peak_excess = grid import above contracted power
```

Optional battery variables:

```text
charge_t
discharge_t
SOC_t
```

---

## Slide 10: Main Constraints

Assignment:

```text
sum_{k,s} x[i,k,s] = 1
```

Compatibility:

```text
x[i,k,s] exists only when resources and operational rules allow job i on partition k
```

Partition capacity:

```text
power_load[k,t] <= power_capacity[k]
gpu_load[k,t] <= gpu_capacity[k]
cpu_load[k,t] <= cpu_capacity[k]
memory_load[k,t] <= memory_capacity[k]
```

Energy balance:

```text
renewable_t + grid_t + discharge_t = facility_load_t + charge_t
```

Grid-import peak:

```text
P_peak >= grid_t
P_peak_excess >= P_peak - contracted_power
```

---

## Slide 11: Current Data and Scenarios

Workload pressure scenarios:

- `jobs_light`
- `jobs_tense`
- `jobs_limit`

Energy day scenarios:

- `clear_sky`
- `base`
- `overcast`

Experiment controls:

- PUE.
- Battery on/off, capacity, efficiency, SOC.
- Renewable price.
- Grid price scenario.
- Peak price and contracted grid-import threshold.
- GPU constraints on/off for ablation.

---

## Slide 12: Alibaba Trace Direction

Alibaba GPU traces are useful for realism because they provide:

- GPU demand,
- GPU type constraints,
- CPU and memory requests,
- runtime distributions,
- arrival/scheduling/completion timestamps,
- QoS and phase fields.

They do not directly provide:

- job electrical power,
- PUE,
- electricity prices,
- renewable availability.

Recommended role:

```text
Use Alibaba to calibrate workload/resource distributions.
Use external assumptions for energy/power.
Use synthetic generation for controlled Gurobi scaling tests.
```

---

## Slide 13: Validation Plan

Before large scenarios, validate tiny hand-checkable cases:

- single job assignment,
- GPU type compatibility,
- power/GPU/CPU/memory capacity binding,
- PUE scaling,
- renewable vs grid economic dispatch,
- grid-import peak charge,
- battery charge/discharge/SOC,
- infeasibility and fail-fast metadata checks.

Reference: `docs/TINY_ABSURD_INSTANCE_VALIDATION.md`.

---

## Slide 14: QUBO and Hybrid Roadmap

The QUBO formulation should not immediately mirror the full MILP.

Recommended path:

1. Start with a simplified assignment/timing QUBO.
2. Validate decoded schedules against the full MILP.
3. Add capacity and energy penalties gradually.
4. Consider decomposition:
   - between compute partitions,
   - within one GPU-type pool,
   - time-window subproblems,
   - workload-family subproblems.

Research question:

```text
Can simplified QUBO or hybrid methods reproduce good feasible schedules on small energy-aware instances?
```

---

## Slide 15: Immediate Next Steps

1. Share current branch with colleague for modeling review.
2. Complete tiny/absurd instance test suite.
3. Decide whether the next GPU modeling step is:
   - aggregate partitions with one GPU type,
   - partitions with GPU-type capacity buckets,
   - one-cluster GPU-type pool,
   - or node/GPU-level decomposition.
4. Calibrate CPU, memory, duration, and GPU-type assumptions using Alibaba traces.
5. Run workload-by-energy scenario matrix with and without battery.
6. Update QUBO notes only after the classical formulation stabilizes.

---

## Slide 16: Questions For Supervisor

- Is aggregate compute-partition scheduling the right abstraction for the thesis
  baseline?
- Should contracted power remain a grid-import billing threshold, or should we
  also add a hard site power cap?
- Should battery final SOC be equality rather than minimum for fair comparisons?
- Should battery be allowed to charge from grid, or only from surplus renewable?
- What GPU modeling level is most defensible: partition, GPU-type bucket, or
  node/GPU-level decomposition?
- Which Alibaba-derived workload features should be used for the first
  real-data scenarios?

---

## Slide 17: Closing Message

The project has moved from a category-based toy scheduling model to a
resource-aware, energy-aware deterministic MILP baseline.

Current thesis position:

```text
Validate the classical formulation rigorously.
Use synthetic and Alibaba-derived instances for scaling and realism.
Then build simplified QUBO/hybrid comparisons against the MILP baseline.
```
