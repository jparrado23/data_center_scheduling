# Mathematical Assumption Alignment Meeting

Date: 2026-06-22

Purpose: align the final classical scheduling formulation before expanding the
tiny-instance checks, redesigning the instance generator, and benchmarking MILP,
GA, QUBO, and hybrid approaches.

## Meeting Goal

By the end of the meeting, we should agree on the baseline mathematical model
that becomes the source of truth for:

- Gurobi implementation,
- tiny and absurd instance validation,
- synthetic instance generation,
- GA comparison,
- future QUBO/QAOA reductions.

The main risk is building benchmarks or quantum formulations on top of a
classical model whose assumptions are still moving.

## Decisions To Align

### 1. GPU Requirement Assumption

Current working assumption:

- Every flexible job requires a positive number of GPUs.
- Non-GPU workloads are outside the first classical scheduling formulation.
- Alibaba non-GPU jobs may be useful later, but not in the first solver
  benchmark unless we explicitly add CPU-only workload support.

Decision needed:

- Confirm that all jobs in the current optimization require GPU capacity.

Recommended default:

- Keep `gpu_count_required > 0` mandatory for all optimized jobs.
- Treat non-GPU/preprocessing-only workloads as a later extension or as jobs
  that still reserve GPU resources if included.

### 2. GPU Capacity Constraint

Current working assumption:

- Each cluster or partition has aggregate GPU capacity.
- A job consumes `gpu_count_required` GPUs while active.
- For every cluster and hour:

```text
sum(active job GPU requirements on cluster k at hour t)
<= cluster GPU capacity
```

Decision needed:

- Confirm GPU capacity must always be enforced.
- Confirm there should be no manual option to silently disable it in the main
  experiments.

Recommended default:

- Enforce GPU capacity by default.
- Fail fast if job GPU requirements or cluster GPU capacities are missing.
- Allow disabling only for explicit ablation experiments, not for baseline
  results.

### 3. GPU Type Compatibility

Current working assumption:

- Jobs may specify one or more allowed GPU types.
- Clusters currently have one main GPU type.
- A job can run on a cluster only if:

```text
job allowed GPU types is empty
or cluster GPU type is in job allowed GPU types
```

Decision needed:

- Confirm whether the first formulation should remain at cluster/partition
  level, or move to GPU-type capacity buckets.

Options:

| Option | Meaning | Pros | Cons |
|---|---|---|---|
| Cluster-level with one GPU type | Each cluster has one aggregate GPU type and capacity | Simple, already close to current implementation | Less realistic for mixed-GPU clusters |
| Cluster-level with GPU-type buckets | Each cluster has capacities per GPU type | More realistic without individual GPU placement | More variables/constraints and more metadata |
| One-cluster GPU pool | Ignore clusters and schedule inside one heterogeneous GPU pool | Good for within-cluster scheduler study | Loses between-cluster energy/location interpretation |
| Node/GPU-level scheduler | Assign jobs to actual nodes or GPUs | Most realistic | Likely too complex for current thesis stage |

Recommended default:

- Keep the current cluster/partition-level formulation for the next milestone.
- Document that each partition has one main GPU type.
- Add GPU-type bucket modeling as a future extension if colleague/supervisor
  believes mixed-GPU clusters are essential.

### 4. CPU And Memory Constraints

Current working assumption:

- Jobs have CPU and memory requirements.
- Clusters have CPU and memory capacities.
- These are aggregate capacity constraints checked per cluster and hour.

For every cluster and hour:

```text
sum(active job CPU requirements) <= cluster CPU capacity
sum(active job memory requirements) <= cluster memory capacity
```

Decision needed:

- Confirm CPU and memory should be included in the baseline model.
- Confirm how to estimate CPU/memory when source traces do not provide them.

Recommended default:

- Include CPU and memory in the baseline mathematical formulation.
- For synthetic instances, generate CPU/memory consistently with GPU demand.
- For imported traces without CPU/memory, document the imputation rule and keep
  sensitivity analysis for later.

Open assumption to document:

- Current imported synthetic jobs estimate CPU and memory from GPU count.
- This is acceptable for stress testing, but not yet a validated real workload
  assumption.

### 5. Cluster Compatibility Versus Resource Compatibility

Previous approach:

- Jobs had direct compatibility flags such as `alpha_B`, `alpha_C`, `alpha_D`.

Current preferred approach:

- Compatibility should primarily come from resource profiles:
  - GPU count,
  - GPU type,
  - CPU,
  - memory,
  - inference reservation if relevant.

Decision needed:

- Should `alpha_*` remain as an additional business rule, or should it be
  replaced entirely by resource compatibility?

Recommended default:

- Use resource compatibility as the main rule.
- Keep `alpha_*` only as an optional additional restriction when provided by a
  scenario.
- Do not rely on `alpha_*` as the only compatibility model.

### 6. Inference Workloads

Current working assumption:

- Online inference is treated as fixed baseline load, not flexible jobs.
- Flexible optimized jobs are training, fine-tuning, batch inference, or
  preprocessing-like GPU jobs.

Decision needed:

- Confirm whether online inference is outside the flexible scheduler.
- Confirm whether forecasted inference should be reserved first and then the
  remaining capacity scheduled.

Recommended default:

- Keep online inference outside the first flexible scheduler.
- Represent it as baseline load and, if needed, reserved GPU capacity.
- Add forecasted inference scheduling as a later extension.

### 7. PUE Modeling

Current working assumption:

- Job power is IT power.
- Cluster capacity is IT-side power capacity.
- PUE converts total IT load into facility load before energy sourcing.

```text
IT_load_t = baseline_IT_load_t + flexible_IT_load_t
facility_load_t = PUE * IT_load_t
```

Decision needed:

- Confirm PUE should be mandatory in the model.
- Confirm units are MW throughout the optimization.

Recommended default:

- Keep all optimization quantities in MW/MWh.
- Make PUE mandatory with default `PUE = 1.0`.
- Run realistic scenarios with `PUE > 1.0`.

### 8. Renewable, Grid, And Peak Charge

Current working assumption:

- Facility load must be served by renewable energy, grid energy, and optionally
  battery discharge.
- Peak charge is based on grid import, not total facility load.

Without battery:

```text
renewable_t + grid_t = facility_load_t
renewable_t <= renewable_available_t
peak >= grid_t
```

Decision needed:

- Confirm peak charge should be applied to grid import only.
- Confirm renewable energy can be curtailed.

Recommended default:

- Charge peak on grid import only.
- Allow renewable curtailment.

### 9. Battery Modeling

Current working assumption:

- Battery is optional.
- Battery is controlled by scenario configuration.
- If enabled, it has charge power limit, discharge power limit, energy capacity,
  efficiency, initial SOC, and optional final SOC.

Energy balance with battery:

```text
renewable_t + grid_t + battery_discharge_t
= facility_load_t + battery_charge_t
```

SOC balance:

```text
SOC_t = SOC_{t-1}
        + charge_efficiency * charge_t * delta_t
        - discharge_t * delta_t / discharge_efficiency
```

Decision needed:

- Should battery be part of the baseline model or only scenario experiments?
- Should battery be allowed to charge from the grid?
- Should final SOC be a minimum, equality, or unconstrained?

Recommended default:

- Keep battery as an optional scenario knob.
- Baseline comparisons should run without battery first.
- If battery is enabled for comparative studies, use a final SOC rule to avoid
  artificially draining the battery at the end.
- Discuss whether grid charging should be allowed. If the business context is
  behind-the-meter solar storage, renewable-only charging may be more realistic.
  If the business context is tariff arbitrage, grid charging is reasonable.

## Tiny And Absurd Instance Proposals

These are not large benchmark instances. They are deliberately small cases where
we can predict the expected answer manually.

### A. One Job, One Cluster, One Feasible Start

Purpose:

- Verify assignment once, duration, power, and basic energy balance.

Expected outcome:

- Job is scheduled in the only possible slot.
- Flexible load equals job IT power during active hours.
- Facility load equals `PUE * IT load`.

### B. One Job With No Compatible GPU Type

Purpose:

- Verify GPU-type compatibility fails correctly.

Setup:

- Job requires `V100M32`.
- Only cluster has `T4`.

Expected outcome:

- Model fails before solve or is declared infeasible with a clear reason.

### C. One Job Exceeds GPU Count

Purpose:

- Verify individual job GPU requirement cannot exceed cluster GPU capacity.

Setup:

- Job requires 8 GPUs.
- Cluster has 4 GPUs.

Expected outcome:

- Job has no compatible cluster.

### D. Two Jobs Fit Separately But Not Together

Purpose:

- Verify aggregate capacity constraints.

Setup:

- Two jobs each require 4 GPUs.
- Cluster has 4 GPUs.
- Both jobs have windows that allow either overlap or separation.

Expected outcome:

- Solver schedules them in different hours if possible.
- If forced to overlap, instance is infeasible.

### E. CPU Capacity Binds While GPU Does Not

Purpose:

- Verify CPU is not ignored.

Setup:

- Two jobs fit GPU and power capacity.
- Combined CPU exceeds cluster CPU capacity.

Expected outcome:

- Jobs cannot overlap.

### F. Memory Capacity Binds While GPU Does Not

Purpose:

- Verify memory is not ignored.

Setup:

- Two jobs fit GPU, CPU, and power.
- Combined memory exceeds cluster memory capacity.

Expected outcome:

- Jobs cannot overlap.

### G. Power Capacity Binds While GPU Does Not

Purpose:

- Verify MW capacity is enforced independently from GPU count.

Setup:

- Two jobs fit GPU count.
- Combined IT power exceeds cluster IT power capacity.

Expected outcome:

- Jobs cannot overlap.

### H. PUE Equals 1 Versus PUE Greater Than 1

Purpose:

- Verify PUE is applied exactly once.

Setup:

- Same one-job instance solved twice.
- Run once with `PUE = 1.0`.
- Run once with `PUE = 1.5`.

Expected outcome:

- IT load is unchanged.
- Facility load and energy cost scale according to PUE.

### I. Renewable Cheaper Than Grid

Purpose:

- Verify renewable dispatch economics.

Setup:

- Renewable available during one candidate hour.
- Renewable price lower than grid price.

Expected outcome:

- Solver prefers scheduling into renewable-rich/cheaper hours when time windows
  allow it.

### J. Grid Cheaper Than Renewable

Purpose:

- Verify renewable is not forced if it is more expensive and curtailment is
  allowed.

Setup:

- Renewable available.
- Renewable price higher than grid price.

Expected outcome:

- Solver may curtail renewable and use grid if that is cheaper.

### K. Peak Charge Moves A Job

Purpose:

- Verify contracted-power logic uses grid import.

Setup:

- Two possible schedules have similar energy cost.
- One schedule creates a higher grid import peak above contracted power.

Expected outcome:

- Solver avoids the peak if the peak charge is large enough.

### L. Battery Disabled

Purpose:

- Verify battery variables do not affect baseline results.

Setup:

- Same scenario with battery off.

Expected outcome:

- No battery charge, discharge, or SOC values are used.
- Energy balance is renewable plus grid equals facility load.

### M. Battery Enabled With Final SOC

Purpose:

- Verify battery does not create free energy.

Setup:

- Battery initial SOC equals final SOC.
- Battery can shift energy but cannot finish emptier than it started.

Expected outcome:

- Battery may charge/discharge only if it improves cost while respecting final
  SOC.

### N. Forced Infeasible Time Window

Purpose:

- Verify infeasibility is detected for time, not energy.

Setup:

- Too many jobs forced into the same hour.
- Cluster resources are insufficient.
- No alternative start is available.

Expected outcome:

- Model is infeasible.

### O. Feasible-By-Construction Stress Case

Purpose:

- Verify the instance generator can create hard but feasible cases.

Setup:

- Start from a known feasible hidden schedule.
- Generate jobs from that schedule.
- Then shuffle windows and compatibility enough to create search complexity.

Expected outcome:

- Model has at least one known feasible solution.
- MILP/GA performance can be benchmarked without confusing hardness with
  infeasibility.

## Proposed Meeting Output

At the end of the meeting, fill this checklist:

- [ ] All optimized jobs require GPUs.
- [ ] GPU count constraints are mandatory.
- [ ] GPU type compatibility rule is agreed.
- [ ] CPU and memory constraints are included.
- [ ] Cluster/partition modeling level is agreed.
- [ ] `alpha_*` compatibility role is agreed.
- [ ] Online inference treatment is agreed.
- [ ] PUE is mandatory and MW/MWh units are confirmed.
- [ ] Peak charge is based on grid import.
- [ ] Battery is optional and its charging/final-SOC policy is agreed.
- [ ] Tiny absurd cases to implement first are selected.

