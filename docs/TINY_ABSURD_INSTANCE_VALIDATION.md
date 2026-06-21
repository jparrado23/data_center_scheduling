# Tiny and Absurd Instance Validation Plan

## Purpose

Before using large synthetic or trace-derived scenarios, validate the MILP on
tiny hand-checkable instances. These cases should be small enough that the
expected schedule, feasibility status, and cost behavior can be reasoned about
manually before running Gurobi.

The goal is to catch formulation mistakes early:

- wrong compatibility logic,
- missing capacity constraints,
- incorrect PUE scaling,
- incorrect renewable/grid/battery balance,
- peak-charge mistakes,
- infeasibility caused by data rather than the formulation.

These tests are not performance benchmarks. They are mathematical sanity checks.

## General Method

For each tiny instance:

1. Define 1-3 jobs, 1-2 compute partitions, and 1-3 time slots.
2. Write the expected result by hand before solving.
3. Solve with Gurobi.
4. Check:
   - whether the model is feasible or infeasible as expected,
   - selected job start times,
   - selected partitions,
   - per-hour partition loads,
   - IT load,
   - facility load after PUE,
   - renewable/grid/battery balance,
   - objective and cost breakdown.
5. Add the case as a unit test once the expected behavior is confirmed.

Use deliberately simple numbers. Example: one job with `power = 1 MW`,
`duration = 1`, renewable availability of `1 MW`, and grid price of `100`.

## Core Assignment Cases

### 1. Single Job, Single Partition, Single Hour

Purpose: verify the assignment constraint and basic energy balance.

Expected behavior:

- One assignment variable is selected.
- Flexible load equals job power.
- Facility load equals `PUE * IT load`.
- Grid plus renewable plus battery discharge equals facility load plus battery charge.

### 2. Single Job, Two Possible Start Hours

Purpose: verify the objective can move flexible work to cheaper hours.

Setup:

- One job, duration 1.
- Start window: hour 0 or hour 1.
- Same renewable availability in both hours.
- Grid price low in hour 0, high in hour 1.

Expected behavior:

- Job starts in the cheaper hour unless renewable/PUE/battery changes the economics.

### 3. Assignment Exactly Once

Purpose: verify a job cannot be dropped or duplicated.

Expected behavior:

- Sum of all selected assignment variables for each job equals 1.
- Schedule output has exactly one row per job.

## Compatibility Cases

### 4. GPU Type Match

Purpose: verify `gpu_type_required` filters assignments.

Setup:

- One T4 partition and one G2 partition.
- Job requires `G2`.

Expected behavior:

- Variables exist only for the G2 partition.

### 5. Multiple Allowed GPU Types

Purpose: verify pipe-separated GPU requirements.

Setup:

- Job has `gpu_type_required = "T4|G2"`.

Expected behavior:

- Variables exist for both T4 and G2 partitions.

### 6. No Compatible Partition

Purpose: verify model builder fails early.

Setup:

- Job requires `V100`.
- Only T4 partition exists.

Expected behavior:

- Model construction raises a clear error before optimization.

### 7. Inference-Reserved Partition

Purpose: verify online inference reservation logic.

Setup:

- One partition has `reserved_for_online_inference = True`.
- A training job and an inference job are tested separately.

Expected behavior:

- Training job cannot use the reserved partition.
- `inference` and `online_inference` labels are allowed.

## Capacity Cases

### 8. Power Capacity Binds

Purpose: verify partition power capacity.

Setup:

- Two jobs, each 1 MW, duration 1.
- One partition has capacity 1 MW.
- Both jobs can run in either of two hours.

Expected behavior:

- Jobs are separated across hours.

### 9. GPU Capacity Binds

Purpose: verify aggregate GPU capacity.

Setup:

- Two jobs require 4 GPUs each.
- Partition has 4 GPUs.

Expected behavior:

- Jobs cannot overlap.

### 10. CPU Capacity Binds

Purpose: verify aggregate CPU capacity.

Setup:

- Two jobs fit GPU and power limits but exceed CPU capacity if simultaneous.

Expected behavior:

- Jobs are separated across hours.

### 11. Memory Capacity Binds

Purpose: verify aggregate memory capacity.

Setup:

- Two jobs fit GPU, CPU, and power limits but exceed memory capacity if simultaneous.

Expected behavior:

- Jobs are separated across hours.

### 12. Missing Resource Metadata

Purpose: verify fail-fast behavior.

Setup:

- Jobs have `cpu_required > 0`.
- A partition omits `cpu_capacity` or sets it to zero.

Expected behavior:

- Model construction raises an error instead of silently disabling CPU constraints.

## Energy and PUE Cases

### 13. PUE Equals 1

Purpose: verify backward-compatible energy balance.

Expected behavior:

- Facility load equals IT load.

### 14. PUE Greater Than 1

Purpose: verify facility-load scaling.

Setup:

- One 1 MW IT job.
- `PUE = 1.5`.

Expected behavior:

- IT load is 1 MW.
- Facility load is 1.5 MW.
- Energy sources serve 1.5 MW.

### 15. Renewable Cheaper Than Grid

Purpose: verify economic renewable dispatch.

Setup:

- Renewable availability is enough to serve all load.
- Renewable price lower than grid price.

Expected behavior:

- Renewable consumption serves the load.
- Grid consumption is zero or minimized.

### 16. Grid Cheaper Than Renewable

Purpose: verify renewable is not forced.

Setup:

- Grid price lower than renewable price.

Expected behavior:

- Model may choose grid and curtail renewable.

## Peak Charge Cases

### 17. Contracted Power Is Soft

Purpose: verify contracted power is not a hard feasibility cap.

Setup:

- Grid import must exceed contracted power.

Expected behavior:

- Model remains feasible.
- Peak excess is positive.
- Peak cost equals `peak_price * peak_over_contracted`.

### 18. Peak Charge Uses Grid Import

Purpose: verify peak charge is based on grid import, not total facility load.

Setup:

- Facility load is high but renewable serves most of it.

Expected behavior:

- Peak grid import is lower than facility peak.
- Peak cost uses grid-import peak.

## Battery Cases

### 19. Battery Disabled

Purpose: verify default behavior.

Setup:

- Battery power and energy capacities are zero.

Expected behavior:

- No battery variables appear in hourly results.
- Energy balance is renewable plus grid equals facility load.

### 20. Battery Charges In Cheap Hour

Purpose: verify price arbitrage behavior.

Setup:

- Hour 0 grid price is low.
- Hour 1 grid price is high.
- Job runs in hour 1.
- Battery capacity is enough to charge in hour 0 and discharge in hour 1.

Expected behavior:

- Battery charges in hour 0.
- Battery discharges in hour 1.
- Grid import in hour 1 decreases.

### 21. Battery Final SOC

Purpose: avoid free use of initially stored energy.

Setup:

- Battery starts with positive SOC.
- `battery_final_soc` equals initial SOC.

Expected behavior:

- Battery cannot simply deplete for free unless it recharges later.

### 22. Battery Cannot Exceed Power or Energy Capacity

Purpose: verify battery physical limits.

Expected behavior:

- `battery_charge[t] <= battery_power_capacity`.
- `battery_discharge[t] <= battery_power_capacity`.
- `0 <= battery_soc[t] <= battery_energy_capacity`.

## Feasibility Cases

### 23. Impossible Time Window

Purpose: verify infeasible windows are detected.

Setup:

- Job duration exceeds available horizon or latest start plus duration exceeds horizon.

Expected behavior:

- Data validation or model construction fails.

### 24. Insufficient Total Capacity

Purpose: verify true infeasibility.

Setup:

- Too many jobs must run in one hour for available power/GPU/CPU/memory capacity.

Expected behavior:

- Model is infeasible.

### 25. Feasible By Construction

Purpose: verify the synthetic generator creates solvable instances.

Setup:

- Use the hidden schedule from the instance generator.

Expected behavior:

- Hidden schedule satisfies all capacity limits.
- Gurobi finds a feasible solution.

## Reporting Checklist

For every tiny case, record:

- expected feasibility status,
- expected selected job starts,
- expected selected partitions,
- expected IT load,
- expected facility load,
- expected grid/renewable use,
- expected battery charge/discharge/SOC,
- expected peak grid import,
- expected objective or cost components where easy to calculate.

## Priority Order

Implement these first:

1. Single job assignment and energy balance.
2. GPU type compatibility.
3. Power/GPU/CPU/memory capacity binding.
4. PUE scaling.
5. Grid-import peak charge.
6. Battery cheap-hour charge and expensive-hour discharge.
7. Missing resource metadata fail-fast.

After these pass, use larger synthetic and Alibaba-derived instances.
