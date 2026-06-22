# MILP Formulation: AI Data-Center Energy Scheduling

## 1. Problem Overview

We model a deterministic static scheduling problem for an AI data center with
flexible GPU jobs, heterogeneous compute partitions, time-varying grid prices,
forecast renewable availability, PUE overhead, optional battery storage, and a
contracted-power peak charge.

Layer 0 v2 keeps the compact non-preemptive assignment variable:

```text
x[i,k,s] = 1 if job i starts on compute partition k at start time s.
```

The main modeling change is that job feasibility is now driven by resource
profiles rather than high-level workload labels. Business labels such as
`training`, `fine_tuning`, or `preprocessing` remain useful for reporting and
scenario interpretation, but compatibility is based on GPU type, GPU count,
CPU, memory, and optional operational rules.

The model is still an aggregate compute-partition model. It does not solve
node-level bin packing. The current Alibaba-aligned scenario builds one
aggregate partition per GPU type. CPU and memory feasibility are approximated
through aggregate capacity constraints per partition and time slot.

## 2. Assumptions

1. The horizon is divided into discrete time slots, currently hours.
2. Each flexible job is non-preemptive and runs continuously once started.
3. A job stays on one compute partition for its full duration.
4. Each generated or imported job requires GPU capacity.
5. Job IT power is constant while active.
6. The Alibaba-aligned baseline uses seven compute partitions, one per GPU
   type observed in the trace summary: A10, G2, G3, P100, T4, V100M16, and
   V100M32.
7. GPU capacity is mandatory in the baseline. CPU and memory constraints are
   part of the formulation and enabled by default, but may be disabled for
   explicit ablation experiments.
8. GPU, CPU, and memory capacity constraints are aggregate partition-level
   approximations, not node-level placement guarantees.
9. PUE always scales IT load into facility load before energy-source balancing.
   PUE may be a scalar or a time-varying coefficient.
10. Renewable availability and grid prices are known over the horizon.
11. Battery operation is continuous at the time-slot level with fixed charge
    and discharge efficiencies.
12. Contracted power is a soft economic threshold on grid import, not a hard
    physical site limit.

## 3. Sets

- $\mathcal{I}$: flexible jobs.
- $\mathcal{K}$: compute partitions.
- $\mathcal{T}$: time slots.
- $\mathcal{S}_i$: feasible start slots for job $i$.

## 4. Job Parameters

For each job $i$:

- $d_i$: duration in time slots.
- $p_i$: IT power demand in MW.
- $g_i$: required GPU count.
- $c_i$: required CPU capacity.
- $m_i$: required memory in GB.
- $\Gamma_i$: allowed GPU types. Empty means no explicit GPU-type restriction.
- $w_i$: workload family used for reporting.
- $A_{i,s,t}$: activity indicator equal to 1 if job $i$ is active at time $t$
  when started at $s$.

The activity indicator is:

$$
A_{i,s,t} =
\begin{cases}
1, & s \leq t \leq s+d_i-1 \\
0, & \text{otherwise}
\end{cases}
$$

## 5. Compute-Partition Parameters

For each compute partition $k$:

- $P_k$: IT power capacity in MW.
- $G_k$: GPU capacity.
- $CPU_k$: aggregate CPU capacity.
- $MEM_k$: aggregate memory capacity in GB.
- $\gamma_k$: GPU type of the partition.
- $\rho_k$: optional role, such as `serving_pool` or `mainstream_gpu_pool`.

A job can run on partition $k$ if:

1. $g_i \leq G_k$,
2. $c_i \leq CPU_k$ when CPU capacity is modeled,
3. $m_i \leq MEM_k$ when memory capacity is modeled,
4. $\Gamma_i$ is empty or $\gamma_k \in \Gamma_i$,
5. optional operational rules allow the assignment.

The implementation still accepts legacy `alpha_B`, `alpha_C`, and `alpha_D`
columns as operational rules. When present, they are combined with resource
compatibility.

## 6. Energy Parameters

For each time slot $t$:

- $B_t$: fixed IT baseline load in MW.
- $G^{ren}_t$: renewable availability in MW.
- $\pi^{grid}_t$: grid price per MWh.
- $\eta^{pue}_t$: PUE multiplier. If no time-varying PUE profile is supplied,
  $\eta^{pue}_t = \eta^{pue}$ for all $t$.

Other scalar parameters:

- $\pi^{ren}$: renewable price per MWh.
- $\pi^{peak}$: peak charge per MW.
- $P^{contracted}$: contracted grid-import threshold in MW.
- $\eta^{pue}$: default scalar PUE multiplier.
- $\Delta t$: slot length in hours.

Battery parameters:

- $P^{bat}$: battery charge/discharge power capacity in MW.
- $E^{bat}$: battery energy capacity in MWh.
- $SOC_0$: initial state of charge in MWh.
- $SOC^{final}$: optional minimum final state of charge.
- $\eta^c$: charge efficiency.
- $\eta^d$: discharge efficiency.

## 7. Decision Variables

Scheduling:

$$
x_{i,k,s} \in \{0,1\}
$$

Energy-source variables:

$$
R_t \geq 0,\quad Q_t \geq 0
$$

where $R_t$ is renewable consumption and $Q_t$ is grid import.

Battery variables, enabled only when battery capacity is positive:

$$
C_t \geq 0,\quad D_t \geq 0,\quad SOC_t \geq 0
$$

where $C_t$ is battery charge power and $D_t$ is battery discharge power.

Peak variables:

$$
P^{peak} \geq 0,\quad E^{peak} \geq 0
$$

## 8. Load Expressions

Compute-partition IT load:

$$
L_{k,t}(x)
=
\sum_i \sum_{s \in \mathcal{S}_i}
p_i A_{i,s,t}x_{i,k,s}
$$

Flexible IT load:

$$
L^{flex}_t(x) = \sum_k L_{k,t}(x)
$$

Total IT load:

$$
L^{IT}_t(x) = B_t + L^{flex}_t(x)
$$

Facility load after PUE:

$$
L^{facility}_t(x) = \eta^{pue}_t L^{IT}_t(x)
$$

## 9. Constraints

Each job is scheduled exactly once:

$$
\sum_k \sum_{s \in \mathcal{S}_i} x_{i,k,s} = 1
\quad \forall i
$$

Variables are created only for compatible job-partition pairs:

$$
x_{i,k,s} \text{ exists only if } C_{i,k}=1
$$

Partition power capacity:

$$
\sum_i \sum_s p_i A_{i,s,t}x_{i,k,s} \leq P_k
\quad \forall k,t
$$

Partition GPU capacity:

$$
\sum_i \sum_s g_i A_{i,s,t}x_{i,k,s} \leq G_k
\quad \forall k,t
$$

Partition CPU capacity:

$$
\sum_i \sum_s c_i A_{i,s,t}x_{i,k,s} \leq CPU_k
\quad \forall k,t
$$

This constraint is enabled by default and may be disabled only for explicit
ablation experiments.

Partition memory capacity:

$$
\sum_i \sum_s m_i A_{i,s,t}x_{i,k,s} \leq MEM_k
\quad \forall k,t
$$

This constraint is enabled by default and may be disabled only for explicit
ablation experiments.

Energy balance without battery:

$$
R_t + Q_t = L^{facility}_t(x)
\quad \forall t
$$

Energy balance with battery:

$$
R_t + Q_t + D_t = L^{facility}_t(x) + C_t
\quad \forall t
$$

Renewable availability:

$$
R_t \leq G^{ren}_t
\quad \forall t
$$

Battery limits:

$$
0 \leq C_t \leq P^{bat},\quad
0 \leq D_t \leq P^{bat},\quad
0 \leq SOC_t \leq E^{bat}
$$

Battery state of charge:

$$
SOC_t =
SOC_{t-1}
+ \eta^c C_t\Delta t
- \frac{D_t\Delta t}{\eta^d}
$$

The first slot uses $SOC_0$ as the previous state. If a final SOC target is
configured:

$$
SOC_T \geq SOC^{final}
$$

Grid-import peak:

$$
P^{peak} \geq Q_t
\quad \forall t
$$

Contracted-power excess:

$$
E^{peak} \geq P^{peak} - P^{contracted}
$$

## 10. Objective

The model minimizes renewable energy cost, grid energy cost, and grid-import
peak cost:

$$
\min
\left[
\sum_t
\left(
\pi^{ren}R_t
+ \pi^{grid}_tQ_t
\right)\Delta t
+ \pi^{peak}E^{peak}
\right]
$$

Battery cycling currently has no degradation cost. If this becomes important,
add a small charge/discharge throughput cost.

## 11. Implementation Schema

`jobs_df` minimum columns:

```text
job_id,category,duration,power,earliest_start,latest_start
```

Resource-profile columns:

```text
workload_family,gpu_type_required,gpu_count_required,cpu_required,memory_required_gb
```

`gpu_type_required` may be empty or pipe-separated, for example:

```text
T4|G2|V100M32
```

`clusters_df` minimum legacy columns:

```text
cluster_id,capacity,compatible_categories
```

Compute-partition columns:

```text
cluster_role,power_capacity_kw,gpu_type,gpu_count,cpu_capacity,memory_capacity_gb
```

The model uses MW internally. If `power_capacity_kw` is provided, it is converted
to MW. Job imports may preserve `power_kw`, but `power` is the canonical MW
column consumed by the MILP.

`hourly_df` columns:

```text
hour,renewable_available,grid_price
```

Optional:

```text
baseline_load
```

`ModelConfig` includes:

```text
contracted_power, renewable_price, peak_price, delta_t, pue,
battery_power_capacity, battery_energy_capacity, battery_initial_soc,
battery_final_soc, battery_charge_efficiency, battery_discharge_efficiency
```

## 12. Current Scope

This formulation is still a static deterministic MILP. It does not yet include:

- rolling-horizon rescheduling,
- uncertainty,
- node-level placement,
- preemption,
- QUBO conversion of the full resource/PUE/battery model.

The immediate objective is a correct, inspectable classical baseline before
quantum or hybrid reformulations are extended.

For the implementation-level compatibility rules, see
[`RESOURCE_COMPATIBILITY.md`](RESOURCE_COMPATIBILITY.md).
