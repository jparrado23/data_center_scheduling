# MILP Formulation: AI Data-Center Energy Scheduling

## 1. Problem Overview

We model the operational scheduling problem of an AI-dedicated data center that runs flexible computational workloads such as model training, fine-tuning, preprocessing, batch inference, and data preparation.

The data center has heterogeneous compute clusters. Each cluster has its own power capacity and can run only compatible job categories. The scheduling horizon is divided into hourly time slots. Flexible jobs can be shifted within their allowed start windows, but once started they run continuously on one compatible cluster until completion.

The goal is to schedule jobs to minimize total energy-related cost:

1. Contracted renewable energy cost.
2. Grid energy consumption cost.
3. Peak demand cost.

The base model explicitly captures:

- time-dependent grid prices,
- forecasted renewable availability,
- optional fixed baseline load,
- job categories,
- cluster-category compatibility,
- heterogeneous cluster capacities,
- contracted power as a soft peak-charge threshold,
- peak demand charges on load above contracted power,
- non-preemptive flexible jobs,
- renewable curtailment,
- peak demand cost.

The formulation distinguishes fixed non-shiftable baseline load from optimized
flexible load. In the thesis setting, inference and Zone A can be represented as
baseline load outside the decision variable, while fine-tuning, training, and
preprocessing remain schedulable over Zones B, C, and D.

## 2. Modelling Assumptions

1. The scheduling horizon is 24 hourly slots.
2. Each flexible workload is represented as one job.
3. Each job belongs to one category, such as `training`, `inference`, `data_processing` or `fine_tuning`
4. Each job is non-preemptive: after it starts, it runs for `d_i` consecutive hours without interruption.
5. A non-preemptive job stays on the same cluster for its full duration.
6. Each job has constant power demand while running.
7. Compute clusters are heterogeneous.
8. Each cluster has its own power capacity.
9. A compatibility matrix determines which job categories can run on which clusters.
10. Renewable availability is forecasted and exogenous.
11. Grid prices are known over the horizon.
12. Renewable price is a fixed contracted/PPA price.
13. Peak demand charge is a real economic billing term, not an artificial penalty.
14. Contracted power is a soft economic threshold.
15. Peak demand charge is applied to the maximum facility load above contracted power.

Non-preemption is relevant because the decision variable can be a compact start-time assignment `x_{i,k,s}`. If preemption were allowed, the model would need additional run-state variables by job, cluster, and hour, plus constraints for remaining processing time, migration, continuity, and possibly checkpointing overhead.

Possible future extensions include:

- finer time resolution,
- heterogeneous job power profiles,
- battery storage,
- multiple renewable sources,
- uncertainty and stochastic optimization.

## 3. Sets and Indices

Let:

$$
\mathcal{T} = \{1,\dots,24\}
$$

be the set of hourly time slots.

Let:

$$
\mathcal{I} = \{1,\dots,N\}
$$

be the set of flexible jobs.

Let:

$$
\mathcal{K} = \{1,\dots,K\}
$$

be the set of compute clusters.

Let:

$$
\mathcal{C}
$$

be the set of job categories.

Let:

$$
\mathcal{S}_i \subseteq \mathcal{T}
$$

be the set of feasible start hours for job `i`.

Indices:

- $t \in \mathcal{T}$: time slot.
- $i \in \mathcal{I}$: flexible job.
- $k \in \mathcal{K}$: compute cluster.
- $c \in \mathcal{C}$: job category.
- $s \in \mathcal{S}_i$: feasible start hour for job `i`.

## 4. Parameters

For each job $i$:

$$
c_i \in \mathcal{C}
$$

is the category of job $i$.

$$
d_i
$$

is the duration of job $i$, in hours.

$$
p_i
$$

is the power required by job $i$ while running.

The activity indicator is:

$$
A_{i,s,t} =
\begin{cases}
1, & \text{if } s \leq t \leq s+d_i-1 \\
0, & \text{otherwise}
\end{cases}
$$

This known parameter is what enforces continuous non-preemptive execution after choosing a start time.

For each cluster $k$:

$$
P^{cluster}_k
$$

is the power capacity of cluster $k$.

Compatibility can be written either by category:

$$
M_{k,c} =
\begin{cases}
1, & \text{if cluster } k \text{ can run category } c \\
0, & \text{otherwise}
\end{cases}
$$

or directly by job and cluster:

$$
C_{i,k} = M_{k,c_i}
$$

For each hour $t$:

$$
G_t
$$

is the renewable power available.

$$
B_t
$$

is the fixed non-shiftable baseline load, such as inference load.

$$
\pi^{grid}_t
$$

is the grid electricity price.

Other parameters:

$$
\pi^{ren}
$$

is the renewable electricity price. Renewable use is chosen economically against
the grid price subject to availability.

$$
\pi^{peak}
$$

is the price charged per unit of peak power demand.

$$
P^{contracted}
$$

is the contracted-power threshold above which peak charges apply.

$$
\Delta t = 1
$$

is the duration of each time slot in hours.

## 5. Decision Variables

Binary scheduling variable:

$$
x_{i,k,s} \in \{0,1\}
$$

where:

$$
x_{i,k,s}=1
$$

if job $i$ starts in cluster $k$ at hour $s$, and 0 otherwise.

Continuous energy-source variables:

$$
R_t \geq 0
$$

renewable power consumed at hour $t$.

$$
Q_t \geq 0
$$

grid power consumed at hour $t$.

Peak variables:

$$
P^{peak} \geq 0
$$

maximum data-center power reached over the horizon.

$$
E^{peak} \geq 0
$$

power above the contracted-power threshold.

## 6. Physical Relationships

### 6.1 Cluster Load

The load of cluster $k$ at hour $t$ is:

$$
L_{k,t}(x)
=
\sum_{i \in \mathcal{I}}
\sum_{s \in \mathcal{S}_i}
p_i A_{i,s,t}x_{i,k,s}
$$

This is the sum of compatible flexible jobs assigned to cluster $k$ and active at hour $t$.

### 6.2 Flexible and Total Data-Center Load

The flexible load of the data center at hour $t$ is:

$$
L^{flex}_t(x)
=
\sum_{k \in \mathcal{K}} L_{k,t}(x)
$$

Equivalently:

$$
L^{flex}_t(x)
=
\sum_{k \in \mathcal{K}}
\sum_{i \in \mathcal{I}}
\sum_{s \in \mathcal{S}_i}
p_i A_{i,s,t}x_{i,k,s}
$$

The total facility load includes fixed baseline load:

$$
L_t(x) = B_t + L^{flex}_t(x)
$$

### 6.3 Energy-Source Balance

At each hour, total facility load is served by renewable and grid energy:

$$
R_t + Q_t = L_t(x)
\quad \forall t \in \mathcal{T}
$$

### 6.4 Renewable Availability

Renewable consumption cannot exceed renewable availability:

$$
R_t \leq G_t
\quad \forall t \in \mathcal{T}
$$

Curtailment is not a decision variable in the current MILP. It is reported after
solving as:

$$
U_t = G_t - R_t
$$

where positive $U_t$ is available renewable energy not consumed by the optimal
economic dispatch.

### 6.5 Peak Load Definition

The peak variable must be at least as large as total load in every hour:

$$
P^{peak} \geq L_t(x)
\quad \forall t \in \mathcal{T}
$$

Since peak demand appears in the objective with positive coefficient
$\pi^{peak}$, the optimizer has no incentive to set $P^{peak}$ above the maximum
hourly load:

$$
P^{peak} = \max_{t \in \mathcal{T}} L_t(x)
$$

## 7. Constraints

### 7.1 Job Assignment Constraint

Each flexible job must be scheduled exactly once:

$$
\sum_{k \in \mathcal{K}}
\sum_{s \in \mathcal{S}_i}
x_{i,k,s}
=
1
\quad \forall i \in \mathcal{I}
$$

### 7.2 Compatibility Domain Reduction

Compatibility is imposed before model construction by creating variables only
for compatible job-cluster pairs:

$$
x_{i,k,s} \text{ exists only when } C_{i,k}=1
$$

### 7.3 Energy-Source Balance

$$
R_t + Q_t
=
B_t
+
\sum_{k \in \mathcal{K}}
\sum_{i \in \mathcal{I}}
\sum_{s \in \mathcal{S}_i}
p_i A_{i,s,t}x_{i,k,s}
\quad \forall t \in \mathcal{T}
$$

### 7.4 Renewable Availability

$$
R_t \leq G_t
\quad \forall t \in \mathcal{T}
$$

### 7.5 Peak-Load Relationship

$$
P^{peak}
\geq
\sum_{k \in \mathcal{K}}
\sum_{i \in \mathcal{I}}
\sum_{s \in \mathcal{S}_i}
p_i A_{i,s,t}x_{i,k,s}
\;+\;B_t
\quad \forall t \in \mathcal{T}
$$

### 7.6 Cluster-Capacity Constraint

Each heterogeneous cluster has its own capacity:

$$
\sum_{i \in \mathcal{I}}
\sum_{s \in \mathcal{S}_i}
p_i A_{i,s,t}x_{i,k,s}
\leq
P^{cluster}_k
\quad \forall k \in \mathcal{K},\; t \in \mathcal{T}
$$

### 7.7 Contracted-Power Excess Relationship

The full data-center load may exceed contracted power, but excess peak is
charged:

$$
E^{peak} \geq P^{peak} - P^{contracted}
$$

## 8. Objective Function

The objective is to minimize renewable cost, grid cost, and
peak demand cost:

$$
\min
\left[
\sum_{t \in \mathcal{T}}
\left(
\pi^{ren}R_t
+
\pi^{grid}_tQ_t
\right)\Delta t
+
\pi^{peak}E^{peak}
\right]
$$

Renewable and grid consumption are optimized economically. If grid energy is
cheaper than renewable energy in a specific hour, the model can choose grid
energy instead of consuming all available renewable energy.

## 9. Implementation Notes for Gurobi

Recommended Python inputs:

`jobs_df` columns:

```text
job_id,category,duration,power,earliest_start,latest_start
```

`hourly_df` columns:

```text
hour,renewable_available,grid_price
```

Optional:

```text
baseline_load
```

`clusters_df` columns:

```text
cluster_id,capacity,compatible_categories
```

`compatible_categories` should be a list in generated data or a comma-separated string when loaded from CSV.

`config.json` schema:

```json
{
  "contracted_power": 70,
  "renewable_price": 50,
  "peak_price": 1000,
  "delta_t": 1
}
```

Implementation functions:

1. `is_active(start, duration, hour)` computes $A_{i,s,t}$.
2. `build_feasible_starts(jobs_df)` computes $\mathcal{S}_i$.
3. `build_milp_model(jobs_df, hourly_df, clusters_df, config)` builds variables, expressions, constraints, and objective.
4. `solve_model(model)` runs Gurobi.
5. `extract_schedule(...)` and `extract_hourly_results(...)` produce readable result tables.

## 10. Units

- Power variables are in MW.
- Energy over an hourly slot is MW x 1 hour = MWh.
- Grid and renewable prices are in currency per MWh.
- Peak price is in currency per MW.
- Total cost is in currency units.

## 11. Current Scope

Do not implement QUBO in the first version.

The current goal is to establish a correct, inspectable classical MILP baseline. Once validated, later work will address:

- heuristic baselines,
- QUBO reformulation,
- variable-count analysis,
- quantum feasibility,
- hybrid quantum-classical decomposition,
- benchmarking against classical baselines.
