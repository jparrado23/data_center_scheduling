# QUBO Formulation

This document derives the first QUBO draft for the AI data-center scheduling
problem. The goal is to map the validated MILP structure into an unconstrained
binary quadratic objective that can be tested with classical QUBO solvers before
any quantum or hybrid backend is used.

The formulation below is intentionally conservative. It keeps the same
non-preemptive start-time assignment used by the MILP and converts hard
constraints into quadratic penalties.

## 1. Scope

The first QUBO should encode:

1. job assignment,
2. cluster compatibility,
3. cluster capacity,
4. contracted-power capacity,
5. peak-demand pressure based on maximum facility load,
6. time-varying energy cost,
7. an optional peak-load proxy.

The first QUBO should not yet try to encode every MILP feature exactly. In
particular, exact renewable allocation and exact peak-demand billing require
auxiliary binary variables. Those can be added after the core scheduling QUBO is
validated.

## 2. Sets and Parameters

Use the same sets as the MILP:

- $\mathcal{I}$: jobs.
- $\mathcal{K}$: clusters.
- $\mathcal{T}$: hourly time slots.
- $\mathcal{S}_i$: feasible start slots for job $i$.

For each job:

- $p_i$: job power in MW.
- $d_i$: duration in hours.
- $A_{i,s,t}$: known activity indicator equal to 1 if job $i$ is active at hour
  $t$ after starting at hour $s$.

For each cluster:

- $P^{cluster}_k$: cluster power capacity.
- $C_{i,k}$: compatibility indicator.

For each hour:

- $\pi^{grid}_t$: grid electricity price.
- $G_t$: available renewable power.

Other parameters:

- $P^{contracted}$: hard operational cap on total facility load.
- $B_t$: fixed non-shiftable baseline load, such as inference.
- $\Delta t$: time-slot length, normally 1 hour.
- $\lambda_{assign}$, $\lambda_{cluster}$, $\lambda_{peak}$, and
  $\lambda_{compat}$: penalty weights.

## 3. Binary Decision Variables

The core binary variable is:

$$
x_{i,k,s} \in \{0,1\}
$$

where:

$$
x_{i,k,s}=1
$$

if job $i$ starts on cluster $k$ at hour $s$.

Only create $x_{i,k,s}$ for $s \in \mathcal{S}_i$. Incompatible cluster choices
can either be omitted from the variable set or included and penalized. Omitting
them is smaller and should be preferred for implementation.

## 4. Load Expressions

Cluster load is a linear expression in the binary variables:

$$
L_{k,t}(x)
=
\sum_{i \in \mathcal{I}}
\sum_{s \in \mathcal{S}_i}
p_i A_{i,s,t}x_{i,k,s}
$$

Flexible load is:

$$
L^{flex}_t(x)
=
\sum_{k \in \mathcal{K}} L_{k,t}(x)
$$

Total facility load is $L_t(x)=B_t+L^{flex}_t(x)$. These expressions are reused
in the objective and penalties.

## 5. Energy-Cost Term

The simplest QUBO energy-cost term charges the residual grid demand after
contracted renewable energy is absorbed. Fixed baseline load must be included in
the hourly demand before this residual is computed.

$$
H_{energy}(x)
=
\sum_{t \in \mathcal{T}}
\pi^{grid}_t Q_t(x)\Delta t
$$

where $Q_t(x)=\max(0,L_t(x)-G_t)$ is the grid residual after contracted
renewable energy is consumed. A simple effective-price approximation can be
expanded over scheduling variables:

$$
H_{energy}(x)
=
\sum_{i,k,s}
\left(
\sum_{t \in \mathcal{T}}
\pi^{grid}_t p_i A_{i,s,t}\Delta t
\right)
x_{i,k,s}
$$

This approximation is linear and therefore QUBO-compatible.

Renewable availability can be incorporated in three increasing levels of
fidelity:

1. Use an adjusted hourly price $\pi^{eff}_t$ that approximates residual grid
   cost after expected contracted renewable absorption.
2. Add auxiliary binary variables for renewable consumption or curtailment and
   enforce the renewable-first split explicitly.
3. Compare simplified QUBO schedules against exact MILP renewable-first
   accounting in post-processing.

The exact renewable-first split is closest to the MILP but increases variable
count and coefficient-scaling risk.

## 6. Assignment Penalty

Each job must be scheduled exactly once:

$$
\sum_{k \in \mathcal{K}}
\sum_{s \in \mathcal{S}_i}
x_{i,k,s}
=
1
$$

The QUBO penalty is:

$$
H_{assign}(x)
=
\lambda_{assign}
\sum_{i \in \mathcal{I}}
\left(
1 -
\sum_{k \in \mathcal{K}}
\sum_{s \in \mathcal{S}_i}
x_{i,k,s}
\right)^2
$$

Because $x^2=x$ for binary variables, this expands into linear and quadratic
terms.

## 7. Compatibility Handling

Preferred implementation: omit incompatible variables from the QUBO:

$$
x_{i,k,s} \text{ exists only if } C_{i,k}=1
$$

If incompatible variables are kept for debugging or uniform indexing, add:

$$
H_{compat}(x)
=
\lambda_{compat}
\sum_{i,k,s}
(1-C_{i,k})x_{i,k,s}
$$

This is a linear penalty that discourages incompatible assignments.

## 8. Capacity Penalties

Cluster capacity is:

$$
L_{k,t}(x) \leq P^{cluster}_k
$$

Peak demand is billed on the maximum facility load, not only on excess above
contracted power. Inequality constraints are not directly QUBO constraints.
There are two viable encodings for the hard cluster capacity terms.

### 8.1 Soft Overload Penalty

A simple first draft penalizes squared overload pressure:

$$
H_{cluster}(x)
=
\lambda_{cluster}
\sum_{k,t}
\left(
\max(0, L_{k,t}(x)-P^{cluster}_k)
\right)^2
$$

and:

$$
H_{contracted}(x)
=
\lambda_{contracted}
\sum_t
\left(
\max(0, L_t(x)-P^{contracted})
\right)^2
$$

The contracted-power term is a hard operational constraint penalty. Separately,
the peak-demand objective can use:

$$
H_{peak-load}(x)
=
\lambda_{peak}
\sum_t
\left(L_t(x)\right)^2
$$

This is a load-smoothing proxy, not an exact max-demand charge. Exact QUBO
encoding of the maximum load requires auxiliary threshold or selection bits.

### 8.2 Pairwise Conflict Approximation

For an initial practical QUBO, capacity can be approximated by penalizing pairs
of assignments that overload a cluster or the full data center when active
together.

For each pair of assignment variables $a=(i,k,s)$ and $b=(j,k,s')$ active in
the same cluster and hour, add a quadratic penalty when their combined load
contributes to capacity pressure:

$$
H_{cluster-pair}(x)
=
\sum_{a<b}
q^{cluster}_{a,b} x_a x_b
$$

with $q^{cluster}_{a,b} > 0$ when the pair is active in the same cluster-hour.

This approximation is smaller and easier to implement, but it does not exactly
enforce multi-job capacity constraints. It should be tested against MILP
feasibility checks after decoding.

## 9. Peak-Load Proxy

The MILP peak term uses:

$$
P^{peak} = \max_t L_t(x)
$$

Exact QUBO encoding needs auxiliary variables for peak thresholds or binary
load levels. For the first QUBO, use a quadratic load-smoothing proxy:

$$
H_{peak-proxy}(x)
=
\lambda_{peak}
\sum_{t \in \mathcal{T}}
L_t(x)^2
$$

This discourages concentrated load and tends to reduce peak demand, but it is
not identical to a billing maximum. Exact peak encoding should be added only
after the core assignment and capacity QUBO is stable.

## 10. Full First-Draft QUBO

The initial QUBO objective is:

$$
\min_x
H(x)
=
H_{energy}(x)
+ H_{assign}(x)
+ H_{capacity}(x)
+ H_{peak-proxy}(x)
$$

where $H_{capacity}$ is either:

- an auxiliary-bit inequality encoding, or
- a pairwise conflict approximation followed by feasibility repair.

Recommended first implementation:

$$
H(x)
=
H_{energy}(x)
+ H_{assign}(x)
+ H_{cluster-pair}(x)
+ H_{contracted}(x)
+ H_{peak-load}(x)
+ H_{peak-proxy}(x)
$$

Then decode the selected schedule and validate it with the existing MILP-style
feasibility checks.

## 11. Variable Count

The core variable count is:

$$
N_x =
\sum_{i \in \mathcal{I}}
|\mathcal{S}_i|
\cdot
|\mathcal{K}^{compatible}_i|
$$

The shared workload samples report approximate QUBO variable counts:

- `jobs_light.csv`: 1,628 variables.
- `jobs_tense.csv`: 435 variables.
- `jobs_limit.csv`: 175 variables.

This means the light workload is not necessarily the easiest QUBO instance. It
has lower energy demand, but more scheduling freedom and therefore more binary
variables.

## 12. Penalty Scaling

Penalty weights must dominate any cost improvement gained by violating a hard
constraint.

Practical starting rules:

- $\lambda_{assign}$ should be larger than the maximum possible energy-cost
  saving from dropping or duplicating any one job assignment.
- $\lambda_{cluster}$ should exceed the largest
  plausible energy-cost saving from overloading capacity in one hour.
- $\lambda_{peak}$ should be smaller than hard-constraint penalties so it shapes
  feasible schedules instead of encouraging assignment violations.

All coefficients should be normalized before passing the QUBO to annealing or
QAOA backends. Large differences between energy prices, MW values, and penalty
weights can make the QUBO numerically difficult.

## 13. Implementation Plan

1. Build a QUBO coefficient dictionary from processed `jobs.csv`,
   `hourly_inputs.csv`, `clusters.csv`, and `config.json`.
2. Omit incompatible cluster variables during variable creation.
3. Add assignment penalties exactly.
4. Add linear energy-cost coefficients.
5. Add pairwise cluster penalties and the contracted-power hard-cap penalty.
6. Add the peak-load proxy.
7. Decode binary samples into schedules.
8. Reuse existing validation and metrics code to compare decoded QUBO schedules
   against MILP feasibility and cost.

## 14. Open Questions

- Should renewable energy be represented exactly in the QUBO or handled through
  post-processing in the first version?
- Should peak demand be approximated with load smoothing or encoded with
  auxiliary threshold variables?
- Should `jobs_limit.csv` be used as a QUBO benchmark even though many starts
  are nearly fixed?
- Should the QUBO use only workload jobs or include fixed inference/facility
  load as an exogenous hourly profile?
- What coefficient normalization should be used before passing the model to
  each solver backend?
