# QUBO Formulation

> Current status: this document describes the implemented reduced QUBO in
> `src/quantum/qubo_builder.py`. The canonical MILP remains the source of truth
> for full feasibility and final cost evaluation. The reduced QUBO intentionally
> omits renewable/grid dispatch variables, exact contracted-peak billing,
> battery dynamics, and exact CPU/memory/power capacity penalties.

This document derives the reduced QUBO used for early quantum and
quantum-inspired experiments on the data-center scheduling problem. It keeps
the MILP's non-preemptive start-time assignment structure, omits incompatible
job-partition choices, adds exact one-hot assignment penalties, encodes
aggregate GPU capacity with binary unused-capacity slack, and adds linear
fixed-PUE energy plus a quadratic peak-smoothing proxy.

## 1. Scope

The implemented reduced QUBO encodes:

1. job assignment,
2. feasible start windows,
3. resource or explicit cluster compatibility by variable omission,
4. linear fixed-PUE energy cost using an effective time price,
5. aggregate GPU capacity with binary slack variables,
6. a fixed-PUE facility-load smoothing proxy.

The full MILP should remain the source of truth for feasibility and cost while
QUBO experiments are developed. Exact renewable/grid dispatch, exact
contracted-peak billing, battery state of charge, and exact CPU/memory/power
capacity penalties require additional variables or higher-order encodings.

## 2. Sets and Parameters

Use the same high-level sets as the MILP:

- $\mathcal{I}$: jobs.
- $\mathcal{K}$: compute partitions.
- $\mathcal{T}$: hourly time slots.
- $\mathcal{S}_i$: feasible start slots for job $i$.

For each job:

- $p_i$: job IT power in MW.
- $d_i$: duration in hours.
- $g_i$: required GPU count.
- $A_{i,s,t}$: known activity indicator equal to 1 if job $i$ is active at hour
  $t$ after starting at hour $s$.

For each partition:

- $G_k$: aggregate GPU capacity.
- $C_{i,k}$: compatibility indicator.

For each hour:

- $\pi^{grid}_t$: grid electricity price.
- $G^{ren}_t$: available renewable power.
- $\eta^{pue}_t$: fixed or exogenous PUE multiplier.

Other parameters:

- $\pi^{ren}$: renewable price.
- $\Delta t$: time-slot length, normally 1 hour.
- $\lambda_{assign}$, $\lambda_{gpu}$, and $\lambda_{peak}$: penalty weights.

## 3. Binary Decision Variables

The core assignment variable is:

$$
x_{i,k,s} \in \{0,1\}
$$

where $x_{i,k,s}=1$ if job $i$ starts on partition $k$ at hour $s$.

Only create $x_{i,k,s}$ for $s \in \mathcal{S}_i$ and compatible job-partition
pairs. In the current builder, `compatible_clusters` takes precedence when it
is provided; otherwise compatibility falls back to GPU count, GPU type, CPU, and
memory metadata.

GPU-capacity slack variables are:

$$
y_{k,t,b} \in \{0,1\}
$$

where bit $b$ represents $2^b$ unused GPUs on partition $k$ during time slot
$t$.

## 4. Load Expressions

Flexible IT load is:

$$
L^{flex}_t(x)
=
\sum_{i,k,s}
p_i A_{i,s,t}x_{i,k,s}
$$

Facility load after PUE, restricted to the flexible jobs encoded in the QUBO,
is:

$$
L^{facility,QUBO}_t(x)
=
\eta^{pue}_t L^{flex}_t(x)
$$

Fixed baseline load is better handled in post-decode cost evaluation because it
does not affect assignment choice unless coupled to dispatch or exact peak
billing.

## 5. Energy-Cost Term

The implemented QUBO energy term approximates renewable/grid dispatch with an
effective hourly price:

$$
\pi^{eff}_t =
\begin{cases}
\min(\pi^{grid}_t,\pi^{ren}), & G^{ren}_t > 0 \\
\pi^{grid}_t, & G^{ren}_t = 0
\end{cases}
$$

The resulting linear term is:

$$
H_{energy}(x)
=
\sum_{i,k,s}
\left(
\sum_{t \in \mathcal{T}}
\pi^{eff}_t \eta^{pue}_t p_i A_{i,s,t}\Delta t
\right)
x_{i,k,s}
$$

This is intentionally a proxy. The exact MILP renewable/grid split is recovered
only by solving or re-evaluating the decoded schedule under the shared project
evaluator.

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

Incompatible variables are omitted from the QUBO:

$$
x_{i,k,s} \text{ exists only if } C_{i,k}=1
$$

For generated instances with an explicit `compatible_clusters` field, that
field is treated as the final compatibility relation by the QUBO builder and
stale GPU-type metadata is ignored. Otherwise the builder checks GPU count, GPU
type, CPU, and memory metadata during variable creation.

## 8. GPU Capacity Penalty

Aggregate GPU usage on partition $k$ at time $t$ is:

$$
D^G_{k,t}(x)
=
\sum_i\sum_s g_i A_{i,s,t}x_{i,k,s}
$$

The hard inequality $D^G_{k,t}(x) \leq G_k$ is converted into an equality with
binary unused-capacity slack:

$$
D^G_{k,t}(x)+\sum_b2^b y_{k,t,b}=G_k
$$

The QUBO penalty is:

$$
H_{GPU}(x)
=
\lambda_{gpu}
\sum_{k,t}
\left(
D^G_{k,t}(x)+\sum_b2^b y_{k,t,b}-G_k
\right)^2
$$

When GPU usage is at or below capacity, a non-negative slack assignment can
make the penalty zero. When usage exceeds capacity, no non-negative slack can
repair the equality, so the assignment is penalized.

CPU, memory, and IT power capacity are currently validated after decoding with
`validate_decoded_schedule`. They are not encoded as QUBO penalties yet.

## 9. Peak-Load Proxy

The MILP peak charge uses maximum grid import above contracted power:

$$
P^{peak} = \max_t Q_t
$$

Exact QUBO encoding needs dispatch and peak-threshold auxiliary variables. The
implemented reduced QUBO uses a quadratic fixed-PUE load-smoothing proxy:

$$
H_{peak}(x)
=
\lambda_{peak}
\sum_{t \in \mathcal{T}}
\left(
\eta^{pue}_t
\sum_{i,k,s}p_i A_{i,s,t}x_{i,k,s}
\right)^2
$$

This discourages concentrated load and tends to reduce peak demand, but it is
not identical to the MILP billing maximum.

## 10. Full Reduced QUBO

The implemented objective is:

$$
\min_{x,y}
H(x,y)
=
H_{energy}(x)
+ H_{assign}(x)
+ H_{GPU}(x,y)
+ H_{peak}(x)
$$

After solving, decode selected assignment variables into a schedule and validate
the schedule against assignment, start-window, compatibility, GPU, CPU, memory,
and power feasibility checks.

## 11. Variable Count

The assignment-variable count is:

$$
N_x =
\sum_{i \in \mathcal{I}}
|\mathcal{S}_i|
\cdot
|\mathcal{K}^{compatible}_i|
$$

The builder also adds GPU slack variables. For each cluster-hour with active
assignment choices, the number of slack bits is:

$$
\lceil\log_2(G_k+1)\rceil
$$

QUBO size therefore depends on both assignment freedom and the number of
cluster-hour GPU-capacity constraints that need slack bits.

## 12. Penalty Scaling

Penalty weights must dominate any cost improvement gained by violating a hard
constraint.

Practical starting rules:

- $\lambda_{assign}$ should be larger than the maximum possible energy-cost
  saving from dropping or duplicating any one job assignment.
- $\lambda_{gpu}$ should exceed the largest plausible energy-cost or
  peak-smoothing improvement from overloading GPU capacity in one hour.
- $\lambda_{peak}$ should be smaller than hard-constraint penalties so it
  shapes feasible schedules instead of encouraging assignment violations.

All coefficients should be normalized before passing the QUBO to annealing or
QAOA backends. Large differences between energy prices, MW values, and penalty
weights can make the QUBO numerically difficult.

## 13. Implementation Plan

The current implementation follows this path:

1. Build a QUBO coefficient dictionary from processed `jobs.csv`,
   `hourly_inputs.csv`, `clusters.csv`, and `config.json`.
2. Omit incompatible cluster variables during variable creation.
3. Add assignment penalties exactly.
4. Add linear fixed-PUE energy-cost coefficients.
5. Add GPU-capacity penalties with binary unused-capacity slack.
6. Add the peak-load proxy.
7. Decode binary samples into schedules.
8. Reuse validation and metrics code to compare decoded QUBO schedules against
   MILP feasibility and cost.

## 14. Open Questions

- Should renewable energy be represented exactly in the QUBO or handled through
  post-processing in the first version?
- Should peak demand be approximated with load smoothing or encoded with
  auxiliary threshold variables?
- Should CPU, memory, and IT power capacity be encoded with slack variables or
  left as post-decode validation checks for the first benchmark ladder?
- What coefficient normalization should be used before passing the model to
  each solver backend?
