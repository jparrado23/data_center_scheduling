# Energy-Aware AI Data Center Scheduling

**A research-grade optimization framework for scheduling flexible AI workloads around compute capacity, electricity prices, renewable generation, facility overhead, and grid-import peaks.**

This repository is the implementation backbone of a Master's thesis at the intersection of operations research and quantum optimization. Its central question is deliberately practical:

> Can an energy-aware scheduler move flexible, multi-node AI workloads to better time slots without violating the physical and operational constraints of an AI data center?

The project starts from a complete classical mixed-integer linear programming (MILP) model in Gurobi. It then derives, validates, and benchmarks a reduced binary QUBO formulation for annealing, tensor-network, Pauli Correlation Encoding (PCE), and future gate-based quantum experiments.

The objective is not to claim a quantum advantage prematurely. It is to identify precisely which scheduling structures are difficult, which can be represented faithfully in QUBO form, and where quantum or quantum-inspired methods may become technically credible.

## Why This Problem Matters

AI data centers are constrained by more than GPU availability. A useful scheduler must reconcile:

- non-preemptive jobs with release times, durations, and deadlines;
- discrete GPU or node requirements and aggregate CPU/memory limits;
- technical compatibility between workloads and compute partitions;
- volatile electricity prices and intermittent renewable supply;
- Power Usage Effectiveness (PUE), which turns IT load into facility load;
- grid-import peak charges and, optionally, battery state of charge.

This is a decision layer above a low-level cluster scheduler. It decides **when** and **where** a flexible job should run; it does not pretend to replace Kubernetes, Slurm, or individual-node placement.

## Research Architecture

~~~
Workload traces / synthetic stress instances
                  |
                  v
    Job windows, resource demand, compatibility
                  |
                  +------------------------------+
                  |                              |
                  v                              v
        Full MILP baseline                Reduced scheduling QUBO
        Gurobi source of truth            Binary assignment + GPU slack
                  |                              |
                  v                              v
   Physical dispatch and cost          SA / D-Wave / Digital Annealer
   renewable, grid, PUE, battery       tensor networks / PCE / QAOA
                  |                              |
                  +-------------+----------------+
                                v
              Common decoded-schedule feasibility audit
              and physical energy-cost post-evaluation
~~~

The split is intentional. The MILP models the full operational problem; the QUBO isolates its discrete scheduling core. Every QUBO sample is decoded and checked against assignment windows and GPU capacity before it is treated as a candidate schedule.

## What Is Implemented

| Component | Status | What it establishes |
|---|---|---|
| Full Gurobi MILP | Implemented and tested | Feasible non-preemptive scheduling, resource constraints, energy dispatch, PUE, peak charges, and optional battery model |
| Tiny/absurd validation suite | Implemented | Hand-checkable cases validate objective and constraint behavior incrementally |
| Synthetic stress generator | Implemented | Feasible-by-construction 15-minute, multi-node A100 scheduling instances with configurable conflict pressure |
| Trace exploration | Implemented | Alibaba, Kalos, and Seren workload analyses inform realistic workload and hardware assumptions |
| Reduced QUBO builder | Implemented and tested | Explicit binary assignment, GPU-slack capacity encoding, effective-price energy term, and peak-smoothing proxy |
| Local simulated annealing | Implemented | Reproducible QUBO sampling through D-Wave Ocean's simulated annealing sampler |
| Tensor-network solver | Implemented | Exact min-sum variable elimination for small QUBOs; exposes graph-width limits rather than hiding them |
| PCE notebook | Exploratory implementation | Tests compressed Pauli-correlation encodings on small reduced QUBOs |
| QAOA and hardware runs | Planned | Kept separate until the encoding, scale, and feasibility protocol are validated |

## The Full Optimization Model

For each job \(i\), compatible compute partition \(k\), and feasible start slot \(s\), the MILP creates one binary decision:

\[
x_{i,k,s}=1 \quad \text{iff job } i \text{ starts on partition } k \text{ at slot } s.
\]

The model enforces:

- exactly one start decision per job;
- non-preemptive execution throughout the selected duration;
- explicit job-to-partition compatibility;
- aggregate GPU, CPU, memory, and IT-power capacity by partition and slot;
- facility load equal to IT load multiplied by scalar or time-varying PUE;
- renewable/grid energy balance;
- renewable availability and grid-import peak charge;
- optional battery charge, discharge, efficiency, and state-of-charge limits.

The objective minimizes renewable cost, grid cost, and excess grid-import peak cost. Contracted power is modeled as an economic threshold on **grid import**, not as a hard cap on total facility demand.

The complete formulation, notation, and assumptions are documented in [MILP_FORMULATION.md](docs/MILP_FORMULATION.md).

### Modeling Scope

The canonical model supports multiple heterogeneous compute partitions. The current quantum stress benchmark intentionally uses one A100 partition with 256 node units (2,048 A100 GPUs, eight GPUs per node). This removes multi-cluster routing from the first quantum experiment and concentrates on the hard interaction between multi-node jobs, long durations, flexible windows, and time-varying energy opportunity.

CPU and memory are represented as aggregate constraints in the MILP. This is an appropriate planning-layer approximation, not a guarantee of individual machine-level placement feasibility.

## QUBO: A Deliberate Reduction, Not a Hidden MILP Rewrite

The reduced QUBO uses binary assignment variables:

\[
x_{i,k,s}\in\{0,1\}.
\]

Its Hamiltonian is:

\[
H = H_{energy} + \lambda_{assign}H_{assign}
  + \lambda_{gpu}H_{gpu} + \lambda_{peak}H_{peak}.
\]

| Term | Purpose | Encoding |
|---|---|---|
| \(H_{energy}\) | Prefer lower-cost execution slots | Linear coefficient for each assignment option using IT power, fixed/exogenous PUE, duration, and an instance-specific effective price |
| \(H_{assign}\) | Schedule each job exactly once | One-hot squared penalty over all options of each job |
| \(H_{gpu}\) | Enforce aggregate GPU capacity | Unused-capacity binary slack converts \(D_{k,t}^{GPU}\leq G_k\) into an equality penalty |
| \(H_{peak}\) | Discourage concentrated load | Squared fixed-PUE load proxy; it is not exact demand-charge billing |

Compatibility is handled structurally: incompatible job-partition-start variables are never created. This both reduces QUBO size and avoids relying on a soft penalty for a hard technical rule.

The QUBO intentionally omits exact renewable/grid dispatch, contracted-peak billing, battery dynamics, and CPU/memory penalties. Those choices would add many auxiliary variables or higher-order terms. They remain in the MILP and in the physical post-evaluation of any decoded QUBO schedule.

Read the derivation in [QUBO_FORMULATION.md](docs/QUBO_FORMULATION.md) or walk through the small constructive example in [notebook 13](notebooks/13_qubo_step_by_step_formulation_simplified.ipynb).

## Benchmarking Discipline

The project treats feasibility, optimality, and hardware scale as different questions.

1. **Validate the formulation.** Tiny absurd instances establish expected behavior before scale experiments. See [notebook 12](notebooks/12_tiny_absurd_formulation_checks.ipynb).
2. **Generate feasible stress instances.** A hidden schedule certifies that an instance has at least one feasible schedule, but is never supplied to a solver.
3. **Measure classical difficulty.** Report Gurobi incumbent, bound, MIP gap, runtime, variable count, and constraint count rather than only an objective value.
4. **Calibrate QUBO penalties on development instances.** Keep held-out target instances out of the penalty-selection process.
5. **Audit every decoded QUBO sample.** Assignment count, start windows, and GPU capacity are hard gates before a cost comparison.
6. **Use a common physical evaluator.** Renewable use, grid import, PUE, and peak metrics are recomputed from a fixed schedule for both solver families.

### Current 15-Minute Stress Target

The benchmark pack includes long_wide_window_seed_7602, a one-cluster A100 instance designed to create conflicting energy-aware alternatives:

| Measure | Value |
|---|---:|
| Jobs | 42 |
| Horizon | 24 hours / 96 slots |
| Node capacity | 256 A100-node units |
| Assignment variables | 703 |
| GPU-slack bits | 864 |
| Logical QUBO variables | 1,567 |
| Quadratic couplers | 113,091 |
| Gurobi budget | 60 seconds |
| Gurobi outcome | Feasible incumbent; 1.02% MIP gap at time limit |

This is not presented as evidence of quantum advantage. It is a controlled target for studying the trade-off between classical optimality proof, QUBO feasibility, coupling density, and future hardware embedding.

The reproducible protocol is in [notebook 21](notebooks/21_qubo_single_cluster_15min_benchmark.ipynb).

## Quantum Research Path

The shared QUBO coefficient representation is deliberately solver-agnostic.

- **Simulated annealing:** local baseline for annealing behavior and penalty calibration.
- **D-Wave / Leap:** hardware or hybrid annealing only after recording logical graph size, coupling density, and embedding feasibility.
- **Fujitsu Digital Annealer:** same QUBO coefficients, enabling an architecture-neutral annealing comparison.
- **Tensor networks:** exact small-QUBO reference and a way to study whether interaction graph structure, not merely variable count, determines tractability.
- **PCE:** compresses many classical binary variables into Pauli-correlation observables on fewer qubits. It reduces qubit count but creates its own measurement, optimization, and post-decoding feasibility challenges.
- **QAOA:** reserved for small, verified QUBO slices where one logical binary variable per qubit is still realistic.
- **Hybrid decomposition:** the likely operational destination: classical rolling-horizon control for the full system and targeted QUBO extraction for structurally dense subproblems.

The [PCE note](docs/PCE_APPROACH.md) explains the compression hypothesis and its constraints. The [tensor-network guide](docs/tensor_network_qubo_study_guide.html) explains the complementary graph-structure perspective.

## Static and Dynamic Operation

The current experiments are static: known jobs, forecasts, and capacities over a fixed horizon. The intended deployment mode is a rolling horizon:

~~~
every hour:
  update job arrivals, energy prices, and renewable forecast
  preserve running / committed work
  optimize the next 3-4 hours
  release near-term decisions to the cluster scheduler
~~~

This division is operationally realistic and provides a disciplined place to test hybrid methods: quantum encodings are candidates for selected short, high-interaction windows, not a substitute for the full control plane.

## Repository Guide

~~~
src/milp/                    Canonical Gurobi model and solve wrapper
src/instance_generator/      Trace-calibrated and synthetic stress generators
src/quantum/                 QUBO builder, decoder, tensor network, benchmarking
notebooks/12_...             Tiny/absurd MILP validation cases
notebooks/13_...             Term-by-term QUBO construction
notebooks/20_...             One-cluster 15-minute synthetic stress generation
notebooks/21_...             QUBO calibration and held-out benchmark protocol
docs/                        Mathematical formulations and research notes
tests/                       Unit and regression tests
~~~

## Reproducibility

Requirements:

- Python 3.12 or newer;
- a licensed local or server-based Gurobi installation for MILP solves;
- D-Wave Ocean dependencies are declared for local QUBO simulation;
- Leap credentials are required only for D-Wave hardware runs.

~~~
# Install project and notebook/test dependencies
uv sync --extra dev

# Validate the repository
uv run --extra dev pytest -q

# Open the benchmark notebooks
uv run jupyter lab
~~~

The data and experiment artifacts in this repository are intentionally kept separate from source code. Generated CSV and JSON outputs are ignored by default; regenerate them from notebooks or scripts rather than treating them as opaque results.

## Further Reading

- [Full MILP formulation](docs/MILP_FORMULATION.md)
- [Reduced QUBO formulation](docs/QUBO_FORMULATION.md)
- [Resource compatibility assumptions](docs/RESOURCE_COMPATIBILITY.md)
- [Tiny-instance validation plan](docs/TINY_ABSURD_INSTANCE_VALIDATION.md)
- [PCE approach](docs/PCE_APPROACH.md)
- [Tensor-network QUBO guide](docs/tensor_network_qubo_study_guide.html)
- [Project roadmap](docs/PROJECT_ROADMAP.md)

## Research Position

This repository does not frame quantum computing as a generic replacement for classical optimization. It uses a classical model as a rigorous source of truth, derives an auditable reduced quantum encoding, measures the resulting combinatorial graph, and reports feasibility separately from objective value. That is the standard required to make an energy-aware quantum optimization claim worth taking seriously.
