# Energy-Aware AI Data Center Scheduling

This repository develops an optimization and quantum-computing research
framework for reducing the energy cost of AI data centers. The project studies
how flexible GPU workloads can be scheduled around renewable availability,
electricity prices, PUE overhead, battery storage, and grid-import peak charges.

The business goal is direct: shift flexible AI computation toward cheaper and
cleaner energy periods while respecting the physical limits of heterogeneous GPU
infrastructure. The technical challenge is that useful scheduling decisions are
combinatorial: each job may have several feasible start times and several
compatible GPU partitions, and each choice interacts with resource capacity and
energy cost over time.

The current source-of-truth model is a Gurobi MILP. It chooses:

- when each flexible job starts;
- which compatible GPU partition runs it;
- how much energy is supplied by renewables, grid import, and optional battery
  discharge;
- the resulting grid-import peak used for demand charges.

The research track then maps reduced versions of the same scheduling problem to
QUBO/Ising form for quantum and quantum-inspired methods: D-Wave quantum
annealing, Fujitsu Digital Annealer-style QUBO execution, Qiskit-based Pauli
Correlation Encoding, QAOA characterization, tensor-network methods, and hybrid
classical/quantum decomposition.

## Business and Research Objective

AI data centers increasingly face two linked constraints:

- GPU demand is high and bursty, especially for training, fine-tuning, and
  batch inference workloads.
- Energy cost and carbon intensity vary over time because of market prices,
  renewable generation, cooling overhead, and grid-connection limits.

This project asks whether a scheduler can reduce energy cost without violating
job deadlines or resource constraints. The expected product direction is an
energy-aware scheduling layer that sits above the low-level cluster scheduler:
it decides when and where flexible jobs should be placed, while respecting GPU,
CPU, memory, and power-system constraints.

The research direction is to compare classical optimization with quantum and
quantum-inspired solvers on the same scheduling structure. The MILP establishes
the trusted baseline; QUBO, annealing, PCE, tensor networks, and hybrid methods
test which parts of the problem are suitable for quantum approaches.

## What This Repository Demonstrates

- Formulating an energy-aware scheduling problem from business requirements to
  mathematical constraints.
- Implementing a Gurobi MILP with resource compatibility, PUE, renewable/grid
  energy split, optional battery behavior, and peak charges.
- Building feasible synthetic and Alibaba-calibrated instances for solver
  stress testing.
- Translating a reduced scheduling problem into QUBO form with explicit penalty
  design and decoded-schedule validation.
- Comparing classical, quantum, and quantum-inspired approaches without
  overstating current hardware capabilities.

## Model At A Glance

The scheduler receives flexible AI jobs with:

- workload family for interpretation, such as `training`, `fine_tuning`,
  `preprocessing`, or inference-like workloads;
- duration and feasible start window;
- IT power demand in MW;
- GPU type, GPU count, CPU, and memory requirements.

The data center is represented as aggregate compute partitions. In the current
Alibaba-aligned baseline, the scenario builder creates one partition per GPU
type observed in the trace summary:

```text
A10, G2, G3, P100, T4, V100M16, V100M32
```

Each partition has IT power capacity, GPU capacity, CPU capacity, and memory
capacity. The model does not assign jobs to individual machines or individual
GPUs; it is an aggregate scheduling model. Job `category` values are labels for
reporting only and do not control compatibility.

For each time slot, the model uses:

- renewable availability;
- grid price;
- fixed IT baseline load, when present;
- scalar or hourly PUE;
- renewable price, peak price, and contracted grid-import threshold.

## Static and Dynamic Scheduling Modes

The project is designed around two operating modes.

### Static Planning

The static mode solves a fixed planning horizon with known jobs, prices,
renewable forecasts, and cluster capacities. This is the current experimental
baseline and the right setting for:

- validating the mathematical formulation;
- comparing MILP, heuristics, QUBO, and quantum-inspired methods fairly;
- stress-testing solver performance as jobs, time slots, and compatibility
  options increase;
- building benchmark instances from Alibaba traces and synthetic generators.

### Dynamic Rolling Horizon

The dynamic mode is the intended operational extension. Instead of solving one
large static instance, the scheduler repeatedly optimizes the next 3-4 hours and
updates the plan every hour as forecasts change.

At each re-optimization step:

1. collect newly arrived jobs and updated job forecasts;
2. update renewable generation, electricity price, and demand forecasts;
3. fix or protect decisions that are already running or too close to execution;
4. solve the next short horizon;
5. pass the selected near-term schedule to the cluster execution layer.

This rolling-horizon design is more realistic for production data centers,
where workload arrivals and renewable forecasts are uncertain. It also creates
a natural hybrid strategy: use robust classical optimization for the full
rolling workflow, and apply QUBO/quantum methods to selected short-horizon
subproblems where the combinatorial structure is dense.

## Current MILP Capabilities

- Non-preemptive job scheduling.
- Resource-profile compatibility using GPU type, GPU count, CPU, and memory.
- Aggregate partition limits for IT power, GPU count, CPU, and memory.
- CPU and memory constraints enabled by default, with explicit ablation flags.
- Scalar or hourly PUE from IT load to facility load.
- Optional battery charge, discharge, and SOC constraints.
- Renewable/grid dispatch with renewable curtailment.
- Grid-import peak charge above contracted power.
- Result extraction for schedule, hourly energy balance, cluster load, and
  summary metrics.

## Repository Layout

```text
data/                  Input data, processed examples, solar profiles
docs/                  Formulation notes, roadmap, meeting notes, price files
experiments/           Experiment outputs and configs
notebooks/             MILP, Alibaba EDA, GA, and quantum exploration notebooks
scripts/               Import and solve entry points
src/data/              Scenario builders, data importers, validation
src/evaluation/        Result extraction, metrics, plotting
src/heuristic/         Classical heuristic placeholders and baselines
src/instance_generator/
                       Feasible-by-construction synthetic generator
src/milp/              Gurobi MILP model and solver wrapper
src/quantum/           Reduced QUBO, tensor-network, QAOA, and annealing code
tests/                 Unit and regression tests
```

## Quantum Approach

The quantum track does **not** attempt to translate the full MILP directly into
QUBO. The full MILP contains continuous dispatch variables, battery state of
charge, renewable/grid split decisions, and peak-charge logic. Encoding all of
that exactly as binary variables would make the first quantum problem too large
and difficult to validate.

Instead, the current quantum approach uses a reduced scheduling QUBO whose
purpose is to test whether quantum or quantum-inspired methods can reproduce
good feasible schedules on small but combinatorially difficult instances.

### Reduced QUBO Scope

The reduced QUBO uses binary variables:

```text
x[job, cluster, start]
```

where the variable is 1 when a job is assigned to a compatible cluster and
starts at a specific time slot.

The implemented QUBO includes:

- fixed or exogenous-PUE energy cost;
- one-hot assignment penalties so every job is scheduled exactly once;
- explicit job-to-cluster compatibility by omitting incompatible variables;
- aggregate GPU-capacity penalties using binary unused-GPU slack variables;
- a fixed-PUE squared load/peak-smoothing proxy.

The first QUBO intentionally excludes:

- exact renewable/grid dispatch variables;
- exact contracted peak billing variables;
- battery charge, discharge, and SOC dynamics;
- CPU and memory capacity penalties;
- full load-dependent PUE in the peak term.

Those features remain in the MILP baseline. Decoded QUBO schedules must
therefore be validated against the same feasibility checks used elsewhere in
the project.

### Why This Reduction Is Useful

The QUBO size grows quickly even before modeling the full MILP:

```text
assignment variables ~= jobs x compatible clusters x feasible starts
slack variables      ~= clusters x time slots x GPU slack bits
```

The number of quadratic interactions can grow faster than the number of
variables. For example, if one job has `m` feasible assignment options, its
assignment penalty creates:

```text
m choose 2
```

pairwise interactions among those options. This matters for both D-Wave and
gate-based methods because hardware must represent not only logical variables,
but also the couplers or Hamiltonian terms between them.

### Implemented Quantum Components

- `src/quantum/qubo_builder.py` builds the reduced scheduling QUBO, exports a
  `dimod.BinaryQuadraticModel`, decodes binary samples, and validates decoded
  schedules.
- `notebooks/13_qubo_step_by_step_formulation.ipynb` rebuilds the QUBO term by
  term for explanation and validation. It progressively solves:
  cost only, cost plus assignment, cost plus GPU capacity, and cost plus peak
  smoothing using D-Wave Ocean's simulated annealing sampler when available.
- `src/quantum/tensor_network.py` solves small QUBOs exactly through min-sum
  tensor-network variable elimination. This is a quantum-inspired reference
  method and a way to study whether QUBO graph structure is exploitable.
- `tests/test_qubo_builder.py` and `tests/test_tensor_network_solver.py`
  provide regression coverage for QUBO construction, compatibility handling,
  decoding, feasibility validation, and tensor-network correctness.

### QUBO Penalties and Their Meaning

The reduced QUBO has the form:

```text
H(z) = offset + linear terms + quadratic interactions
```

Each binary variable represents one scheduling option:

```text
x[job, cluster, start] = 1
```

The Hamiltonian is built from interpretable terms:

- **Energy cost**: adds a linear cost to each scheduling option according to
  power, PUE, electricity price, and active time slots. Alone, this term prefers
  scheduling nothing, so it must be combined with assignment penalties.
- **Assignment penalty**: forces every job to be scheduled exactly once. It
  rewards selecting one option for a job and penalizes selecting zero or more
  than one.
- **Compatibility**: incompatible job-cluster pairs are omitted before QUBO
  construction. This reduces variables and avoids needing a separate
  compatibility penalty.
- **GPU capacity penalty**: converts
  `used GPUs <= available GPUs` into
  `used GPUs + unused-GPU slack = available GPUs`, then squares the violation.
  This is the main physical feasibility term in the first QUBO.
- **Peak/load smoothing penalty**: penalizes concentrated load by squaring
  simultaneous IT or fixed-PUE facility load. This encourages spreading work
  when energy-cost savings do not justify a load spike.
- **PUE handling**: fixed or hourly PUE is a multiplier in the energy and peak
  terms. Load-dependent PUE is supported in the energy term when it remains
  quadratic; it is not inserted into the squared peak term because that would
  create higher-order terms outside QUBO.

The important modeling point is that constraints become penalties. Penalty
weights must be large enough that feasible schedules dominate cheap but invalid
schedules, and every decoded solution is validated after solving.

### Quantum Roadmap

The quantum roadmap is organized around a shared reduced QUBO benchmark ladder:

1. **QUBO construction and validation**: build the Hamiltonian term by term,
   verify each penalty on tiny hand-checkable instances, and validate decoded
   schedules.
2. **Classical QUBO solvers**: use simulated annealing and exact tensor-network
   contraction where tractable as controlled references.
3. **D-Wave quantum annealing**: execute the reduced QUBO as a
   `dimod.BinaryQuadraticModel`, first with local samplers and then with Leap
   hardware access.
4. **Fujitsu Digital Annealer**: use the same QUBO coefficient representation
   as a target for Digital Annealer execution and compare feasibility, energy,
   and runtime against D-Wave and Gurobi baselines.
5. **QAOA and Qiskit**: convert QUBO to Ising form and characterize small
   instances under gate-based quantum circuits.
6. **Pauli Correlation Encoding**: implement a Qiskit-based PCE path to reduce
   the number of qubits required relative to one-QUBO-variable-per-qubit QAOA.
7. **Tensor networks**: use tensor-network contraction as a quantum-inspired
   method for exact small-QUBO solving and for studying how interaction graph
   structure affects difficulty.
8. **Hybrid decomposition**: combine classical rolling-horizon optimization
   with QUBO subproblem extraction, using quantum methods only where they are
   structurally plausible.

This roadmap keeps the thesis technically grounded: quantum methods are not
presented as a replacement for Gurobi today, but as a disciplined investigation
of where quantum encodings may become useful as hardware and hybrid methods
improve.

## Setup

Use Python 3.12.

```bash
pip install -r requirements.txt
pip install -e .
```

Gurobi must be installed and licensed locally, or configured through a license
server.

The commands below assume the local conda environment used during development:

```bash
conda run -n quantum_py312 ...
```

The core project does not require D-Wave Ocean packages. For the QUBO notebook
sections that export or sample a `dimod.BinaryQuadraticModel`, install the
optional Ocean dependencies in the active environment:

```bash
pip install dimod dwave-samplers
```

D-Wave hardware execution will also require `dwave-system` and Leap
credentials; the current notebook path uses local simulation when available.

## Quick Start

Solve the default repo-local thesis scenario:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --workload tense \
  --scenario base \
  --quiet
```

The script reads:

- workload instances from `data/instances/`;
- monthly solar profiles from `data/solar_profile/monthly_solar_profiles.csv`;
- representative OMIE price files from `docs/energy_price/`.

It writes:

```text
schedule.csv
hourly_results.csv
cluster_hourly_results.csv
metrics.csv
```

under `experiments/outputs/<workload>_<scenario>/`, unless `--output-dir` is
provided.

Available workload cases:

```text
light, tense, limit
```

Available energy scenarios:

```text
clear_sky, overcast, base
```

Available cluster modes:

```text
alibaba_gpu_types, document
```

The default is `alibaba_gpu_types`. Use `document` to recover the earlier
three-partition thesis setup.

To solve a generated or processed instance directory directly:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --processed-dir experiments/generated/alibaba_100 \
  --quiet
```

The directory must contain model-ready `jobs.csv`, `hourly_inputs.csv`,
`clusters.csv`, and `config.json`.

## Main Configuration Knobs

### Resource Constraints

GPU, CPU, and memory constraints are enabled by default. For ablation
experiments:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --workload tense \
  --scenario base \
  --no-cpu-constraints \
  --no-memory-constraints
```

`--no-gpu-constraints` also exists, but GPU constraints should remain enabled
for baseline experiments.

### PUE

Use scalar PUE with:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --workload tense \
  --scenario base \
  --pue 1.2
```

If `hourly_df` contains a positive `pue` column, those hourly coefficients
override the scalar `--pue` value for each time slot.

### Battery

Battery storage is disabled by default. Enable it explicitly:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --workload tense \
  --scenario base \
  --battery \
  --battery-power-capacity 0.05 \
  --battery-energy-capacity 0.10 \
  --battery-initial-soc 0.00
```

Optional battery settings include final SOC, charge efficiency, and discharge
efficiency.

## Instance Generation

Generate an Alibaba-calibrated feasible instance:

```bash
conda run -n quantum_py312 python -m src.instance_generator.cli \
  --num-jobs 100 \
  --output-dir experiments/generated/alibaba_100 \
  --sampling-mode alibaba \
  --slot-minutes 60
```

The generator samples empirical Alibaba job profiles and then places them into a
hidden feasible schedule. It exports the hidden schedule for diagnostics, but
the solver only uses the flexible model-ready input files.

For 15-minute or 30-minute slots, regenerate the Alibaba calibration with the
same `slot_minutes` first. The generator fails fast if calibration slot length
and instance slot length differ.

## Input Schema

The model-ready tables use these core columns.

Jobs:

```text
job_id
category
workload_family
duration
power
earliest_start
latest_start
gpu_type_required
gpu_count_required
cpu_required
memory_required_gb
```

Clusters:

```text
cluster_id
capacity
cluster_role
gpu_type
gpu_count / gpu_capacity
cpu_capacity
memory_capacity_gb
```

Hourly inputs:

```text
hour
renewable_available
grid_price
baseline_load          optional
pue                    optional hourly override
```

Configuration:

```text
contracted_power
renewable_price
peak_price
delta_t
pue
battery_power_capacity
battery_energy_capacity
battery_initial_soc
battery_final_soc
battery_charge_efficiency
battery_discharge_efficiency
```

## Units

- Power is represented in MW.
- Energy is represented in MWh.
- With hourly slots, `MW * 1 hour = MWh`.
- Grid and renewable prices use currency per MWh.
- Peak price uses currency per MW.
- Cost values use the same abstract currency units as the input prices.

## Documentation

- [MILP formulation](docs/MILP_FORMULATION.md)
- [Resource compatibility](docs/RESOURCE_COMPATIBILITY.md)
- [Tiny and absurd validation plan](docs/TINY_ABSURD_INSTANCE_VALIDATION.md)
- [Project roadmap](docs/PROJECT_ROADMAP.md)
- [QUBO formulation notes](docs/QUBO_FORMULATION.md)
- [QUBO Hamiltonian formulation](docs/qubo_hamiltonian_formulation.html)
- [Quantum agent architecture](docs/QUANTUM_AGENT_ARCHITECTURE.md)
- [Pauli Correlation Encoding approach](docs/PCE_APPROACH.md)
- [Tensor-network QUBO study guide](docs/tensor_network_qubo_study_guide.html)
- [Data notes](data/README.md)

## Tests

Run the full test suite:

```bash
conda run -n quantum_py312 pytest
```

Current coverage includes data import, scenario building, resource
compatibility, PUE behavior, optional battery behavior, metrics, and the
instance generator. Quantum-specific tests cover reduced QUBO construction,
decoded-schedule validation, explicit compatibility handling, and exact
tensor-network contraction on small QUBOs.

## Current State

Implemented:

- MILP model construction and solve wrapper;
- Alibaba GPU-type aggregate cluster mode;
- resource-profile compatibility;
- aggregate power, GPU, CPU, and memory constraints;
- scalar and hourly PUE;
- optional battery modeling;
- feasible-by-construction synthetic generator;
- GA baseline notebook;
- reduced QUBO builder, decoder, and D-Wave Ocean BQM export;
- step-by-step QUBO explanation notebook with simulated annealing execution;
- exact tensor-network solver for small QUBOs;
- schedule, hourly, cluster, and metric extraction;
- regression tests for the current formulation.

Under development:

- Alibaba-derived instance generator redesign;
- broader MILP versus GA benchmarking;
- QAOA experiments on the reduced Ising/QUBO model;
- Pauli Correlation Encoding implementation;
- D-Wave hardware execution path;
- hybrid classical/quantum decomposition strategy.
