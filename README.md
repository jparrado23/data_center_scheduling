# Energy-Aware Scheduling for AI Data Centers

This repository develops an optimization framework for scheduling flexible AI
workloads in a data center with heterogeneous GPU resources, renewable
availability, electricity prices, PUE overhead, optional battery storage, and
grid-import peak charges.

The current source-of-truth model is a Gurobi MILP. It chooses:

- when each flexible job starts;
- which compatible GPU partition runs it;
- how much energy is supplied by renewables, grid import, and optional battery
  discharge;
- the resulting grid-import peak used for demand charges.

The project is also preparing heuristic and quantum comparisons: genetic
algorithm baselines, QUBO formulations, QAOA characterization, quantum
annealing, and hybrid approaches.

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
src/quantum/           QUBO, QAOA, and annealing scaffolding
tests/                 Unit and regression tests
```

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

## Notebooks

Open the toy MILP notebook:

```bash
jupyter notebook notebooks/01_milp_toy_example.ipynb
```

Solve a repo-local or processed scenario:

```bash
jupyter notebook notebooks/05_solve_processed_instance.ipynb
```

Explore Alibaba traces:

```bash
jupyter notebook notebooks/06_alibaba_2023_trace_eda.ipynb
```

Run the current genetic algorithm baseline:

```bash
jupyter notebook notebooks/07_genetic_algorithm_baseline.ipynb
```

Run Gurobi stress tests on generated Alibaba-calibrated instances:

```bash
jupyter notebook notebooks/08_gurobi_generated_instance_stress_test.ipynb
```

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
- [Quantum agent architecture](docs/QUANTUM_AGENT_ARCHITECTURE.md)
- [Data notes](data/README.md)

## Tests

Run the full test suite:

```bash
conda run -n quantum_py312 pytest
```

Current coverage includes data import, scenario building, resource
compatibility, PUE behavior, optional battery behavior, metrics, and the
instance generator.

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
- schedule, hourly, cluster, and metric extraction;
- regression tests for the current formulation.

Under development:

- Alibaba-derived instance generator redesign;
- broader MILP versus GA benchmarking;
- QUBO formulation update for the current resource model;
- QAOA and annealing experiments;
- hybrid classical/quantum decomposition strategy.
