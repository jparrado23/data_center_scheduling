# Energy-Aware Scheduling for AI Data Centers

AI data centers run workloads with very different timing requirements. Some
jobs must be served immediately, while others can be delayed within a limited
window without affecting the final service. At the same time, operators face
power-cap limits, volatile electricity prices, renewable availability, and peak
demand charges. The result is a scheduling problem: decide when and where to run
flexible AI jobs so that compute demand is served while energy cost and power
stress are reduced.

This project builds an optimization framework for that problem. The current
focus is a mixed-integer linear programming (MILP) model that schedules flexible
AI workloads over a 24-hour horizon across heterogeneous compute partitions.
The model uses Gurobi to choose job start times, partition assignments,
renewable energy use, grid energy use, optional battery operation, and
grid-import peak behavior.

The goal is to create a reliable baseline that can answer questions such as:

- Which jobs should be shifted to cheaper or cleaner hours?
- Which compute partition should run each workload?
- How much grid energy is needed after using available renewable energy?
- What is the tradeoff between energy cost, peak demand, and scheduling
  flexibility?
- How should exact optimization compare against heuristic and future
  quantum-inspired approaches?

The quantum-related directories are present because they are part of the planned
method comparison, but that work has not started yet.

## Problem Being Solved

The system receives a set of AI jobs. Each job has:

- a workload family, such as training, inference, preprocessing, or
  fine-tuning, used for interpretation;
- a duration;
- an IT power requirement in MW;
- an earliest and latest allowed start time;
- resource requirements such as GPU type, GPU count, CPU, and memory.

The data center contains multiple compute partitions. Each partition has:

- an IT power capacity;
- GPU type and GPU capacity;
- aggregate CPU and memory capacity;
- an operational role in the scheduling decision.

For every hour in the planning horizon, the model also receives:

- available renewable energy;
- optional fixed IT baseline load;
- grid energy price;
- contracted power as a soft grid-import peak-charge threshold;
- PUE, which scales IT load into facility load;
- renewable and peak-demand pricing parameters.

The scheduler must assign every flexible job to exactly one compatible partition
and one feasible start time. Once a job starts, it runs continuously until its
duration is complete. Across the full schedule, partition power, GPU, CPU, and
memory capacities must not be violated. Peak demand charges are measured from
maximum grid import above contracted power.

## How The Project Addresses It

The current implementation formulates the scheduling task as a MILP. Binary
decision variables select flexible job start times and cluster assignments.
Continuous variables track renewable consumption, grid consumption,
curtailment, optional battery charge/discharge/SOC, grid-import peak, and peak
import above contracted power. The objective
minimizes total operator cost, combining:

- grid energy cost;
- renewable energy cost;
- peak-demand cost.

The model distinguishes fixed baseline load from controllable flexible demand.
For the thesis problem, inference can be represented as fixed baseline load
outside the optimized job set, while fine-tuning, training, and preprocessing
remain schedulable.

## Current MILP Capabilities

The implemented model includes:

- non-preemptive job scheduling;
- heterogeneous compute-partition capacities;
- resource-profile compatibility using GPU type, GPU count, CPU, and memory;
- per-partition power, GPU, CPU, and memory limits;
- PUE scaling from IT load to facility load;
- optional battery storage with charge/discharge/SOC constraints;
- contracted power as a soft grid-import peak-charge threshold;
- peak demand charges on grid import above contracted power;
- economic renewable/grid dispatch based on input prices;
- renewable curtailment reporting;
- grid-import peak minimization through the cost function;
- result extraction, metrics, and plotting utilities.

## Repository Layout

```text
data/                  Raw and processed input data
docs/                  Formulation notes, project roadmap, and price files
notebooks/             Toy examples and experiment notebooks
src/data/              Synthetic data, processed instances, and loaders
src/milp/              Gurobi model construction and solve helpers
src/heuristic/         Early classical heuristic code
src/quantum/           QUBO and quantum-method placeholders
src/evaluation/        Result extraction, metrics, and plots
src/utils/             Shared utilities
experiments/           Experiment configs and generated outputs
tests/                 Unit tests
```

## Units

- Power is represented in MW.
- One hourly time slot converts MW directly to MWh.
- Grid and renewable prices are in currency per MWh.
- Peak price is in currency per MW.
- Reported cost values use the same abstract currency units as the input
  prices.

## Setup

Use Python 3.12.

```bash
pip install -r requirements.txt
pip install -e .
```

Gurobi must also be installed and licensed locally, or configured to use a
license server.

## Running The Main Examples

Solve one thesis scenario from terminal with:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --workload tense \
  --scenario base
```

This command reads repo-local inputs from:

- `data/instances/` for workload instances;
- `data/solar_profile/monthly_solar_profiles.csv` for monthly hourly solar
  availability;
- `docs/energy_price/` for OMIE representative-day prices.

It writes `schedule.csv`, `hourly_results.csv`,
`cluster_hourly_results.csv`, and `metrics.csv` under
`experiments/outputs/<workload>_<scenario>/` unless `--output-dir` is provided.

Available workload cases are `light`, `tense`, and `limit`. Available energy
scenarios are `clear_sky`, `overcast`, and `base`.

GPU capacity constraints are enabled by default when job and cluster GPU
columns are available. For the simplest MVP run without GPU capacity limits,
use:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --workload tense \
  --scenario base \
  --no-gpu-constraints
```

Battery storage is disabled by default. Enable it explicitly with:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --workload tense \
  --scenario base \
  --battery \
  --battery-power-capacity 0.05 \
  --battery-energy-capacity 0.10 \
  --battery-initial-soc 0.00
```

Use `--pue` to scale IT load into facility load, for example `--pue 1.2`.

Run the toy MILP notebook with:

```bash
jupyter notebook notebooks/01_milp_toy_example.ipynb
```

The notebook builds a synthetic instance, solves it with Gurobi, extracts the
selected schedule, computes cost and load metrics, and plots the hourly
profiles.

The synthetic input tables use the following columns:

- `jobs_df`: `job_id`, `category`, `duration`, `power`, `earliest_start`,
  `latest_start`
- `clusters_df`: `cluster_id`, `capacity`, `compatible_categories`, plus
  optional resource columns such as `cluster_role`, `gpu_type`, `gpu_count`,
  `cpu_capacity`, and `memory_capacity_gb`
- `hourly_df`: `hour`, `renewable_available`, `grid_price`, optional
  `baseline_load`
- `config`: `contracted_power` soft peak-charge threshold, `renewable_price`,
  `peak_price` excess-demand rate, `delta_t`, `pue`, and optional battery
  capacity/efficiency settings

To solve a processed instance, use:

```bash
jupyter notebook notebooks/05_solve_processed_instance.ipynb
```

By default, that notebook builds the same repo-local thesis scenario as the
terminal script. Set `BUILD_FROM_SOURCE = False` only if you want to solve a
preassembled processed folder containing:

```text
jobs.csv
hourly_inputs.csv
clusters.csv
config.json
```

The current processed data folder also includes:

- `clusters.csv`, with cluster assumptions in both source kW and model-ready
  MW;
- `job_types.csv`, with power, duration, and start-delay ranges for each job
  type;
- `jobs.csv`, a generated instance based on those job-type ranges.

More detail on the data files is in `data/README.md`.

## Current State

Implemented:

- synthetic data generation;
- processed-instance generation;
- MILP model construction;
- resource-profile compatibility and aggregate capacity constraints;
- PUE and optional battery modeling;
- Gurobi solve wrapper;
- schedule extraction;
- load and cost metrics;
- plotting helpers;
- basic tests.

Still under development:

- stronger heuristic baselines;
- the full QUBO formulation;
- QAOA and quantum annealing experiments;
- broader experiment automation.

The project roadmap is tracked in
[`docs/PROJECT_ROADMAP.md`](docs/PROJECT_ROADMAP.md).
