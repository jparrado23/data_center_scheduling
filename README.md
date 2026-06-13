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
AI workloads over a 24-hour horizon across heterogeneous data-center clusters.
The model uses Gurobi to choose job start times, cluster assignments, renewable
energy use, grid energy use, and peak-load behavior.

The goal is to create a reliable baseline that can answer questions such as:

- Which jobs should be shifted to cheaper or cleaner hours?
- Which cluster should run each workload type?
- How much grid energy is needed after using available renewable energy?
- What is the tradeoff between energy cost, peak demand, and scheduling
  flexibility?
- How should exact optimization compare against heuristic and future
  quantum-inspired approaches?

The quantum-related directories are present because they are part of the planned
method comparison, but that work has not started yet.

## Problem Being Solved

The system receives a set of AI jobs. Each job has:

- a workload category, such as training, inference, preprocessing, or
  fine-tuning;
- a duration;
- a power requirement;
- an earliest and latest allowed start time.

The data center contains multiple clusters. Each cluster has:

- a maximum power capacity;
- a set of compatible job categories;
- its own role in the scheduling decision.

For every hour in the planning horizon, the model also receives:

- available renewable energy;
- optional fixed baseline load;
- grid energy price;
- contracted power as a hard operational cap;
- peak demand charges based on maximum hourly load;
- renewable and peak-demand pricing parameters.

The scheduler must assign every flexible job to exactly one compatible cluster
and one feasible start time. Once a job starts, it runs continuously until its
duration is complete. Across the full schedule, cluster capacities must not be
violated. Peak demand is measured from the maximum total facility load.

## How The Project Addresses It

The current implementation formulates the scheduling task as a MILP. Binary
decision variables select flexible job start times and cluster assignments.
Continuous variables track contracted renewable consumption, grid residual
consumption, curtailment, and peak load. The objective minimizes total operator
cost, combining:

- grid energy cost;
- contracted renewable cost;
- peak-demand cost.

The model distinguishes fixed baseline load from controllable flexible demand.
For the thesis problem, inference can be represented as fixed baseline load
outside the optimized job set, while fine-tuning, training, and preprocessing
remain schedulable.

## Current MILP Capabilities

The implemented model includes:

- non-preemptive job scheduling;
- heterogeneous cluster capacities;
- job-category compatibility constraints;
- per-cluster power limits;
- contracted power as a hard operational cap;
- peak demand charges based on maximum hourly load;
- contracted renewable first, with grid consumption as residual demand;
- renewable curtailment reporting;
- peak-load minimization through the cost function;
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
- `clusters_df`: `cluster_id`, `capacity`, `compatible_categories`
- `hourly_df`: `hour`, `renewable_available`, `grid_price`, optional
  `baseline_load`
- `config`: `contracted_power` hard operational cap, `renewable_price`,
  `peak_price` demand-charge rate, `delta_t`

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
