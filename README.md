# Energy-Aware Scheduling for AI Data Centers

This repository contains the code used for my master thesis work on scheduling
flexible AI workloads against data-center power and energy constraints.

At this stage the useful part of the project is the classical MILP baseline. It
uses Gurobi to schedule jobs over a 24-hour horizon on four heterogeneous
clusters. The quantum and QUBO directories are kept in the tree because they are
part of the thesis plan, but they should be treated as work in progress rather
than finished implementations.

## What The MILP Does

The model schedules jobs with a category, duration, power demand, and feasible
start window. Each job is assigned to one compatible cluster and one start time.
Once started, a job runs without interruption on the same cluster until it
finishes.

The current formulation includes:

- cluster-specific capacity limits;
- job-category compatibility constraints;
- a contracted-power limit across all clusters;
- renewable and grid energy balance variables;
- grid, renewable, and peak-demand costs in the objective.

The baseline intentionally ignores fixed, non-flexible facility load for now.
That keeps the first experiments focused on the controllable AI workload. A
fixed load profile can be added later without changing the main scheduling
structure.

## Repository Layout

```text
data/                  Raw and processed input data
docs/                  Formulation notes and project roadmap
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

The project is set up for Python 3.12. In my local workspace I use the Conda
environment `quantum_py312`.

```bash
conda activate quantum_py312
pip install -r requirements.txt
pip install -e .
```

Gurobi must also be installed and licensed locally, or configured to use a
license server.

## Running The Main Examples

The small MILP example is in:

```bash
jupyter notebook notebooks/01_milp_toy_example.ipynb
```

It builds a synthetic instance, solves it with Gurobi, extracts the selected
schedule, computes basic cost and load metrics, and plots the hourly profiles.

The synthetic input tables use the following columns:

- `jobs_df`: `job_id`, `category`, `duration`, `power`, `earliest_start`,
  `latest_start`
- `clusters_df`: `cluster_id`, `capacity`, `compatible_categories`
- `hourly_df`: `hour`, `renewable_available`, `grid_price`
- `config`: `contracted_power`, `renewable_price`, `peak_price`, `delta_t`

For a processed instance, use:

```bash
jupyter notebook notebooks/05_solve_processed_instance.ipynb
```

Point `DATA_DIR` in that notebook to a folder containing:

```text
jobs.csv
hourly_inputs.csv
clusters.csv
config.json
```

The current processed data folder also includes:

- `clusters.csv`, with the thesis cluster assumptions in both source kW and
  model-ready MW;
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

The overall thesis roadmap is tracked in
[`docs/PROJECT_ROADMAP.md`](docs/PROJECT_ROADMAP.md).
