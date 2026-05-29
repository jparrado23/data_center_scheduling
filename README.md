# Energy AI Data-Center Optimization

This repository supports a master thesis project on optimization for AI data-center energy scheduling.

The long-term research roadmap covers:

1. Classical exact optimization with MILP/Gurobi.
2. Classical heuristic baselines.
3. Future QUBO, quantum, and hybrid quantum-classical implementations.

The current working deliverable is the classical MILP baseline using Gurobi. It schedules flexible AI jobs over a 24-hour horizon across four heterogeneous clusters. QUBO and quantum files are placeholders only.

## Current Implementation

The implemented MILP uses:

- Job categories: each job has a `category`, such as `training`, `inference`, or `data_processing`.
- Four heterogeneous clusters: each cluster has its own `cluster_id`, power `capacity`, and list of `compatible_categories`.
- Compatibility constraints: a job can only be assigned to a cluster that supports its category.
- Non-preemptive scheduling: once a job starts, it runs continuously for its full duration on the same cluster.
- Per-cluster capacity constraints: each cluster's active workload load must stay below that cluster's capacity.
- Total contracted-power constraint: aggregate scheduled load must stay below `contracted_power`.
- Energy-source balance: scheduled load is supplied by renewable consumption `R[t]` and grid consumption `Q[t]`.
- Peak-load cost: the model minimizes energy cost plus peak demand cost.

The base model intentionally does not include non-flexible baseline load. The first implementation focuses on controllable AI workload scheduling; fixed facility load can be added later as an extension.

## Repository Structure

```text
data/                  Raw and processed input data placeholders
docs/                  MILP formulation, QUBO placeholder, roadmap
notebooks/             Research notebooks and toy examples
src/data/              Synthetic data and future real-data loaders
src/milp/              Gurobi MILP model construction and solving
src/heuristic/         Future classical heuristic baselines
src/quantum/           Future QUBO and quantum/hybrid methods
src/evaluation/        Result extraction, metrics, and plotting
src/utils/             Shared utilities
experiments/           Configs and generated experiment outputs
tests/                 Unit tests
```

## Units

- Power variables are in MW.
- Energy over an hourly slot is MW x 1 hour = MWh.
- Grid and renewable prices are in currency per MWh.
- Peak price is in currency per MW.
- Total cost is in currency units.

## Install

Use Python 3.12. In this workspace, the intended Conda environment is `quantum_py312`.

```bash
conda activate quantum_py312
pip install -r requirements.txt
pip install -e .
```

Gurobi also requires a valid local license or access to a configured license server.

## Run The Toy MILP Notebook

```bash
jupyter notebook notebooks/01_milp_toy_example.ipynb
```

The notebook generates a small synthetic dataset with heterogeneous clusters and job-category compatibility, builds and solves the MILP, extracts the selected schedule, computes cost and load metrics, and plots the resulting hourly profiles.

The synthetic input schema is:

- `jobs_df`: `job_id`, `category`, `duration`, `power`, `earliest_start`, `latest_start`
- `clusters_df`: `cluster_id`, `capacity`, `compatible_categories`
- `hourly_df`: `hour`, `renewable_available`, `grid_price`
- `config`: `contracted_power`, `renewable_price`, `peak_price`, `delta_t`

## Current Status

- Implemented: synthetic data generation, MILP model construction, Gurobi solve wrapper, schedule extraction, metrics, plotting, and basic tests.
- Placeholder only: heuristic baselines beyond simple stubs, QUBO formulation, QAOA, and quantum annealing.

## Roadmap

See [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md).
