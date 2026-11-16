# Data

This project can now run the thesis scenarios directly from repo-local data.
The shared files are stored inside the repository so notebooks and terminal
commands do not depend on sibling folders outside the project.

## Repo-local Scenario Inputs

The main thesis scenario builder uses:

- `data/instances/jobs_light.csv`
- `data/instances/jobs_tense.csv`
- `data/instances/jobs_limit.csv`
- `data/solar_profile/monthly_solar_profiles.csv`
- `docs/energy_price/marginalpdbc_20230404.1`
- `docs/energy_price/marginalpdbc_20230607.1`
- `docs/energy_price/marginalpdbc_20230927.1`

The solver combines one workload file, one OMIE price file, and the monthly
average solar profile matching the price scenario month.

Run a repo-local scenario from terminal with:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --workload tense \
  --scenario base
```

Battery storage is off by default. To enable it from the terminal, pass
`--battery` plus positive power and energy capacities:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --workload tense \
  --scenario base \
  --battery \
  --battery-power-capacity 0.05 \
  --battery-energy-capacity 0.10
```

Available workloads are `light`, `tense`, and `limit`. Available energy
scenarios are:

- `clear_sky`: April 4, 2023 prices and April monthly solar profile.
- `overcast`: June 7, 2023 prices and June monthly solar profile.
- `base`: September 27, 2023 prices and September monthly solar profile.

The default output folder is:

```text
experiments/outputs/<workload>_<scenario>/
```

It contains:

```text
schedule.csv
hourly_results.csv
cluster_hourly_results.csv
metrics.csv
```

The same repo-local scenario path is used by
`notebooks/05_solve_processed_instance.ipynb` when `BUILD_FROM_SOURCE = True`.

## Processed Inputs

Processed inputs can still be used for generated or manually assembled
instances:

- `data/processed/jobs.csv`
- `data/processed/hourly_inputs.csv`
- `data/processed/clusters.csv`
- `data/processed/job_types.csv`
- `data/processed/config.json`

The catalog files `clusters.csv` and `job_types.csv` are now initialized from the thesis assumptions. Concrete experiment instances still need `jobs.csv`, where each row is one actual job sampled or selected from the job-type ranges.

The current `jobs.csv` instance was generated with seed `42` using:

- 8 inference jobs
- 15 fine-tuning jobs
- 20 training jobs
- 15 preprocessing jobs

Regenerate it with:

```bash
conda run -n quantum_py312 python -m src.data.instances
```

Expected `jobs.csv` schema:

```text
job_id,category,duration,power,earliest_start,latest_start
```

Resource-profile jobs may also include:

```text
workload_family,gpu_type_required,gpu_count_required,cpu_required,memory_required_gb
```

`power` is the canonical model-ready IT power in MW. `power_kw` may be kept for
audit, but the MILP consumes `power`.

Shared job-instance CSV files can be converted or normalized with:

```bash
python scripts/import_job_instance.py data/instances/jobs_tense.csv --output data/processed/jobs.csv
```

The importer expects:

- `job_id`: job identifier.
- `tier`: workload type, mapped to the model `category`.
- `gpus`: dedicated GPUs required while the job runs.
- `e_kw`: execution power in kW, converted to model-ready MW `power`.
- `duration_h`: non-preemptive duration in hours.
- `t_min`, `t_max`: earliest and latest allowed start times. By default these
  are treated as 1-based source hours and converted to zero-based model hours.
- `alpha_B`, `alpha_C`, `alpha_D`: job-level compatibility with clusters B, C,
  and D. Cluster A remains reserved for inference.

The normalized output keeps `gpus`, `power_kw`, `e_kw`, and the `alpha_*`
columns alongside the model-required columns. It also adds
`gpu_count_required`, `cpu_required`, and `memory_required_gb` so the MILP can
enforce aggregate resource constraints. Current shared job files do not contain
GPU type requirements, so `gpu_type_required` is left empty unless supplied by a
trace-derived importer.

Expected `hourly_inputs.csv` schema:

```text
hour,renewable_available,grid_price
```

`baseline_load` is optional. When present, it represents fixed non-shiftable IT
load, such as inference in Zone A. The model adds it to optimized flexible IT
load, then applies PUE before renewable/grid/battery balancing.

```text
hour,renewable_available,grid_price,baseline_load
```

`renewable_available` is renewable power available in MW. The model chooses the
renewable/grid split economically from `renewable_price` and `grid_price`, while
all demand must be served. Unused renewable availability is reported as
curtailment.

OMIE `MARGINALPDBC` files are stored under `docs/energy_price/`. They can be
converted into grid prices with:

```bash
python scripts/import_marginalpdbc.py docs/energy_price/marginalpdbc_20230404.1 \
  --output data/processed/hourly_inputs.csv
```

The source format is semicolon-delimited:

```text
MARGINALPDBC;
year;month;day;hour;price_1;price_2;
...
```

The importer treats source hours as 1-based by default and converts them to
model hours `0..23`. It uses the last price column by default as `grid_price`,
while keeping both raw price columns available in the parser output for audit.
When an existing hourly file already contains a renewable profile, preserve it
and replace only prices with:

```bash
python scripts/import_marginalpdbc.py "/path/to/marginalpdbc_20230404.1" \
  --existing-hourly-inputs data/processed/hourly_inputs.csv \
  --output data/processed/hourly_inputs.csv
```

The monthly solar profile is already stored at
`data/solar_profile/monthly_solar_profiles.csv`. If the original PVGIS
time-series source is updated, regenerate monthly average hourly profiles with:

```bash
python scripts/import_solar_profile.py "/path/to/Timeseries_...csv" \
  --output data/solar_profile/monthly_solar_profiles.csv
```

The output contains one row per `month,hour` pair:

```text
month,hour,renewable_available,avg_power_kw,sample_count
```

`renewable_available` is model-ready MW, computed from PVGIS `P` in W. The
source file is a 250 kWp PV system by default. To scale the same profile to a
different dedicated plant size:

```bash
python scripts/import_solar_profile.py "/path/to/Timeseries_...csv" \
  --target-capacity-kwp 500 \
  --output data/processed/monthly_solar_profiles_500kwp.csv
```

To combine one month's solar profile with an existing price instance, preserve
the price column and replace only `renewable_available`:

```bash
python scripts/import_solar_profile.py "/path/to/Timeseries_...csv" \
  --month 4 \
  --existing-hourly-inputs data/processed/hourly_inputs.csv \
  --output data/processed/hourly_inputs.csv
```

For the main representative-day scenarios, pair the price file with the actual
same-day PVGIS solar profile rather than a monthly average:

```bash
python scripts/import_solar_profile.py "/path/to/Timeseries_...csv" \
  --date 2023-04-04 \
  --existing-hourly-inputs data/processed/hourly_inputs.csv \
  --output data/processed/hourly_inputs.csv
```

Expected `clusters.csv` legacy schema:

```text
cluster_id,capacity_kw,capacity,compatible_categories
```

`capacity_kw` preserves the source value. `capacity` is the model-ready value in MW.

Resource-profile compute partitions may also include:

```text
cluster_role,power_capacity_kw,gpu_type,gpu_count,gpu_capacity,cpu_capacity,memory_capacity_gb,reserved_for_online_inference
```

`gpu_count` and `gpu_capacity` are currently treated as equivalent GPU-capacity
columns. If jobs request GPU/CPU/memory resources, all compute partitions must
provide the corresponding positive capacity metadata; otherwise the model fails
fast rather than silently disabling the resource constraint.

Expected `job_types.csv` schema:

```text
job_type,power_min_kw,power_max_kw,power_min,power_max,max_start_delay_hours,duration_min_hours,duration_max_hours,start_window_note
```

`power_min_kw` and `power_max_kw` preserve the source values. `power_min` and `power_max` are model-ready values in MW.

Expected `config.json` schema:

```json
{
  "contracted_power": 0.222,
  "renewable_price": 40,
  "peak_price": 1000,
  "delta_t": 1,
  "pue": 1.0,
  "battery_power_capacity": 0.0,
  "battery_energy_capacity": 0.0,
  "battery_initial_soc": 0.0,
  "battery_final_soc": null,
  "battery_charge_efficiency": 0.95,
  "battery_discharge_efficiency": 0.95
}
```

`contracted_power` is a soft grid-import peak-charge threshold. The model may
exceed it, but pays `peak_price` on the maximum grid import above the contracted
threshold. Battery variables are enabled only when both battery power and energy
capacities are positive.
