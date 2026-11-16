# Project Roadmap

- [x] 1. Problem definition
- [x] 2. MILP formulation
- [x] 3. Synthetic data generation
- [x] 4. MILP implementation
- [ ] 5. Classical heuristic baselines
- [ ] 6. QUBO formulation
- [ ] 7. Hybrid quantum-classical workflow
- [ ] 8. Experiments and evaluation
- [ ] 9. Thesis writing

## Discussion Checkpoints

These are the open modelling points to clarify before converting the shared data
into final experiment instances.

### Energy and Price Scenarios

- Confirm the representative-day design: base, clear-sky, and overcast days
  should pair solar production and OMIE prices from the same calendar date.
- Confirm the OMIE price column for Madrid. The `MARGINALPDBC` files include
  Portuguese and Spanish marginal prices; Madrid should use the Spanish column.
- Decide the economic treatment of PV production:
  - zero marginal cost,
  - fixed renewable/LCOE price,
  - or opportunity-cost pricing against exported/curtailed energy.
- Decide whether the model minimizes only flexible workload cost or includes
  fixed non-flexible facility load.
- Confirm the hourly time convention for PVGIS timestamps ending in `:10` and
  OMIE periods `1..24`, then document the conversion to model hours.
- Decide whether peak-demand charges are evaluated inside each 24-hour
  representative day or scaled/interpreted as a billing-period proxy.

### Workload Scenarios

- Confirm whether the shared job files intentionally exclude continuous
  inference jobs or whether inference should be added as fixed baseline load.
- Treat `e_kw` as IT workload power. The model applies PUE separately to convert
  IT load into facility load.
- Confirm whether `t_min` and `t_max` are 1-based start-hour windows; the model
  currently uses zero-based hours.
- Compatibility is now resource-profile driven: GPU type, GPU count, CPU,
  memory, and optional operational rules. Existing `alpha_B`, `alpha_C`, and
  `alpha_D` columns are retained as additional allow/deny rules for current
  shared job files.
- Clarify the role of `delta`: deadline/end time, delay allowance, or an
  explanatory field not needed by the current model.
- Treat `jobs_light.csv`, `jobs_tense.csv`, and `jobs_limit.csv` as workload
  pressure scenarios, separate from energy/price scenarios.

### Experiment Matrix

- Combine workload and energy scenarios as independent axes:
  `workload_case x energy_day`.
- Use `jobs_light.csv` as the normal/flexible baseline.
- Use `jobs_tense.csv` as the main stress-test workload.
- Use `jobs_limit.csv` as a near-limit feasibility and peak-pressure case.
- Keep scenario inputs in named folders rather than overwriting the current
  processed synthetic instance.
- Run each scenario with battery disabled by default, then repeat selected
  scenarios with explicit battery settings.
- Report IT energy, facility energy after PUE, grid import peak, renewable
  curtailment, and battery charge/discharge totals separately.

### Current Model Caveats

- The current MILP is aggregate compute-partition scheduling, not node-level
  placement or individual-GPU bin packing.
- PUE is implemented as a multiplier from IT load to facility load. It can be a
  scalar from `ModelConfig` or an hourly `pue` column in `hourly_inputs.csv`.
- Battery storage is optional and simplified: no degradation cost and no binary
  same-hour charge/discharge exclusion.
- Contracted power is modeled as a soft grid-import billing threshold, not a
  hard physical facility cap.
