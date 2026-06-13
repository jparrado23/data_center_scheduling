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
- Confirm whether `e_kw` is IT workload power or already facility/PUE-adjusted
  power.
- Confirm whether `t_min` and `t_max` are 1-based start-hour windows; the model
  currently uses zero-based hours.
- Confirm whether compatibility should come from `alpha_B`, `alpha_C`,
  `alpha_D` in the job files or from the existing cluster compatibility table.
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
