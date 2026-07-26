# Formulation v0.2 Alignment Table

This note compares the formulation proposed in `Formulation_v0.2.docx` with
the current repository implementation. The goal is to decide what should be
adopted now, what should be deferred, and what should remain only in the
research roadmap.

## Summary Recommendation

Treat v0.2 as a publishable formulation roadmap, not as a single immediate
implementation task. The current MILP is a correct baseline with resource
compatibility, aggregate GPU/CPU/memory constraints, PUE, optional battery, and
grid-import peak charge. The v0.2 proposal adds richer physical realism, but
several parts introduce new binary variables, big-M constraints, calibration
requirements, and QUBO complexity.

Recommended order:

1. Keep the current core assignment/resource model.
2. Add soft-deadline weighted tardiness.
3. Improve PUE first with exogenous time-varying or thermal coefficients.
4. Add battery no-simultaneous-charge/discharge if battery scenarios become
   important.
5. Defer load-dependent PUE tranches and SOC-dependent battery efficiency until
   baseline experiments are stable.
6. Keep the first QUBO/PCE target simpler than the full MILP.

## Alignment Table

| v0.2 Component | Current Implementation Status | Recommendation | Reason | Required Data | Implementation Risk |
|---|---|---|---|---|---|
| Core binary assignment `x[job, partition, start]` | Implemented as `x[job_id, cluster, start]` in `src/milp/gurobi_model.py`. | Keep | This is the right non-preemptive scheduling primitive and already supports domain reduction. | Jobs, compatible partitions, feasible starts. | Low |
| Domain reduction before solving | Implemented for feasible starts and compatible clusters. | Keep and emphasize | Critical for Gurobi, QUBO, QAOA, and PCE scalability. | Job windows, GPU type, GPU count, CPU, memory, partition capacities. | Low |
| Resource compatibility based on GPU/CPU/memory | Implemented. GPU type/count plus optional CPU/memory filtering and aggregate capacity constraints. | Keep | Matches the Alibaba GPU-type partition approach and avoids invented workload categories. | Job GPU type/count, CPU, memory; partition capacities. | Low |
| One partition per GPU type | Implemented in scenario builders for Alibaba-aligned experiments. | Keep as baseline | Reasonable aggregate abstraction for current thesis scope. | Alibaba GPU type counts, aggregate CPU/memory, GPU power specs. | Low |
| Fixed inference baseline | Implemented as `baseline_load` in hourly inputs. | Keep | Useful for representing non-flexible load without adding fixed jobs to the combinatorial model. | Baseline MW profile or scalar assumption. | Low |
| Constant PUE | Implemented through `ModelConfig.pue`. | Keep as fallback | Simple, transparent baseline. | Scalar PUE assumption. | Low |
| Exogenous time-varying PUE | Implemented if `hourly_df["pue"]` is present. | Adopt now / keep | Best near-term improvement: more realistic than scalar PUE without new binary variables. | Hourly PUE profile or proxy from weather/cooling assumptions. | Low |
| Thermal PUE from ambient temperature | Not implemented as a generator/preprocessing step. | Adopt next, as exogenous coefficient | Good realism gain while preserving MILP linearity. Compute outside the MILP and pass as `hourly_df["pue"]`. | Ambient temperature, cooling regime, threshold, slope, cap, floor. | Medium |
| Load-dependent PUE tranches | Not implemented. | Defer | Adds tranche binaries and big-M constraints. Good research extension, but increases MILP complexity and numerical risk. | Partial-load PUE curve, utilization tranche bounds, tight big-M calibration. | High |
| Multiplicative thermal × load PUE | Not implemented. | Defer until thermal-only PUE is stable | Conceptually reasonable, but only safe once double counting and calibration are clear. | Thermal PUE profile plus load-utilization PUE factors. | High |
| Partition-specific PUE utilization | Not implemented. Current PUE applies at facility time-slot level. | Defer | Partition-level utilization PUE may be physically questionable unless cooling/power overhead is truly partition-local. | Mapping from partitions to cooling/power infrastructure. | High |
| Facility-energy auxiliary variables for PUE tranches | Not implemented. | Defer with load-dependent PUE | Necessary only if load-dependent PUE is adopted. | Tight bounds for each partition/time slot. | High |
| Battery charge/discharge/SOC | Implemented with continuous charge, discharge, and SOC variables. | Keep optional | Good scenario knob; current model is simple and understandable. | Battery power, energy capacity, initial/final SOC, efficiencies. | Medium |
| Constant battery efficiency | Implemented. | Keep as baseline | Good first-order representation. | Charge/discharge efficiency assumptions. | Low |
| No simultaneous charge/discharge binary | Not implemented. | Adopt if battery is used seriously | Prevents non-physical cycling, especially once variable efficiency or costs are added. | None beyond battery config. | Medium |
| SOC-dependent battery efficiency tranches | Not implemented. | Defer | More realistic, but adds tranche binaries and big-M constraints. It is not needed for the first battery comparison. | Efficiency by SOC tranche, SOC thresholds, tight big-M values. | High |
| Clean-surplus-only battery charging | Not implemented as a hard rule. Current battery can charge whenever energy balance and prices make it attractive. | Decide business assumption before coding | This changes interpretation: curtailment recovery only versus market/grid arbitrage. Both are valid but answer different questions. | Policy choice; renewable surplus definition. | Medium |
| Battery final SOC rule | Implemented as optional minimum final SOC. | Keep; consider equality for fair scenarios | Avoids draining the battery at the end of the horizon for artificial savings. | Desired final SOC or equality rule. | Low |
| Soft-deadline tardiness | Not implemented. Current domains enforce hard feasible windows. | Adopt next | Adds useful service differentiation with low modeling risk. Tardiness is linear because completion is fixed for each `x[j,k,s]`. | Original deadlines, soft/hard job flag, priority weights. | Low |
| Hard-deadline jobs | Implemented through feasible start domain. | Keep | Cleanest way to enforce hard deadlines. | Arrival/latest start/deadline. | Low |
| Priority weights | Not implemented. | Adopt with soft tardiness | Necessary to represent service differentiation. | Priority or SLA weight per job. | Low |
| Peak charge on worst grid-import slot | Implemented with `P_peak >= Q[t]` and excess over contracted power. | Keep | Correctly charges grid import, not total facility load. | Contracted power, peak price. | Low |
| Renewable and grid energy split | Implemented through `R[t]` and `Q[t]`. | Keep | Gives interpretable energy sourcing and cost components. | Renewable availability, grid price, renewable price. | Low |
| Renewable curtailment recovery lever | Partially implicit through optional battery and renewable availability. | Keep as metric; avoid extra objective term initially | Battery can reduce curtailment if enabled, but curtailment should be reported separately before adding more objective weights. | Renewable availability and solved renewable consumption. | Low |
| Integrated euro scalar objective | Partially implemented: renewable cost, grid cost, peak cost. Tardiness missing. | Extend after tardiness | Maintain one scalar objective, but report components separately. | Prices, peak tariff, tardiness weights. | Low |
| Explicit reported cost components | Partially implemented in evaluation metrics. | Keep and expand | Essential for thesis interpretation. Add tardiness and battery metrics if implemented. | Solved hourly and schedule outputs. | Low |
| Big-M linearizations | Not used for current PUE/battery because current model avoids tranche products. | Use only where unavoidable | Big-M must be tightly bounded. Poor bounds will hurt Gurobi and make results harder to trust. | Physical bounds by variable family. | Medium/High |
| Full MILP-to-QUBO conversion | Not implemented; docs mark QUBO as simplified/legacy. | Do not attempt full v0.2 QUBO first | Encoding continuous PUE, SOC, energy split, and peak variables can explode binary count. | Binary encodings, penalty calibration, coefficient normalization. | Very High |
| QUBO hard-constraint penalties | Partially discussed in `docs/QUBO_FORMULATION.md`. | Adopt only for reduced model | Start with assignment, capacity, compatibility, and simplified energy/peak proxy. | Penalty multipliers and validation suite. | High |
| PCE/QAOA target | Not in v0.2 directly. Current quantum docs propose reduced QUBO first. | Keep reduced quantum target | PCE may reduce qubits, but constraint-heavy full MILP is not a good first quantum target. | Small benchmark instances and QUBO/Ising coefficients. | High |

## Suggested Implementation Phases

### Phase 1: Stabilize Current Baseline

- Keep current resource-compatible MILP.
- Confirm all tiny absurd tests for assignment, GPU/CPU/memory, PUE, battery,
  and peak charge.
- Keep battery disabled in main baseline unless explicitly testing storage.

### Phase 2: Low-Risk v0.2 Additions

- Add soft-deadline weighted tardiness:
  - `deadline`
  - `allow_late` or `deadline_type`
  - `priority_weight`
  - precomputed tardiness for each feasible placement.
- Add objective reporting:
  - energy cost,
  - peak cost,
  - tardiness cost,
  - total cost.

### Phase 3: Exogenous Physical Realism

- Add preprocessing for thermal PUE:
  - ambient temperature profile,
  - cooling regime,
  - threshold,
  - slope,
  - floor,
  - cap.
- Pass the result as `hourly_df["pue"]`.
- Avoid load-dependent PUE until thermal-only PUE is validated.

### Phase 4: Battery Refinement

- Add no-simultaneous-charge/discharge binary if battery experiments are central.
- Decide whether charging is:
  - surplus-renewable-only, or
  - allowed from grid based on price.
- Keep constant efficiency first.
- Defer SOC-dependent efficiency tranches.

### Phase 5: Advanced MILP Extensions

- Load-dependent PUE tranches.
- SOC-dependent battery efficiency.
- Tighter big-M bound derivations.
- More detailed facility/cooling structure.

### Phase 6: Quantum-Compatible Reduction

- Do not translate the full v0.2 MILP into QUBO first.
- Build a reduced QUBO with:
  - assignment penalties,
  - capacity penalties,
  - compatibility/domain reduction,
  - simplified energy coefficients,
  - optional peak/load-smoothing proxy.
- Validate reduced QUBO against MILP on tiny and constrained benchmark
  instances before adding battery or load-dependent PUE.

## Open Decisions For The Next Meeting

1. Should battery charge only from renewable surplus, or may it charge from the
   grid when prices are low?
2. Should final SOC be a minimum or equality constraint for fair scenario
   comparison?
3. Is PUE intended to be facility-wide, partition-specific, or cooling-regime
   specific?
4. What data source will calibrate thermal PUE?
5. Do soft-deadline jobs exist in the business context, or are all jobs hard
   deadline?
6. Should load-dependent PUE be part of the thesis core or an extension?
7. What is the first quantum target: reduced assignment/capacity QUBO, or a
   richer energy-aware QUBO?

