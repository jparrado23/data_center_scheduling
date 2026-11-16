# Quantum Agent Architecture

This document defines how Codex subagents should be used for the quantum part of
the data-center scheduling project. The goal is not to create permanent agents
inside Codex, but to make every delegated quantum task follow the same structure,
assumptions, and validation rules.

## 1. Purpose

The quantum work should compare alternative methods on the same scheduling
problem, using the same generated instances and the same feasibility checks.

The intended methodology stack is:

- MILP baseline with Gurobi.
- Genetic algorithm baseline.
- Reduced QUBO formulation.
- QAOA on the reduced QUBO or Ising model.
- Tensor-network simulation or analysis of QAOA circuits.
- Later hybrid classical-quantum approaches.

The MILP remains the source of truth for feasibility and final cost evaluation.
Quantum methods may initially solve simplified versions, but decoded schedules
must be validated against the project feasibility checks.

## 2. Supervisor Agent

The main Codex conversation acts as the supervisor.

Responsibilities:

- Own the final project direction.
- Keep the mathematical assumptions aligned with `docs/MILP_FORMULATION.md`.
- Decide which generated instances are official benchmarks.
- Ensure all methods use the same processed input format.
- Compare MILP, GA, QUBO, QAOA, and tensor-network results fairly.
- Review subagent outputs before implementation.
- Reject shortcuts that make methods incomparable.

The supervisor should not allow each methodology to invent its own data model,
objective definition, or feasibility standard.

## 3. Domain Agents

### 3.1 QUBO Agent

Role:

Design and implement the reduced binary optimization model.

Initial scope:

- Binary variable: `x[j, k, s] = 1` if job `j` starts on cluster/GPU-type pool
  `k` at slot `s`.
- Omit infeasible variables for incompatible job-cluster pairs.
- Enforce or penalize each job being scheduled exactly once.
- Include feasible start windows.
- Include aggregate GPU capacity.
- Treat aggregate IT power capacity as a decoded-schedule validation check
  unless a dedicated power-capacity penalty is added.
- Include PUE as a deterministic multiplier in energy coefficients.
- Decode QUBO samples back into schedules.

Postpone initially:

- Battery state of charge.
- Exact renewable/grid dispatch.
- Exact peak-demand billing.
- CPU and memory capacity.
- Exact aggregate IT power-capacity encoding.
- 15-minute time granularity.

Required outputs:

- Variable indexing.
- Objective and penalty terms.
- Penalty-scaling strategy.
- Decoding function.
- Feasibility report after decoding.
- Comparison against MILP for tiny instances.

### 3.2 QAOA Agent

Role:

Run QAOA experiments on the reduced QUBO or equivalent Ising model.

Initial scope:

- Consume the QUBO produced by the QUBO agent.
- Convert QUBO to Ising form when needed.
- Start with very small generated instances.
- Compare exact brute force, classical QUBO solver, and QAOA outputs when
  possible.
- Track qubit count, circuit depth, optimizer calls, runtime, and feasibility
  after decoding.

Postpone initially:

- Full MILP-to-QAOA translation.
- Battery modeling.
- Large generated instances.
- Real hardware runs before simulator behavior is understood.

Required outputs:

- Qubit count per instance.
- Circuit depth per QAOA depth `p`.
- Best measured bitstring or sample distribution.
- Decoded schedule.
- Feasibility and cost under the shared evaluator.
- Simulator limitations and hardware-readiness assessment.

### 3.3 Tensor-Network Agent

Role:

Use tensor networks as a simulation and analysis tool for QAOA-like circuits.

Initial scope:

- Analyze the QUBO interaction graph.
- Estimate contraction difficulty.
- Simulate low-depth QAOA circuits when statevector simulation becomes too
  expensive.
- Compare tensor-network simulation results against statevector results on small
  cases.

Postpone initially:

- Treating tensor networks as a standalone replacement optimizer.
- Modeling the full MILP directly.
- Claims of scalability without graph-width diagnostics.

Required outputs:

- QUBO graph statistics.
- Contraction-width or contraction-cost estimates.
- Maximum instance size tested.
- Agreement check against smaller exact/statevector experiments.
- Clear statement of when tensor networks help and when they do not.

### 3.4 Reviewer Agent

Role:

Audit methodology-specific work before it is treated as a valid result.

Responsibilities:

- Check that generated instances are used consistently.
- Check that decoded schedules are validated.
- Check that no hidden hardcoded values define the result.
- Check that simplifications are documented.
- Check that reported costs are recomputed under the shared evaluator.
- Check that tests or tiny examples support the implementation.

The reviewer should prioritize bugs, invalid comparisons, missing constraints,
and misleading conclusions.

## 4. Shared Inputs

All quantum and heuristic methods should consume generated or processed instance
directories, not legacy ad hoc job files.

Expected instance directory contents:

- `jobs.csv`
- `clusters.csv`
- `hourly_inputs.csv`
- `config.json`
- optional generator metadata

Generated instances should be created through `src.instance_generator`.

Example:

```bash
conda run -n quantum_py312 python -m src.instance_generator.cli \
  --num-jobs 10 \
  --output-dir experiments/quantum_instances/qaoa_10 \
  --sampling-mode alibaba \
  --slot-minutes 60
```

The same instance can then be solved with the MILP path:

```bash
conda run -n quantum_py312 python scripts/solve_thesis_scenario.py \
  --processed-dir experiments/quantum_instances/qaoa_10
```

## 5. Shared Validation Rules

Every method must report:

- whether every job is scheduled exactly once;
- whether start windows are respected;
- whether GPU-type compatibility is respected;
- whether aggregate GPU capacity is respected;
- whether aggregate power capacity is respected;
- whether CPU and memory are ignored, relaxed, or enforced;
- whether battery is disabled, approximated, or modeled;
- objective value under the method's internal objective;
- final evaluated cost under the shared project evaluator when available.

For early quantum methods, it is acceptable for the internal objective to be a
simplified proxy. It is not acceptable to compare that proxy directly against the
full MILP objective without recomputing common metrics after decoding.

## 6. Recommended Instance Ladder

Use only new generated instances for future work.

Start with 60-minute slots. Move to 30-minute or 15-minute slots only after the
quantum encoding works on hourly instances.

| Stage | Jobs | GPU-type pools | Slots | Purpose |
|---|---:|---:|---:|---|
| Tiny sanity | 3-5 | 1-2 | 6-8 | Manual validation of variables and penalties |
| First quantum | 6-10 | 2-3 | 8-12 | QUBO and QAOA simulator feasibility |
| Small benchmark | 10-20 | 2-4 | 12-24 | MILP vs GA vs QUBO/QAOA comparison |
| Method stress | 20-40 | 4-7 | 24 | Tensor-network and annealing experiments |
| Classical stress | 50+ | 7 | 24+ | MILP and GA scaling, not first QAOA tests |

The most important early quantity is not just number of jobs. It is the number
of binary choices:

```text
QUBO variables = sum over jobs of feasible (cluster, start_slot) choices
```

Wide scheduling windows and unrestricted GPU compatibility can make a small job
set quantum-infeasible.

## 7. Delegation Templates

### QUBO Exploration Prompt

```text
Act as the QUBO methodology agent for this data-center scheduling repo.
Read docs/QUANTUM_AGENT_ARCHITECTURE.md, docs/MILP_FORMULATION.md, and
docs/QUBO_FORMULATION.md.

Task:
Propose the smallest QUBO formulation that preserves job assignment, start time,
GPU-type or explicit cluster compatibility, aggregate GPU capacity, and a
PUE-adjusted energy proxy. Treat aggregate power capacity as a post-decode
validation requirement unless you propose an explicit penalty for it.

Return:
1. variables,
2. objective terms,
3. penalty terms,
4. decoding procedure,
5. instance sizes to test first,
6. risks and simplifications.

Do not implement code yet.
```

### QAOA Exploration Prompt

```text
Act as the QAOA methodology agent for this data-center scheduling repo.
Read docs/QUANTUM_AGENT_ARCHITECTURE.md and the current QUBO notes.

Task:
Define the first QAOA feasibility experiment using generated processed
instances only.

Return:
1. whether QAOA should consume QUBO or Ising form,
2. simulator plan,
3. qubit-count limits,
4. instance ladder,
5. hardware-readiness criteria,
6. risks.

Do not implement code yet.
```

### Tensor-Network Exploration Prompt

```text
Act as the tensor-network methodology agent for this data-center scheduling
repo.
Read docs/QUANTUM_AGENT_ARCHITECTURE.md and the current QUBO notes.

Task:
Evaluate whether tensor networks are useful for simulating or analyzing QAOA on
the reduced scheduling QUBO.

Return:
1. where tensor networks fit,
2. graph diagnostics needed,
3. instance sizes to try,
4. relation to QUBO/QAOA,
5. risks.

Do not implement code yet.
```

### Reviewer Prompt

```text
Act as a reviewer for the quantum methodology implementation.
Read docs/QUANTUM_AGENT_ARCHITECTURE.md and inspect the changed files.

Check:
1. bugs,
2. inconsistent assumptions,
3. hardcoded values,
4. missing feasibility validation,
5. invalid objective comparisons,
6. missing tests or tiny sanity cases.

Return findings ordered by severity with file and line references.
```

## 8. Implementation Order

Recommended order:

1. Freeze the generated tiny instance ladder.
2. Implement QUBO builder for the smallest reduced scope.
3. Implement QUBO decoder and shared feasibility report.
4. Validate QUBO against hand-checkable tiny cases.
5. Run classical QUBO solvers or brute force on tiny cases.
6. Add QAOA simulation on the same tiny cases.
7. Add tensor-network diagnostics once QAOA circuits exist.
8. Compare MILP, GA, QUBO, and QAOA on the small benchmark ladder.
9. Only then consider annealer or real hardware runs.

## 9. Non-Negotiable Assumptions

- The MILP formulation is the reference model.
- New generated instances are the official input source.
- Quantum methods may simplify the model, but must document simplifications.
- Decoded schedules must be validated.
- Internal quantum objective values are not directly comparable to full MILP
  objective values unless recomputed under the same evaluator.
- Battery should remain disabled for first quantum experiments.
- CPU and memory can be postponed for first quantum experiments, but the report
  must state that clearly.
- PUE should be included at least as a deterministic multiplier.
