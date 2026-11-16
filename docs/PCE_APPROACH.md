# Pauli Correlation Encoding Approach

This document defines a proposed Pauli Correlation Encoding (PCE) path for the
data-center scheduling QUBO. It is intended for review before implementation in
a new notebook.

The goal is not to replace the MILP. The MILP remains the source of truth for
full feasibility and business cost. PCE is a quantum-compatible experiment on
the reduced QUBO formulation:

- assignment variables `x[job, cluster, start]`;
- fixed/exogenous PUE energy coefficients;
- one-hot assignment penalties;
- aggregate GPU capacity penalties with binary slack;
- fixed-PUE peak/load-smoothing proxy.

## 1. Goal

The near-term quantum problem is that QAOA needs one qubit per QUBO binary
variable. Our scheduling QUBO can quickly reach hundreds or thousands of binary
variables because:

```text
assignment variables ~= jobs x compatible clusters x feasible starts
slack variables      ~= clusters x time slots x GPU slack bits
```

Direct QAOA therefore becomes unrealistic very quickly.

PCE offers a different encoding:

> Instead of assigning one qubit to each binary variable, encode many binary
> variables as signs of Pauli-string expectation values measured on a smaller
> quantum register.

For our project, PCE should answer this research question:

> Can we solve, or at least produce useful feasible schedules for, reduced
> scheduling QUBOs whose binary-variable count is larger than the number of
> available qubits?

## 2. Theoretical Justification

### 2.1 Classical Binary Variables

Start with a binary optimization problem over `m` variables:

```math
z_i \in \{0,1\}, \quad i = 1,\dots,m
```

Our QUBO is:

```math
E(z) = q_0 + \sum_i q_i z_i + \sum_{i<j} q_{ij} z_i z_j
```

For PCE it is usually more convenient to use spin variables:

```math
s_i \in \{-1,+1\}
```

with:

```math
z_i = \frac{1+s_i}{2}
```

After conversion, the QUBO becomes an Ising-style energy:

```math
E(s) = c + \sum_i h_i s_i + \sum_{i<j} J_{ij} s_i s_j
```

### 2.2 PCE Encoding

PCE assigns each classical spin variable `s_i` to a Pauli string
`Pi_i` acting on `n` qubits:

```math
\Pi_i \in \{I,X,Y,Z\}^{\otimes n}
```

Then the binary variable is decoded from the sign of the expectation value:

```math
s_i = \operatorname{sgn}(\langle \Pi_i \rangle)
```

where:

```math
\langle \Pi_i \rangle
=
\langle \psi(\theta) | \Pi_i | \psi(\theta) \rangle
```

and `|psi(theta)>` is the variational quantum state produced by a
parameterized circuit.

In practice, during training we do not optimize the discontinuous sign
function directly. We use a smooth relaxation:

```math
\tilde{s}_i(\theta)
=
\tanh(\alpha \langle \Pi_i \rangle)
```

where `alpha` controls binarization strength.

The PCE loss becomes:

```math
L(\theta)
=
c
+
\sum_i h_i \tilde{s}_i(\theta)
+
\sum_{i<j} J_{ij}
\tilde{s}_i(\theta)\tilde{s}_j(\theta)
```

After optimization, decode:

```math
\hat{s}_i = \operatorname{sgn}(\langle \Pi_i \rangle)
```

and convert back:

```math
\hat{z}_i = \frac{1+\hat{s}_i}{2}
```

### 2.3 Qubit Compression

If we use `k`-body Pauli correlations on `n` qubits, a common PCE construction
uses Pauli strings such as:

```text
X on k selected qubits, identity elsewhere
Y on k selected qubits, identity elsewhere
Z on k selected qubits, identity elsewhere
```

The number of encodable variables is:

```math
m_{\max} = 3 {n \choose k}
```

So the qubit count required for `m` binary variables is the smallest `n` such
that:

```math
3 {n \choose k} \ge m
```

Examples:

| Binary variables `m` | QAOA qubits | PCE `k=2` qubits | PCE `k=3` qubits |
|---:|---:|---:|---:|
| 50 | 50 | 7 | 6 |
| 100 | 100 | 9 | 7 |
| 250 | 250 | 14 | 9 |
| 500 | 500 | 19 | 12 |
| 1000 | 1000 | 27 | 14 |
| 5000 | 5000 | 59 | 23 |

This is the main reason PCE is attractive for our scheduling QUBO.

## 3. Why PCE May Be Better Than QAOA For This Project

### QAOA Limitation

Standard QAOA maps one binary variable to one qubit. If the reduced QUBO has:

```text
m = 500 binary variables
```

then QAOA needs approximately:

```text
500 logical qubits
```

before considering circuit depth, connectivity, measurement cost, or noise.

Our problem can reach that size with modest assumptions:

```text
100 jobs x 2 start options x 2-3 compatible clusters + slack bits
```

That makes full QAOA mainly useful for tiny slices.

### PCE Advantage

PCE can encode those `m` binary variables into far fewer qubits by using
multi-qubit correlations.

For example, with `k=2`:

```math
m_{\max} = 3 {n \choose 2}
```

So `n=20` qubits can encode:

```math
3 {20 \choose 2} = 570
```

binary variables.

With `k=3`, `n=20` can encode:

```math
3 {20 \choose 3} = 3420
```

binary variables.

### Training Landscape

PCE literature also argues that polynomial compression can reduce barren
plateau severity compared with one-variable-per-qubit encodings. The intuition
is that the number of qubits `n` grows as roughly `m^(1/k)`, so trainability
depends on a smaller quantum register than QAOA's `n=m` register.

### Important Caveat

PCE is not automatically better in every sense.

It reduces qubits, but it introduces:

- many Pauli expectation values to estimate;
- a nonlinear relaxed objective;
- sensitivity to the binarization parameter `alpha`;
- potential constraint-satisfaction problems after decoding;
- need for classical post-processing or repair.

For constrained optimization, recent PCE work reports that standard PCE can
struggle to enforce constraints reliably. Progressive Binarization PCE
(PB-PCE) was introduced to improve feasibility by progressively increasing
the binarization parameter and re-optimizing from the previous solution.

## 4. How PCE Applies To Our Scheduling QUBO

### 4.1 Input

The starting point should be the existing reduced QUBO:

```python
qubo = build_scheduling_qubo(...)
bqm = to_dimod_bqm(qubo)
```

The QUBO contains:

- `qubo.linear`: coefficients `q_i`;
- `qubo.quadratic`: coefficients `q_ij`;
- `qubo.offset`;
- `qubo.variables`: metadata needed to decode a bitstring back into a schedule.

### 4.2 Convert QUBO To Ising

PCE should operate on spin variables:

```math
s_i \in \{-1,+1\}
```

Use:

```math
z_i = \frac{1+s_i}{2}
```

Substitute into the QUBO:

```math
E(z) = q_0 + \sum_i q_i z_i + \sum_{i<j} q_{ij} z_i z_j
```

For each linear term:

```math
q_i z_i
=
\frac{q_i}{2}
+
\frac{q_i}{2}s_i
```

For each quadratic term:

```math
q_{ij}z_i z_j
=
\frac{q_{ij}}{4}
\left(
1 + s_i + s_j + s_i s_j
\right)
```

So the Ising coefficients are:

```math
c = q_0 + \frac{1}{2}\sum_i q_i + \frac{1}{4}\sum_{i<j}q_{ij}
```

```math
h_i = \frac{q_i}{2} + \frac{1}{4}\sum_{j \ne i} q_{ij}
```

```math
J_{ij} = \frac{q_{ij}}{4}
```

### 4.3 Assign QUBO Variables To Pauli Strings

Let:

```math
m = \text{number of QUBO variables}
```

Choose:

- compression order `k`, normally `k=2` first;
- number of qubits `n`, smallest such that `3 * comb(n, k) >= m`;
- an ordered list of Pauli strings.

Example for `k=2`, `n=4`:

```text
XXII
XIXI
XIIX
IXXI
IXIX
IIXX
YYII
YIYI
...
ZZII
...
```

Each QUBO variable index maps to one Pauli string:

```text
variable 0 -> XXII
variable 1 -> XIXI
variable 2 -> XIIX
...
```

### 4.4 Build The PCE Loss

For every variable:

```math
r_i(\theta) = \langle \Pi_i \rangle_\theta
```

Smooth relaxed spin:

```math
\tilde{s}_i(\theta)=\tanh(\alpha r_i(\theta))
```

Loss:

```math
L(\theta)
=
c
+
\sum_i h_i \tilde{s}_i(\theta)
+
\sum_{i<j} J_{ij}
\tilde{s}_i(\theta)\tilde{s}_j(\theta)
```

This loss mirrors the QUBO/Ising energy but uses Pauli correlations instead of
direct binary variables.

### 4.5 Decode And Validate

After optimization:

```math
\hat{s}_i = \operatorname{sgn}(r_i)
```

Then:

```math
\hat{z}_i = \frac{1+\hat{s}_i}{2}
```

Decode the bitstring with the existing QUBO decoder:

```python
schedule = decode_qubo_sample(qubo, z_hat)
report = validate_decoded_schedule(schedule, jobs_df, clusters_df, hours)
```

This is non-negotiable. PCE output must be validated like every other solver.

## 5. Progressive Binarization For Constraints

Our QUBO has constraints encoded as penalties. This makes PCE a constrained
optimization problem in practice.

Standard PCE may produce relaxed correlators that look good continuously but
decode to infeasible binary schedules.

Progressive Binarization PCE addresses this by solving a sequence:

```text
alpha_1 < alpha_2 < ... < alpha_R
```

At each stage:

1. optimize the PCE loss with current `alpha`;
2. use the optimized parameters as initialization for the next `alpha`;
3. increase `alpha`, making `tanh(alpha * r_i)` more binary;
4. decode and validate at each stage.

Suggested initial schedule:

```text
alpha_values = [0.5, 1, 2, 4, 8, 16, 32]
```

or:

```text
alpha_values = geometric progression until decoded feasibility stabilizes
```

This is likely more relevant than one-shot PCE for our problem because
assignment and GPU constraints must survive decoding.

## 6. Qiskit Implementation Plan

### 6.1 Dependencies

Expected packages:

```python
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
```

For hardware/runtime later:

```python
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Session
```

### 6.2 Build Pauli Strings

Pseudo-code:

```python
from itertools import combinations

def generate_pce_paulis(num_qubits: int, order: int, max_terms: int):
    labels = []
    for pauli_char in ["X", "Y", "Z"]:
        for support in combinations(range(num_qubits), order):
            label = ["I"] * num_qubits
            for q in support:
                label[q] = pauli_char
            labels.append("".join(reversed(label)))  # Qiskit little-endian convention
            if len(labels) == max_terms:
                return labels
    return labels
```

Then:

```python
observables = [SparsePauliOp(label) for label in labels]
```

### 6.3 Choose Qubit Count

```python
from math import comb

def pce_qubits_required(num_variables: int, order: int) -> int:
    n = order
    while 3 * comb(n, order) < num_variables:
        n += 1
    return n
```

### 6.4 Variational Circuit

Start simple:

```python
ansatz = EfficientSU2(
    num_qubits=n,
    reps=2,
    entanglement="linear",
)
```

Later test:

- `reps=1,2,3`;
- `linear` versus `full` entanglement;
- hardware-efficient ansatz matched to backend coupling map.

### 6.5 Estimate Correlators

For simulator:

```python
estimator = StatevectorEstimator()
```

For a parameter vector `theta`, estimate:

```python
r_i(theta) = <Pi_i>
```

Then compute:

```python
s_tilde = np.tanh(alpha * correlators)
loss = c + h @ s_tilde + sum(J_ij * s_tilde[i] * s_tilde[j])
```

### 6.6 Optimization Loop

Pseudo-code:

```python
theta = random_initial_parameters()

for alpha in alpha_values:
    result = minimize(
        fun=lambda theta: pce_loss(theta, alpha),
        x0=theta,
        method="COBYLA",
        options={"maxiter": maxiter},
    )
    theta = result.x

    correlators = estimate_correlators(theta)
    z_hat = decode_correlators_to_bits(correlators)
    schedule = decode_qubo_sample(qubo, z_hat)
    report = validate_decoded_schedule(schedule, jobs, clusters, hours)
```

Record per stage:

- `alpha`;
- loss;
- QUBO energy of decoded bitstring;
- assignment violations;
- GPU violations;
- schedule cost under shared evaluator if available.

### 6.7 Hardware Path

Once the simulator path works:

1. select a backend with enough qubits for chosen `n`;
2. transpile ansatz;
3. group Pauli measurements if possible;
4. use `EstimatorV2`;
5. compare shot-based correlators against statevector correlators;
6. decode and validate.

## 7. Suggested Notebook Structure

Create:

```text
notebooks/14_pce_scheduling_qubo.ipynb
```

Recommended sections:

1. Load or create tiny QUBO instance.
2. Build QUBO with `src.quantum.qubo_builder`.
3. Convert QUBO to Ising coefficients.
4. Compute QAOA qubit count versus PCE qubit count.
5. Generate PCE Pauli strings.
6. Build Qiskit ansatz.
7. Statevector PCE objective.
8. One-shot PCE optimization.
9. Progressive-Binarization PCE optimization.
10. Decode schedules.
11. Validate schedules.
12. Compare:
    - D-Wave simulator,
    - QAOA tiny feasibility if available,
    - PCE,
    - MILP on same tiny case.

## 8. Expected Qubit Reduction For Our QUBO

Use the current QUBO size calculation:

```python
m = qubo.num_variables
```

Then compare:

```text
QAOA qubits = m
PCE k=2 qubits = min n with 3*C(n,2) >= m
PCE k=3 qubits = min n with 3*C(n,3) >= m
```

Example table:

| QUBO variables | QAOA | PCE k=2 | PCE k=3 |
|---:|---:|---:|---:|
| 14 | 14 | 4 | 5 |
| 50 | 50 | 7 | 6 |
| 100 | 100 | 9 | 7 |
| 500 | 500 | 19 | 12 |
| 1000 | 1000 | 27 | 14 |
| 5000 | 5000 | 59 | 23 |

The compression is substantial. But this is logical-qubit compression only.
It does not guarantee lower runtime or better solution quality.

## 9. Risks And Validation Requirements

### Risk 1: Constraint Feasibility

PCE optimizes a relaxed loss. Decoded bitstrings may violate assignment or GPU
capacity constraints.

Mitigation:

- start with tiny cases;
- use PB-PCE alpha schedule;
- add repair heuristics after decoding;
- validate with `validate_decoded_schedule`.

### Risk 2: Measurement Overhead

PCE reduces qubits but requires many Pauli expectation values. For `m`
variables, we need `m` correlators per loss evaluation.

Mitigation:

- use commuting groups from X/Y/Z construction;
- begin with statevector simulation;
- measure shot cost separately.

### Risk 3: Loss Does Not Match Discrete Objective

The relaxed objective uses:

```math
\tilde{s}_i = \tanh(\alpha \langle \Pi_i \rangle)
```

The final schedule uses:

```math
\operatorname{sgn}(\langle \Pi_i \rangle)
```

The continuous optimum may not decode to the best discrete bitstring.

Mitigation:

- track both relaxed loss and decoded QUBO energy;
- use progressive binarization;
- apply local search or schedule repair around decoded bitstring.

### Risk 4: Penalty Scaling

The QUBO penalty weights must be scaled so feasibility is preferred after
decoding. If penalties are too weak, PCE may return low-cost infeasible
schedules.

Mitigation:

- reuse progressive tiny examples;
- normalize QUBO coefficients before PCE;
- report feasibility first, cost second.

### Risk 5: PCE Is Not A Full MILP Encoding

The first PCE target should not include full battery, exact renewable/grid
dispatch, or load-dependent PUE. It should target the reduced QUBO.

Mitigation:

- clearly state scope;
- evaluate decoded schedules with shared project metrics;
- compare against MILP only on the same simplified objective unless using MILP
  as post-evaluation.

## 10. Recommended First Implementation

Implement the first PCE notebook in this order:

1. Use the existing tiny QUBO from notebook 13.
2. Convert QUBO to Ising.
3. Generate `k=2` PCE strings.
4. Use statevector estimator first.
5. Optimize with `COBYLA`.
6. Decode signs to QUBO bits.
7. Validate schedule.
8. Add PB-PCE alpha schedule.
9. Compare qubits:
   - QAOA: `m`;
   - PCE: `n`.
10. Only then try a slightly larger generated QUBO.

Do not start with hardware execution. First prove the encoding, loss, decoding,
and validation loop on simulator.

## 11. Open Decisions Before Coding

1. Should the first PCE target include slack variables, or should we first
   encode only assignment variables and repair capacity classically?
2. Should we use `k=2` only initially, or compare `k=2` and `k=3`?
3. Should the loss use the full penalized QUBO or separate feasibility-first
   terms?
4. Should we normalize QUBO coefficients before constructing the PCE loss?
5. Which classical optimizer should be the default: `COBYLA`, `SPSA`, or
   `L-BFGS-B` on statevector?
6. Should decoded bitstrings be repaired before validation or reported raw
   first?

## 12. Initial Recommendation

Use PCE as a **qubit-reduction experiment for the reduced QUBO**, not as a full
replacement for MILP.

The first implementation should be:

- QUBO from `src.quantum.qubo_builder`;
- QUBO-to-Ising conversion;
- `k=2` PCE;
- statevector Qiskit simulation;
- PB-PCE alpha schedule;
- decoded schedule validation;
- optional local repair after raw results are understood.

This gives a defensible path:

> QAOA shows the direct one-qubit-per-variable limitation. PCE tests whether
> correlation encoding can handle larger scheduling QUBOs with fewer qubits,
> while still being evaluated under the same feasibility rules.

## Sources

- IBM Quantum documentation, "Pauli correlation encoding to reduce max-cut
  requirements": https://quantum.cloud.ibm.com/docs/en/tutorials/pauli-correlation-encoding-for-qaoa
- Sciorilli et al., "Towards large-scale quantum optimization solvers with few
  qubits", Nature Communications 2025: https://www.nature.com/articles/s41467-024-55346-z
- Padin-Martinez et al., "Progressive Binarization - Pauli Correlation Encoding:
  a Continuation Method for Constrained Optimization", arXiv:2602.17479:
  https://arxiv.org/abs/2602.17479
