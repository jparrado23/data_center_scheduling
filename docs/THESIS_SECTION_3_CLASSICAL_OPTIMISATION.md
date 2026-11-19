# 3. Classical optimisation approach

## 3.1. Operations-research perspective

The problem studied in this thesis is a deterministic, finite-horizon scheduling problem. A data-centre operator receives a set of flexible jobs. Each job has a release time, a latest permissible start time, a processing duration, a power demand, and a resource profile. Jobs are non-preemptive: after starting, a job runs continuously for its full duration. The operator must choose a start time and a compatible compute partition for every job, while respecting resource capacities in every time slot. Electricity prices and renewable availability vary over the horizon, so the resulting schedule also affects operating cost.

This structure connects to several established operations-research problem classes.

### Scheduling with time windows

A job's release time and deadline restrict the set of permissible start times. In the implementation, `earliest_start` and `latest_start` define an inclusive set of candidate start slots. For job \(i\), with discrete release slot \(r_i\), latest start \(\ell_i\), and duration \(d_i\), the feasible start set is

\[
\mathcal S_i=\{s\in\mathcal T: r_i\le s\le \ell_i,\; s+d_i-1\le t_{\max}\}.
\]

The deadline convention here is expressed through the latest **start** slot; the completion condition ensures that a job finishes within the modeled horizon. This is a time-window scheduling model: jobs need not start immediately, and the scheduler exploits flexibility inside each window. A start-time binary formulation is a natural representation because it encodes both the selected start and the whole, contiguous execution interval.

### Resource-constrained project scheduling

The model also has the central structure of a resource-constrained scheduling problem (RCPSP): activities consume limited renewable resources while they execute. Here, the resources are not only GPUs. A job can consume GPU, CPU, memory, and IT power, and the capacities are indexed by compute partition and time slot. The usual precedence relations of a project network are absent in the current model; jobs are independent except for their shared resource constraints. Thus, the model is best described as a resource-constrained, multi-resource scheduling problem with time windows, rather than a full precedence-constrained RCPSP.

Compute partitions are heterogeneous. A partition may represent a pool of one GPU type and has its own GPU, CPU, memory, and power capacities. Compatibility determines which job–partition assignments are allowed. The model uses aggregate partition capacities. It does not assign jobs to individual machines or GPUs, so it does not capture fragmentation or node-level placement constraints. This is a planning-level approximation: aggregate feasibility is necessary but may not guarantee that a concrete node-level placement exists.

### Energy-aware and data-centre scheduling

Energy-aware scheduling adds time-dependent operating conditions to the resource-constrained problem. A schedule determines when flexible IT load is present; facility load is then obtained by applying Power Usage Effectiveness (PUE). The load can be supplied by renewable energy and grid imports, subject to renewable availability. Grid prices vary over time, and the maximum grid import may incur a contracted-power excess charge. Consequently, the objective may favour shifting jobs to lower-cost periods, but these shifts remain bounded by time windows, hardware compatibility, and resource capacities.

This thesis therefore combines three decisions that are often treated separately: job start-time and partition assignment, shared compute-resource allocation, and energy-cost-aware operation. The full classical model represents these decisions in a single mixed-integer linear program (MILP). The optional battery extension adds inter-temporal energy storage and state-of-charge constraints. These elements are kept distinct from the reduced QUBO used later: the QUBO focuses on binary scheduling choices and a GPU-capacity representation, while the full MILP remains the operational reference model.

The classical optimisation literature provides the methodological foundation for this formulation: discrete start-time variables represent non-preemptive scheduling; time-indexed resource inequalities enforce renewable resource capacities; and energy costs can be incorporated by coupling scheduled IT demand to energy-supply and tariff variables. The contribution here is not a new general scheduling class, but an integrated formulation adapted to flexible AI workloads, heterogeneous GPU partitions, time-varying energy conditions, and the subsequent comparison with a reduced binary model.

## 3.2. Full MILP formulation

### 3.2.1. Modeling assumptions and boundary

The formulation below describes the full deterministic scheduling model implemented in the project. The horizon is divided into discrete slots of equal length \(\Delta t\) hours. Job parameters and energy inputs are known in advance. Each job runs without interruption on one compatible compute partition for its entire duration. Its resource demands remain constant while it is active.

The MILP uses aggregate compute-partition capacities. GPU, CPU, memory, and power capacities are enforced by partition and time slot. Compatibility is determined from job resource requirements and partition metadata (and, for legacy data, supported allow/deny indicators). The energy balance applies PUE to total IT load, including any fixed baseline load, before balancing renewable supply, grid import, and optional battery operation. Contracted power is an economic threshold on grid import, not a hard upper bound on facility demand.

The model is static and deterministic. It does not represent uncertain prices or renewable forecasts, rolling-horizon rescheduling, job preemption, or individual-node placement. Battery variables are included only when positive battery power and energy capacities are configured. In the main quantum comparison, the QUBO is a reduction of the scheduling core and does not encode all energy-dispatch and battery decisions of this full MILP.

### 3.2.2. Sets and indices

\[
\begin{aligned}
&\mathcal I &&\text{set of jobs, indexed by }i,\\
&\mathcal K &&\text{set of compute partitions, indexed by }k,\\
&\mathcal T=\{0,1,\ldots,H-1\} &&\text{set of time slots, indexed by }t,\\
&\mathcal S_i &&\text{feasible start slots for job }i,\\
&\mathcal K_i\subseteq\mathcal K &&\text{partitions compatible with job }i.
\end{aligned}
\]

Define an activity coefficient for each candidate start:

\[
a_{i,s,t}=\begin{cases}
1,&s\le t\le s+d_i-1,\\
0,&\text{otherwise.}
\end{cases}
\]

A start is included in \(\mathcal S_i\) only if it satisfies the job's release/latest-start window and the full execution interval lies within the horizon. Let \(C_{ik}\in\{0,1\}\) indicate compatibility. Then

\[
\mathcal K_i=\{k\in\mathcal K:C_{ik}=1\}.
\]

The implementation prunes incompatible job–partition pairs when constructing decision variables, rather than creating variables for them and fixing them to zero.

### 3.2.3. Parameters

For every job \(i\in\mathcal I\):

| Parameter | Meaning | Units |
|---|---|---|
| \(r_i\) | Earliest permissible start slot (release time) | slots |
| \(\ell_i\) | Latest permissible start slot | slots |
| \(d_i\) | Continuous processing duration | slots |
| \(p_i\) | IT power while active | MW |
| \(g_i\) | GPU requirement | GPU units |
| \(c_i\) | CPU requirement | CPU units, as represented in the input data |
| \(m_i\) | Memory requirement | GB |
| \(C_{ik}\) | 1 if job \(i\) is compatible with partition \(k\), 0 otherwise | binary |

For every partition \(k\in\mathcal K\):

| Parameter | Meaning | Units |
|---|---|---|
| \(P_k\) | IT power capacity | MW |
| \(G_k\) | GPU capacity | GPU units |
| \(CPU_k\) | Aggregate CPU capacity | CPU units |
| \(MEM_k\) | Aggregate memory capacity | GB |

For every time slot \(t\in\mathcal T\):

| Parameter | Meaning | Units |
|---|---|---|
| \(B_t\) | Fixed baseline IT load | MW |
| \(G^{ren}_t\) | Renewable power available | MW |
| \(\pi^{grid}_t\) | Grid electricity price | currency/MWh |
| \(\eta^{pue}_t\) | PUE multiplier | dimensionless |

Global energy and battery parameters:

| Parameter | Meaning | Units |
|---|---|---|
| \(\pi^{ren}\) | Renewable-energy price | currency/MWh |
| \(\pi^{peak}\) | Charge per MW of contracted-power excess | currency/MW over the modeled billing period |
| \(P^{contracted}\) | Contracted grid-import threshold | MW |
| \(\Delta t\) | Slot duration | hours |
| \(P^{bat}\) | Battery charge/discharge power limit | MW |
| \(E^{bat}\) | Battery energy capacity | MWh |
| \(SOC_0\) | Initial battery state of charge | MWh |
| \(SOC^{final}\) | Optional minimum terminal state of charge | MWh |
| \(\eta^c,\eta^d\) | Battery charge and discharge efficiencies | dimensionless |

If a time-varying PUE profile is unavailable, \(\eta^{pue}_t\) is replaced by the configured scalar PUE for every slot. If the battery is disabled, battery variables and constraints are omitted.

### 3.2.4. Decision variables

The principal scheduling variable is

\[
x_{i,k,s}\in\{0,1\},\qquad i\in\mathcal I,\ k\in\mathcal K_i,\ s\in\mathcal S_i,
\]

where \(x_{i,k,s}=1\) means job \(i\) starts at slot \(s\) on partition \(k\). Variables are defined only for compatible pairs \((i,k)\).

The energy-dispatch variables for each slot are

\[
R_t\ge0,\qquad Q_t\ge0,
\]

where \(R_t\) is renewable power consumed and \(Q_t\) is grid-import power. When the battery is enabled, the model also has

\[
C_t,D_t\ge0,\qquad 0\le SOC_t\le E^{bat},
\]

where \(C_t\) and \(D_t\) are charging and discharging power, respectively, and \(SOC_t\) is stored energy at the end of slot \(t\). Finally,

\[
P^{peak}\ge0,\qquad E^{peak}\ge0
\]

represent the maximum grid import and the excess above contracted power.

### 3.2.5. Load expressions

The IT power assigned to partition \(k\) in slot \(t\) is

\[
L_{k,t}(x)=\sum_{i\in\mathcal I}\sum_{s\in\mathcal S_i}
 p_i a_{i,s,t}x_{i,k,s}.
\]

GPU, CPU, and memory loads are defined analogously:

\[
\begin{aligned}
G_{k,t}(x)&=\sum_{i\in\mathcal I}\sum_{s\in\mathcal S_i}
 g_i a_{i,s,t}x_{i,k,s},\\
CPU_{k,t}(x)&=\sum_{i\in\mathcal I}\sum_{s\in\mathcal S_i}
 c_i a_{i,s,t}x_{i,k,s},\\
MEM_{k,t}(x)&=\sum_{i\in\mathcal I}\sum_{s\in\mathcal S_i}
 m_i a_{i,s,t}x_{i,k,s}.
\end{aligned}
\]

Total flexible IT load and total facility load are

\[
L^{flex}_t(x)=\sum_{k\in\mathcal K}L_{k,t}(x),
\]

\[
L^{IT}_t(x)=B_t+L^{flex}_t(x),\qquad
L^{facility}_t(x)=\eta^{pue}_tL^{IT}_t(x).
\]

These load quantities are powers. Multiplication by \(\Delta t\) converts power into energy for each slot's cost contribution.

### 3.2.6. Objective function

The objective minimises renewable-energy cost, grid-energy cost, and the charge on grid-import power exceeding the contracted threshold:

\[
\min\quad
\sum_{t\in\mathcal T}
\left(\pi^{ren}R_t+\pi^{grid}_tQ_t\right)\Delta t
+\pi^{peak}E^{peak}.
\tag{1}
\]

The energy terms are prices per MWh multiplied by power and slot duration. The peak term is based on the maximum grid import across the modeled horizon, not on maximum total facility load. This is the billing proxy implemented by the model. No battery degradation cost is currently included; if battery use is studied, throughput costs could be added as an extension.

### 3.2.7. Constraints

**Each job is scheduled exactly once.**

\[
\sum_{k\in\mathcal K_i}\sum_{s\in\mathcal S_i}x_{i,k,s}=1
\qquad\forall i\in\mathcal I.
\tag{2}
\]

Because all candidate starts lie inside the job's time window and variables exist only for compatible partitions, this single constraint selects exactly one feasible partition–start combination per job.

**Partition IT power capacity.**

\[
L_{k,t}(x)\le P_k
\qquad\forall k\in\mathcal K,\ t\in\mathcal T.
\tag{3}
\]

**Partition GPU capacity.**

\[
G_{k,t}(x)\le G_k
\qquad\forall k\in\mathcal K,\ t\in\mathcal T.
\tag{4}
\]

**Partition CPU capacity.**

\[
CPU_{k,t}(x)\le CPU_k
\qquad\forall k\in\mathcal K,\ t\in\mathcal T.
\tag{5}
\]

**Partition memory capacity.**

\[
MEM_{k,t}(x)\le MEM_k
\qquad\forall k\in\mathcal K,\ t\in\mathcal T.
\tag{6}
\]

Constraints (4)–(6) are enforced when the corresponding resource is represented and enabled. These are aggregate capacities, so they do not resolve per-node fragmentation.

**Energy balance.** When no battery is present,

\[
R_t+Q_t=L^{facility}_t(x)
\qquad\forall t\in\mathcal T.
\tag{7a}
\]

When battery operation is enabled,

\[
R_t+Q_t+D_t=L^{facility}_t(x)+C_t
\qquad\forall t\in\mathcal T.
\tag{7b}
\]

Thus, renewable and grid supply, together with battery discharge, cover facility demand and any simultaneous battery charging. The model contains no export variable; excess renewable availability is simply not consumed.

**Renewable availability.**

\[
0\le R_t\le G^{ren}_t
\qquad\forall t\in\mathcal T.
\tag{8}
\]

**Grid-import peak and contracted-power excess.**

\[
P^{peak}\ge Q_t
\qquad\forall t\in\mathcal T,
\tag{9}
\]

\[
E^{peak}\ge P^{peak}-P^{contracted},\qquad E^{peak}\ge0.
\tag{10}
\]

Since \(E^{peak}\) is minimised with a non-negative coefficient, at an optimum it equals \(\max\{0,P^{peak}-P^{contracted}\}\). Since \(P^{peak}\) appears only in these constraints and the excess calculation, it can be set to the maximum grid import at an optimum.

**Battery power and energy bounds (if enabled).**

\[
0\le C_t\le P^{bat},\qquad 0\le D_t\le P^{bat},
\qquad 0\le SOC_t\le E^{bat}
\qquad\forall t\in\mathcal T.
\tag{11}
\]

The state-of-charge transition is

\[
SOC_t=SOC_{t-1}+\eta^c C_t\Delta t-
\frac{D_t\Delta t}{\eta^d}
\qquad\forall t\in\mathcal T,
\tag{12}
\]

where the previous state at the first modeled slot is the parameter \(SOC_0\). If a terminal target is specified,

\[
SOC_{H-1}\ge SOC^{final}.
\tag{13}
\]

Battery charge and discharge are modeled as continuous powers. The current formulation does not include a binary mode variable to prohibit simultaneous charging and discharging; with positive losses and the current cost structure, simultaneous operation is generally dominated, but it is not structurally forbidden. A strict operating rule could be added with binary mode variables and suitable bounds.

**Variable domains.**

\[
x_{i,k,s}\in\{0,1\},\quad
R_t,Q_t,P^{peak},E^{peak}\ge0,
\tag{14}
\]

with the non-negativity and upper bounds for battery variables as specified above. All expressions and constraints are linear, so (1)–(14) form a mixed-integer linear program.

### 3.2.8. Interpretation and relationship to the reduced QUBO

The MILP makes a global choice for every job: a compatible partition and a start slot. Equations (2)–(6) determine whether the resulting schedule can be supported by the modeled compute resources. Equations (7)–(13) determine the energy dispatch and its operating cost for that schedule. Gurobi can exploit the linear structure and return a feasible incumbent, a bound, and—if it closes the optimality gap—a proof of optimality.

The later QUBO experiments do not encode this entire model. They retain binary job–cluster–start assignment choices and GPU-capacity feasibility, while simplifying or omitting parts of the energy system. In particular, continuous renewable/grid dispatch and optional battery state are outside the reduced QUBO used in the main comparison. Therefore, QUBO energy is not directly the same quantity as the MILP objective. Candidate schedules must be decoded and evaluated with a common physical cost calculation before comparing their schedule quality.

The full MILP is consequently used as the classical operational baseline where the instance and configuration match. For fair comparison, the thesis should state explicitly which constraints and objective terms are shared with each reduced QUBO experiment, and report whether Gurobi proved optimality or only supplied a reference incumbent.

## References for the related-work framing

The discussion above deliberately characterises the problem using established model classes rather than claiming a new scheduling taxonomy. The final thesis bibliography should cite authoritative sources for (i) time-indexed formulations of non-preemptive scheduling with release dates and deadlines, (ii) resource-constrained project scheduling and renewable resource constraints, and (iii) energy-aware data-centre/workload scheduling with time-varying electricity prices and renewable supply. These citations should be selected to match the scope and literature review required by the programme; they are not inferred from the codebase and should be verified against the papers actually reviewed for the thesis.
