# Resource Compatibility and Compute-Partition Scheduling

## Purpose

This document explains how the current branch decides whether a job can run on
a compute partition and how resource usage is constrained over time.

The current model is an aggregate compute-partition scheduler. It does not
assign jobs to individual machines or individual GPUs. In the Alibaba-aligned
baseline, each compute partition represents one GPU type pool.

## Job Resource Profile

Each schedulable job has a descriptive label and an operational resource
profile.

Business/reporting field:

```text
workload_family
```

Examples:

```text
training
fine_tuning
preprocessing
```

These labels are for reporting and analysis only. The optimizer uses the
resource fields:

```text
gpu_type_required
gpu_count_required
cpu_required
memory_required_gb
power
duration
earliest_start
latest_start
```

`power` is IT power in MW. If source data contains `power_kw` or `e_kw`, it must
be converted to model-ready MW in `power`.

`gpu_type_required` can be empty or pipe-separated:

```text
T4
G2
T4|G2|V100M32
```

An empty GPU type means the job has no explicit GPU-type restriction. Jobs in
the current thesis branch are still assumed to require positive GPU count.

## Compute-Partition Profile

Each compute partition can include:

```text
cluster_id
cluster_role
capacity
gpu_type
gpu_count
gpu_capacity
cpu_capacity
memory_capacity_gb
reserved_for_online_inference
compatible_categories
```

`reserved_for_online_inference` and `compatible_categories` may appear in
legacy files, but they are ignored by the active compatibility model.

`capacity` is IT power capacity in MW. `power_capacity_kw` may be supplied by
source data, but the MILP converts it to MW internally.

`gpu_count` and `gpu_capacity` are treated as equivalent aggregate GPU-capacity
columns. The Alibaba-aligned scenario creates seven partitions, one for each GPU
type observed in the trace summary: A10, G2, G3, P100, T4, V100M16, and
V100M32.

## Compatibility Rules

The model creates assignment variables only for compatible job-partition pairs:

```text
x[i,k,s] exists only if job i can run on partition k.
```

Current compatibility checks:

1. If the job requires GPUs, the partition must have positive GPU capacity.
2. The job GPU count must fit within the partition GPU capacity:

   ```text
   gpu_count_required <= gpu_capacity
   ```

3. If the job specifies allowed GPU types, the partition GPU type must be one
   of them:

   ```text
   cluster.gpu_type in job.gpu_type_required
   ```

4. If CPU constraints are enabled and requirements are present, they must fit:

   ```text
   cpu_required <= cpu_capacity
   ```

5. If memory constraints are enabled and requirements are present, they must fit:

   ```text
   memory_required_gb <= memory_capacity_gb
   ```

6. If legacy `alpha_B`, `alpha_C`, or `alpha_D` columns are present, they are
   treated as additional operational allow/deny rules for matching legacy
   clusters only.

## Aggregate Capacity Constraints

After compatible variables are created, the MILP enforces per-partition,
per-time-slot capacity constraints.

Power:

```text
sum active job power on partition k at time t <= partition power capacity
```

GPU:

```text
sum active job GPU count on partition k at time t <= partition GPU capacity
```

CPU:

```text
sum active job CPU on partition k at time t <= partition CPU capacity
```

Memory:

```text
sum active job memory on partition k at time t <= partition memory capacity
```

CPU and memory constraints are enabled by default. They can be disabled for
ablation experiments, but the baseline model should keep them active.

If jobs request a resource but any partition is missing the corresponding
capacity metadata, model construction fails fast. The model does not silently
disable the resource constraint.

## Relationship to Workload Labels

`workload_family` and `category` are not a feasibility mechanism. They are only
useful for reporting, analysis, and business interpretation.

If business rules must forbid a job from a partition even when resources fit,
encode that rule through a future explicit operational-compatibility table. Do
not reuse invented workload labels as compatibility constraints.

## Limitations

This model is not node-level bin packing.

A schedule can satisfy aggregate GPU, CPU, memory, and power limits but still be
impossible on individual machines if resources are fragmented across nodes. This
is an accepted Layer 0 approximation.

The current branch also assumes each partition has one main GPU type. A more
detailed extension would model GPU-type capacity buckets:

```text
gpu_capacity[partition, gpu_type]
gpu_load[partition, gpu_type, time]
```

That would improve realism for mixed-GPU clusters while avoiding full
individual-GPU placement.
