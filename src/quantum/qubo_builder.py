"""QUBO construction helpers for reduced scheduling instances.

The builder encodes the assignment/start decision variables used by the MILP:
``x[job, cluster, start]``. It is intended for quantum and quantum-inspired
experiments on small slices of the hard-instance benchmark pack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.data.validation import validate_clusters, validate_hourly_inputs, validate_jobs
from src.milp.expressions import build_feasible_starts, is_active


QuboKey = tuple[int, int]


@dataclass(frozen=True)
class QuboPenaltyWeights:
    """Penalty and proxy weights used by the reduced scheduling QUBO."""

    assignment: float = 100.0
    power_capacity: float = 50.0
    gpu_capacity: float = 50.0
    cpu_capacity: float = 20.0
    memory_capacity: float = 20.0
    peak_smoothing: float = 1.0


@dataclass
class SchedulingQubo:
    """QUBO coefficients and metadata needed to decode binary samples."""

    linear: dict[int, float] = field(default_factory=dict)
    quadratic: dict[QuboKey, float] = field(default_factory=dict)
    offset: float = 0.0
    variables: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_variables(self) -> int:
        """Return the number of QUBO binary variables."""

        return len(self.variables)


def _is_blank(value: Any) -> bool:
    """Return whether a dataframe value should be treated as missing."""

    return pd.isna(value) or str(value).strip() in {"", "None", "nan"}


def _normalize_gpu_types(value: Any) -> set[str]:
    """Parse a pipe-separated GPU-type requirement."""

    if _is_blank(value):
        return set()
    return {token.strip() for token in str(value).split("|") if token.strip()}


def _add_linear(qubo: SchedulingQubo, variable: int, coefficient: float) -> None:
    """Accumulate one linear QUBO coefficient."""

    if coefficient:
        qubo.linear[variable] = qubo.linear.get(variable, 0.0) + float(coefficient)


def _add_quadratic(qubo: SchedulingQubo, left: int, right: int, coefficient: float) -> None:
    """Accumulate one quadratic QUBO coefficient."""

    if left == right:
        _add_linear(qubo, left, coefficient)
        return
    key = (left, right) if left < right else (right, left)
    if coefficient:
        qubo.quadratic[key] = qubo.quadratic.get(key, 0.0) + float(coefficient)


def _compatible(job: pd.Series, cluster: pd.Series) -> bool:
    """Return whether a job can run on a cluster under GPU-type/count metadata."""

    gpu_count = int(job.get("gpu_count_required", job.get("gpus", 0)))
    if gpu_count > int(cluster.get("gpu_count", cluster.get("gpu_capacity", 0))):
        return False
    required_gpu_types = _normalize_gpu_types(job.get("gpu_type_required", ""))
    if required_gpu_types and str(cluster.get("gpu_type", "")).strip() not in required_gpu_types:
        return False
    if float(job.get("cpu_required", 0.0)) > float(cluster.get("cpu_capacity", np.inf)):
        return False
    if float(job.get("memory_required_gb", 0.0)) > float(cluster.get("memory_capacity_gb", np.inf)):
        return False
    return True


def _variable_activity(variable: dict[str, Any], hour: int) -> bool:
    """Return whether a QUBO assignment variable is active in a time slot."""

    return is_active(int(variable["start"]), int(variable["duration"]), hour)


def _linear_energy_cost(
    variable: dict[str, Any],
    hourly_lookup: dict[int, dict[str, float]],
    config: ModelConfig,
) -> float:
    """Approximate MILP energy dispatch with a time-varying effective price."""

    cost = 0.0
    for hour in hourly_lookup:
        if not _variable_activity(variable, hour):
            continue
        grid_price = hourly_lookup[hour]["grid_price"]
        renewable_available = hourly_lookup[hour]["renewable_available"]
        effective_price = min(grid_price, config.renewable_price) if renewable_available > 0 else grid_price
        cost += effective_price * config.pue * float(variable["power"]) * config.delta_t
    return float(cost)


def _add_assignment_penalties(qubo: SchedulingQubo, weights: QuboPenaltyWeights) -> None:
    """Add exact one-hot assignment penalties for each job."""

    by_job: dict[str, list[int]] = {}
    for index, variable in enumerate(qubo.variables):
        by_job.setdefault(str(variable["job_id"]), []).append(index)

    for indices in by_job.values():
        qubo.offset += weights.assignment
        for index in indices:
            _add_linear(qubo, index, -weights.assignment)
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                _add_quadratic(qubo, left, right, 2.0 * weights.assignment)


def _active_variables_by_cluster_hour(qubo: SchedulingQubo, hours: list[int]) -> dict[tuple[str, int], list[int]]:
    """Index variables by cluster and active hour."""

    active: dict[tuple[str, int], list[int]] = {}
    for index, variable in enumerate(qubo.variables):
        cluster = str(variable["cluster"])
        for hour in hours:
            if _variable_activity(variable, hour):
                active.setdefault((cluster, hour), []).append(index)
    return active


def _add_pairwise_capacity_pressure(
    qubo: SchedulingQubo,
    indices: list[int],
    coefficients: dict[int, float],
    capacity: float,
    weight: float,
) -> None:
    """Add pairwise resource-contention pressure for a capacity constraint.

    A direct ``(load - capacity)^2`` term is a poor inequality proxy because it
    rewards filling otherwise idle capacity. This pairwise term only adds cost
    when active variables compete for the same resource in the same slot.
    Exact feasibility is still checked after decoding.
    """

    if capacity <= 0:
        return
    for left_position, left in enumerate(indices):
        for right in indices[left_position + 1 :]:
            combined = coefficients[left] + coefficients[right]
            normalized_pressure = coefficients[left] * coefficients[right] / (capacity * capacity)
            overload_pressure = max(0.0, combined - capacity) ** 2 / (capacity * capacity)
            _add_quadratic(qubo, left, right, weight * (normalized_pressure + overload_pressure))


def _add_resource_capacity_proxies(
    qubo: SchedulingQubo,
    clusters_df: pd.DataFrame,
    hours: list[int],
    weights: QuboPenaltyWeights,
) -> None:
    """Add soft squared-capacity proxies for cluster resources.

    These terms guide the quantum optimizer away from overloads, but feasibility
    must still be checked after decoding because inequality constraints are only
    approximated here.
    """

    cluster_lookup = clusters_df.set_index("cluster_id").to_dict(orient="index")
    active = _active_variables_by_cluster_hour(qubo, hours)
    specs = [
        ("power", "capacity", weights.power_capacity),
        ("gpu_count_required", "gpu_count", weights.gpu_capacity),
        ("cpu_required", "cpu_capacity", weights.cpu_capacity),
        ("memory_required_gb", "memory_capacity_gb", weights.memory_capacity),
    ]
    for (cluster, hour), indices in active.items():
        cluster_data = cluster_lookup[cluster]
        for variable_field, capacity_field, weight in specs:
            capacity = float(cluster_data.get(capacity_field, 0.0))
            if capacity <= 0 or weight <= 0:
                continue
            coefficients = {index: float(qubo.variables[index].get(variable_field, 0.0)) for index in indices}
            if any(value > 0 for value in coefficients.values()):
                _add_pairwise_capacity_pressure(qubo, indices, coefficients, capacity, weight)


def _add_peak_smoothing_proxy(
    qubo: SchedulingQubo,
    hours: list[int],
    weight: float,
) -> None:
    """Add a quadratic load-smoothing proxy over facility IT power."""

    if weight <= 0:
        return
    for hour in hours:
        indices = [index for index, variable in enumerate(qubo.variables) if _variable_activity(variable, hour)]
        coefficients = {index: float(qubo.variables[index]["power"]) for index in indices}
        for index in indices:
            _add_linear(qubo, index, weight * coefficients[index] * coefficients[index])
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                _add_quadratic(qubo, left, right, 2.0 * weight * coefficients[left] * coefficients[right])


def build_scheduling_qubo(
    jobs_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: ModelConfig,
    weights: QuboPenaltyWeights | None = None,
) -> SchedulingQubo:
    """Build a reduced scheduling QUBO from model-ready instance tables."""

    validate_jobs(jobs_df)
    validate_hourly_inputs(hourly_df)
    validate_clusters(clusters_df)

    penalty_weights = weights or QuboPenaltyWeights()
    feasible_starts = build_feasible_starts(jobs_df)
    hours = [int(hour) for hour in hourly_df["hour"].tolist()]
    hourly_lookup = {
        int(row.hour): {
            "grid_price": float(row.grid_price),
            "renewable_available": float(row.renewable_available),
        }
        for row in hourly_df.itertuples(index=False)
    }

    qubo = SchedulingQubo(metadata={"weights": penalty_weights.__dict__.copy(), "hours": hours})
    for job in jobs_df.itertuples(index=False):
        job_series = pd.Series(job._asdict())
        for cluster in clusters_df.itertuples(index=False):
            cluster_series = pd.Series(cluster._asdict())
            if not _compatible(job_series, cluster_series):
                continue
            for start in feasible_starts[str(job.job_id)]:
                variable = {
                    "job_id": str(job.job_id),
                    "cluster": str(cluster.cluster_id),
                    "start": int(start),
                    "duration": int(job.duration),
                    "power": float(job.power),
                    "gpu_count_required": float(getattr(job, "gpu_count_required", getattr(job, "gpus", 0.0))),
                    "cpu_required": float(getattr(job, "cpu_required", 0.0)),
                    "memory_required_gb": float(getattr(job, "memory_required_gb", 0.0)),
                }
                index = len(qubo.variables)
                qubo.variables.append(variable)
                _add_linear(qubo, index, _linear_energy_cost(variable, hourly_lookup, config))

    _add_assignment_penalties(qubo, penalty_weights)
    _add_resource_capacity_proxies(qubo, clusters_df, hours, penalty_weights)
    _add_peak_smoothing_proxy(qubo, hours, penalty_weights.peak_smoothing)
    qubo.metadata["num_linear_terms"] = len(qubo.linear)
    qubo.metadata["num_quadratic_terms"] = len(qubo.quadratic)
    return qubo


def qubo_energy(qubo: SchedulingQubo, sample: dict[int, int] | list[int] | np.ndarray) -> float:
    """Evaluate a binary sample under the QUBO objective."""

    if isinstance(sample, dict):
        value = lambda index: int(sample.get(index, 0))
    else:
        array = np.asarray(sample)
        value = lambda index: int(array[index])

    energy = qubo.offset
    energy += sum(coefficient * value(index) for index, coefficient in qubo.linear.items())
    energy += sum(coefficient * value(left) * value(right) for (left, right), coefficient in qubo.quadratic.items())
    return float(energy)


def decode_qubo_sample(qubo: SchedulingQubo, sample: dict[int, int] | list[int] | np.ndarray) -> pd.DataFrame:
    """Decode selected QUBO variables into a schedule dataframe."""

    if isinstance(sample, dict):
        selected = [index for index, bit in sample.items() if int(bit) == 1]
    else:
        selected = [index for index, bit in enumerate(np.asarray(sample)) if int(bit) == 1]
    rows = []
    for index in selected:
        variable = qubo.variables[index]
        rows.append(
            {
                "job_id": variable["job_id"],
                "assigned_cluster": variable["cluster"],
                "start_hour": variable["start"],
                "duration": variable["duration"],
                "power": variable["power"],
                "gpu_count_required": variable["gpu_count_required"],
                "cpu_required": variable["cpu_required"],
                "memory_required_gb": variable["memory_required_gb"],
                "qubo_variable": index,
            }
        )
    return pd.DataFrame(rows).sort_values(["start_hour", "job_id"]).reset_index(drop=True) if rows else pd.DataFrame(rows)


def validate_decoded_schedule(
    schedule_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    hours: list[int],
) -> dict[str, Any]:
    """Validate assignment and resource feasibility for a decoded schedule."""

    jobs = jobs_df.set_index("job_id")
    clusters = clusters_df.set_index("cluster_id")
    violations: list[str] = []
    if len(schedule_df) != len(jobs_df):
        violations.append(f"expected {len(jobs_df)} assigned jobs, found {len(schedule_df)}")
    if not schedule_df.empty:
        assignment_counts = schedule_df["job_id"].value_counts()
        duplicated = assignment_counts[assignment_counts > 1].index.tolist()
        missing = sorted(set(jobs.index) - set(schedule_df["job_id"]))
        if duplicated:
            violations.append(f"duplicated jobs: {duplicated}")
        if missing:
            violations.append(f"missing jobs: {missing}")

    usage = {
        cluster_id: {
            "power": {hour: 0.0 for hour in hours},
            "gpu": {hour: 0.0 for hour in hours},
            "cpu": {hour: 0.0 for hour in hours},
            "memory": {hour: 0.0 for hour in hours},
        }
        for cluster_id in clusters.index
    }
    for row in schedule_df.itertuples(index=False):
        job = jobs.loc[row.job_id]
        if not (int(job.earliest_start) <= int(row.start_hour) <= int(job.latest_start)):
            violations.append(f"{row.job_id} starts outside its window")
        if row.assigned_cluster not in clusters.index:
            violations.append(f"{row.job_id} assigned to unknown cluster {row.assigned_cluster}")
            continue
        cluster = clusters.loc[row.assigned_cluster]
        if not _compatible(job, cluster):
            violations.append(f"{row.job_id} assigned to incompatible cluster {row.assigned_cluster}")
        for hour in hours:
            if is_active(int(row.start_hour), int(row.duration), hour):
                usage[row.assigned_cluster]["power"][hour] += float(row.power)
                usage[row.assigned_cluster]["gpu"][hour] += float(row.gpu_count_required)
                usage[row.assigned_cluster]["cpu"][hour] += float(row.cpu_required)
                usage[row.assigned_cluster]["memory"][hour] += float(row.memory_required_gb)

    for cluster_id, cluster in clusters.iterrows():
        capacities = {
            "power": float(cluster.capacity),
            "gpu": float(cluster.get("gpu_count", cluster.get("gpu_capacity", 0.0))),
            "cpu": float(cluster.get("cpu_capacity", 0.0)),
            "memory": float(cluster.get("memory_capacity_gb", 0.0)),
        }
        for resource, capacity in capacities.items():
            if capacity <= 0:
                continue
            max_usage = max(usage[cluster_id][resource].values()) if usage[cluster_id][resource] else 0.0
            if max_usage > capacity + 1e-8:
                violations.append(f"{cluster_id} exceeds {resource} capacity: {max_usage:.6g} > {capacity:.6g}")

    return {"feasible": not violations, "violations": violations}
