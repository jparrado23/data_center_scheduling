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
    gpu_capacity: float = 50.0
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


def _normalize_cluster_ids(value: Any) -> set[str]:
    """Parse an explicit job-to-cluster compatibility field."""

    if isinstance(value, (list, tuple, set)):
        return {str(token).strip() for token in value if str(token).strip()}
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
    """Return whether a job can run on a cluster.

    Prefer an explicit ``compatible_clusters`` relation when present. When that
    relation exists, it is treated as the final business/technical
    compatibility matrix and stale GPU-type metadata is ignored. GPU-type
    metadata remains supported only as a preprocessing fallback for older
    instances.
    """

    gpu_count = int(job.get("gpu_count_required", job.get("gpus", 0)))
    if gpu_count > int(cluster.get("gpu_count", cluster.get("gpu_capacity", 0))):
        return False
    compatible_clusters = _normalize_cluster_ids(job.get("compatible_clusters", ""))
    if compatible_clusters:
        return str(cluster.get("cluster_id", "")).strip() in compatible_clusters
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
    """Return the fixed-PUE linear energy coefficient for one assignment.

    The reduced QUBO does not decide renewable/grid dispatch explicitly. It uses
    an effective time price and multiplies IT load by fixed or exogenous PUE.
    """

    cost = 0.0
    for hour in hourly_lookup:
        if not _variable_activity(variable, hour):
            continue
        grid_price = hourly_lookup[hour]["grid_price"]
        renewable_available = hourly_lookup[hour]["renewable_available"]
        pue = hourly_lookup[hour]["pue"]
        effective_price = min(grid_price, config.renewable_price) if renewable_available > 0 else grid_price
        cost += effective_price * pue * float(variable["power"]) * config.delta_t
    return float(cost)


def build_hourly_lookup(hourly_df: pd.DataFrame, config: ModelConfig) -> dict[int, dict[str, float]]:
    """Build the time-slot coefficient lookup used by the reduced QUBO."""

    return {
        int(row.hour): {
            "grid_price": float(row.grid_price),
            "renewable_available": float(row.renewable_available),
            "pue": float(getattr(row, "pue", config.pue)),
        }
        for row in hourly_df.itertuples(index=False)
    }


def add_assignment_variables(
    qubo: SchedulingQubo,
    jobs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    feasible_starts: dict[str, list[int]] | None = None,
) -> None:
    """Create one binary variable for each feasible ``(job, cluster, start)`` option."""

    starts_by_job = feasible_starts or build_feasible_starts(jobs_df)
    for job in jobs_df.itertuples(index=False):
        job_series = pd.Series(job._asdict())
        for cluster in clusters_df.itertuples(index=False):
            cluster_series = pd.Series(cluster._asdict())
            if not _compatible(job_series, cluster_series):
                continue
            for start in starts_by_job[str(job.job_id)]:
                qubo.variables.append(
                    {
                        "variable_type": "assignment",
                        "job_id": str(job.job_id),
                        "cluster": str(cluster.cluster_id),
                        "start": int(start),
                        "duration": int(job.duration),
                        "power": float(job.power),
                        "gpu_count_required": float(getattr(job, "gpu_count_required", getattr(job, "gpus", 0.0))),
                        "cpu_required": float(getattr(job, "cpu_required", 0.0)),
                        "memory_required_gb": float(getattr(job, "memory_required_gb", 0.0)),
                    }
                )


def add_fixed_pue_energy_cost(
    qubo: SchedulingQubo,
    hourly_lookup: dict[int, dict[str, float]],
    config: ModelConfig,
) -> None:
    """Add ``H_energy`` with fixed or exogenous PUE as linear QUBO coefficients."""

    for index in _assignment_indices(qubo):
        _add_linear(qubo, index, _linear_energy_cost(qubo.variables[index], hourly_lookup, config))


def add_assignment_constraint_penalty(qubo: SchedulingQubo, weight: float) -> None:
    """Add the exact one-hot assignment penalty for every job."""

    _add_assignment_penalties(qubo, QuboPenaltyWeights(assignment=weight))


def add_gpu_capacity_constraint_penalty(
    qubo: SchedulingQubo,
    clusters_df: pd.DataFrame,
    hours: list[int],
    weight: float,
) -> None:
    """Add exact GPU capacity penalties using binary slack variables."""

    _add_gpu_capacity_penalties_with_slack(qubo, clusters_df, hours, weight)


def add_fixed_pue_peak_penalty(
    qubo: SchedulingQubo,
    hours: list[int],
    hourly_lookup: dict[int, dict[str, float]],
    weight: float,
) -> None:
    """Add the fixed-PUE squared facility-load peak proxy."""

    _add_peak_smoothing_proxy(qubo, hours, hourly_lookup, weight)


def _add_assignment_penalties(qubo: SchedulingQubo, weights: QuboPenaltyWeights) -> None:
    """Add exact one-hot assignment penalties for each job."""

    by_job: dict[str, list[int]] = {}
    for index in _assignment_indices(qubo):
        variable = qubo.variables[index]
        by_job.setdefault(str(variable["job_id"]), []).append(index)

    for indices in by_job.values():
        qubo.offset += weights.assignment
        for index in indices:
            _add_linear(qubo, index, -weights.assignment)
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                _add_quadratic(qubo, left, right, 2.0 * weights.assignment)


def _assignment_indices(qubo: SchedulingQubo) -> list[int]:
    """Return indices for assignment variables only."""

    return [index for index, variable in enumerate(qubo.variables) if variable.get("variable_type") == "assignment"]


def _active_assignments_by_cluster_hour(qubo: SchedulingQubo, hours: list[int]) -> dict[tuple[str, int], list[int]]:
    """Index variables by cluster and active hour."""

    active: dict[tuple[str, int], list[int]] = {}
    for index in _assignment_indices(qubo):
        variable = qubo.variables[index]
        cluster = str(variable["cluster"])
        for hour in hours:
            if _variable_activity(variable, hour):
                active.setdefault((cluster, hour), []).append(index)
    return active


def _add_squared_equality_penalty(
    qubo: SchedulingQubo,
    terms: dict[int, float],
    rhs: float,
    weight: float,
) -> None:
    """Add ``weight * (sum_i terms_i z_i - rhs)^2`` to the QUBO."""

    if weight <= 0:
        return
    qubo.offset += weight * rhs * rhs
    indices = list(terms)
    for index, coefficient in terms.items():
        _add_linear(qubo, index, weight * (coefficient * coefficient - 2.0 * rhs * coefficient))
    for left_position, left in enumerate(indices):
        for right in indices[left_position + 1 :]:
            _add_quadratic(qubo, left, right, 2.0 * weight * terms[left] * terms[right])


def _cluster_gpu_capacity(cluster: pd.Series | dict[str, Any]) -> int:
    """Return aggregate GPU capacity for a cluster row."""

    return int(cluster.get("gpu_count", cluster.get("gpu_capacity", 0)))


def _add_gpu_capacity_penalties_with_slack(
    qubo: SchedulingQubo,
    clusters_df: pd.DataFrame,
    hours: list[int],
    weight: float,
) -> None:
    """Add exact GPU capacity penalties using binary unused-GPU slack.

    For every cluster and time slot, the inequality
    ``D_gpu[k,t] <= G[k]`` is converted to
    ``D_gpu[k,t] + slack[k,t] = G[k]``. The slack is binary encoded, so a zero
    penalty exists exactly when GPU usage does not exceed capacity.
    """

    if weight <= 0:
        return
    cluster_lookup = clusters_df.set_index("cluster_id").to_dict(orient="index")
    active = _active_assignments_by_cluster_hour(qubo, hours)
    for cluster_id, cluster_data in cluster_lookup.items():
        capacity = _cluster_gpu_capacity(cluster_data)
        if capacity < 0:
            raise ValueError(f"cluster {cluster_id} has negative GPU capacity")
        bit_count = int(np.ceil(np.log2(capacity + 1))) if capacity > 0 else 1
        for hour in hours:
            active_indices = active.get((str(cluster_id), hour), [])
            if not active_indices:
                continue
            terms: dict[int, float] = {
                index: float(qubo.variables[index]["gpu_count_required"])
                for index in active_indices
            }
            for bit in range(bit_count):
                slack_index = len(qubo.variables)
                coefficient = float(2**bit)
                qubo.variables.append(
                    {
                        "variable_type": "gpu_slack",
                        "cluster": str(cluster_id),
                        "hour": int(hour),
                        "bit": int(bit),
                        "coefficient": coefficient,
                    }
                )
                terms[slack_index] = coefficient
            _add_squared_equality_penalty(qubo, terms, rhs=float(capacity), weight=weight)


def _add_peak_smoothing_proxy(
    qubo: SchedulingQubo,
    hours: list[int],
    hourly_lookup: dict[int, dict[str, float]],
    weight: float,
) -> None:
    """Add the fixed-PUE squared facility-load proxy."""

    if weight <= 0:
        return
    for hour in hours:
        indices = [index for index in _assignment_indices(qubo) if _variable_activity(qubo.variables[index], hour)]
        if not indices:
            continue
        pue = float(hourly_lookup[hour]["pue"])
        terms = {index: pue * float(qubo.variables[index]["power"]) for index in indices}
        _add_squared_equality_penalty(qubo, terms, rhs=0.0, weight=weight)


def build_scheduling_qubo(
    jobs_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: ModelConfig,
    weights: QuboPenaltyWeights | None = None,
) -> SchedulingQubo:
    """Build the reduced scheduling QUBO from model-ready instance tables.

    The implemented Hamiltonian is:

    ``H = H_energy_fixed_pue + H_assignment + H_gpu_slack + H_peak_fixed_pue``.

    Compatibility is handled by omitting infeasible assignment variables. GPU
    capacity is encoded exactly with binary unused-GPU slack. CPU, memory,
    renewable/grid dispatch, exact contracted peak billing, and battery dynamics
    remain outside this first reduced QUBO and should be checked or modeled
    separately.
    """

    validate_jobs(jobs_df)
    validate_hourly_inputs(hourly_df)
    validate_clusters(clusters_df)

    penalty_weights = weights or QuboPenaltyWeights()
    feasible_starts = build_feasible_starts(jobs_df)
    hours = [int(hour) for hour in hourly_df["hour"].tolist()]
    hourly_lookup = build_hourly_lookup(hourly_df, config)

    qubo = SchedulingQubo(
        metadata={
            "formulation": "fixed_pue_assignment_gpu_slack_energy_peak_qubo",
            "weights": penalty_weights.__dict__.copy(),
            "hours": hours,
            "terms": {},
        }
    )
    add_assignment_variables(qubo, jobs_df, clusters_df, feasible_starts)
    add_fixed_pue_energy_cost(qubo, hourly_lookup, config)

    qubo.metadata["terms"]["energy_fixed_pue"] = {
        "linear_terms_after_term": len(qubo.linear),
        "quadratic_terms_after_term": len(qubo.quadratic),
        "description": "Linear fixed-PUE energy cost using effective time price.",
    }
    add_assignment_constraint_penalty(qubo, penalty_weights.assignment)
    qubo.metadata["terms"]["assignment"] = {
        "linear_terms_after_term": len(qubo.linear),
        "quadratic_terms_after_term": len(qubo.quadratic),
        "description": "Exact one-hot job assignment penalty.",
    }
    add_gpu_capacity_constraint_penalty(qubo, clusters_df, hours, penalty_weights.gpu_capacity)
    qubo.metadata["terms"]["gpu_capacity_with_slack"] = {
        "linear_terms_after_term": len(qubo.linear),
        "quadratic_terms_after_term": len(qubo.quadratic),
        "description": "Exact GPU capacity penalty using binary unused-GPU slack per cluster and time.",
    }
    add_fixed_pue_peak_penalty(qubo, hours, hourly_lookup, penalty_weights.peak_smoothing)
    qubo.metadata["terms"]["peak_fixed_pue"] = {
        "linear_terms_after_term": len(qubo.linear),
        "quadratic_terms_after_term": len(qubo.quadratic),
        "description": "Quadratic fixed-PUE facility-load smoothing proxy.",
    }
    qubo.metadata["num_assignment_variables"] = len(_assignment_indices(qubo))
    qubo.metadata["num_slack_variables"] = qubo.num_variables - qubo.metadata["num_assignment_variables"]
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


def to_dimod_bqm(qubo: SchedulingQubo):
    """Convert a ``SchedulingQubo`` into a D-Wave Ocean binary quadratic model."""

    try:
        import dimod
    except ImportError as exc:  # pragma: no cover - depends on optional Ocean install
        raise ImportError("Install D-Wave Ocean's 'dimod' package to build a BinaryQuadraticModel") from exc

    bqm = dimod.BinaryQuadraticModel({}, {}, qubo.offset, dimod.BINARY)
    for index, coefficient in qubo.linear.items():
        bqm.add_variable(index, coefficient)
    for (left, right), coefficient in qubo.quadratic.items():
        bqm.add_interaction(left, right, coefficient)
    return bqm


def decode_qubo_sample(qubo: SchedulingQubo, sample: dict[int, int] | list[int] | np.ndarray) -> pd.DataFrame:
    """Decode selected QUBO variables into a schedule dataframe."""

    if isinstance(sample, dict):
        selected = [index for index, bit in sample.items() if int(bit) == 1]
    else:
        selected = [index for index, bit in enumerate(np.asarray(sample)) if int(bit) == 1]
    rows = []
    for index in selected:
        variable = qubo.variables[index]
        if variable.get("variable_type") != "assignment":
            continue
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
