"""Gurobi MILP model for AI data-center energy scheduling."""

from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from src.config import ModelConfig
from src.data.validation import validate_clusters, validate_hourly_inputs, validate_jobs
from src.milp.expressions import build_feasible_starts, is_active


JOB_ALPHA_BY_CLUSTER = {
    "cluster_b": "alpha_B",
    "cluster_c": "alpha_C",
    "cluster_d": "alpha_D",
}


def _is_blank(value: Any) -> bool:
    return pd.isna(value) or str(value).strip() in {"", "None", "nan"}


def _optional_float(row: Any, column: str, default: float = 0.0) -> float:
    if not hasattr(row, column):
        return default
    value = getattr(row, column)
    return default if _is_blank(value) else float(value)


def _optional_int(row: Any, column: str, default: int = 0) -> int:
    if not hasattr(row, column):
        return default
    value = getattr(row, column)
    return default if _is_blank(value) else int(value)


def _normalize_gpu_types(value: Any) -> set[str]:
    if _is_blank(value):
        return set()
    return {item.strip() for item in str(value).split("|") if item.strip()}


def _optional_bool(row: Any, column: str, default: bool = False) -> bool:
    if not hasattr(row, column):
        return default
    value = getattr(row, column)
    if _is_blank(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _job_lookup(jobs_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Normalize the jobs table into a dictionary keyed by job identifier.

    The helper strips the model down to the fields repeatedly accessed during
    constraint construction so the MILP builder can avoid repeated dataframe
    lookups inside nested loops.
    """

    jobs: dict[str, dict[str, Any]] = {}
    for row in jobs_df.itertuples(index=False):
        workload_family = str(getattr(row, "workload_family", getattr(row, "category", "")))
        gpu_count = _optional_int(row, "gpu_count_required", _optional_int(row, "gpus", 0))
        job = {
            "category": str(getattr(row, "category", workload_family)),
            "workload_family": workload_family,
            "duration": int(row.duration),
            "power": float(row.power),
            "earliest_start": int(row.earliest_start),
            "latest_start": int(row.latest_start),
            "gpu_type_required": _normalize_gpu_types(getattr(row, "gpu_type_required", None)),
            "gpu_count_required": gpu_count,
            "cpu_required": _optional_float(row, "cpu_required", 0.0),
            "memory_required_gb": _optional_float(row, "memory_required_gb", 0.0),
        }
        if gpu_count > 0:
            job["gpus"] = gpu_count
        for cluster, alpha_column in JOB_ALPHA_BY_CLUSTER.items():
            if hasattr(row, alpha_column):
                job[cluster] = int(getattr(row, alpha_column))
        jobs[str(row.job_id)] = job
    return jobs


def _normalize_categories(value: Any) -> set[str]:
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in value}


def _cluster_lookup(clusters_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    clusters: dict[str, dict[str, Any]] = {}
    for row in clusters_df.itertuples(index=False):
        capacity = float(getattr(row, "capacity", getattr(row, "power_capacity_mw", 0.0)))
        if hasattr(row, "power_capacity_kw"):
            capacity = float(row.power_capacity_kw) / 1000.0
        gpu_capacity = _optional_int(row, "gpu_count", _optional_int(row, "gpu_capacity", 0))
        cluster = {
            "capacity": capacity,
            "cluster_role": str(getattr(row, "cluster_role", "")),
            "gpu_type": str(getattr(row, "gpu_type", "")).strip(),
            "gpu_capacity": gpu_capacity,
            "cpu_capacity": _optional_float(row, "cpu_capacity", 0.0),
            "memory_capacity_gb": _optional_float(row, "memory_capacity_gb", 0.0),
            "reserved_for_online_inference": _optional_bool(row, "reserved_for_online_inference", False),
            "compatible_categories": _normalize_categories(getattr(row, "compatible_categories", "")),
        }
        clusters[str(row.cluster_id)] = cluster
    return clusters


def _validate_resource_metadata(
    jobs: dict[str, dict[str, Any]],
    cluster_data: dict[str, dict[str, Any]],
    *,
    enforce_gpu_constraints: bool,
    enforce_cpu_constraints: bool,
    enforce_memory_constraints: bool,
) -> None:
    if enforce_gpu_constraints and any(job["gpu_count_required"] > 0 for job in jobs.values()):
        missing = [cluster for cluster, data in cluster_data.items() if data["gpu_capacity"] <= 0]
        if missing:
            raise ValueError(f"clusters missing positive GPU capacity metadata: {missing}")
    if enforce_cpu_constraints and any(job["cpu_required"] > 0 for job in jobs.values()):
        missing = [cluster for cluster, data in cluster_data.items() if data["cpu_capacity"] <= 0]
        if missing:
            raise ValueError(f"clusters missing positive CPU capacity metadata: {missing}")
    if enforce_memory_constraints and any(job["memory_required_gb"] > 0 for job in jobs.values()):
        missing = [cluster for cluster, data in cluster_data.items() if data["memory_capacity_gb"] <= 0]
        if missing:
            raise ValueError(f"clusters missing positive memory capacity metadata: {missing}")


def _has_resource_profile(jobs: dict[str, Any], cluster_data: dict[str, Any]) -> bool:
    return (
        jobs["gpu_count_required"] > 0
        or bool(jobs["gpu_type_required"])
        or jobs["cpu_required"] > 0
        or jobs["memory_required_gb"] > 0
        or cluster_data["gpu_capacity"] > 0
        or bool(cluster_data["gpu_type"])
        or cluster_data["cpu_capacity"] > 0
        or cluster_data["memory_capacity_gb"] > 0
    )


def _is_compatible(job: dict[str, Any], cluster: str, cluster_data: dict[str, Any]) -> int:
    """Return compatibility using resource profiles when available."""

    cluster_profile = cluster_data[cluster]
    if _has_resource_profile(job, cluster_profile):
        if cluster_profile["reserved_for_online_inference"] and job["workload_family"] not in {"online_inference", "inference"}:
            return 0
        if job["gpu_count_required"] > 0:
            if cluster_profile["gpu_capacity"] <= 0:
                return 0
            if job["gpu_count_required"] > cluster_profile["gpu_capacity"]:
                return 0
            required_gpu_types = job["gpu_type_required"]
            if required_gpu_types and cluster_profile["gpu_type"] not in required_gpu_types:
                return 0
        if cluster_profile["cpu_capacity"] > 0 and job["cpu_required"] > cluster_profile["cpu_capacity"]:
            return 0
        if cluster_profile["memory_capacity_gb"] > 0 and job["memory_required_gb"] > cluster_profile["memory_capacity_gb"]:
            return 0
        if cluster in job:
            return int(job[cluster])
        return 1
    if cluster in job:
        return int(job[cluster])
    return int(job["category"] in cluster_data[cluster]["compatible_categories"])


def build_milp_model(
    jobs_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: ModelConfig,
    model_name: str = "energy_ai_datacenter_milp",
    enforce_gpu_constraints: bool = True,
    enforce_cpu_constraints: bool = True,
    enforce_memory_constraints: bool = True,
) -> tuple[gp.Model, dict[str, Any]]:
    """Build the full scheduling MILP and return the model plus helper objects.

    The model schedules each job exactly once across clusters and feasible start
    times, chooses the lowest-cost feasible renewable/grid split, enforces
    heterogeneous cluster capacities, and minimizes energy cost plus
    contracted-power excess charges.
    The companion dictionary exposes the model variables and derived
    expressions used by result-extraction helpers.
    """

    validate_jobs(jobs_df)
    validate_hourly_inputs(hourly_df)
    validate_clusters(clusters_df)

    jobs = _job_lookup(jobs_df)
    cluster_data = _cluster_lookup(clusters_df)
    _validate_resource_metadata(
        jobs,
        cluster_data,
        enforce_gpu_constraints=enforce_gpu_constraints,
        enforce_cpu_constraints=enforce_cpu_constraints,
        enforce_memory_constraints=enforce_memory_constraints,
    )
    job_ids = list(jobs)
    hours = [int(hour) for hour in hourly_df["hour"].tolist()]
    clusters = list(cluster_data)
    feasible_starts = build_feasible_starts(jobs_df)
    compatibility = {
        (job_id, cluster): _is_compatible(jobs[job_id], cluster, cluster_data)
        for job_id in job_ids
        for cluster in clusters
    }
    compatible_clusters = {
        job_id: [cluster for cluster in clusters if compatibility[(job_id, cluster)] == 1]
        for job_id in job_ids
    }
    jobs_without_compatible_cluster = [job_id for job_id, allowed_clusters in compatible_clusters.items() if not allowed_clusters]
    if jobs_without_compatible_cluster:
        raise ValueError(f"jobs have no compatible cluster: {jobs_without_compatible_cluster}")
    enforce_gpu_capacity = (
        enforce_gpu_constraints
        and any(jobs[job_id]["gpu_count_required"] > 0 for job_id in job_ids)
    )
    enforce_cpu_capacity = (
        enforce_cpu_constraints
        and any(jobs[job_id]["cpu_required"] > 0 for job_id in job_ids)
    )
    enforce_memory_capacity = (
        enforce_memory_constraints
        and any(jobs[job_id]["memory_required_gb"] > 0 for job_id in job_ids)
    )
    battery_enabled = config.battery_power_capacity > 0 and config.battery_energy_capacity > 0

    renewable_available = dict(zip(hours, hourly_df["renewable_available"].astype(float), strict=True))
    grid_price = dict(zip(hours, hourly_df["grid_price"].astype(float), strict=True))
    if "baseline_load" in hourly_df.columns:
        baseline_load = dict(zip(hours, hourly_df["baseline_load"].astype(float), strict=True))
    else:
        baseline_load = {hour: 0.0 for hour in hours}

    model = gp.Model(model_name)

    x = {
        (job_id, cluster, start): model.addVar(vtype=GRB.BINARY, name=f"x[{job_id},{cluster},{start}]")
        for job_id in job_ids
        for cluster in compatible_clusters[job_id]
        for start in feasible_starts[job_id]
    }
    R = {hour: model.addVar(lb=0.0, name=f"R[{hour}]") for hour in hours}
    Q = {hour: model.addVar(lb=0.0, name=f"Q[{hour}]") for hour in hours}
    B_charge = {
        hour: model.addVar(lb=0.0, ub=config.battery_power_capacity, name=f"B_charge[{hour}]")
        for hour in hours
    } if battery_enabled else {}
    B_discharge = {
        hour: model.addVar(lb=0.0, ub=config.battery_power_capacity, name=f"B_discharge[{hour}]")
        for hour in hours
    } if battery_enabled else {}
    B_soc = {
        hour: model.addVar(lb=0.0, ub=config.battery_energy_capacity, name=f"B_soc[{hour}]")
        for hour in hours
    } if battery_enabled else {}
    P_peak = model.addVar(lb=0.0, name="P_peak")
    P_peak_excess = model.addVar(lb=0.0, name="P_peak_excess")

    model.update()

    cluster_load: dict[tuple[str, int], gp.LinExpr] = {}
    cluster_gpu_load: dict[tuple[str, int], gp.LinExpr] = {}
    cluster_cpu_load: dict[tuple[str, int], gp.LinExpr] = {}
    cluster_memory_load: dict[tuple[str, int], gp.LinExpr] = {}
    flexible_load: dict[int, gp.LinExpr] = {}
    it_load: dict[int, gp.LinExpr | float] = {}
    total_load: dict[int, gp.LinExpr | float] = {}

    for cluster in clusters:
        for hour in hours:
            cluster_load[(cluster, hour)] = gp.quicksum(
                jobs[job_id]["power"] * x[(job_id, cluster, start)]
                for job_id in job_ids
                for start in feasible_starts[job_id]
                if (job_id, cluster, start) in x
                if is_active(start, jobs[job_id]["duration"], hour)
            )
            if enforce_gpu_capacity:
                cluster_gpu_load[(cluster, hour)] = gp.quicksum(
                    jobs[job_id]["gpu_count_required"] * x[(job_id, cluster, start)]
                    for job_id in job_ids
                    for start in feasible_starts[job_id]
                    if (job_id, cluster, start) in x
                    if is_active(start, jobs[job_id]["duration"], hour)
                )
            if enforce_cpu_capacity:
                cluster_cpu_load[(cluster, hour)] = gp.quicksum(
                    jobs[job_id]["cpu_required"] * x[(job_id, cluster, start)]
                    for job_id in job_ids
                    for start in feasible_starts[job_id]
                    if (job_id, cluster, start) in x
                    if is_active(start, jobs[job_id]["duration"], hour)
                )
            if enforce_memory_capacity:
                cluster_memory_load[(cluster, hour)] = gp.quicksum(
                    jobs[job_id]["memory_required_gb"] * x[(job_id, cluster, start)]
                    for job_id in job_ids
                    for start in feasible_starts[job_id]
                    if (job_id, cluster, start) in x
                    if is_active(start, jobs[job_id]["duration"], hour)
                )

    for hour in hours:
        flexible_load[hour] = gp.quicksum(
            jobs[job_id]["power"] * x[(job_id, cluster, start)]
            for job_id in job_ids
            for cluster in compatible_clusters[job_id]
            for start in feasible_starts[job_id]
            if is_active(start, jobs[job_id]["duration"], hour)
        )
        it_load[hour] = flexible_load[hour] + baseline_load[hour]
        total_load[hour] = config.pue * it_load[hour]

    # 1. Each job scheduled exactly once.
    for job_id in job_ids:
        model.addConstr(
            gp.quicksum(
                x[(job_id, cluster, start)]
                for cluster in compatible_clusters[job_id]
                for start in feasible_starts[job_id]
            )
            == 1,
            name=f"assign_once[{job_id}]",
        )

    for hour in hours:
        # 3. Energy-source balance. Renewable and grid usage are optimized
        # economically while all demand is served.
        battery_charge = B_charge[hour] if battery_enabled else 0.0
        battery_discharge = B_discharge[hour] if battery_enabled else 0.0
        model.addConstr(
            R[hour] + Q[hour] + battery_discharge == total_load[hour] + battery_charge,
            name=f"energy_balance[{hour}]",
        )

        # 4. Renewable availability limit.
        model.addConstr(R[hour] <= renewable_available[hour], name=f"renewable_available[{hour}]")

        # 5. Contracted power is modeled as a grid-import peak charge.
        model.addConstr(P_peak >= Q[hour], name=f"grid_peak[{hour}]")

        if battery_enabled:
            previous_soc = config.battery_initial_soc if hour == hours[0] else B_soc[hours[hours.index(hour) - 1]]
            model.addConstr(
                B_soc[hour]
                == previous_soc
                + config.battery_charge_efficiency * B_charge[hour] * config.delta_t
                - (B_discharge[hour] * config.delta_t) / config.battery_discharge_efficiency,
                name=f"battery_soc_balance[{hour}]",
            )

    if battery_enabled and config.battery_final_soc is not None:
        model.addConstr(B_soc[hours[-1]] >= config.battery_final_soc, name="battery_final_soc")

    # 6. Cluster-capacity constraint.
    for cluster in clusters:
        for hour in hours:
            model.addConstr(
                cluster_load[(cluster, hour)] <= cluster_data[cluster]["capacity"],
                name=f"cluster_capacity[{cluster},{hour}]",
            )
            if enforce_gpu_capacity:
                model.addConstr(
                    cluster_gpu_load[(cluster, hour)] <= cluster_data[cluster]["gpu_capacity"],
                    name=f"cluster_gpu_capacity[{cluster},{hour}]",
                )
            if enforce_cpu_capacity:
                model.addConstr(
                    cluster_cpu_load[(cluster, hour)] <= cluster_data[cluster]["cpu_capacity"],
                    name=f"cluster_cpu_capacity[{cluster},{hour}]",
                )
            if enforce_memory_capacity:
                model.addConstr(
                    cluster_memory_load[(cluster, hour)] <= cluster_data[cluster]["memory_capacity_gb"],
                    name=f"cluster_memory_capacity[{cluster},{hour}]",
                )

    renewable_cost = gp.quicksum(config.renewable_price * R[hour] * config.delta_t for hour in hours)
    grid_cost = gp.quicksum(grid_price[hour] * Q[hour] * config.delta_t for hour in hours)
    model.addConstr(P_peak_excess >= P_peak - config.contracted_power, name="peak_excess")
    peak_cost = config.peak_price * P_peak_excess
    model.setObjective(renewable_cost + grid_cost + peak_cost, GRB.MINIMIZE)

    variables = {
        "x": x,
        "R": R,
        "Q": Q,
        "B_charge": B_charge,
        "B_discharge": B_discharge,
        "B_soc": B_soc,
        "P_peak": P_peak,
        "P_peak_excess": P_peak_excess,
        "cluster_load": cluster_load,
        "cluster_gpu_load": cluster_gpu_load,
        "cluster_cpu_load": cluster_cpu_load,
        "cluster_memory_load": cluster_memory_load,
        "flexible_load": flexible_load,
        "it_load": it_load,
        "total_load": total_load,
        "baseline_load": baseline_load,
        "renewable_available": renewable_available,
        "feasible_starts": feasible_starts,
        "compatibility": compatibility,
        "compatible_clusters": compatible_clusters,
        "jobs": jobs,
        "cluster_data": cluster_data,
        "enforce_gpu_capacity": enforce_gpu_capacity,
        "enforce_cpu_capacity": enforce_cpu_capacity,
        "enforce_memory_capacity": enforce_memory_capacity,
        "battery_enabled": battery_enabled,
        "hours": hours,
        "clusters": clusters,
    }
    return model, variables
