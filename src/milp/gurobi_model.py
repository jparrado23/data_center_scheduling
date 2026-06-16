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


def _job_lookup(jobs_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Normalize the jobs table into a dictionary keyed by job identifier.

    The helper strips the model down to the fields repeatedly accessed during
    constraint construction so the MILP builder can avoid repeated dataframe
    lookups inside nested loops.
    """

    jobs: dict[str, dict[str, Any]] = {}
    for row in jobs_df.itertuples(index=False):
        job = {
            "category": str(row.category),
            "duration": int(row.duration),
            "power": float(row.power),
            "earliest_start": int(row.earliest_start),
            "latest_start": int(row.latest_start),
        }
        if hasattr(row, "gpus"):
            job["gpus"] = int(row.gpus)
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
        cluster = {
            "capacity": float(row.capacity),
            "compatible_categories": _normalize_categories(row.compatible_categories),
        }
        if hasattr(row, "gpu_capacity"):
            cluster["gpu_capacity"] = int(row.gpu_capacity)
        clusters[str(row.cluster_id)] = cluster
    return clusters


def _is_compatible(job: dict[str, Any], cluster: str, cluster_data: dict[str, Any]) -> int:
    """Return compatibility using job-level alpha columns when available."""

    if cluster in job:
        return int(job[cluster])
    return int(job["category"] in cluster_data[cluster]["compatible_categories"])


def build_milp_model(
    jobs_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: ModelConfig,
    model_name: str = "energy_ai_datacenter_milp",
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
    enforce_gpu_capacity = all("gpus" in jobs[job_id] for job_id in job_ids) and all(
        "gpu_capacity" in cluster_data[cluster] for cluster in clusters
    )

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
    P_peak = model.addVar(lb=0.0, name="P_peak")
    P_peak_excess = model.addVar(lb=0.0, name="P_peak_excess")

    model.update()

    cluster_load: dict[tuple[str, int], gp.LinExpr] = {}
    cluster_gpu_load: dict[tuple[str, int], gp.LinExpr] = {}
    flexible_load: dict[int, gp.LinExpr] = {}
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
                    jobs[job_id]["gpus"] * x[(job_id, cluster, start)]
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
        total_load[hour] = flexible_load[hour] + baseline_load[hour]

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
        model.addConstr(R[hour] + Q[hour] == total_load[hour], name=f"energy_balance[{hour}]")

        # 4. Renewable availability limit.
        model.addConstr(R[hour] <= renewable_available[hour], name=f"renewable_available[{hour}]")

        # 5. Peak-load relationship.
        model.addConstr(P_peak >= total_load[hour], name=f"peak_load[{hour}]")

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

    renewable_cost = gp.quicksum(config.renewable_price * R[hour] * config.delta_t for hour in hours)
    grid_cost = gp.quicksum(grid_price[hour] * Q[hour] * config.delta_t for hour in hours)
    model.addConstr(P_peak_excess >= P_peak - config.contracted_power, name="peak_excess")
    peak_cost = config.peak_price * P_peak_excess
    model.setObjective(renewable_cost + grid_cost + peak_cost, GRB.MINIMIZE)

    variables = {
        "x": x,
        "R": R,
        "Q": Q,
        "P_peak": P_peak,
        "P_peak_excess": P_peak_excess,
        "cluster_load": cluster_load,
        "cluster_gpu_load": cluster_gpu_load,
        "flexible_load": flexible_load,
        "total_load": total_load,
        "baseline_load": baseline_load,
        "renewable_available": renewable_available,
        "feasible_starts": feasible_starts,
        "compatibility": compatibility,
        "compatible_clusters": compatible_clusters,
        "jobs": jobs,
        "cluster_data": cluster_data,
        "hours": hours,
        "clusters": clusters,
    }
    return model, variables
