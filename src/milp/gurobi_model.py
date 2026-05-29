"""Gurobi MILP model for AI data-center energy scheduling."""

from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from src.config import ModelConfig
from src.data.validation import validate_clusters, validate_hourly_inputs, validate_jobs
from src.milp.expressions import build_feasible_starts, is_active


def _job_lookup(jobs_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Normalize the jobs table into a dictionary keyed by job identifier.

    The helper strips the model down to the fields repeatedly accessed during
    constraint construction so the MILP builder can avoid repeated dataframe
    lookups inside nested loops.
    """

    return {
        str(row.job_id): {
            "category": str(row.category),
            "duration": int(row.duration),
            "power": float(row.power),
            "earliest_start": int(row.earliest_start),
            "latest_start": int(row.latest_start),
        }
        for row in jobs_df.itertuples(index=False)
    }


def _normalize_categories(value: Any) -> set[str]:
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in value}


def _cluster_lookup(clusters_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row.cluster_id): {
            "capacity": float(row.capacity),
            "compatible_categories": _normalize_categories(row.compatible_categories),
        }
        for row in clusters_df.itertuples(index=False)
    }


def build_milp_model(
    jobs_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: ModelConfig,
    model_name: str = "energy_ai_datacenter_milp",
) -> tuple[gp.Model, dict[str, Any]]:
    """Build the full scheduling MILP and return the model plus helper objects.

    The model schedules each job exactly once across clusters and feasible start
    times, balances renewable and grid supply against total load, enforces the
    heterogeneous cluster capacities and contracted-power limits, and minimizes
    operating plus peak costs. The companion dictionary exposes the model
    variables and derived expressions used by result-extraction helpers.
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
        (job_id, cluster): int(jobs[job_id]["category"] in cluster_data[cluster]["compatible_categories"])
        for job_id in job_ids
        for cluster in clusters
    }

    renewable_available = dict(zip(hours, hourly_df["renewable_available"].astype(float), strict=True))
    grid_price = dict(zip(hours, hourly_df["grid_price"].astype(float), strict=True))

    model = gp.Model(model_name)

    x = {
        (job_id, cluster, start): model.addVar(vtype=GRB.BINARY, name=f"x[{job_id},{cluster},{start}]")
        for job_id in job_ids
        for cluster in clusters
        for start in feasible_starts[job_id]
    }
    R = {hour: model.addVar(lb=0.0, name=f"R[{hour}]") for hour in hours}
    Q = {hour: model.addVar(lb=0.0, name=f"Q[{hour}]") for hour in hours}
    P_peak = model.addVar(lb=0.0, name="P_peak")

    model.update()

    cluster_load: dict[tuple[str, int], gp.LinExpr] = {}
    total_load: dict[int, gp.LinExpr] = {}

    for cluster in clusters:
        for hour in hours:
            cluster_load[(cluster, hour)] = gp.quicksum(
                jobs[job_id]["power"] * x[(job_id, cluster, start)]
                for job_id in job_ids
                for start in feasible_starts[job_id]
                if is_active(start, jobs[job_id]["duration"], hour)
            )

    for hour in hours:
        total_load[hour] = gp.quicksum(
            jobs[job_id]["power"] * x[(job_id, cluster, start)]
            for cluster in clusters
            for job_id in job_ids
            for start in feasible_starts[job_id]
            if is_active(start, jobs[job_id]["duration"], hour)
        )

    # 1. Each job scheduled exactly once.
    for job_id in job_ids:
        model.addConstr(
            gp.quicksum(x[(job_id, cluster, start)] for cluster in clusters for start in feasible_starts[job_id]) == 1,
            name=f"assign_once[{job_id}]",
        )

    # 2. Cluster compatibility: jobs can only run on compatible clusters.
    for job_id in job_ids:
        for cluster in clusters:
            for start in feasible_starts[job_id]:
                model.addConstr(
                    x[(job_id, cluster, start)] <= compatibility[(job_id, cluster)],
                    name=f"compatibility[{job_id},{cluster},{start}]",
                )

    for hour in hours:
        # 3. Energy-source balance.
        model.addConstr(R[hour] + Q[hour] == total_load[hour], name=f"energy_balance[{hour}]")

        # 4. Renewable availability.
        model.addConstr(R[hour] <= renewable_available[hour], name=f"renewable_available[{hour}]")

        # 5. Peak-load relationship.
        model.addConstr(P_peak >= total_load[hour], name=f"peak_load[{hour}]")

        # 7. Total contracted-power constraint.
        model.addConstr(total_load[hour] <= config.contracted_power, name=f"contracted_power[{hour}]")

    # 6. Cluster-capacity constraint.
    for cluster in clusters:
        for hour in hours:
            model.addConstr(
                cluster_load[(cluster, hour)] <= cluster_data[cluster]["capacity"],
                name=f"cluster_capacity[{cluster},{hour}]",
            )

    energy_cost = gp.quicksum(
        (config.renewable_price * R[hour] + grid_price[hour] * Q[hour]) * config.delta_t for hour in hours
    )
    peak_cost = config.peak_price * P_peak
    model.setObjective(energy_cost + peak_cost, GRB.MINIMIZE)

    variables = {
        "x": x,
        "R": R,
        "Q": Q,
        "P_peak": P_peak,
        "cluster_load": cluster_load,
        "total_load": total_load,
        "feasible_starts": feasible_starts,
        "compatibility": compatibility,
        "jobs": jobs,
        "cluster_data": cluster_data,
        "hours": hours,
        "clusters": clusters,
    }
    return model, variables
