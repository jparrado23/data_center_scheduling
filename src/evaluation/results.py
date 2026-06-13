"""Extract readable results from solved MILP models."""

from __future__ import annotations

from typing import Any

import pandas as pd


def extract_schedule(jobs_df: pd.DataFrame, variables: dict[str, Any], tolerance: float = 0.5) -> pd.DataFrame:
    """Convert the solved assignment variables into a readable schedule table.

    Parameters
    ----------
    jobs_df:
        Original jobs table used to recover job attributes.
    variables:
        Dictionary returned by the model builder, including the binary `x`
        variables.
    tolerance:
        Threshold used to treat a binary variable as selected.

    Returns
    -------
    pd.DataFrame
        A schedule with job identifiers, assigned clusters, start hours, and
        copied job attributes sorted by execution time.
    """

    selected_rows = []
    job_info = {str(row.job_id): row for row in jobs_df.itertuples(index=False)}

    for (job_id, cluster, start), var in variables["x"].items():
        if var.X > tolerance:
            job = job_info[job_id]
            selected_rows.append(
                {
                    "job_id": job_id,
                    "category": str(job.category),
                    "assigned_cluster": cluster,
                    "start_hour": start,
                    "duration": int(job.duration),
                    "power": float(job.power),
                }
            )

    return pd.DataFrame(selected_rows).sort_values(["start_hour", "job_id"]).reset_index(drop=True)


def extract_hourly_results(hourly_df: pd.DataFrame, variables: dict[str, Any]) -> pd.DataFrame:
    """Build a per-hour result table from solved model variables.

    The output combines the original hourly inputs with the optimized flexible
    load, optional fixed baseline load, total facility load, renewable
    consumption, curtailment, and grid residual demand.
    """

    rows = []
    hourly_lookup = hourly_df.set_index("hour")
    for hour in variables["hours"]:
        flexible = variables["flexible_load"][hour].getValue()
        total = variables["total_load"][hour].getValue()
        renewable_available = float(hourly_lookup.loc[hour, "renewable_available"])
        renewable_consumption = variables["R"][hour].X
        rows.append(
            {
                "hour": hour,
                "baseline_load": float(variables["baseline_load"][hour]),
                "flexible_load": flexible,
                "total_load": total,
                "renewable_available": renewable_available,
                "renewable_consumption": renewable_consumption,
                "renewable_curtailment": max(0.0, renewable_available - renewable_consumption),
                "grid_consumption": variables["Q"][hour].X,
                "grid_price": float(hourly_lookup.loc[hour, "grid_price"]),
            }
        )
    return pd.DataFrame(rows)


def extract_cluster_hourly_results(variables: dict[str, Any]) -> pd.DataFrame:
    """Build a per-cluster, per-hour load table from solved expressions."""

    rows = []
    for cluster in variables["clusters"]:
        capacity = variables["cluster_data"][cluster]["capacity"]
        gpu_capacity = variables["cluster_data"][cluster].get("gpu_capacity")
        for hour in variables["hours"]:
            row = {
                "cluster_id": cluster,
                "hour": hour,
                "cluster_load": variables["cluster_load"][(cluster, hour)].getValue(),
                "capacity": capacity,
            }
            if gpu_capacity is not None and (cluster, hour) in variables["cluster_gpu_load"]:
                row["cluster_gpu_load"] = variables["cluster_gpu_load"][(cluster, hour)].getValue()
                row["gpu_capacity"] = gpu_capacity
            rows.append(row)
    return pd.DataFrame(rows)
