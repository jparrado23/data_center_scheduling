"""Validation helpers for model input data."""

from __future__ import annotations

import pandas as pd


JOB_COLUMNS = {"job_id", "category", "duration", "power", "earliest_start", "latest_start"}
CLUSTER_COLUMNS = {"cluster_id", "capacity", "compatible_categories"}
HOURLY_COLUMNS = {"hour", "renewable_available", "grid_price"}
OPTIONAL_HOURLY_NONNEGATIVE_COLUMNS = {"baseline_load"}


def validate_jobs(jobs_df: pd.DataFrame) -> None:
    """Validate the jobs table used by the scheduling model.

    The input must contain the expected columns, at least one job, positive
    duration and power values, and a feasible start window for every row.
    """

    missing = JOB_COLUMNS.difference(jobs_df.columns)
    if missing:
        raise ValueError(f"jobs_df is missing columns: {sorted(missing)}")
    if jobs_df.empty:
        raise ValueError("jobs_df must contain at least one job")
    if (jobs_df["duration"] <= 0).any():
        raise ValueError("all job durations must be positive")
    if (jobs_df["power"] <= 0).any():
        raise ValueError("all job powers must be positive")
    if "gpus" in jobs_df.columns and (jobs_df["gpus"] <= 0).any():
        raise ValueError("all GPU requirements must be positive")
    if (jobs_df["earliest_start"] > jobs_df["latest_start"]).any():
        raise ValueError("earliest_start must be <= latest_start for every job")
    if jobs_df["category"].isna().any() or (jobs_df["category"].astype(str).str.len() == 0).any():
        raise ValueError("every job must have a non-empty category")


def validate_clusters(clusters_df: pd.DataFrame) -> None:
    """Validate heterogeneous cluster capacities and category compatibility."""

    missing = CLUSTER_COLUMNS.difference(clusters_df.columns)
    if missing:
        raise ValueError(f"clusters_df is missing columns: {sorted(missing)}")
    if clusters_df.empty:
        raise ValueError("clusters_df must contain at least one cluster")
    if (clusters_df["capacity"] <= 0).any():
        raise ValueError("all cluster capacities must be positive")
    if "gpu_capacity" in clusters_df.columns and (clusters_df["gpu_capacity"] <= 0).any():
        raise ValueError("all cluster GPU capacities must be positive")


def validate_hourly_inputs(hourly_df: pd.DataFrame) -> None:
    """Validate the hourly exogenous inputs used by the MILP model.

    The frame must provide renewable availability and grid-price columns,
    contain at least one hour, and keep physical power quantities non-negative.
    `grid_price` may be negative in high-renewable market conditions.
    """

    missing = HOURLY_COLUMNS.difference(hourly_df.columns)
    if missing:
        raise ValueError(f"hourly_df is missing columns: {sorted(missing)}")
    if hourly_df.empty:
        raise ValueError("hourly_df must contain at least one hour")
    nonnegative_columns = ["renewable_available"]
    nonnegative_columns.extend(
        column for column in OPTIONAL_HOURLY_NONNEGATIVE_COLUMNS if column in hourly_df.columns
    )
    if (hourly_df[nonnegative_columns] < 0).any().any():
        raise ValueError("renewable availability and baseline load must be non-negative")
