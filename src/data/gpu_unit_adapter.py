"""Helpers to select the GPU aggregation unit used by solvers."""

from __future__ import annotations

from typing import Literal

import pandas as pd


GpuUnitMode = Literal["node", "block"]


def adapt_gpu_unit(
    jobs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    mode: GpuUnitMode,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return solver-ready jobs and clusters for a GPU unit mode.

    Generated Pretrain instances keep both physical node units and reduced
    QUBO block units. Solvers consume the generic columns
    ``gpu_count_required``, ``gpu_capacity``, and ``gpu_count``. This adapter
    maps the selected representation into those generic columns.
    """

    if mode not in {"node", "block"}:
        raise ValueError("mode must be either 'node' or 'block'")

    jobs = jobs_df.copy()
    clusters = clusters_df.copy()

    if mode == "node":
        job_column = "node_units_required"
        capacity_column = "node_capacity"
    else:
        job_column = "block_units_required"
        capacity_column = "block_capacity"

    missing_job_columns = [column for column in ("gpu_count_required", job_column) if column not in jobs.columns]
    if missing_job_columns:
        raise ValueError(f"jobs_df is missing columns required for {mode!r} mode: {missing_job_columns}")

    missing_cluster_columns = [
        column for column in ("gpu_capacity", "gpu_count", capacity_column) if column not in clusters.columns
    ]
    if missing_cluster_columns:
        raise ValueError(f"clusters_df is missing columns required for {mode!r} mode: {missing_cluster_columns}")

    jobs["gpu_count_required"] = jobs[job_column].astype(int)
    clusters["gpu_capacity"] = clusters[capacity_column].astype(int)
    clusters["gpu_count"] = clusters[capacity_column].astype(int)

    return jobs, clusters
