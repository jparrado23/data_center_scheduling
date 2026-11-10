"""Reusable MILP execution and schedule extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from src.config import ModelConfig
from src.milp.gurobi_model import build_milp_model
from src.milp.solve import solve_model


@dataclass
class MilpResult:
    """Gurobi solve summary and its decoded assignment schedule."""

    schedule: pd.DataFrame
    runtime_s: float
    objective: float
    mip_gap: float
    status: int
    model: Any
    variables: dict[str, Any]


def extract_milp_schedule(variables: dict[str, Any], jobs_df: pd.DataFrame) -> pd.DataFrame:
    """Decode selected MILP assignment variables into the common schedule schema."""

    jobs = jobs_df.set_index("job_id")
    rows: list[dict[str, Any]] = []
    for (job_id, cluster, start), variable in variables["x"].items():
        if variable.X <= 0.5:
            continue
        job = jobs.loc[job_id]
        rows.append(
            {
                "job_id": job_id,
                "assigned_cluster": cluster,
                "start_hour": int(start),
                "duration": int(job.duration),
                "power": float(job.power),
                "gpu_count_required": float(job.gpu_count_required),
                "cpu_required": float(job.get("cpu_required", 0.0)),
                "memory_required_gb": float(job.get("memory_required_gb", 0.0)),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["start_hour", "job_id"]).reset_index(drop=True)
        if rows
        else pd.DataFrame(rows)
    )


def run_milp(
    jobs_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: ModelConfig,
    *,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    model_name: str = "scheduling_milp",
    enforce_gpu_constraints: bool = True,
    enforce_cpu_constraints: bool = False,
    enforce_memory_constraints: bool = False,
) -> MilpResult:
    """Build, solve, and decode one scheduling MILP instance."""

    started = perf_counter()
    model, variables = build_milp_model(
        jobs_df,
        hourly_df,
        clusters_df,
        config,
        model_name=model_name,
        enforce_gpu_constraints=enforce_gpu_constraints,
        enforce_cpu_constraints=enforce_cpu_constraints,
        enforce_memory_constraints=enforce_memory_constraints,
    )
    model.Params.OutputFlag = 0
    solve_model(model, time_limit=time_limit, mip_gap=mip_gap)
    return MilpResult(
        schedule=extract_milp_schedule(variables, jobs_df),
        runtime_s=perf_counter() - started,
        objective=float(model.ObjVal),
        mip_gap=float(model.MIPGap),
        status=int(model.Status),
        model=model,
        variables=variables,
    )
