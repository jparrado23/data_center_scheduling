"""Physical validation and duplicate-only repair for decoded PCE schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.quantum.benchmarking import evaluate_physical_schedule
from src.quantum.pce_diagnostics import PcePenaltyTrace
from src.quantum.qubo_builder import validate_decoded_schedule


@dataclass
class PceScheduleSelection:
    """A direct or repaired schedule selected from one fixed-alpha trace."""

    status: str
    schedule: pd.DataFrame
    source_evaluation: int | None
    direct_candidate_count: int
    repairable_candidate_count: int
    removed_assignments: int
    physical_gpu_excess: float
    physical_metrics: dict[str, float | None] | None


def restore_physical_gpu_units(
    schedule_df: pd.DataFrame,
    raw_jobs_df: pd.DataFrame,
    raw_clusters_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restore raw GPU requirements and capacities after node/block optimization."""

    required_job_columns = {"job_id", "raw_gpu_num"}
    required_cluster_columns = {"cluster_id", "raw_gpu_count"}
    missing_jobs = required_job_columns.difference(raw_jobs_df.columns)
    missing_clusters = required_cluster_columns.difference(raw_clusters_df.columns)
    if missing_jobs:
        raise ValueError(f"raw jobs are missing physical GPU columns: {sorted(missing_jobs)}")
    if missing_clusters:
        raise ValueError(
            f"raw clusters are missing physical GPU columns: {sorted(missing_clusters)}"
        )

    jobs = raw_jobs_df.copy()
    clusters = raw_clusters_df.copy()
    schedule = schedule_df.copy()
    jobs["gpu_count_required"] = jobs["raw_gpu_num"].astype(float)
    clusters["gpu_capacity"] = clusters["raw_gpu_count"].astype(float)
    clusters["gpu_count"] = clusters["raw_gpu_count"].astype(float)
    if not schedule.empty:
        requirements = jobs.set_index("job_id")["gpu_count_required"]
        unknown = sorted(set(schedule["job_id"]) - set(requirements.index))
        if unknown:
            raise ValueError(f"schedule contains unknown jobs: {unknown}")
        schedule["gpu_count_required"] = schedule["job_id"].map(requirements).astype(float)
    return schedule, jobs, clusters


def physical_gpu_summary(
    schedule_df: pd.DataFrame,
    raw_jobs_df: pd.DataFrame,
    raw_clusters_df: pd.DataFrame,
    hours: Sequence[int],
) -> dict[str, object]:
    """Summarize assignment counts and raw-GPU capacity violations."""

    schedule, jobs, clusters = restore_physical_gpu_units(
        schedule_df, raw_jobs_df, raw_clusters_df
    )
    expected_jobs = set(jobs["job_id"])
    counts = (
        schedule["job_id"].value_counts()
        if not schedule.empty
        else pd.Series(dtype=int)
    )
    missing = sorted(expected_jobs - set(counts.index))
    duplicate_assignments = int((counts - 1).clip(lower=0).sum())

    capacities = clusters.set_index("cluster_id")["raw_gpu_count"].astype(float)
    max_excess = 0.0
    violated_slots = 0
    for cluster_id, capacity in capacities.items():
        selected = schedule[schedule["assigned_cluster"] == cluster_id]
        for hour in hours:
            active = selected[
                (selected["start_hour"] <= hour)
                & (hour < selected["start_hour"] + selected["duration"])
            ]
            excess = max(0.0, float(active["gpu_count_required"].sum()) - capacity)
            max_excess = max(max_excess, excess)
            violated_slots += int(excess > 1e-8)

    validation = validate_decoded_schedule(
        schedule, jobs, clusters, [int(hour) for hour in hours]
    )
    return {
        "missing_jobs": missing,
        "missing_job_count": len(missing),
        "duplicate_assignments": duplicate_assignments,
        "physical_gpu_max_excess": max_excess,
        "physical_gpu_violated_slots": violated_slots,
        "physical_gpu_feasible": violated_slots == 0,
        "fully_feasible": bool(validation["feasible"]),
        "validation": validation,
        "physical_schedule": schedule,
    }


def _assignment_effective_costs(
    schedule_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    *,
    delta_t: float,
) -> pd.Series:
    """Score each selected assignment with the QUBO effective-price signal."""

    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    price_column = "effective_price" if "effective_price" in energy_df else "grid_price"
    lookup = energy_df.set_index("hour")
    scores: dict[int, float] = {}
    for index, row in schedule_df.iterrows():
        score = 0.0
        for hour in range(int(row.start_hour), int(row.start_hour + row.duration)):
            if hour not in lookup.index:
                raise ValueError(f"energy profile is missing hour {hour}")
            hourly = lookup.loc[hour]
            score += (
                float(row.power)
                * float(hourly.get("pue", 1.0))
                * float(hourly[price_column])
                * delta_t
            )
        scores[index] = score
    return pd.Series(scores, dtype=float)


def repair_duplicate_only_schedule(
    schedule_df: pd.DataFrame,
    raw_jobs_df: pd.DataFrame,
    raw_clusters_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    hours: Sequence[int],
    *,
    delta_t: float,
) -> pd.DataFrame:
    """Keep the cheapest assignment per job for a duplicate-only candidate."""

    before = physical_gpu_summary(schedule_df, raw_jobs_df, raw_clusters_df, hours)
    if before["missing_job_count"]:
        raise ValueError("duplicate-only repair cannot add missing jobs")
    if not before["physical_gpu_feasible"]:
        raise ValueError("candidate exceeds physical GPU capacity before repair")
    if not before["duplicate_assignments"]:
        return before["physical_schedule"].copy()

    physical_schedule = before["physical_schedule"].copy()
    physical_schedule["repair_score"] = _assignment_effective_costs(
        physical_schedule, energy_df, delta_t=delta_t
    )
    tie_breakers = ["repair_score", "start_hour"]
    if "qubo_variable" in physical_schedule:
        tie_breakers.append("qubo_variable")
    repaired = (
        physical_schedule.sort_values(tie_breakers)
        .drop_duplicates("job_id", keep="first")
        .drop(columns="repair_score")
        .sort_values(["start_hour", "job_id"])
        .reset_index(drop=True)
    )
    after = physical_gpu_summary(repaired, raw_jobs_df, raw_clusters_df, hours)
    if not after["fully_feasible"]:
        raise RuntimeError(
            f"duplicate-only repair produced an invalid schedule: {after['validation']['violations']}"
        )
    return repaired


def trace_physical_diagnostics(
    trace: PcePenaltyTrace,
    raw_jobs_df: pd.DataFrame,
    raw_clusters_df: pd.DataFrame,
    hours: Sequence[int],
) -> pd.DataFrame:
    """Classify every decoded theta evaluation by its physical failure mode."""

    if not hasattr(trace, "candidates"):
        raise ValueError(
            "the PCE trace does not retain per-evaluation candidates; regenerate it "
            "with the current run_fixed_alpha_penalty_trace implementation"
        )
    rows: list[dict[str, object]] = []
    for evaluation, candidate in enumerate(trace.candidates, start=1):
        summary = physical_gpu_summary(
            candidate.schedule, raw_jobs_df, raw_clusters_df, hours
        )
        missing = int(summary["missing_job_count"])
        duplicates = int(summary["duplicate_assignments"])
        gpu_feasible = bool(summary["physical_gpu_feasible"])
        if missing > 0:
            category = "missing_jobs"
        elif not gpu_feasible:
            category = "physical_gpu_violation"
        elif duplicates > 0:
            category = "duplicate_only_repairable"
        elif bool(summary["fully_feasible"]):
            category = "direct_feasible"
        else:
            category = "other_physical_violation"
        trace_row = trace.evaluations.iloc[evaluation - 1]
        rows.append(
            {
                "evaluation": evaluation,
                "evaluation_type": str(
                    trace_row.get("evaluation_type", "optimizer")
                ),
                "category": category,
                "missing_jobs": missing,
                "duplicate_assignments": duplicates,
                "assignment_errors": missing + duplicates,
                "physical_gpu_feasible": gpu_feasible,
                "physical_gpu_violated_slots": int(
                    summary["physical_gpu_violated_slots"]
                ),
                "physical_gpu_max_excess": float(summary["physical_gpu_max_excess"]),
                "pce_loss": float(candidate.pce_loss),
                "qubo_energy": float(candidate.qubo_energy),
                "relaxed_energy": float(trace_row["relaxed_energy"]),
                "relaxed_assignment": float(trace_row["relaxed_assignment"]),
                "relaxed_gpu": float(trace_row["relaxed_gpu"]),
            }
        )
    diagnostics = pd.DataFrame(rows)
    if not diagnostics.empty:
        diagnostics["running_min_missing_jobs"] = diagnostics["missing_jobs"].cummin()
        diagnostics["running_min_assignment_errors"] = diagnostics[
            "assignment_errors"
        ].cummin()
    return diagnostics


def select_direct_or_repaired_schedule(
    trace: PcePenaltyTrace,
    raw_jobs_df: pd.DataFrame,
    raw_clusters_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    hours: Sequence[int],
    *,
    delta_t: float,
) -> PceScheduleSelection:
    """Prefer direct feasibility, then the cheapest duplicate-only repair."""

    if not hasattr(trace, "candidates"):
        raise ValueError(
            "the PCE trace does not retain per-evaluation candidates; regenerate it "
            "with the current run_fixed_alpha_penalty_trace implementation"
        )
    direct: list[tuple[float, int, pd.DataFrame, dict[str, float | None]]] = []
    repairable: list[tuple[float, int, pd.DataFrame, dict[str, float | None], int]] = []
    for evaluation, candidate in enumerate(trace.candidates, start=1):
        summary = physical_gpu_summary(
            candidate.schedule, raw_jobs_df, raw_clusters_df, hours
        )
        if summary["fully_feasible"]:
            schedule = summary["physical_schedule"]
            _, metrics = evaluate_physical_schedule(
                schedule, energy_df, delta_t=delta_t
            )
            direct.append((float(metrics["physical_total_cost"]), evaluation, schedule, metrics))
            continue
        if (
            summary["missing_job_count"] == 0
            and summary["physical_gpu_feasible"]
            and summary["duplicate_assignments"] > 0
        ):
            schedule = repair_duplicate_only_schedule(
                candidate.schedule,
                raw_jobs_df,
                raw_clusters_df,
                energy_df,
                hours,
                delta_t=delta_t,
            )
            _, metrics = evaluate_physical_schedule(
                schedule, energy_df, delta_t=delta_t
            )
            repairable.append(
                (
                    float(metrics["physical_total_cost"]),
                    evaluation,
                    schedule,
                    metrics,
                    int(summary["duplicate_assignments"]),
                )
            )

    if direct:
        _, evaluation, schedule, metrics = min(direct, key=lambda item: item[0])
        return PceScheduleSelection(
            "direct", schedule, evaluation, len(direct), len(repairable), 0, 0.0, metrics
        )
    if repairable:
        _, evaluation, schedule, metrics, removed = min(repairable, key=lambda item: item[0])
        return PceScheduleSelection(
            "repaired", schedule, evaluation, 0, len(repairable), removed, 0.0, metrics
        )
    return PceScheduleSelection(
        "none", pd.DataFrame(), None, 0, 0, 0, float("nan"), None
    )
