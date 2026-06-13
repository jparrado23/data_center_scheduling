"""Import externally defined job-instance CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.validation import validate_jobs


SOURCE_JOB_COLUMNS = {
    "job_id",
    "tier",
    "gpus",
    "e_kw",
    "duration_h",
    "t_min",
    "t_max",
    "alpha_B",
    "alpha_C",
    "alpha_D",
}

TIER_TO_CATEGORY = {
    "finetune": "fine_tuning",
    "fine_tuning": "fine_tuning",
    "training": "training",
    "prep": "preprocessing",
    "preprocessing": "preprocessing",
    "inference": "inference",
}

ALPHA_COLUMNS = ("alpha_B", "alpha_C", "alpha_D")


def _normalize_tier(value: object) -> str:
    tier = str(value).strip().lower()
    if tier not in TIER_TO_CATEGORY:
        raise ValueError(f"unknown workload tier: {value!r}")
    return TIER_TO_CATEGORY[tier]


def _validate_source_schema(source_df: pd.DataFrame) -> None:
    missing = SOURCE_JOB_COLUMNS.difference(source_df.columns)
    if missing:
        raise ValueError(f"job instance CSV is missing columns: {sorted(missing)}")
    if source_df.empty:
        raise ValueError("job instance CSV must contain at least one job")


def load_job_instance_csv(
    path: str | Path,
    *,
    source_hours_are_one_based: bool = True,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Load a shared job-instance CSV into the model-ready jobs schema.

    The external files use `tier`, `e_kw`, `duration_h`, `t_min`, `t_max`, and
    per-cluster compatibility columns. The returned dataframe keeps those source
    fields where useful, while adding the normalized columns consumed by the
    solver: `category`, `duration`, `power`, `earliest_start`, and
    `latest_start`.
    """

    source_df = pd.read_csv(path)
    _validate_source_schema(source_df)

    hour_offset = 1 if source_hours_are_one_based else 0
    jobs_df = pd.DataFrame(
        {
            "job_id": source_df["job_id"].astype(str),
            "category": source_df["tier"].map(_normalize_tier),
            "tier": source_df["tier"].astype(str),
            "gpus": source_df["gpus"].astype(int),
            "duration": source_df["duration_h"].astype(int),
            "duration_h": source_df["duration_h"].astype(int),
            "power_kw": source_df["e_kw"].astype(float),
            "e_kw": source_df["e_kw"].astype(float),
            "power": source_df["e_kw"].astype(float) / 1000.0,
            "earliest_start": source_df["t_min"].astype(int) - hour_offset,
            "latest_start": source_df["t_max"].astype(int) - hour_offset,
        }
    )

    for column in ALPHA_COLUMNS:
        jobs_df[column] = source_df[column].astype(int)

    if (jobs_df["gpus"] <= 0).any():
        raise ValueError("all jobs must require at least one GPU")
    if (jobs_df["earliest_start"] < 0).any():
        raise ValueError("earliest_start cannot be negative after hour conversion")
    if ((jobs_df["latest_start"] + jobs_df["duration"]) > horizon_hours).any():
        raise ValueError("latest_start plus duration must fit inside the scheduling horizon")

    validate_jobs(jobs_df)
    return jobs_df


def write_imported_job_instance(
    source_path: str | Path,
    output_path: str | Path,
    *,
    source_hours_are_one_based: bool = True,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Convert a shared job-instance CSV and write the normalized jobs file."""

    jobs_df = load_job_instance_csv(
        source_path,
        source_hours_are_one_based=source_hours_are_one_based,
        horizon_hours=horizon_hours,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    jobs_df.to_csv(output, index=False)
    return jobs_df
