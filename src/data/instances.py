"""Generate concrete job instances from job-type catalogs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_JOB_COUNTS = {
    "inference": 8,
    "fine_tuning": 15,
    "training": 20,
    "preprocessing": 15,
}


def _sample_received_hour(rng: np.random.Generator, duration: int, max_start_delay: int, horizon_hours: int) -> int:
    """Sample a received hour that leaves at least one feasible start time."""

    latest_received = max(0, horizon_hours - duration - max_start_delay)
    return int(rng.integers(0, latest_received + 1))


def generate_jobs_from_catalog(
    job_types_df: pd.DataFrame,
    counts: dict[str, int] | None = None,
    horizon_hours: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate one concrete `jobs.csv` instance from job-type ranges."""

    rng = np.random.default_rng(seed)
    counts = counts or DEFAULT_JOB_COUNTS
    catalog = job_types_df.set_index("job_type")
    jobs = []

    for job_type, count in counts.items():
        spec = catalog.loc[job_type]
        max_start_delay = int(spec.max_start_delay_hours)

        for idx in range(1, count + 1):
            duration = int(rng.integers(int(spec.duration_min_hours), int(spec.duration_max_hours) + 1))
            power_kw = float(rng.uniform(float(spec.power_min_kw), float(spec.power_max_kw)))

            if job_type == "inference":
                received_hour = 0
                earliest_start = 0
                latest_start = 0
            elif job_type == "preprocessing":
                received_hour = int(rng.integers(0, horizon_hours - duration + 1))
                earliest_start = received_hour
                latest_start = horizon_hours - duration
            else:
                received_hour = _sample_received_hour(rng, duration, max_start_delay, horizon_hours)
                earliest_start = received_hour
                latest_start = min(received_hour + max_start_delay, horizon_hours - duration)

            jobs.append(
                {
                    "job_id": f"{job_type}_{idx:02d}",
                    "category": job_type,
                    "received_hour": received_hour,
                    "duration": duration,
                    "power_kw": round(power_kw, 4),
                    "power": round(power_kw / 1000.0, 8),
                    "earliest_start": earliest_start,
                    "latest_start": latest_start,
                }
            )

    return pd.DataFrame(jobs)


def write_processed_jobs(
    data_dir: str | Path = "data/processed",
    counts: dict[str, int] | None = None,
    horizon_hours: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate and write `data/processed/jobs.csv`."""

    data_path = Path(data_dir)
    job_types_df = pd.read_csv(data_path / "job_types.csv")
    jobs_df = generate_jobs_from_catalog(job_types_df, counts=counts, horizon_hours=horizon_hours, seed=seed)
    jobs_df.to_csv(data_path / "jobs.csv", index=False)
    return jobs_df


if __name__ == "__main__":
    write_processed_jobs()
