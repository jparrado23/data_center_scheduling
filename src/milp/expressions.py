"""Small helpers that mirror the MILP mathematical formulation."""

from __future__ import annotations

import pandas as pd


def is_active(start: int, duration: int, hour: int) -> bool:
    """Check whether a job is active during a given hour.

    A job is considered active from its start hour through the final hour of
    its duration, inclusive.
    """

    return start <= hour <= start + duration - 1


def build_feasible_starts(jobs_df: pd.DataFrame) -> dict[str, list[int]]:
    """Build the feasible start-hour set for each job.

    The returned dictionary maps each job identifier to the inclusive range of
    start hours allowed by its earliest and latest start constraints.
    """

    feasible_starts: dict[str, list[int]] = {}
    for row in jobs_df.itertuples(index=False):
        feasible_starts[str(row.job_id)] = list(range(int(row.earliest_start), int(row.latest_start) + 1))
    return feasible_starts
