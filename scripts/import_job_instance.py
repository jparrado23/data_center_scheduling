"""Convert a shared job-instance CSV into the model-ready jobs schema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.job_instances import write_imported_job_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path, help="Path to jobs_light/jobs_tense/jobs_limit style CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/jobs.csv"),
        help="Path for the normalized jobs.csv output.",
    )
    parser.add_argument(
        "--zero-based-source-hours",
        action="store_true",
        help="Use this only if t_min/t_max are already 0-based model hours.",
    )
    parser.add_argument("--horizon-hours", type=int, default=24, help="Scheduling horizon length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_df = write_imported_job_instance(
        args.source_csv,
        args.output,
        source_hours_are_one_based=not args.zero_based_source_hours,
        horizon_hours=args.horizon_hours,
    )
    print(f"Wrote {len(jobs_df)} jobs to {args.output}")


if __name__ == "__main__":
    main()
