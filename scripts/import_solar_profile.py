"""Build monthly hourly solar profiles from a PVGIS time-series CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.solar_profiles import (
    write_daily_hourly_inputs,
    write_monthly_hourly_inputs,
    write_monthly_hourly_solar_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pvgis_csv", type=Path, help="Path to the PVGIS time-series CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/monthly_solar_profiles.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--month",
        type=int,
        help="If provided, write a 24-row hourly_inputs.csv-style file for this month.",
    )
    parser.add_argument(
        "--date",
        help="If provided, write a 24-row hourly_inputs.csv-style file for this actual date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--existing-hourly-inputs",
        type=Path,
        help="Optional hourly_inputs.csv whose renewable_available column should be replaced.",
    )
    parser.add_argument("--base-capacity-kwp", type=float, default=250.0, help="PVGIS source system size.")
    parser.add_argument(
        "--target-capacity-kwp",
        type=float,
        help="Optional target PV system size. Profiles are scaled linearly from the base capacity.",
    )
    parser.add_argument(
        "--grid-price",
        type=float,
        default=0.0,
        help="Grid price used only when creating a new month-specific hourly_inputs.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.month is not None and args.date is not None:
        raise ValueError("use either --month or --date, not both")
    if args.date is not None:
        hourly_df = write_daily_hourly_inputs(
            args.pvgis_csv,
            args.output,
            args.date,
            existing_hourly_inputs_path=args.existing_hourly_inputs,
            base_capacity_kwp=args.base_capacity_kwp,
            target_capacity_kwp=args.target_capacity_kwp,
            grid_price=args.grid_price,
        )
        print(f"Wrote {len(hourly_df)} hourly rows for {args.date} to {args.output}")
        return

    if args.month is None:
        profile_df = write_monthly_hourly_solar_profile(
            args.pvgis_csv,
            args.output,
            base_capacity_kwp=args.base_capacity_kwp,
            target_capacity_kwp=args.target_capacity_kwp,
        )
        print(f"Wrote {len(profile_df)} monthly hourly solar rows to {args.output}")
        return

    hourly_df = write_monthly_hourly_inputs(
        args.pvgis_csv,
        args.output,
        args.month,
        existing_hourly_inputs_path=args.existing_hourly_inputs,
        base_capacity_kwp=args.base_capacity_kwp,
        target_capacity_kwp=args.target_capacity_kwp,
        grid_price=args.grid_price,
    )
    print(f"Wrote {len(hourly_df)} hourly rows for month {args.month} to {args.output}")


if __name__ == "__main__":
    main()
